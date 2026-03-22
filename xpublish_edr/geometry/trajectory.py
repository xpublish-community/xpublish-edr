"""
Handle selection and formatting for trajectory queries
"""

from __future__ import annotations

from typing import Literal

import geopandas as gpd
import numpy as np
import shapely
import xarray as xr
from rasterix.rasterize import rasterize

from xpublish_edr.geometry.common import (
    VECTORIZED_DIM,
    dataset_crs,
    is_regular_xy_coords,
)
from xpublish_edr.logger import logger


def select_by_trajectory(
    ds: xr.Dataset,
    line: shapely.LineString | shapely.MultiLineString,
    method: Literal["nearest", "linear"] = "nearest",
) -> xr.Dataset:
    """
    Return a dataset with grid cells intersected by the line, ordered by distance
    along the line through cell center points (stable tie-break by index order).
    """
    if not is_regular_xy_coords(ds):
        # TODO: Handle 2D coordinates
        raise NotImplementedError("Only  2D coordinates are supported")

    if line.is_empty:
        return _empty_trajectory_dataset(ds)

    ds_sub = _subset_to_line_bbox(ds, line)
    if ds_sub is None:
        return _empty_trajectory_dataset(ds)

    x_dim = ds.cf["X"].dims[0]
    y_dim = ds.cf["Y"].dims[0]

    data_crs = dataset_crs(ds_sub)
    gdf = gpd.GeoDataFrame(geometry=[line], crs=data_crs)

    try:
        y_nz, x_nz = _intersected_cell_indices(ds_sub, gdf, x_dim, y_dim)
    except Exception as e:
        logger.error("Rasterize failed for trajectory query: %s", e, exc_info=True)
        raise KeyError(
            "Trajectory rasterization failed; check CRS, grid metadata, and geometry.",
        ) from e

    return _sort_cells_along_line(ds_sub, y_nz, x_nz, line, x_dim, y_dim, method)


def _subset_to_line_bbox(
    ds: xr.Dataset,
    line: shapely.LineString | shapely.MultiLineString,
) -> xr.Dataset | None:
    """
    Return a spatial subset of ds covering the line bounding box, or None when empty.

    Falls back to the full dataset when the subset is too small for rasterix to infer
    pixel size (fewer than two samples on either axis).
    """
    x_dim = ds.cf["X"].dims[0]
    y_dim = ds.cf["Y"].dims[0]

    minx, miny, maxx, maxy = line.bounds
    indexes = ds.cf.indexes
    x_sel = (
        slice(minx, maxx) if indexes["X"].is_monotonic_increasing else slice(maxx, minx)
    )
    y_sel = (
        slice(miny, maxy) if indexes["Y"].is_monotonic_increasing else slice(maxy, miny)
    )

    try:
        ds_sub = ds.cf.sel(X=x_sel, Y=y_sel)
    except Exception as e:
        logger.debug("Trajectory bbox subset failed: %s", e)
        return None

    if ds_sub.sizes.get(x_dim, 0) == 0 or ds_sub.sizes.get(y_dim, 0) == 0:
        return None

    # rasterix needs at least two samples per axis to infer pixel size
    if ds_sub.sizes.get(x_dim, 0) < 2 or ds_sub.sizes.get(y_dim, 0) < 2:
        return ds

    return ds_sub


def _sort_cells_along_line(
    ds_sub: xr.Dataset,
    y_nz: np.ndarray,
    x_nz: np.ndarray,
    line: shapely.LineString | shapely.MultiLineString,
    x_dim: str,
    y_dim: str,
    method: Literal["nearest", "linear"],
) -> xr.Dataset:
    """
    Select and order the rasterized cells by their projected distance along the line.
    """
    if y_nz.size == 0:
        return _empty_trajectory_dataset(ds_sub)

    x_coords = np.asarray(ds_sub.cf["X"].values)[x_nz]
    y_coords = np.asarray(ds_sub.cf["Y"].values)[y_nz]
    dists = np.array(
        [
            line.project(shapely.Point(float(x), float(y)))
            for x, y in zip(x_coords, y_coords)
        ],
    )
    order = np.argsort(dists, kind="stable")

    x_var = xr.Variable(dims=VECTORIZED_DIM, data=x_nz[order])
    y_var = xr.Variable(dims=VECTORIZED_DIM, data=y_nz[order])

    if method == "linear":
        logger.warning(
            "Trajectory selection does not interpolate along the path; "
            "'linear' is treated like 'nearest' for spatial indexing.",
        )
    return ds_sub.cf.isel(X=x_var, Y=y_var)


def _intersected_cell_indices(
    ds_sub: xr.Dataset,
    gdf: gpd.GeoDataFrame,
    x_dim: str,
    y_dim: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return `(y_idx, x_idx)` for cells intersected by the input line.

    Strategy:
    1. Prefer exactextract coverage because it provides a sparse mask and avoids
       building dense intermediates.
    2. Fall back to rusterize for a fast rasterization path without GDAL.
    3. Fall back to rasterio with all_touched for parity with legacy behavior.
    """
    try:
        cover = _exactextract_coverage(ds_sub, gdf, xdim=x_dim, ydim=y_dim, clip=False)
        if "geometry" in cover.dims:
            cover = cover.isel(geometry=0, drop=True)
        return _nonzero_indices(cover, x_dim, y_dim)
    except Exception as e:
        logger.debug("exactextract coverage failed for trajectory query: %s", e)

    try:
        burned = rasterize(
            ds_sub,
            gdf,
            xdim=x_dim,
            ydim=y_dim,
            clip=False,
            merge_alg="replace",
            engine="rusterize",
        )
        return _burned_nonfill_indices(burned)
    except Exception as e:
        logger.debug("rusterize failed for trajectory query: %s", e)

    burned = rasterize(
        ds_sub,
        gdf,
        xdim=x_dim,
        ydim=y_dim,
        clip=False,
        all_touched=True,
        merge_alg="replace",
        engine="rasterio",
    )
    return _burned_nonfill_indices(burned)


def _nonzero_indices(
    data: xr.DataArray,
    x_dim: str,
    y_dim: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return non-zero cell indices for a 2D mask data array.

    Uses sparse-array coordinate indices directly when available to avoid densifying.
    """
    y_axis = data.get_axis_num(y_dim)
    x_axis = data.get_axis_num(x_dim)
    mask_data = data.data
    if hasattr(mask_data, "coords"):
        return (
            np.asarray(mask_data.coords[y_axis], dtype=np.intp),
            np.asarray(mask_data.coords[x_axis], dtype=np.intp),
        )

    dense = np.asarray(data.values)
    return np.nonzero(dense > 0)


def _burned_nonfill_indices(burned: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return indices for rasterized pixels that are not equal to fill value.
    """
    arr = np.asarray(burned.values)
    fill = int(arr.max()) if arr.size else 0
    return np.nonzero(arr != fill)


def _exactextract_coverage(
    ds_sub: xr.Dataset,
    gdf: gpd.GeoDataFrame,
    xdim: str,
    ydim: str,
    clip: bool,
) -> xr.DataArray:
    """
    Lazily import exactextract coverage support.

    rasterix' exactextract path depends on optional extras (`sparse`, exactextract).
    Delaying the import keeps trajectory queries importable even when extras are
    unavailable; runtime fallback handles the missing dependency.
    """
    from rasterix.rasterize.exact import coverage

    return coverage(ds_sub, gdf, xdim=xdim, ydim=ydim, clip=clip)


def _empty_trajectory_dataset(ds: xr.Dataset) -> xr.Dataset:
    """
    Return an empty spatial subset (no intersected cells), encodable as CoverageJSON.
    """
    x_dim = ds.cf["X"].dims[0]
    y_dim = ds.cf["Y"].dims[0]
    return ds.isel({x_dim: slice(0, 0), y_dim: slice(0, 0)})
