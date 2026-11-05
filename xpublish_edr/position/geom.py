"""Handle selection and formatting for position queries"""

from __future__ import annotations

from typing import Literal

import numpy as np
import shapely
import xarray as xr

from xpublish_edr.geometry.common import (
    VECTORIZED_DIM,
    SpatialRef,
    prepare_spatial_grid,
)


def select_by_position(
    ds: xr.Dataset,
    point: shapely.Point | shapely.MultiPoint,
    method: Literal["nearest", "linear"] = "nearest",
    spatial_ref: SpatialRef | None = None,
) -> xr.Dataset:
    """
    Return a dataset with the position nearest to the given coordinates
    """
    grid = prepare_spatial_grid(ds, spatial_ref=spatial_ref, require_regular=True)
    ds = grid.ds
    X, Y = grid.spatial_ref.X, grid.spatial_ref.Y

    if not isinstance(point, (shapely.Point, shapely.MultiPoint)):
        raise ValueError(
            f"Invalid point type {point.geom_type}, must be Point or MultiPoint",
        )

    _validate_points_in_grid(ds, point, grid.spatial_ref)

    if isinstance(point, shapely.Point):
        return _select_by_position_regular_xy_grid(ds, point, X, Y, method)
    return _select_by_multiple_positions_regular_xy_grid(ds, point, X, Y, method)


def _padded_bounds(coord: np.ndarray) -> tuple[float, float]:
    """
    Return the (min, max) extent of a 1D coordinate, padded by half a cell on
    each side so points that fall within the edge cells are accepted
    """
    lo, hi = float(np.nanmin(coord)), float(np.nanmax(coord))
    if coord.size < 2:
        return lo, hi
    half_step = float(np.nanmax(np.abs(np.diff(coord)))) / 2
    return lo - half_step, hi + half_step


def _validate_points_in_grid(
    ds: xr.Dataset,
    points: shapely.Point | shapely.MultiPoint,
    spatial_ref: SpatialRef,
) -> None:
    """
    Ensure the query point(s), already projected to the dataset's CRS, fall
    within the dataset's spatial extent.

    Raises a ValueError rather than letting nearest neighbor selection
    silently snap the point(s) to the edge of the grid. No attempt is made to
    reinterpret out of range coordinates (e.g. wrapping longitudes between
    -180..180 and 0..360); the caller is asked to convert them instead.
    """
    coords = shapely.get_coordinates(points)
    x, y = coords[:, 0], coords[:, 1]

    X, Y = spatial_ref.X, spatial_ref.Y
    x_lo, x_hi = _padded_bounds(ds[X].values)
    y_lo, y_hi = _padded_bounds(ds[Y].values)

    out_of_bounds = (x < x_lo) | (x > x_hi) | (y < y_lo) | (y > y_hi)
    if out_of_bounds.any():
        bad = ", ".join(f"({px:g} {py:g})" for px, py in coords[out_of_bounds])
        raise ValueError(
            f"Point(s) {bad} fall outside the dataset's spatial extent "
            f"({X}: [{float(ds[X].min()):g}, {float(ds[X].max()):g}], "
            f"{Y}: [{float(ds[Y].min()):g}, {float(ds[Y].max()):g}]). "
            "Convert the coordinates to the dataset's range and try again.",
        )


def _select_by_position_regular_xy_grid(
    ds: xr.Dataset,
    point: shapely.Point,
    X: str,
    Y: str,
    method: Literal["nearest", "linear"] = "nearest",
) -> xr.Dataset:
    """
    Return a dataset with the position nearest to the given coordinates
    """
    # Find the nearest X and Y coordinates to the point
    if method == "nearest":
        return ds.sel({X: [point.x], Y: [point.y]}, method=method)
    else:
        return ds.interp({X: [point.x], Y: [point.y]}, method=method)


def _select_by_multiple_positions_regular_xy_grid(
    ds: xr.Dataset,
    points: shapely.MultiPoint,
    X: str,
    Y: str,
    method: Literal["nearest", "linear"] = "nearest",
) -> xr.Dataset:
    """
    Return a dataset with the positions nearest to the given coordinates
    """
    # Find the nearest X and Y coordinates to the point using vectorized indexing
    x, y = np.array(list(zip(*[(point.x, point.y) for point in points.geoms])))

    # When using vectorized indexing with interp, we need to persist the attributes explicitly
    sel_x = xr.Variable(data=x, dims=VECTORIZED_DIM, attrs=ds[X].attrs)
    sel_y = xr.Variable(data=y, dims=VECTORIZED_DIM, attrs=ds[Y].attrs)
    if method == "nearest":
        return ds.sel({X: sel_x, Y: sel_y}, method=method)
    else:
        return ds.interp({X: sel_x, Y: sel_y}, method=method)
