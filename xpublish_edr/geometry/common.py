"""
Common geometry handling functions
"""

import itertools
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Mapping, Optional, Union

import cf_xarray  # noqa: F401  (registers the ``.cf`` dataset accessor)
import numpy as np
import pyproj
import rioxarray  # noqa
import shapely
import xarray as xr
from shapely import Geometry

from xpublish_edr.logger import logger

VECTORIZED_DIM = "pts"

# https://pyproj4.github.io/pyproj/stable/advanced_examples.html#caching-pyproj-objects
transformer_from_crs = lru_cache(partial(pyproj.Transformer.from_crs, always_xy=True))


DEFAULT_CRS = pyproj.CRS.from_epsg(4326)


def coord_is_regular(da: xr.DataArray) -> bool:
    """
    Check if the DataArray has a regular grid
    """
    return len(da.shape) == 1 and da.name in da.dims


@dataclass
class SpatialRef:
    """Resolved spatial reference for a dataset.

    Holds the CRS together with the names of the X (x/longitude) and Y
    (y/latitude) coordinate variables, so that CRS detection and coordinate
    identification stay consistent across the query pipeline.
    """

    crs: pyproj.CRS
    X: str
    Y: str


def _is_rotated_pole(crs: pyproj.CRS) -> bool:
    """Whether the CRS is a CF rotated latitude/longitude grid."""
    return crs.to_cf().get("grid_mapping_name") == "rotated_latitude_longitude"


def _parse_proj_convention_crs(attrs: Mapping) -> Optional[pyproj.CRS]:
    """Parse a CRS from GeoZarr ``proj:`` convention attributes.

    The ``proj:`` convention (https://github.com/zarr-conventions/proj) encodes
    the CRS as ``proj:code`` (an authority:code string such as ``"EPSG:4326"``)
    or ``proj:wkt2`` (a WKT2 string). ``proj:projjson`` is intentionally not
    handled here, matching the reader in xpublish-tiles.
    """
    code = attrs.get("proj:code")
    if code is not None:
        try:
            return pyproj.CRS.from_user_input(code)
        except Exception as e:
            logger.error(f"Failed to parse proj:code {code!r}: {e}")
    wkt = attrs.get("proj:wkt2")
    if wkt is not None:
        try:
            return pyproj.CRS.from_wkt(wkt)
        except Exception as e:
            logger.error(f"Failed to parse proj:wkt2: {e}")
    return None


def _geozarr_spatial_dimensions(ds: xr.Dataset) -> Optional[tuple[str, str]]:
    """Return ``(Xname, Yname)`` from the GeoZarr ``spatial:dimensions`` attribute.

    Per the ``spatial:`` convention (https://github.com/zarr-conventions/spatial)
    ``spatial:dimensions`` is in ``[Y, X]`` order, and array-level attrs override
    group (dataset) level attrs.
    """
    for var in ds.data_vars:
        spatial_dims = ds[var].attrs.get("spatial:dimensions")
        if spatial_dims is not None and len(spatial_dims) == 2:
            return str(spatial_dims[1]), str(spatial_dims[0])
    spatial_dims = ds.attrs.get("spatial:dimensions")
    if spatial_dims is not None and len(spatial_dims) == 2:
        return str(spatial_dims[1]), str(spatial_dims[0])
    return None


def _xy_from_cf(
    ds: xr.Dataset,
    crs: pyproj.CRS,
    restrict: Optional[tuple[str, ...]] = None,
) -> Optional[tuple[str, str]]:
    """Identify X/Y coordinate variable names via cf_xarray, keyed off CRS type.

    Mirrors the logic in xpublish-tiles' ``guess_coordinate_vars``: geographic
    CRSs use longitude/latitude, rotated-pole CRSs use grid_longitude/
    grid_latitude, and projected CRSs use the X/Y axes. When ``restrict`` is
    given (the coordinate names from a grid mapping), candidates are limited to
    it. Returns ``None`` if a single unambiguous (X, Y) pair cannot be found.
    """
    try:
        if _is_rotated_pole(crs):
            stdnames = ds.cf.standard_names
            xs = list(stdnames.get("grid_longitude", []))
            ys = list(stdnames.get("grid_latitude", []))
        elif crs.is_geographic:
            coords = ds.cf.coordinates
            xs = list(coords.get("longitude", []))
            ys = list(coords.get("latitude", []))
        else:
            axes = ds.cf.axes
            xs = list(axes.get("X", []))
            ys = list(axes.get("Y", []))
    except Exception:
        return None

    if restrict is not None:
        allowed = set(restrict)
        xs = [n for n in xs if n in allowed]
        ys = [n for n in ys if n in allowed]

    # Keep only coordinates that exist as non-scalar variables
    xs = [n for n in xs if n in ds.variables and ds[n].ndim > 0]
    ys = [n for n in ys if n in ds.variables and ds[n].ndim > 0]

    if len(xs) == 1 and len(ys) == 1:
        return str(xs[0]), str(ys[0])
    return None


def _resolve_xy_names(
    ds: xr.Dataset,
    crs: pyproj.CRS,
    coordinates: Optional[tuple[str, ...]] = None,
) -> tuple[str, str]:
    """Resolve the X and Y coordinate variable names for a dataset.

    Priority: grid-mapping coordinates / CF detection, then GeoZarr
    ``spatial:dimensions``, then a final fall back to cf_xarray's ``X``/``Y``
    axes (today's behavior).
    """
    names = _xy_from_cf(ds, crs, restrict=coordinates)
    if names is not None:
        return names

    spatial_dims = _geozarr_spatial_dimensions(ds)
    if spatial_dims is not None:
        return spatial_dims

    try:
        return str(ds.cf["X"].name), str(ds.cf["Y"].name)
    except KeyError as e:
        raise ValueError(
            "Could not determine X/Y coordinate variables for the dataset. "
            "Provide CF axis/standard_name attributes, a grid_mapping variable, "
            "or GeoZarr 'proj:'/'spatial:' convention attributes.",
        ) from e


def _select_grid_mapping(ds: xr.Dataset, grid_mappings):
    """Pick the grid mapping describing the dataset's native coordinates.

    For datasets with multiple grid mappings (e.g. GeoZarr alternate CRSs),
    prefer the mapping whose coordinates are present as indexed (then regular
    1D) coordinate variables -- i.e. the grid the data is actually stored on.
    """
    if len(grid_mappings) == 1:
        return grid_mappings[0]
    for mapping in grid_mappings:
        coords = mapping.coordinates or ()
        if coords and all(c in ds.indexes for c in coords):
            return mapping
    for mapping in grid_mappings:
        coords = mapping.coordinates or ()
        if coords and all(
            c in ds.variables and coord_is_regular(ds[c]) for c in coords
        ):
            return mapping
    return grid_mappings[0]


def _resolve_crs(
    ds: xr.Dataset,
) -> tuple[pyproj.CRS, Optional[tuple[str, ...]]]:
    """Resolve the dataset CRS and any grid-mapping-provided coordinate names.

    Detection priority (mirrors xpublish-tiles):

    1. CF ``grid_mapping`` convention via ``ds.cf.grid_mappings`` (cf_xarray),
       which also handles datasets with multiple grid mappings.
    2. GeoZarr ``proj:`` convention attributes.
    3. Legacy ``spatial_ref``/``crs`` variables, else a WGS84 default when
       latitude/longitude coordinates are present.

    The returned coordinate names (if any) come from the selected CF grid
    mapping and are used to disambiguate X/Y; CRS resolution itself never
    depends on the coordinates being resolvable.
    """
    grid_mappings = ds.cf.grid_mappings
    if grid_mappings:
        mapping = _select_grid_mapping(ds, grid_mappings)
        if mapping.crs is not None:
            return mapping.crs, mapping.coordinates

    crs = _parse_proj_convention_crs(ds.attrs)
    if crs is not None:
        return crs, None

    return get_default_grid_mapping(ds), None


def dataset_spatial_ref(ds: xr.Dataset) -> SpatialRef:
    """Resolve the CRS and X/Y coordinate variable names for a dataset."""
    crs, coordinates = _resolve_crs(ds)
    X, Y = _resolve_xy_names(ds, crs, coordinates=coordinates)
    return SpatialRef(crs=crs, X=X, Y=Y)


def dataset_xy_names(ds: xr.Dataset) -> tuple[str, str]:
    """Return the ``(Xname, Yname)`` coordinate variable names for a dataset."""
    spatial_ref = dataset_spatial_ref(ds)
    return spatial_ref.X, spatial_ref.Y


def is_regular_xy_coords(ds: xr.Dataset) -> bool:
    """
    Check if the dataset has regular (1D) X and Y coordinates
    """
    try:
        X, Y = dataset_xy_names(ds)
    except ValueError:
        # X/Y coordinate variables could not be resolved (e.g. an affine-only
        # or 2D grid); not a regular 1D grid we can select on.
        return False
    if X not in ds.variables or Y not in ds.variables:
        return False
    return coord_is_regular(ds[X]) and coord_is_regular(ds[Y])


def spatial_bounds(ds: xr.Dataset) -> tuple[float, float, float, float]:
    """
    Get the spatial bounds of the dataset, naively, in whatever CRS it is in
    """
    X, Y = dataset_xy_names(ds)
    x = ds[X]
    min_x = float(x.min().values)
    max_x = float(x.max().values)

    y = ds[Y]
    min_y = float(y.min().values)
    max_y = float(y.max().values)
    return min_x, min_y, max_x, max_y


def get_default_grid_mapping(ds: xr.Dataset) -> pyproj.CRS:
    """Get a default grid_mapping if no grid_mapping attribute is specified."""
    if "spatial_ref" in ds.variables:
        return pyproj.CRS.from_cf(ds["spatial_ref"].attrs)
    if "crs" in ds.variables:
        return pyproj.CRS.from_cf(ds["crs"].attrs)

    # Default to WGS84, EPSG:4326
    keys = ds.cf.keys()
    if "latitude" in keys and "longitude" in keys:
        crs = pyproj.CRS.from_epsg(4326)
        # Write the crs to the dataset so it is there on export
        ds.rio.write_crs(crs, inplace=True)
        return crs
    else:
        raise ValueError(
            "Unknown coordinate system. "
            "Please set the grid_mapping attribute "
            "on relevant data variables.",
        )


def dataset_crs(ds: xr.Dataset) -> pyproj.CRS:
    """Resolve the dataset's native CRS.

    Supports the CF ``grid_mapping`` convention (including datasets with
    multiple grid mappings) and the GeoZarr ``proj:``/``spatial:`` conventions.
    """
    return _resolve_crs(ds)[0]


def project_geometry(ds: xr.Dataset, geometry_crs: str, geometry: Geometry) -> Geometry:
    """
    Get the projection from the dataset
    """
    data_crs = dataset_crs(ds)

    transformer = transformer_from_crs(
        crs_from=geometry_crs,
        crs_to=data_crs,
    )

    def _transform(coords: np.ndarray) -> np.ndarray:
        """Vectorized callback for shapely.transform: project all coords in one call."""
        x, y = transformer.transform(coords[:, 0], coords[:, 1])
        return np.column_stack([x, y])

    return shapely.transform(geometry, _transform)


def project_bbox(
    ds: xr.Dataset,
    bbox_crs: Union[str, pyproj.CRS],
    bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """
    Project the bbox to the dataset's CRS
    """
    data_crs = dataset_crs(ds)
    if isinstance(bbox_crs, pyproj.CRS):
        target_crs = bbox_crs
    else:
        target_crs = pyproj.CRS.from_string(bbox_crs)
    if data_crs == target_crs:
        return bbox

    transformer: pyproj.Transformer = transformer_from_crs(
        crs_from=target_crs,
        crs_to=data_crs,
    )
    projected_x, projected_y = transformer.transform(
        xx=[bbox[0], bbox[2]],
        yy=[bbox[1], bbox[3]],
    )

    min_x = min(projected_x)
    max_x = max(projected_x)
    min_y = min(projected_y)
    max_y = max(projected_y)
    return min_x, min_y, max_x, max_y


def _ensure_cf_axes(ds: xr.Dataset) -> xr.Dataset:
    """Tag the resolved X/Y coordinates with CF ``axis`` attributes if missing.

    GeoZarr datasets may describe their spatial coordinates purely via the
    ``proj:``/``spatial:`` conventions, leaving the coordinate variables without
    CF ``axis`` attributes. Downstream formatters use ``ds.cf.axes``; tag the
    coordinates so output works even when no reprojection occurs.
    """
    axes = ds.cf.axes
    if "X" in axes and "Y" in axes:
        return ds
    try:
        X, Y = dataset_xy_names(ds)
    except ValueError:
        return ds
    ds = ds.copy(deep=False)
    for name, axis in ((X, "X"), (Y, "Y")):
        if name in ds.coords and "axis" not in ds[name].attrs:
            ds[name].attrs = {**ds[name].attrs, "axis": axis}
    return ds


def project_dataset(ds: xr.Dataset, query_crs: Union[str, pyproj.CRS]) -> xr.Dataset:
    """
    Project the dataset to the given CRS
    """
    data_crs = dataset_crs(ds)
    if isinstance(query_crs, pyproj.CRS):
        target_crs = query_crs
    else:
        target_crs = pyproj.CRS.from_string(query_crs)
    if data_crs == target_crs:
        return _ensure_cf_axes(ds)

    transformer = transformer_from_crs(
        crs_from=data_crs,
        crs_to=target_crs,
    )

    # Unpack the coordinates using the resolved grid-mapping / convention names
    Xname, Yname = dataset_xy_names(ds)
    X = ds[Xname]
    Y = ds[Yname]

    # Transform the coordinates
    # If the data is vectorized, we just transform the points in full
    # TODO: Handle 2D coordinates
    if len(X.dims) > 1 or len(Y.dims) > 1:
        raise NotImplementedError("Only 1D coordinates are supported")

    x, y = xr.broadcast(X, Y)
    target_dims = x.dims

    x, y = transformer.transform(x, y)

    x_dim = X.dims[0]
    y_dim = Y.dims[0]

    coords_to_drop = [
        c for c in ds.coords if x_dim in ds[c].dims or y_dim in ds[c].dims
    ]

    # TODO: Handle rotated pole
    target_cf_coords = target_crs.coordinate_system.to_cf()

    # Get the new X and Y coordinates
    target_x_coord = next(coord for coord in target_cf_coords if coord["axis"] == "X")
    target_y_coord = next(coord for coord in target_cf_coords if coord["axis"] == "Y")

    target_x_coord_name = target_x_coord["standard_name"]
    target_y_coord_name = target_y_coord["standard_name"]

    stdnames = ds.cf.standard_names
    coords_to_drop += list(
        itertools.chain(
            stdnames.get(target_x_coord_name, []),
            stdnames.get(target_y_coord_name, []),
        ),
    )
    ds = ds.drop_vars(coords_to_drop)

    # Create the new dataset with vectorized coordinates
    x_var = xr.Variable(dims=target_dims, data=x, attrs=target_x_coord)
    y_var = xr.Variable(dims=target_dims, data=y, attrs=target_y_coord)
    ds = ds.assign_coords(
        {
            target_x_coord_name: x_var,
            target_y_coord_name: y_var,
        },
    )

    if x_dim != y_dim:
        ds = ds.transpose(..., y_dim, x_dim)

    # Write the crs to the dataset so it is there on export
    ds = ds.rio.write_crs(target_crs, inplace=True)
    return ds
