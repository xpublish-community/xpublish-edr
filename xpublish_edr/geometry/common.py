"""
Common geometry handling functions
"""

import itertools
from functools import lru_cache

import pyproj
import xarray as xr
from shapely import Geometry
from shapely.ops import transform

VECTORIZED_DIM = "pts"

# https://pyproj4.github.io/pyproj/stable/advanced_examples.html#caching-pyproj-objectshttps://pyproj4.github.io/pyproj/stable/advanced_examples.html#caching-pyproj-objects
transformer_from_crs = lru_cache(pyproj.Transformer.from_crs)


def coord_is_regular(da: xr.DataArray) -> bool:
    """
    Check if the DataArray has a regular grid
    """
    return len(da.shape) == 1 and da.name in da.dims


def is_regular_xy_coords(ds: xr.Dataset) -> bool:
    """
    Check if the dataset has 2D coordinates
    """
    return coord_is_regular(ds.cf["X"]) and coord_is_regular(ds.cf["Y"])


def dataset_crs(ds: xr.Dataset) -> pyproj.CRS:
    grid_mapping_names = ds.cf.grid_mapping_names
    if len(grid_mapping_names) == 0:
        # Default to WGS84
        return pyproj.crs.CRS.from_epsg(4326)
    if len(grid_mapping_names) > 1:
        raise ValueError(f"Multiple grid mappings found: {grid_mapping_names!r}!")
    (grid_mapping_var,) = tuple(itertools.chain(*ds.cf.grid_mapping_names.values()))

    grid_mapping = ds[grid_mapping_var]
    return pyproj.CRS.from_cf(grid_mapping.attrs)


def project_geometry(ds: xr.Dataset, geometry_crs: str, geometry: Geometry) -> Geometry:
    """
    Get the projection from the dataset
    """
    data_crs = dataset_crs(ds)

    transformer = transformer_from_crs(
        crs_from=geometry_crs,
        crs_to=data_crs,
        always_xy=True,
    )
    return transform(transformer.transform, geometry)


def project_dataset(ds: xr.Dataset, query_crs: str) -> xr.Dataset:
    """
    Project the dataset to the given CRS
    """
    data_crs = dataset_crs(ds)
    target_crs = pyproj.CRS.from_string(query_crs)
    if data_crs == target_crs:
        return ds

    transformer = transformer_from_crs(
        crs_from=data_crs,
        crs_to=target_crs,
        always_xy=True,
    )

    # TODO: Handle rotated pole
    cf_coords = target_crs.coordinate_system.to_cf()

    # Get the new X and Y coordinates
    target_y_coord = next(coord for coord in cf_coords if coord["axis"] == "Y")
    target_x_coord = next(coord for coord in cf_coords if coord["axis"] == "X")

    # Transform the coordinates
    # If the data is vectorized, we just transform the points in full
    # TODO: Handle 2D coordinates
    if not is_regular_xy_coords(ds):
        raise NotImplementedError("Only 1D coordinates are supported")

    x_dim = ds.cf["X"].dims[0]
    y_dim = ds.cf["Y"].dims[0]
    if x_dim == [VECTORIZED_DIM]:
        x = ds.cf["X"]
        y = ds.cf["Y"]
    else:
        # Otherwise we need to transform the full grid
        x, y = xr.broadcast(ds.cf["X"], ds.cf["Y"])

    x, y = transformer.transform(x, y)

    coords_to_drop = [
        c for c in ds.coords if x_dim in ds[c].dims or y_dim in ds[c].dims
    ]

    target_x_coord_name = target_x_coord["standard_name"]
    target_y_coord_name = target_y_coord["standard_name"]

    if target_x_coord_name in ds:
        target_x_coord_name += "_"
    if target_y_coord_name in ds:
        target_y_coord_name += "_"

    # Create the new dataset with vectorized coordinates
    ds = ds.assign_coords(
        {
            target_x_coord_name: ((x_dim, y_dim), x),
            target_y_coord_name: ((x_dim, y_dim), y),
        },
    )

    ds = ds.drop(coords_to_drop)

    return ds
