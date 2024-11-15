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

    X = ds.cf["X"]
    Y = ds.cf["Y"]

    # Transform the coordinates
    # If the data is vectorized, we just transform the points in full
    # TODO: Handle 2D coordinates
    if len(X.dims) > 1 or len(Y.dims) > 1:
        raise NotImplementedError("Only 1D coordinates are supported")

    x_dim = X.dims[0]
    y_dim = Y.dims[0]
    if x_dim == VECTORIZED_DIM and y_dim == VECTORIZED_DIM:
        x = X
        y = Y
        target_dims: tuple = (VECTORIZED_DIM,)
    else:
        # Otherwise we need to transform the full grid
        # TODO: Is there a better way to handle this? this is quite hacky
        var = [d for d in ds.data_vars if x_dim and y_dim in ds[d].dims][0]
        var_dims = ds[var].dims
        if var_dims.index(x_dim) < var_dims.index(y_dim):
            x, y = xr.broadcast(X, Y)
            target_dims = (x_dim, y_dim)
        else:
            x, y = xr.broadcast(Y, X)
            target_dims = (y_dim, x_dim)

    x, y = transformer.transform(x, y)

    coords_to_drop = [
        c for c in ds.coords if x_dim in ds[c].dims or y_dim in ds[c].dims
    ]

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
    ds = ds.assign_coords(
        {
            target_x_coord_name: (target_dims, x),
            target_y_coord_name: (target_dims, y),
        },
    )

    return ds
