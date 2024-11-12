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


def project_geometry(ds: xr.Dataset, geometry_crs: str, geometry: Geometry) -> Geometry:
    """
    Get the projection from the dataset
    """
    grid_mapping_names = ds.cf.grid_mapping_names
    if len(grid_mapping_names) == 0:
        # TODO: Should we require a grid mapping? For now return as is
        return geometry
    if len(grid_mapping_names) > 1:
        raise ValueError(f"Multiple grid mappings found: {grid_mapping_names!r}!")
    (grid_mapping_var,) = tuple(itertools.chain(*ds.cf.grid_mapping_names.values()))

    grid_mapping = ds[grid_mapping_var]
    data_crs = pyproj.crs.CRS.from_cf(grid_mapping.attrs)
    if not data_crs.is_projected:
        raise ValueError(
            "This method is intended to be used with projected coordinate systems.",
        )

    transformer = transformer_from_crs(
        crs_from=geometry_crs,
        crs_to=data_crs,
        always_xy=True,
    )
    return transform(transformer.transform, geometry)
