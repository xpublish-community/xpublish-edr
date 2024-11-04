"""
Common geometry handling functions
"""

import xarray as xr


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
