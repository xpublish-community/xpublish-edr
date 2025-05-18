"""
Handle selection and formatting for cube queries
"""

import xarray as xr


def select_by_bbox(
    ds: xr.Dataset,
    bbox: tuple[float, float, float, float],
) -> xr.Dataset:
    """
    Return a dataset with the data within the given bbox

    Assumes that the dataset is in the same CRS as the bbox
    """
    return ds.cf.sel(X=slice(bbox[0], bbox[2]), Y=slice(bbox[1], bbox[3]))
