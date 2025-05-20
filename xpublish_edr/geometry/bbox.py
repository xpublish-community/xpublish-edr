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
    indexes = ds.cf.indexes
    if indexes["X"].is_monotonic_increasing:
        x_slice = slice(bbox[0], bbox[2])
    else:
        x_slice = slice(bbox[2], bbox[0])
    if indexes["Y"].is_monotonic_increasing:
        y_slice = slice(bbox[1], bbox[3])
    else:
        y_slice = slice(bbox[3], bbox[1])
    return ds.cf.sel(X=x_slice, Y=y_slice)
