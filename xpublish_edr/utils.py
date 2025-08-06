"""Data utilities for xpublish-edr processing"""

import numpy as np
import xarray as xr


def to_compat_da_dtype(da: xr.DataArray) -> xr.DataArray:
    """Ensures data arrays are valid for rasterio + gdal + netcdf

    Currently float16 is not supported until rasterio ships gdal 3.11, so we need to cast to float32

    NetCDF does not support float16 either, so we need to cast to float32
    """
    if da.dtype.kind == "f" and da.dtype.itemsize < 4:
        da = da.astype(np.float32)
    return da
