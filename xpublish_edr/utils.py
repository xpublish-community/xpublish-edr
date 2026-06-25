"""Data utilities for xpublish-edr processing"""

import asyncio

import numpy as np
import xarray as xr
from fastapi import Request


def to_compat_da_dtype(da: xr.DataArray) -> xr.DataArray:
    """Ensures data arrays are valid for rasterio + gdal + netcdf

    Currently float16 is not supported until rasterio ships gdal 3.11, so we need to cast to float32

    NetCDF does not support float16 either, so we need to cast to float32
    """
    if da.dtype.kind == "f" and da.dtype.itemsize < 4:
        da = da.astype(np.float32)
    return da


def _load_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Eagerly load the selected dataset, preferring asynchronous loading.

    Backends that support it (e.g. zarr) can fetch chunks concurrently, which
    is significantly faster for remote stores. Backends that don't raise
    ``NotImplementedError``, in which case we fall back to standard
    synchronous loading. Safe to call from the sync handlers since they run
    in the threadpool, where no event loop is running.
    """
    try:
        return asyncio.run(ds.load_async())
    except NotImplementedError:
        return ds.load()


async def _raw_body(request: Request) -> bytes:
    """Read the raw request body as bytes.

    Used as a FastAPI dependency so the position/area endpoints can stay
    synchronous (`def`) handlers that run in the threadpool: the body is read
    here in the async layer -- correctly returning raw bytes regardless of
    content-type -- and the result is passed to the sync endpoint. Returns an
    empty ``bytes`` for GET requests, which have no body.
    """
    return await request.body()
