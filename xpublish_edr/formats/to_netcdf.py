"""
Generate a NetCDF from an xarray Dataset for EDR queries
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import xarray as xr
from fastapi import Response

from xpublish_edr.utils import to_compat_da_dtype


def to_compat_nc_attr(value: Any) -> Any:
    """Coerce an attribute value into something that can be stored in a valid netCDF4 file

    h5netcdf cannot write numpy unicode (``<U``) arrays or scalars, so they are
    converted to python strings (or lists of strings, which are written as
    variable-length string attributes).

    Booleans are not a netCDF data type, so they are stored as 0/1 integers.
    """
    if isinstance(value, bool | np.bool_):
        return np.int8(value)
    if isinstance(value, np.str_):
        return str(value)
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "UO":
            return [str(v) for v in value.ravel()]
        if value.dtype.kind == "b":
            return value.astype(np.int8)
    if isinstance(value, list | tuple) and any(isinstance(v, bool | np.bool_) for v in value):
        return np.asarray(value, dtype=np.int8)
    return value


def to_compat_nc_attrs(ds: xr.Dataset) -> xr.Dataset:
    """Return a shallow copy of the dataset with netCDF4 compatible attributes"""
    ds = ds.copy(deep=False)
    ds.attrs = {k: to_compat_nc_attr(v) for k, v in ds.attrs.items()}
    for var in ds.variables.values():
        var.attrs = {k: to_compat_nc_attr(v) for k, v in var.attrs.items()}
    return ds


def to_netcdf(ds: xr.Dataset):
    """Return a NetCDF response from a dataset"""
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "data.nc"

        # Float16 is not supported by netCDF4
        ds = ds.map(to_compat_da_dtype)
        ds = to_compat_nc_attrs(ds)

        # Explicitly use h5netcdf to always write netCDF4, rather than letting xarray
        # fall back to the scipy engine, which can only write netCDF3 and fails on
        # attributes such as arrays of strings.
        ds.to_netcdf(path, engine="h5netcdf", format="NETCDF4")

        with path.open("rb") as f:
            return Response(
                f.read(),
                media_type="application/netcdf",
                headers={"Content-Disposition": 'attachment; filename="data.nc"'},
            )
