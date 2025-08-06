"""
Generate a NetCDF from an xarray Dataset for EDR queries
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import xarray as xr
from fastapi import Response

from xpublish_edr.utils import to_compat_da_dtype


def to_netcdf(ds: xr.Dataset):
    """Return a NetCDF response from a dataset"""
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "data.nc"

        # Float16 is not supported by netCDF4
        ds = ds.map(to_compat_da_dtype)
        ds.to_netcdf(path)

        with path.open("rb") as f:
            return Response(
                f.read(),
                media_type="application/netcdf",
                headers={"Content-Disposition": 'attachment; filename="data.nc"'},
            )
