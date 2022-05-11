"""
Generate a NetCDF from an xarray Dataset for EDR queries
"""
from pathlib import Path
from tempfile import TemporaryDirectory

import xarray as xr
from fastapi import Response


def to_netcdf(ds: xr.Dataset):
    """Return a NetCDF response from a dataset"""
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "position.nc"
        ds.to_netcdf(path)

        with path.open("rb") as f:
            return Response(
                f.read(),
                media_type="application/netcdf",
                headers={"Content-Disposition": 'attachment; filename="position.nc"'},
            )
