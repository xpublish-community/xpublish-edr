"""
Generate CSV responses for an xarray dataset for EDR queries
"""
import xarray as xr
from fastapi import Response


def to_csv(ds: xr.Dataset):
    """Return a CSV response from an xarray dataset"""
    ds = ds.squeeze()
    df = ds.to_pandas()

    csv = df.to_csv()

    return Response(
        csv,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="position.csv"'},
    )
