"""
Generate CSV responses for an xarray dataset for EDR queries
"""

import xarray as xr
from fastapi import Response


def to_parquet(ds: xr.Dataset) -> Response:
    """Return a Parquet response from an xarray dataset"""
    df = ds.to_dataframe()

    pq = df.to_parquet()

    return Response(
        pq,
        media_type="application/parquet",
        headers={"Content-Disposition": 'attachment; filename="data.parquet"'},
    )
