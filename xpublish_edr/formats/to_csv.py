"""
Generate CSV responses for an xarray dataset for EDR queries
"""

import xarray as xr
from fastapi import Response


def to_csv(ds: xr.Dataset):
    """Return a CSV response from an xarray dataset"""
    # Drop CRS grid-mapping variables (scalar integer blobs written by rioxarray)
    crs_vars = [name for name, var in ds.variables.items() if "grid_mapping_name" in var.attrs]
    if crs_vars:
        ds = ds.drop_vars(crs_vars)

    # Dimensions with no coordinate values become meaningless 0-based integer index
    # columns after to_dataframe() (e.g. lat/lon dims after project_dataset drops
    # their original values and replaces them with 2D latitude/longitude coords)
    index_only_dims = [dim for dim in ds.dims if dim not in ds.coords]

    df = ds.to_dataframe().reset_index()
    df = df.drop(columns=[c for c in index_only_dims if c in df.columns])

    csv = df.to_csv(index=False)

    return Response(
        csv,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="data.csv"'},
    )
