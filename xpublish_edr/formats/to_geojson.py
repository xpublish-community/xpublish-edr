"""
Generate GeoJSON responses for an xarray dataset for EDR queries
"""

import geopandas as gpd
import pandas as pd
import xarray as xr
from fastapi import Response


def handle_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Handle date columns in a GeoDataFrame"""
    for col in df.columns:
        if isinstance(df[col].iloc[0], pd.Timestamp):
            df[col] = df[col].map(lambda x: x.isoformat())
        elif isinstance(df[col].iloc[0], pd.Timedelta):
            df[col] = df[col].map(lambda x: x.total_seconds())
    return df


def to_geojson(ds: xr.Dataset):
    """Return a GeoJSON response from an xarray dataset"""
    axes = ds.cf.axes
    (x_col,) = axes["X"]
    (y_col,) = axes["Y"]

    df = ds.to_dataframe().reset_index()
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[x_col], df[y_col]))
    gdf = handle_date_columns(gdf)

    json = gdf.to_json()

    return Response(
        json,
        media_type="application/json",
    )
