"""
Generate GeoJSON responses for an xarray dataset for EDR queries
"""

import geopandas as gpd
import xarray as xr
from fastapi import Response


def to_geojson(ds: xr.Dataset):
    """Return a GeoJSON response from an xarray dataset"""
    ds = ds.squeeze()
    x_col = ds.cf["X"].name
    y_col = ds.cf["Y"].name
    if "T" in ds.cf:
        time_col = ds.cf["T"].name
    else:
        time_col = None

    df = ds.to_dataframe().reset_index()
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[x_col], df[y_col]))

    # Map the time to a string if applicable
    # TODO: Handle timezone?
    if time_col:
        gdf[time_col] = gdf[time_col].map(lambda t: t.isoformat())

    json = gdf.to_json()

    return Response(
        json,
        media_type="application/json",
    )
