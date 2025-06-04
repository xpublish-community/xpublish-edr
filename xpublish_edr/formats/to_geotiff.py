"""
Generate GeoTIFF responses for an xarray dataset for EDR queries
"""

import xarray as xr
from fastapi import Response


def to_geotiff(ds: xr.Dataset) -> Response:
    """Return a GeoTIFF response from an xarray dataset"""
    pass
