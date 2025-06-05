"""
Generate GeoTIFF responses for an xarray dataset for EDR queries
"""

import xarray as xr
from fastapi import HTTPException, Response

from xpublish_edr.logger import logger


def to_geotiff(ds: xr.Dataset) -> Response:
    """Return a GeoTIFF response from an xarray dataset"""
    import io

    import rasterio  # noqa

    # Remove any dimensions that are scalar
    ds = ds.squeeze()

    # Handle data variables
    data_vars = list(ds.data_vars)
    if len(data_vars) == 0:
        raise HTTPException(
            status_code=400,
            detail="No variables with x and y coordinates found.",
        )

    for var in data_vars:
        if len(ds[var].shape) > 2:
            logger.error(
                f"Variable {var} has {ds[var].shape} dimensions. "
                "GeoTIFF export only supports up to 2 dimensions, "
                "add dimensions to the query to reduce the number of dimensions. "
                f"Found dimensions: {', '.join(ds[var].dims)}. "
                f"Found variables: {', '.join(ds.data_vars)}.",
            )
            logger.error("Full dataset", ds)
            raise HTTPException(
                status_code=400,
                detail=f"Variable {var} has {ds[var].shape} dimensions. "
                "GeoTIFF export only supports up to 2 dimensions, "
                "add dimensions to the query to reduce the number of dimensions. "
                f"Found dimensions: {', '.join(ds[var].dims)}. "
                f"Found variables: {', '.join(ds.data_vars)}.",
            )

    # Set the spatial dims and the crs
    axes = ds.cf.axes
    x_coord = axes["X"][0]
    y_coord = axes["Y"][0]
    ds = ds.rio.set_spatial_dims(x_dim=x_coord, y_dim=y_coord, inplace=True)
    ds = ds.transpose(..., y_coord, x_coord)

    # Create in-memory GeoTIFF
    memfile = io.BytesIO()
    ds.rio.to_raster(memfile, driver="GTiff")

    # Reset buffer position
    memfile.seek(0)

    # Return FastAPI response
    return Response(
        content=memfile.getvalue(),
        media_type="image/tiff",
        headers={"Content-Disposition": "attachment; filename=data.tiff"},
    )
