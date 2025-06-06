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
    data_var_count = len(data_vars)
    if data_var_count == 0:
        raise HTTPException(
            status_code=400,
            detail="No variables with x and y coordinates found.",
        )

    if data_var_count == 1:
        var = data_vars[0]
        if len(ds[var].shape) > 3:
            error_message = (
                f"Variable {var} has {ds[var].shape} dimensions. "
                "GeoTIFF export only supports up to 3 dimensions for a single variable, "
                "add dimensions to the query to reduce the number of dimensions. "
            )
            logger.error(error_message)
            raise HTTPException(
                status_code=400,
                detail=error_message,
            )
        ds = ds[var]
    else:
        for var in data_vars:
            if len(ds[var].shape) > 2:
                error_message = (
                    f"Variable {var} has {ds[var].shape} dimensions. "
                    "GeoTIFF export only supports up to 2 dimensions for multiple variables, "
                    "add dimensions to the query to reduce the number of dimensions. "
                )
                logger.error(error_message)
                logger.error("Full dataset", ds)
                raise HTTPException(
                    status_code=400,
                    detail=error_message,
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
