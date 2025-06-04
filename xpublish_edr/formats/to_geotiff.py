"""
Generate GeoTIFF responses for an xarray dataset for EDR queries
"""

import xarray as xr
from fastapi import HTTPException, Response


def to_geotiff(ds: xr.Dataset) -> Response:
    """Return a GeoTIFF response from an xarray dataset"""
    import io

    import rasterio  # noqa

    # Get CF axes
    axes = ds.cf.axes
    x_coord = axes["X"][0]
    y_coord = axes["Y"][0]

    # Ensure all variables have x and y coordinates
    for var in ds.data_vars:
        if x_coord not in ds[var].cf.axes or y_coord not in ds[var].cf.axes:
            ds = ds.drop_vars(var)

    # Handle data variables
    data_vars = list(ds.data_vars)
    if len(data_vars) == 0:
        raise HTTPException(
            status_code=400,
            detail="No variables with x and y coordinates found.",
        )

    if len(data_vars) == 1 and len(ds[data_vars[0]].shape) > 3:
        raise HTTPException(
            status_code=400,
            detail=f"Variable {data_vars[0]} has {ds[data_vars[0]].shape} dimensions. "
            "GeoTIFF export only supports up to 3 dimensions when exporting a single variable. "
            f"Found dimensions: {', '.join(ds[data_vars[0]].dims)}",
        )
    else:
        for var in data_vars:
            if len(ds[var].shape) > 2:
                raise HTTPException(
                    status_code=400,
                    detail=f"Variable {var} has {ds[var].shape} dimensions. "
                    "GeoTIFF export only supports up to 2 dimensions when exporting multiple variables. "
                    f"Found dimensions: {', '.join(ds[var].dims)}",
                )

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
