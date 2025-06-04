"""
Generate GeoTIFF responses for an xarray dataset for EDR queries
"""

import xarray as xr
from fastapi import Response


def to_geotiff(ds: xr.Dataset) -> Response:
    """Return a GeoTIFF response from an xarray dataset"""
    import io

    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    # Get the first data variable
    var_name = list(ds.data_vars)[0]
    data = ds[var_name].values

    # Handle different dimensions
    if len(data.shape) == 2:
        # 2D data (x,y)
        data = np.expand_dims(data, axis=0)
    elif len(data.shape) > 3:
        # More than 3D, take first slice
        data = data[0]

    # Get coordinates
    x_coords = ds[ds[var_name].dims[-1]].values
    y_coords = ds[ds[var_name].dims[-2]].values

    # Calculate pixel size
    x_res = (x_coords[-1] - x_coords[0]) / (len(x_coords) - 1)
    y_res = (y_coords[-1] - y_coords[0]) / (len(y_coords) - 1)

    # Create transform
    transform = from_origin(x_coords[0], y_coords[-1], x_res, y_res)

    # Create in-memory GeoTIFF
    memfile = io.BytesIO()
    with rasterio.open(
        memfile,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)

    # Reset buffer position
    memfile.seek(0)

    # Return FastAPI response
    return Response(
        content=memfile.getvalue(),
        media_type="image/tiff",
        headers={"Content-Disposition": "attachment; filename=data.tiff"},
    )
