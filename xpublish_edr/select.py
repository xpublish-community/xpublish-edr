import numpy as np
import xarray as xr
from shapely import Point, Polygon


def select_postition(ds: xr.Dataset, point: Point) -> xr.Dataset:
    """
    Return a dataset with the position nearest to the given coordinates
    """
    # TODO: Handle 2D coordinates
    return ds.cf.sel(X=point.x, Y=point.y, method="nearest")


def select_area(ds: xr.Dataset, polygon: Polygon) -> xr.Dataset:
    """
    Return a dataset with the area within the given polygon
    """
    contains = np.vectorize(lambda p: polygon.contains(Point(p)), signature="(n)->()")

    # TODO: Handle 2D coordinates
    pts = np.meshgrid(ds.cf['X'], ds.cf['Y'])

    # Create a mask of the points within the polygon
    contains = np.vectorize(lambda p: polygon.contains(Point(p)), signature="(n)->()")
    mask = contains(np.stack(pts, axis=-1))

    # Find the x and y indices that have any points within the polygon
    x_mask = mask.any(axis=0)
    y_mask = mask.any(axis=1)

    # Create a new DataArray with the mask matching the dataset dimensions
    dims = ds.cf['Y'].dims + ds.cf['X'].dims
    da_mask = xr.DataArray(mask, dims=dims)

    # Apply the mask and subset out any data that does not fall within the polygon bounds
    ds_subset = ds.where(da_mask).cf.isel(X=x_mask, Y=y_mask)
    return ds_subset
