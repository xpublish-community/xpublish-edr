import numpy as np
import xarray as xr
from shapely import Point, Polygon


def select_postition(ds: xr.Dataset, point: Point) -> xr.Dataset:
    """
    Return a dataset with the position nearest to the given coordinates
    """
    if _is_regular_xy_coords(ds):
        return _select_position_regular_xy_grid(ds, point)
    else:
        # TODO: Handle 2D coordinates
        raise NotImplementedError("Only 1D coordinates are supported")


def select_area(ds: xr.Dataset, polygon: Polygon) -> xr.Dataset:
    """
    Return a dataset with the area within the given polygon
    """
    if _is_regular_xy_coords(ds):
        return _select_area_regular_xy_grid(ds, polygon)
    else:
        # TODO: Handle 2D coordinates
        raise NotImplementedError("Only 1D coordinates are supported")


def _coord_is_regular(da: xr.DataArray) -> bool:
    """
    Check if the DataArray has a regular grid
    """
    return len(da.shape) == 1 and da.name in da.dims


def _is_regular_xy_coords(ds: xr.Dataset) -> bool:
    """
    Check if the dataset has 2D coordinates
    """
    return _coord_is_regular(ds.cf["X"]) and _coord_is_regular(ds.cf["Y"])


def _select_position_regular_xy_grid(ds: xr.Dataset, point: Point) -> xr.Dataset:
    """
    Return a dataset with the position nearest to the given coordinates
    """
    # Find the nearest X and Y coordinates to the point
    return ds.cf.sel(X=point.x, Y=point.y, method="nearest")


def _select_area_regular_xy_grid(ds: xr.Dataset, polygon: Polygon) -> xr.Dataset:
    """
    Return a dataset with the area within the given polygon
    """
    # For a regular grid, we can create a meshgrid of the X and Y coordinates to create a spatial mask
    pts = np.meshgrid(ds.cf["X"], ds.cf["Y"])

    # Create a mask of the points within the polygon
    contains = np.vectorize(lambda p: polygon.contains(Point(p)), signature="(n)->()")
    mask = contains(np.stack(pts, axis=-1))

    # Find the x and y indices that have any points within the polygon
    x_mask = mask.any(axis=0)
    y_mask = mask.any(axis=1)

    # Create a new DataArray with the mask matching the dataset dimensions
    dims = ds.cf["Y"].dims + ds.cf["X"].dims
    da_mask = xr.DataArray(mask, dims=dims)

    # Apply the mask and subset out any data that does not fall within the polygon bounds
    ds_subset = ds.where(da_mask).cf.isel(X=x_mask, Y=y_mask)
    return ds_subset
