"""
Handle selection and formatting for area queries
"""

import numpy as np
import shapely
import xarray as xr

from xpublish_edr.geometry.common import VECTORIZED_DIM, is_regular_xy_coords


def select_by_area(
    ds: xr.Dataset,
    polygon: shapely.Polygon,
) -> xr.Dataset:
    """
    Return a dataset with the area within the given polygon
    """
    if not is_regular_xy_coords(ds):
        # TODO: Handle 2D coordinates
        raise NotImplementedError("Only 1D coordinates are supported")
    return _select_area_regular_xy_grid(ds, polygon)


def _select_area_regular_xy_grid(
    ds: xr.Dataset,
    polygon: shapely.Polygon,
) -> xr.Dataset:
    """
    Return a dataset with the area within the given polygon
    """
    # To minimize performance impact, we first subset the dataset to the bounding box of the polygon
    (minx, miny, maxx, maxy) = polygon.bounds
    ds = ds.cf.sel(X=slice(minx, maxx), Y=slice(maxy, miny))

    # For a regular grid, we can create a meshgrid of the X and Y coordinates to create a spatial mask
    pts = np.meshgrid(ds.cf["X"], ds.cf["Y"])

    # Create a mask of the points within the polygon
    mask = shapely.intersects_xy(polygon, pts[0], pts[1])

    # Find the x and y indices that have any points within the polygon
    x_inds, y_inds = np.nonzero(mask)
    x_sel = xr.Variable(data=x_inds, dims=VECTORIZED_DIM)
    y_sel = xr.Variable(data=y_inds, dims=VECTORIZED_DIM)

    # Apply the mask and vectorize to a 1d collection of points
    ds_sub = ds.cf.isel(X=x_sel, Y=y_sel)
    return ds_sub
