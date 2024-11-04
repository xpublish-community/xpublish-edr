"""
Handle selection and formatting for position queries
"""

import numpy as np
import shapely
import xarray as xr

from xpublish_edr.geometry.common import VECTORIZED_DIM, is_regular_xy_coords


def select_by_position(
    ds: xr.Dataset, point: shapely.Point | shapely.MultiPoint
) -> xr.Dataset:
    """
    Return a dataset with the position nearest to the given coordinates
    """
    if not is_regular_xy_coords(ds):
        # TODO: Handle 2D coordinates
        raise NotImplementedError("Only 1D coordinates are supported")

    if isinstance(point, shapely.Point):
        return _select_by_position_regular_xy_grid(ds, point)
    elif isinstance(point, shapely.MultiPoint):
        return _select_by_multiple_positions_regular_xy_grid(ds, point)
    else:
        raise ValueError(
            f"Invalid point type {point.geom_type}, must be Point or MultiPoint"
        )


def _select_by_position_regular_xy_grid(
    ds: xr.Dataset,
    point: shapely.Point,
) -> xr.Dataset:
    """
    Return a dataset with the position nearest to the given coordinates
    """
    # Find the nearest X and Y coordinates to the point
    return ds.cf.sel(X=point.x, Y=point.y, method="nearest")


def _select_by_multiple_positions_regular_xy_grid(
    ds: xr.Dataset,
    points: shapely.MultiPoint,
) -> xr.Dataset:
    """
    Return a dataset with the positions nearest to the given coordinates
    """
    # Find the nearest X and Y coordinates to the point using vectorized indexing
    x, y = np.array(list(zip(*[(point.x, point.y) for point in points.geoms])))
    sel_x = xr.Variable(data=x, dims=VECTORIZED_DIM)
    sel_y = xr.Variable(data=y, dims=VECTORIZED_DIM)
    return ds.cf.sel(X=sel_x, Y=sel_y, method="nearest")
