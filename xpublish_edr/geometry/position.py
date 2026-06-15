"""
Handle selection and formatting for position queries
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import shapely
import xarray as xr

from xpublish_edr.geometry.common import (
    SpatialRef,
    VECTORIZED_DIM,
    prepare_spatial_grid,
)


def select_by_position(
    ds: xr.Dataset,
    point: shapely.Point | shapely.MultiPoint,
    method: Literal["nearest", "linear"] = "nearest",
    spatial_ref: SpatialRef | None = None,
) -> xr.Dataset:
    """
    Return a dataset with the position nearest to the given coordinates
    """
    grid = prepare_spatial_grid(ds, spatial_ref=spatial_ref, require_regular=True)
    ds = grid.ds
    X, Y = grid.spatial_ref.X, grid.spatial_ref.Y

    if isinstance(point, shapely.Point):
        return _select_by_position_regular_xy_grid(ds, point, X, Y, method)
    elif isinstance(point, shapely.MultiPoint):
        return _select_by_multiple_positions_regular_xy_grid(ds, point, X, Y, method)
    else:
        raise ValueError(
            f"Invalid point type {point.geom_type}, must be Point or MultiPoint",
        )


def _select_by_position_regular_xy_grid(
    ds: xr.Dataset,
    point: shapely.Point,
    X: str,
    Y: str,
    method: Literal["nearest", "linear"] = "nearest",
) -> xr.Dataset:
    """
    Return a dataset with the position nearest to the given coordinates
    """
    # Find the nearest X and Y coordinates to the point
    if method == "nearest":
        return ds.sel({X: [point.x], Y: [point.y]}, method=method)
    else:
        return ds.interp({X: [point.x], Y: [point.y]}, method=method)


def _select_by_multiple_positions_regular_xy_grid(
    ds: xr.Dataset,
    points: shapely.MultiPoint,
    X: str,
    Y: str,
    method: Literal["nearest", "linear"] = "nearest",
) -> xr.Dataset:
    """
    Return a dataset with the positions nearest to the given coordinates
    """
    # Find the nearest X and Y coordinates to the point using vectorized indexing
    x, y = np.array(list(zip(*[(point.x, point.y) for point in points.geoms])))

    # When using vectorized indexing with interp, we need to persist the attributes explicitly
    sel_x = xr.Variable(data=x, dims=VECTORIZED_DIM, attrs=ds[X].attrs)
    sel_y = xr.Variable(data=y, dims=VECTORIZED_DIM, attrs=ds[Y].attrs)
    if method == "nearest":
        return ds.sel({X: sel_x, Y: sel_y}, method=method)
    else:
        return ds.interp({X: sel_x, Y: sel_y}, method=method)
