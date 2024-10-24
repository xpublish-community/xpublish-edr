import logging
from typing import Optional

import numpy as np
import shapely
import xarray as xr

from xpublish_edr.query import EDRQuery, edr_query_params

logger = logging.getLogger("cf_edr")


def select_query(ds: xr.Dataset, query: EDRQuery, query_params: dict) -> xr.Dataset:
    """Select data from a dataset based on the query"""
    if query.z:
        ds = ds.cf.sel(Z=query.z, method="nearest")

    if query.datetime:
        try:
            datetimes = query.datetime.split("/")
            if len(datetimes) == 1:
                ds = ds.cf.sel(T=datetimes[0], method="nearest")
            elif len(datetimes) == 2:
                ds = ds.cf.sel(T=slice(datetimes[0], datetimes[1]))
            else:
                raise ValueError(
                    f"Invalid datetimes submitted - {datetimes}",
                )
        except ValueError as e:
            logger.error("Error with datetime", exc_info=True)
            raise ValueError(f"Invalid datetime ({e})") from e

    if query.parameters:
        try:
            ds = ds.cf[query.parameters.split(",")]
        except KeyError as e:
            raise ValueError(f"Invalid variable: {e}") from e

    query_param_keys = list(query_params.keys())
    for query_param in query_param_keys:
        if query_param in edr_query_params:
            del query_params[query_param]

    method: Optional[str] = "nearest"

    for key, value in query_params.items():
        split_value = value.split("/")
        if len(split_value) == 1:
            continue
        elif len(split_value) == 2:
            query_params[key] = slice(split_value[0], split_value[1])
            method = None
        else:
            raise ValueError(f"Too many values for selecting {key}")

    ds = ds.sel(query_params, method=method)
    return ds


def select_postition(ds: xr.Dataset, point: shapely.Point) -> xr.Dataset:
    """
    Return a dataset with the position nearest to the given coordinates
    """
    if _is_regular_xy_coords(ds):
        return _select_position_regular_xy_grid(ds, point)
    else:
        # TODO: Handle 2D coordinates
        raise NotImplementedError("Only 1D coordinates are supported")


def select_area(
    ds: xr.Dataset, polygon: shapely.Polygon, buffer: float = 0.0001,
) -> xr.Dataset:
    """
    Return a dataset with the area within the given polygon
    """
    if _is_regular_xy_coords(ds):
        return _select_area_regular_xy_grid(ds, polygon, buffer)
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


def _select_position_regular_xy_grid(
    ds: xr.Dataset,
    point: shapely.Point,
) -> xr.Dataset:
    """
    Return a dataset with the position nearest to the given coordinates
    """
    # Find the nearest X and Y coordinates to the point
    return ds.cf.sel(X=point.x, Y=point.y, method="nearest")


def _select_area_regular_xy_grid(
    ds: xr.Dataset,
    polygon: shapely.Polygon,
    buffer: float = 0.0001,
) -> xr.Dataset:
    """
    Return a dataset with the area within the given polygon
    """
    # To minimize performance impact, we first subset the dataset to the bounding box of the polygon
    polygon = polygon.buffer(buffer)
    (minx, miny, maxx, maxy) = polygon.bounds
    ds = ds.cf.sel(X=slice(minx, maxx), Y=slice(maxy, miny))

    # For a regular grid, we can create a meshgrid of the X and Y coordinates to create a spatial mask
    pts = np.meshgrid(ds.cf["X"], ds.cf["Y"])

    # Create a mask of the points within the polygon
    mask = shapely.contains_xy(polygon, pts[0], pts[1])

    # Find the x and y indices that have any points within the polygon
    x_inds, y_inds = np.nonzero(mask)
    x_sel = xr.Variable(data=x_inds, dims="pts")
    y_sel = xr.Variable(data=y_inds, dims="pts")

    # Apply the mask and vectorize to a 1d collection of points
    ds_sub = ds.cf.isel(X=x_sel, Y=y_sel)
    return ds_sub
