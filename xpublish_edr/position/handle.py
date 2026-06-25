"""Select and format data for an EDR position query."""

import shapely
import xarray as xr
from fastapi import HTTPException
from shapely.errors import GEOSException

from xpublish_edr.format import position_formats
from xpublish_edr.formats.to_covjson import to_cf_covjson
from xpublish_edr.geometry.common import (
    prepare_spatial_grid,
    project_dataset,
    project_geometry,
)
from xpublish_edr.logger import logger
from xpublish_edr.utils import _load_dataset

from .geom import select_by_position
from .params import EDRPositionQueryPost


def handle_query(
    dataset: xr.Dataset,
    query: EDRPositionQueryPost,
    query_params: dict,
    geometry: shapely.Geometry,
):
    """Select and format data for an EDR position query.

    ``geometry`` is the point(s) to query, parsed from the ``coords`` query
    parameter on GET or from the request body on POST by the caller. Shared by
    the dataset-level ``/edr/position`` routes and the OGC-core
    ``/collections/{id}/position`` routes.
    """
    try:
        ds = query.select(dataset, query_params)
    except ValueError as e:
        logger.error(
            f"Error selecting from query while selecting by position: {e}",
        )
        raise HTTPException(
            status_code=404,
            detail=f"Error selecting from query: {e.args[0]}",
        )

    logger.debug(f"Dataset filtered by query params {ds}")

    try:
        grid = prepare_spatial_grid(ds, require_regular=True)
        projected_geometry = project_geometry(
            grid.ds,
            query.crs,
            geometry,
            grid.spatial_ref,
        )
        ds = select_by_position(
            grid.ds,
            projected_geometry,
            query.method,
            grid.spatial_ref,
        )
    except GEOSException as e:
        logger.error(
            f"Error parsing coordinates to geometry while selecting by position: {e}",
        )
        raise HTTPException(
            status_code=422,
            detail="Could not parse coordinates to geometry, "
            + "check the format of the 'coords' query parameter",
        )
    except KeyError as e:
        logger.error(f"Error selecting by position: {e}")
        raise HTTPException(
            status_code=404,
            detail=(
                f"Error selecting by position: {e}. "
                "Ensure that dataset has valid CF metadata and has 1D coordinates."
            ),
        )

    logger.debug(f"Dataset filtered by position ({geometry}): {ds}")

    try:
        ds = project_dataset(ds, query.crs, grid.spatial_ref)
    except Exception as e:
        logger.error(
            f"Error projecting dataset while selecting by position: {e}",
        )
        raise HTTPException(
            status_code=404,
            detail="Error projecting dataset",
        )

    logger.debug(f"Dataset projected to {query.crs}: {ds}")

    ds = _load_dataset(ds)

    logger.debug("Dataset loaded")

    if query.format:
        try:
            format_fn = position_formats()[query.format]
        except KeyError as e:
            logger.error(
                f"Error getting format function while selecting by position: {e}",
            )
            raise HTTPException(
                404,
                f"{query.format} is not a valid format for EDR position queries. "
                "Get `./position/formats` for valid formats",
            )

        return format_fn(ds)

    return to_cf_covjson(ds)
