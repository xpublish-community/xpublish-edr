"""Select and format data for an EDR area query."""

import shapely
import xarray as xr
from fastapi import HTTPException
from shapely.errors import GEOSException

from xpublish_edr.format import area_formats
from xpublish_edr.formats.to_covjson import to_cf_covjson
from xpublish_edr.geometry.common import (
    prepare_spatial_grid,
    project_dataset,
    project_geometry,
)
from xpublish_edr.logger import logger
from xpublish_edr.utils import _load_dataset

from .geom import select_by_area
from .params import EDRAreaQueryPost


def handle_query(
    dataset: xr.Dataset,
    query: EDRAreaQueryPost,
    query_params: dict,
    geometry: shapely.Geometry,
):
    """Select and format data for an EDR area query.

    ``geometry`` is the polygon to query, parsed from the ``coords`` query
    parameter on GET or from the request body on POST by the caller. Shared by
    the dataset-level ``/edr/area`` routes and the OGC-core
    ``/collections/{id}/area`` routes.
    """
    try:
        ds = query.select(dataset, query_params)
    except ValueError as e:
        logger.error(f"Error selecting from query while selecting by area: {e}")
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
        ds = select_by_area(grid.ds, projected_geometry, grid.spatial_ref)
    except GEOSException as e:
        logger.error(
            f"Error parsing coordinates to geometry while selecting by area: {e}",
        )
        raise HTTPException(
            status_code=422,
            detail="Could not parse coordinates to geometry, "
            + "check the format of the 'coords' query parameter",
        )
    except KeyError as e:
        logger.error(f"Error selecting by area: {e}")
        raise HTTPException(
            status_code=404,
            detail="Dataset does not have CF Convention compliant metadata",
        )

    logger.debug(f"Dataset filtered by polygon {geometry.boundary}: {ds}")

    try:
        ds = project_dataset(ds, query.crs, grid.spatial_ref)
    except Exception as e:
        logger.error(f"Error projecting dataset while selecting by area: {e}")
        raise HTTPException(
            status_code=404,
            detail="Error projecting dataset",
        )

    logger.debug(f"Dataset projected to {query.crs}: {ds}")

    ds = _load_dataset(ds)

    logger.debug("Dataset loaded")

    if query.format:
        try:
            format_fn = area_formats()[query.format]
        except KeyError as e:
            logger.error(f"Error getting format function: {e}")
            raise HTTPException(
                404,
                f"{query.format} is not a valid format for EDR area queries. "
                "Get `./area/formats` for valid formats",
            )

        return format_fn(ds)

    return to_cf_covjson(ds)
