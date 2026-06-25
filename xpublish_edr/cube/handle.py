"""Select and format data for an EDR cube query."""

import xarray as xr
from fastapi import HTTPException

from xpublish_edr.format import cube_formats
from xpublish_edr.formats.to_covjson import to_cf_covjson
from xpublish_edr.geometry.bbox import select_by_bbox
from xpublish_edr.geometry.common import (
    prepare_spatial_grid,
    project_bbox,
    project_dataset,
)
from xpublish_edr.logger import logger
from xpublish_edr.utils import _load_dataset

from .params import EDRCubeQuery


def handle_query(
    dataset: xr.Dataset,
    query: EDRCubeQuery,
    query_params: dict,
):
    """Select and format data for an EDR cube query.

    Shared by the dataset-level ``/edr/cube`` route and the OGC-core
    ``/collections/{id}/cube`` route.
    """
    try:
        ds = query.select(dataset, query_params)
    except ValueError as e:
        logger.error(f"Error selecting from query while selecting by cube: {e}")
        raise HTTPException(
            status_code=404,
            detail=f"Error selecting from query: {e.args[0]}",
        )

    logger.debug(f"Dataset filtered by query params {ds}")

    try:
        grid = prepare_spatial_grid(ds, require_regular=True)
        bbox = project_bbox(grid.ds, query.crs, query.bbox, grid.spatial_ref)
        ds = select_by_bbox(grid.ds, bbox, grid.spatial_ref)
    except KeyError as e:
        logger.error(f"Error selecting by bbox: {e}")
        raise HTTPException(
            status_code=404,
            detail="Dataset does not have CF Convention compliant metadata",
        )
    except ValueError as e:
        logger.error(f"Error selecting by bbox: {e}")
        raise HTTPException(
            status_code=404,
            detail="Error selecting by bbox, see logs for more details",
        )

    logger.debug(
        f"Dataset filtered by bbox ({query.bbox}): {ds}",
    )

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
            format_fn = cube_formats()[query.format]
        except KeyError as e:
            logger.error(f"Error getting format function: {e}")
            raise HTTPException(
                404,
                f"{query.format} is not a valid format for EDR cube queries. "
                "Get `./cube/formats` for valid formats",
            )

        return format_fn(ds)

    return to_cf_covjson(ds)
