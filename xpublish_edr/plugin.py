"""
OGC EDR router for datasets with CF convention metadata
"""

import importlib
from typing import Annotated, List

import xarray as xr
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from shapely.errors import GEOSException
from xpublish import Dependencies, Plugin, hookimpl

from xpublish_edr.formats.to_covjson import to_cf_covjson
from xpublish_edr.geometry.area import select_by_area
from xpublish_edr.geometry.common import project_dataset
from xpublish_edr.geometry.position import select_by_position
from xpublish_edr.logger import logger
from xpublish_edr.metadata import collection_metadata
from xpublish_edr.query import EDRAreaQuery, EDRCubeQuery, EDRPositionQuery


def output_formats():
    """
    Return response format functions from registered
    `xpublish_edr_position_formats` entry_points
    """
    formats = {}

    entry_points = importlib.metadata.entry_points()
    for entry_point in entry_points.select(group="xpublish_edr_position_formats"):
        formats[entry_point.name] = entry_point.load()

    return formats


class CfEdrPlugin(Plugin):
    """
    OGC EDR compatible endpoints for Xpublish datasets
    """

    name: str = "cf_edr"

    app_router_prefix: str = "/edr"
    app_router_tags: List[str] = ["edr"]

    dataset_router_prefix: str = "/edr"
    dataset_router_tags: List[str] = ["edr"]

    @hookimpl
    def app_router(self):
        """Register an application level router for EDR format info"""
        router = APIRouter(prefix=self.app_router_prefix, tags=self.app_router_tags)

        @router.get(
            "/position/formats",
            summary="Position query response formats",
        )
        def get_position_formats():
            """
            Returns the various supported formats for position queries
            """
            formats = {key: value.__doc__ for key, value in output_formats().items()}

            return formats

        @router.get(
            "/area/formats",
            summary="Area query response formats",
        )
        def get_area_formats():
            """
            Returns the various supported formats for area queries
            """
            formats = {key: value.__doc__ for key, value in output_formats().items()}

            return formats

        @router.get("/cube/formats", summary="Cube query response formats")
        def get_cube_formats():
            """
            Returns the various supported formats for cube queries
            """
            formats = {key: value.__doc__ for key, value in output_formats().items()}
            return formats

        return router

    @hookimpl
    def dataset_router(self, deps: Dependencies):
        """Register dataset level router for EDR endpoints"""
        router = APIRouter(prefix=self.app_router_prefix, tags=self.dataset_router_tags)

        @router.get("/", summary="Collection metadata")
        def get_collection_metadata(dataset: xr.Dataset = Depends(deps.dataset)):
            """
            Returns the collection metadata for the dataset

            There is no nested hierarchy in our router right now, so instead we return the metadata
            for the current dataset as the a single collection. See the spec for more information:
            https://docs.ogc.org/is/19-086r6/19-086r6.html#_162817c2-ccd7-43c9-b1ea-ad3aea1b4d6b
            """
            available_output_formats = list(output_formats().keys())
            return collection_metadata(dataset, available_output_formats).dict(
                exclude_none=True,
            )

        @router.get("/position", summary="Position query")
        def get_position(
            request: Request,
            query: Annotated[EDRPositionQuery, Query()],
            dataset: xr.Dataset = Depends(deps.dataset),
        ):
            """
            Returns position data based on WKT `Point(lon lat)` coordinates

            Extra selecting/slicing parameters can be provided as extra query parameters
            """
            try:
                ds = query.select(dataset, dict(request.query_params))
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
                ds = select_by_position(ds, query.project_geometry(ds), query.method)
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
                    detail="Dataset does not have CF Convention compliant metadata",
                )

            logger.debug(
                f"Dataset filtered by position ({query.geometry}): {ds}",
            )

            try:
                ds = project_dataset(ds, query.crs)
            except Exception as e:
                logger.error(
                    f"Error projecting dataset while selecting by position: {e}",
                )
                raise HTTPException(
                    status_code=404,
                    detail="Error projecting dataset",
                )

            logger.debug(f"Dataset projected to {query.crs}: {ds}")

            if query.format:
                try:
                    format_fn = output_formats()[query.format]
                except KeyError as e:
                    logger.error(
                        f"Error getting format function while selecting by position: {e}",
                    )
                    raise HTTPException(
                        404,
                        f"{query.format} is not a valid format for EDR position queries. "
                        "Get `./formats` for valid formats",
                    )

                return format_fn(ds)

            return to_cf_covjson(ds)

        @router.get("/area", summary="Area query")
        def get_area(
            request: Request,
            query: Annotated[EDRAreaQuery, Query()],
            dataset: xr.Dataset = Depends(deps.dataset),
        ):
            """
            Returns area data based on WKT `Polygon(lon lat)` coordinates

            Extra selecting/slicing parameters can be provided as extra query parameters
            """
            try:
                ds = query.select(dataset, dict(request.query_params))
            except ValueError as e:
                logger.error(f"Error selecting from query while selecting by area: {e}")
                raise HTTPException(
                    status_code=404,
                    detail=f"Error selecting from query: {e.args[0]}",
                )

            logger.debug(f"Dataset filtered by query params {ds}")

            try:
                ds = select_by_area(ds, query.project_geometry(ds))
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

            logger.debug(f"Dataset filtered by polygon {query.geometry.boundary}: {ds}")

            try:
                ds = project_dataset(ds, query.crs)
            except Exception as e:
                logger.error(f"Error projecting dataset while selecting by area: {e}")
                raise HTTPException(
                    status_code=404,
                    detail="Error projecting dataset",
                )

            logger.debug(f"Dataset projected to {query.crs}: {ds}")

            if query.format:
                try:
                    format_fn = output_formats()[query.format]
                except KeyError as e:
                    logger.error(f"Error getting format function: {e}")
                    raise HTTPException(
                        404,
                        f"{query.format} is not a valid format for EDR position queries. "
                        "Get `./formats` for valid formats",
                    )

                return format_fn(ds)

            return to_cf_covjson(ds)

        @router.get("/cube", summary="Cube query")
        def get_cube(
            request: Request,
            query: Annotated[EDRCubeQuery, Query()],
            dataset: xr.Dataset = Depends(deps.dataset),
        ):
            """
            Returns cube data based on bbox coordinates and optional elevation
            """
            pass

        return router
