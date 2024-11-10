"""
OGC EDR router for datasets with CF convention metadata
"""

import importlib
from typing import List

import xarray as xr
from fastapi import APIRouter, Depends, HTTPException, Request
from xpublish import Dependencies, Plugin, hookimpl

from xpublish_edr.formats.to_covjson import to_cf_covjson
from xpublish_edr.geometry.area import select_by_area
from xpublish_edr.geometry.position import select_by_position
from xpublish_edr.logger import logger
from xpublish_edr.query import EDRQuery, edr_query


def position_formats():
    """
    Return response format functions from registered
    `xpublish_edr_position_formats` entry_points
    """
    formats = {}

    entry_points = importlib.metadata.entry_points()
    for entry_point in entry_points.get("xpublish_edr_position_formats", []):
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
            formats = {key: value.__doc__ for key, value in position_formats().items()}

            return formats

        @router.get(
            "/area/formats",
            summary="Area query response formats",
        )
        def get_area_formats():
            """
            Returns the various supported formats for area queries
            """
            formats = {key: value.__doc__ for key, value in position_formats().items()}

            return formats

        return router

    @hookimpl
    def dataset_router(self, deps: Dependencies):
        """Register dataset level router for EDR endpoints"""
        router = APIRouter(prefix=self.app_router_prefix, tags=self.dataset_router_tags)

        @router.get("/position", summary="Position query")
        def get_position(
            request: Request,
            query: EDRQuery = Depends(edr_query),
            dataset: xr.Dataset = Depends(deps.dataset),
        ):
            """
            Returns position data based on WKT `Point(lon lat)` coordinates

            Extra selecting/slicing parameters can be provided as extra query parameters
            """
            try:
                ds = query.select(dataset, dict(request.query_params))
            except ValueError as e:
                raise HTTPException(
                    status_code=404,
                    detail=f"Error selecting from query: {e.args[0]}",
                )

            logger.debug(f"Dataset filtered by query params {ds}")

            try:
                ds = select_by_position(ds, query.geometry, query.method)
            except KeyError:
                raise HTTPException(
                    status_code=404,
                    detail="Dataset does not have CF Convention compliant metadata",
                )

            logger.debug(
                f"Dataset filtered by position ({query.geometry}): {ds}",
            )

            if query.format:
                try:
                    format_fn = position_formats()[query.format]
                except KeyError:
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
            query: EDRQuery = Depends(edr_query),
            dataset: xr.Dataset = Depends(deps.dataset),
        ):
            """
            Returns area data based on WKT `Polygon(lon lat)` coordinates

            Extra selecting/slicing parameters can be provided as extra query parameters
            """
            try:
                ds = query.select(dataset, dict(request.query_params))
            except ValueError as e:
                raise HTTPException(
                    status_code=404,
                    detail=f"Error selecting from query: {e.args[0]}",
                )

            logger.debug(f"Dataset filtered by query params {ds}")

            try:
                ds = select_by_area(ds, query.geometry)
            except KeyError:
                raise HTTPException(
                    status_code=404,
                    detail="Dataset does not have CF Convention compliant metadata",
                )

            logger.debug(f"Dataset filtered by polygon {query.geometry.boundary}: {ds}")

            if query.format:
                try:
                    format_fn = position_formats()[query.format]
                except KeyError:
                    raise HTTPException(
                        404,
                        f"{query.format} is not a valid format for EDR position queries. "
                        "Get `./formats` for valid formats",
                    )

                return format_fn(ds)

            return to_cf_covjson(ds)

        return router
