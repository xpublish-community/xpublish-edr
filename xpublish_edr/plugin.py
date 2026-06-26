"""
OGC EDR router for datasets with CF convention metadata
"""

from typing import Annotated

import xarray as xr
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from shapely.errors import GEOSException
from xpublish import Dependencies, Plugin, hookimpl

from xpublish_edr import area, cube, position
from xpublish_edr.logger import logger
from xpublish_edr.metadata import (
    EDR_CONFORMANCE_CLASSES,
    area_query_description,
    collection_metadata,
    cube_query_description,
    dataset_height_units,
    position_query_description,
    supported_crs_details,
)
from xpublish_edr.utils import _raw_body


class CfEdrPlugin(Plugin):
    """
    OGC EDR compatible endpoints for Xpublish datasets
    """

    name: str = "cf_edr"

    app_router_prefix: str = "/edr"
    app_router_tags: list[str] = ["edr"]

    dataset_router_prefix: str = "/edr"
    dataset_router_tags: list[str] = ["edr"]

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
            formats = {key: value.__doc__ for key, value in position.formats().items()}

            return formats

        @router.get(
            "/area/formats",
            summary="Area query response formats",
        )
        def get_area_formats():
            """
            Returns the various supported formats for area queries
            """
            formats = {key: value.__doc__ for key, value in area.formats().items()}

            return formats

        @router.get("/cube/formats", summary="Cube query response formats")
        def get_cube_formats():
            """
            Returns the various supported formats for cube queries
            """
            formats = {key: value.__doc__ for key, value in cube.formats().items()}
            return formats

        return router

    @hookimpl
    def ogc_router(self, deps: Dependencies):
        """Register OGC routers at the application level"""
        # This hook only runs in an app composed with xpublish-ogc-core
        from xpublish_ogc_core.plugin import (
            OGC_EXCEPTION_RESPONSES,
            OGCExceptionRoute,
        )

        router = APIRouter(tags=["OGC EDR"], route_class=OGCExceptionRoute)

        @router.get(
            "/collections/{collection_id}/position",
            summary="OGC EDR Position endpoint",
            responses=OGC_EXCEPTION_RESPONSES,
        )
        def get_position(
            collection_id: str,
            request: Request,
            query: Annotated[position.EDRPositionQueryGet, Query()],
        ):
            """
            Returns position data for a collection based on WKT `Point(lon lat)` coordinates

            Extra selecting/slicing parameters can be provided as extra query parameters
            """
            try:
                geometry = query.geometry
            except GEOSException:
                raise HTTPException(
                    status_code=422,
                    detail="Could not parse coordinates to geometry, "
                    "check the format of the 'coords' query parameter",
                )
            dataset = deps.dataset(collection_id)
            return query.run_query(dataset, dict(request.query_params), geometry)

        @router.post(
            "/collections/{collection_id}/position",
            summary="OGC EDR Position endpoint (POST)",
            responses=OGC_EXCEPTION_RESPONSES,
        )
        def post_position(
            collection_id: str,
            request: Request,
            query: Annotated[position.EDRPositionQueryPost, Query()],
            body: Annotated[bytes, Depends(_raw_body)],
        ):
            """
            Returns position data for a collection, with the point(s) submitted
            in the request body as CSV (`text/csv`) or GeoJSON
            (`application/geo+json`); all other selection parameters are passed
            as query parameters, as for GET.
            """
            if not body:
                raise HTTPException(
                    status_code=422,
                    detail="POST position requires a non-empty request body",
                )
            try:
                geometry = position.parse_body(body, request.headers.get("content-type"))
            except ValueError as e:
                logger.error(f"Error parsing position body: {e}")
                raise HTTPException(status_code=422, detail=str(e))

            dataset = deps.dataset(collection_id)
            return query.run_query(dataset, dict(request.query_params), geometry)

        @router.get(
            "/collections/{collection_id}/area",
            summary="OGC EDR Area endpoint",
            responses=OGC_EXCEPTION_RESPONSES,
        )
        def get_area(
            collection_id: str,
            request: Request,
            query: Annotated[area.EDRAreaQueryGet, Query()],
        ):
            """
            Returns area data for a collection based on WKT `Polygon(lon lat)` coordinates

            Extra selecting/slicing parameters can be provided as extra query parameters
            """
            try:
                geometry = query.geometry
            except GEOSException:
                raise HTTPException(
                    status_code=422,
                    detail="Could not parse coordinates to geometry, "
                    "check the format of the 'coords' query parameter",
                )
            dataset = deps.dataset(collection_id)
            return query.run_query(dataset, dict(request.query_params), geometry)

        @router.post(
            "/collections/{collection_id}/area",
            summary="OGC EDR Area endpoint (POST)",
            responses=OGC_EXCEPTION_RESPONSES,
        )
        def post_area(
            collection_id: str,
            request: Request,
            query: Annotated[area.EDRAreaQueryPost, Query()],
            body: Annotated[bytes, Depends(_raw_body)],
        ):
            """
            Returns area data for a collection, with the polygon submitted in
            the request body as WKT (`text/plain`) or GeoJSON
            (`application/geo+json`); all other selection parameters are passed
            as query parameters, as for GET.
            """
            if not body:
                raise HTTPException(
                    status_code=422,
                    detail="POST area requires a non-empty request body",
                )
            try:
                geometry = area.parse_body(body, request.headers.get("content-type"))
            except ValueError as e:
                logger.error(f"Error parsing area body: {e}")
                raise HTTPException(status_code=422, detail=str(e))

            dataset = deps.dataset(collection_id)
            return query.run_query(dataset, dict(request.query_params), geometry)

        @router.get(
            "/collections/{collection_id}/cube",
            summary="OGC EDR Cube endpoint",
            responses=OGC_EXCEPTION_RESPONSES,
        )
        def get_cube(
            collection_id: str,
            request: Request,
            query: Annotated[cube.EDRCubeQuery, Query()],
        ):
            """
            Returns cube data for a collection based on bbox coordinates and optional elevation

            Extra selecting/slicing parameters can be provided as extra query parameters
            """
            dataset = deps.dataset(collection_id)
            return query.run_query(dataset, dict(request.query_params))

        return router

    @hookimpl
    def ogc_conformance_classes(self) -> list[str]:
        """Declare the OGC EDR conformance classes implemented by this plugin"""
        return EDR_CONFORMANCE_CLASSES

    @hookimpl
    def ogc_collection_dataqueries(self, collection_id: str, ds: xr.Dataset):
        """Register data queries for OGC collection metadata

        Every supported EDR geometry query (position, area, and cube) is
        described, reusing the metadata machinery from `xpublish_edr.metadata`.
        """
        try:
            supported_crs = supported_crs_details(ds)
        except Exception as e:
            logger.warning(
                f"Can not describe EDR data queries for {collection_id}: {e}",
            )
            return None

        data_queries = {
            "position": position_query_description(
                list(position.formats().keys()),
                supported_crs,
                href=f"/collections/{collection_id}/position?coords={{coords}}",
            ),
            "area": area_query_description(
                list(area.formats().keys()),
                supported_crs,
                href=f"/collections/{collection_id}/area?coords={{coords}}",
            ),
            "cube": cube_query_description(
                list(cube.formats().keys()),
                supported_crs,
                href=f"/collections/{collection_id}/cube?bbox={{bbox}}",
                height_units=dataset_height_units(ds),
            ),
        }

        return {
            name: query.model_dump(exclude_none=True, by_alias=True)
            for name, query in data_queries.items()
        }

    @hookimpl
    def ogc_collection_metadata(self, collection_id: str, ds: xr.Dataset):
        """Contribute EDR collection metadata to OGC collection objects

        Reuses `collection_metadata` so `/collections/{collection_id}` served by
        xpublish-ogc-core carries the same extent, parameter_names, crs, and
        output_formats as the dataset level EDR metadata endpoint.
        """
        try:
            metadata = collection_metadata(
                ds,
                list(position.formats().keys()),
                list(area.formats().keys()),
                list(cube.formats().keys()),
            ).model_dump(exclude_none=True, by_alias=True)
        except Exception as e:
            logger.warning(
                f"Can not build EDR collection metadata for {collection_id}: {e}",
            )
            return None

        # links and data queries are built by ogc-core (the data queries via the
        # ogc_collection_dataqueries hook, with collection scoped hrefs), and the
        # id is the dataset id rather than the `_xpublish_id` attr
        for key in ("id", "links", "data_queries"):
            metadata.pop(key, None)

        return metadata

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
            position_output_formats = list(position.formats().keys())
            area_output_formats = list(area.formats().keys())
            cube_output_formats = list(cube.formats().keys())
            return collection_metadata(
                dataset,
                position_output_formats,
                area_output_formats,
                cube_output_formats,
            ).dict(
                exclude_none=True,
            )

        # GET and POST are registered as separate routes (not a single
        # multi-method api_route) so that the OpenAPI marks `coords` as a
        # required query parameter on GET while POST omits it entirely (the
        # points come from the body). The OGC EDR CITE suite scans every path
        # ending in `/position` and requires the `coords` parameter to be
        # `required: true`, which a shared GET/POST signature cannot express.
        @router.get("/position", summary="Position query")
        def get_position(
            request: Request,
            query: Annotated[position.EDRPositionQueryGet, Query()],
            dataset: xr.Dataset = Depends(deps.dataset),
        ):
            """
            Returns vectorized position data for one or more points passed as
            WKT via the `coords` query parameter.

            Other selection parameters (datetime, z, parameter-name, crs, f,
            method) are passed as query string parameters.

            This endpoint is intentionally a synchronous (`def`) handler so that
            Starlette runs it in the threadpool. The select/project/format
            pipeline is CPU-bound and blocking; running it on the event loop
            would starve other requests (e.g. ``/health``).
            """
            try:
                geometry = query.geometry
            except GEOSException:
                raise HTTPException(
                    status_code=422,
                    detail="Could not parse coordinates to geometry, "
                    "check the format of the 'coords' query parameter",
                )
            return query.run_query(dataset, dict(request.query_params), geometry)

        @router.post("/position", summary="Position query (POST)")
        def post_position(
            request: Request,
            query: Annotated[position.EDRPositionQueryPost, Query()],
            body: Annotated[bytes, Depends(_raw_body)],
            dataset: xr.Dataset = Depends(deps.dataset),
        ):
            """
            Returns vectorized position data for one or more points submitted in
            the request body as either CSV (Content-Type: text/csv) with x/y,
            lon/lat, or longitude/latitude columns, or GeoJSON (Content-Type:
            application/geo+json) as a Point, MultiPoint, Feature,
            FeatureCollection, or GeometryCollection.

            Other selection parameters (datetime, z, parameter-name, crs, f,
            method) are passed as query string parameters. Synchronous (`def`)
            for the same reason as ``get_position``; the body is injected via a
            dependency so no ``await`` is needed.
            """
            if not body:
                raise HTTPException(
                    status_code=422,
                    detail="POST /position requires a non-empty request body",
                )
            try:
                geometry = position.parse_body(
                    body,
                    request.headers.get("content-type"),
                )
            except ValueError as e:
                logger.error(f"Error parsing position body: {e}")
                raise HTTPException(status_code=422, detail=str(e))

            return query.run_query(dataset, dict(request.query_params), geometry)

        @router.get("/area", summary="Area query")
        def get_area(
            request: Request,
            query: Annotated[area.EDRAreaQueryGet, Query()],
            dataset: xr.Dataset = Depends(deps.dataset),
        ):
            """
            Returns vectorized area data for a Polygon or MultiPolygon passed as
            WKT via the `coords` query parameter.

            Other selection parameters (datetime, z, parameter-name, crs, f,
            method) are passed as query string parameters. Synchronous (`def`)
            for the same reason as ``get_position``: the query pipeline is
            blocking CPU work and must run in the threadpool.
            """
            try:
                geometry = query.geometry
            except GEOSException:
                raise HTTPException(
                    status_code=422,
                    detail="Could not parse coordinates to geometry, "
                    "check the format of the 'coords' query parameter",
                )
            return query.run_query(dataset, dict(request.query_params), geometry)

        @router.post("/area", summary="Area query (POST)")
        def post_area(
            request: Request,
            query: Annotated[area.EDRAreaQueryPost, Query()],
            body: Annotated[bytes, Depends(_raw_body)],
            dataset: xr.Dataset = Depends(deps.dataset),
        ):
            """
            Returns vectorized area data for a polygon submitted in the request
            body as either GeoJSON (Content-Type: application/geo+json) --
            Polygon, MultiPolygon, Feature, FeatureCollection, or
            GeometryCollection -- or raw WKT (Content-Type: application/wkt or
            text/plain).

            Other selection parameters (datetime, z, parameter-name, crs, f,
            method) are passed as query string parameters. Synchronous (`def`)
            for the same reason as ``get_area``; the body is injected via a
            dependency so no ``await`` is needed.
            """
            if not body:
                raise HTTPException(
                    status_code=422,
                    detail="POST /area requires a non-empty request body",
                )
            try:
                geometry = area.parse_body(
                    body,
                    request.headers.get("content-type"),
                )
            except ValueError as e:
                logger.error(f"Error parsing area body: {e}")
                raise HTTPException(status_code=422, detail=str(e))

            return query.run_query(dataset, dict(request.query_params), geometry)

        @router.get("/cube", summary="Cube query")
        def get_cube(
            request: Request,
            query: Annotated[cube.EDRCubeQuery, Query()],
            dataset: xr.Dataset = Depends(deps.dataset),
        ):
            """
            Returns gridded cube data based on bbox coordinates and optional elevation

            Extra selecting/slicing parameters can be provided as extra query parameters
            """
            return query.run_query(dataset, dict(request.query_params))

        return router
