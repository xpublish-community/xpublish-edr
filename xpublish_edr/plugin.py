"""
OGC EDR router for datasets with CF convention metadata
"""

import asyncio
from typing import Annotated

import shapely
import xarray as xr
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from shapely.errors import GEOSException
from xpublish import Dependencies, Plugin, hookimpl

from xpublish_edr.format import area_formats, cube_formats, position_formats
from xpublish_edr.formats.to_covjson import to_cf_covjson
from xpublish_edr.geometry.area import select_by_area
from xpublish_edr.geometry.bbox import select_by_bbox
from xpublish_edr.geometry.common import (
    prepare_spatial_grid,
    project_bbox,
    project_dataset,
    project_geometry,
)
from xpublish_edr.geometry.parse import parse_area_body, parse_position_body
from xpublish_edr.geometry.position import select_by_position
from xpublish_edr.logger import logger
from xpublish_edr.metadata import (
    area_query_description,
    collection_metadata,
    cube_query_description,
    dataset_height_units,
    position_query_description,
    supported_crs_details,
)
from xpublish_edr.query import (
    EDRAreaQueryGet,
    EDRAreaQueryPost,
    EDRCubeQuery,
    EDRPositionQueryGet,
    EDRPositionQueryPost,
)

# EDR 1.1 is backwards compatible with 1.0, so both sets of classes are
# declared; the CITE ets-ogcapi-edr10 suite checks for the 1.0 URIs.
# The geojson class is not declared even though `f=geojson` is supported,
# because the class also requires Locations resources (EDR 1.0 Abstract
# Test 21), which are not implemented.
EDR_CONFORMANCE_CLASSES = [
    f"http://www.opengis.net/spec/ogcapi-edr-1/{version}/conf/{conf_class}"
    for version in ("1.0", "1.1")
    for conf_class in (
        "core",
        "collections",
        "json",
        "covjson",
        "queries",
    )
]


def handle_position_query(
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


def handle_area_query(
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


def handle_cube_query(
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


async def _raw_body(request: Request) -> bytes:
    """Read the raw request body as bytes.

    Used as a FastAPI dependency so the position/area endpoints can stay
    synchronous (`def`) handlers that run in the threadpool: the body is read
    here in the async layer -- correctly returning raw bytes regardless of
    content-type -- and the result is passed to the sync endpoint. Returns an
    empty ``bytes`` for GET requests, which have no body.
    """
    return await request.body()


def _load_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Eagerly load the selected dataset, preferring asynchronous loading.

    Backends that support it (e.g. zarr) can fetch chunks concurrently, which
    is significantly faster for remote stores. Backends that don't raise
    ``NotImplementedError``, in which case we fall back to standard
    synchronous loading. Safe to call from the sync handlers since they run
    in the threadpool, where no event loop is running.
    """
    try:
        return asyncio.run(ds.load_async())
    except NotImplementedError:
        return ds.load()


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
            formats = {key: value.__doc__ for key, value in area_formats().items()}

            return formats

        @router.get("/cube/formats", summary="Cube query response formats")
        def get_cube_formats():
            """
            Returns the various supported formats for cube queries
            """
            formats = {key: value.__doc__ for key, value in cube_formats().items()}
            return formats

        return router

    @hookimpl
    def ogc_router(self, deps: Dependencies):
        """Register OGC routers at the application level"""
        # This hook only runs in an app composed with xpublish-ogc-core, so the
        # import is local; OGCExceptionRoute renders validation/HTTP errors as
        # OGC exception objects, which the official OGC schema requires, and
        # OGC_EXCEPTION_RESPONSES documents that body so the OpenAPI matches.
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
            query: Annotated[EDRPositionQueryGet, Query()],
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
            return handle_position_query(
                dataset,
                query,
                dict(request.query_params),
                geometry,
            )

        @router.post(
            "/collections/{collection_id}/position",
            summary="OGC EDR Position endpoint (POST)",
            responses=OGC_EXCEPTION_RESPONSES,
        )
        def post_position(
            collection_id: str,
            request: Request,
            query: Annotated[EDRPositionQueryPost, Query()],
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
                geometry = parse_position_body(body, request.headers.get("content-type"))
            except ValueError as e:
                logger.error(f"Error parsing position body: {e}")
                raise HTTPException(status_code=422, detail=str(e))

            dataset = deps.dataset(collection_id)
            return handle_position_query(
                dataset,
                query,
                dict(request.query_params),
                geometry=geometry,
            )

        @router.get(
            "/collections/{collection_id}/area",
            summary="OGC EDR Area endpoint",
            responses=OGC_EXCEPTION_RESPONSES,
        )
        def get_area(
            collection_id: str,
            request: Request,
            query: Annotated[EDRAreaQueryGet, Query()],
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
            return handle_area_query(
                dataset,
                query,
                dict(request.query_params),
                geometry,
            )

        @router.post(
            "/collections/{collection_id}/area",
            summary="OGC EDR Area endpoint (POST)",
            responses=OGC_EXCEPTION_RESPONSES,
        )
        def post_area(
            collection_id: str,
            request: Request,
            query: Annotated[EDRAreaQueryPost, Query()],
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
                geometry = parse_area_body(body, request.headers.get("content-type"))
            except ValueError as e:
                logger.error(f"Error parsing area body: {e}")
                raise HTTPException(status_code=422, detail=str(e))

            dataset = deps.dataset(collection_id)
            return handle_area_query(
                dataset,
                query,
                dict(request.query_params),
                geometry=geometry,
            )

        @router.get(
            "/collections/{collection_id}/cube",
            summary="OGC EDR Cube endpoint",
            responses=OGC_EXCEPTION_RESPONSES,
        )
        def get_cube(
            collection_id: str,
            request: Request,
            query: Annotated[EDRCubeQuery, Query()],
        ):
            """
            Returns cube data for a collection based on bbox coordinates and optional elevation

            Extra selecting/slicing parameters can be provided as extra query parameters
            """
            dataset = deps.dataset(collection_id)
            return handle_cube_query(dataset, query, dict(request.query_params))

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
                list(position_formats().keys()),
                supported_crs,
                href=f"/collections/{collection_id}/position?coords={{coords}}",
            ),
            "area": area_query_description(
                list(area_formats().keys()),
                supported_crs,
                href=f"/collections/{collection_id}/area?coords={{coords}}",
            ),
            "cube": cube_query_description(
                list(cube_formats().keys()),
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
                list(position_formats().keys()),
                list(area_formats().keys()),
                list(cube_formats().keys()),
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
            position_output_formats = list(position_formats().keys())
            area_output_formats = list(area_formats().keys())
            cube_output_formats = list(cube_formats().keys())
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
            query: Annotated[EDRPositionQueryGet, Query()],
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
            return handle_position_query(
                dataset,
                query,
                dict(request.query_params),
                geometry,
            )

        @router.post("/position", summary="Position query (POST)")
        def post_position(
            request: Request,
            query: Annotated[EDRPositionQueryPost, Query()],
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
                geometry = parse_position_body(
                    body,
                    request.headers.get("content-type"),
                )
            except ValueError as e:
                logger.error(f"Error parsing position body: {e}")
                raise HTTPException(status_code=422, detail=str(e))

            return handle_position_query(
                dataset,
                query,
                dict(request.query_params),
                geometry,
            )

        @router.get("/area", summary="Area query")
        def get_area(
            request: Request,
            query: Annotated[EDRAreaQueryGet, Query()],
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
            return handle_area_query(
                dataset,
                query,
                dict(request.query_params),
                geometry,
            )

        @router.post("/area", summary="Area query (POST)")
        def post_area(
            request: Request,
            query: Annotated[EDRAreaQueryPost, Query()],
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
                geometry = parse_area_body(
                    body,
                    request.headers.get("content-type"),
                )
            except ValueError as e:
                logger.error(f"Error parsing area body: {e}")
                raise HTTPException(status_code=422, detail=str(e))

            return handle_area_query(
                dataset,
                query,
                dict(request.query_params),
                geometry,
            )

        @router.get("/cube", summary="Cube query")
        def get_cube(
            request: Request,
            query: Annotated[EDRCubeQuery, Query()],
            dataset: xr.Dataset = Depends(deps.dataset),
        ):
            """
            Returns gridded cube data based on bbox coordinates and optional elevation

            Extra selecting/slicing parameters can be provided as extra query parameters
            """
            return handle_cube_query(dataset, query, dict(request.query_params))

        return router
