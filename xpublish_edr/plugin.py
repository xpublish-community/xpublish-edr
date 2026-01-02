"""
OGC EDR router for datasets with CF convention metadata
"""

import asyncio
from typing import Annotated, List

import shapely
import xarray as xr
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from shapely.errors import GEOSException
from xpublish import Dependencies, Plugin, hookimpl

from xpublish_edr.format import area_formats, cube_formats, position_formats
from xpublish_edr.formats.to_covjson import to_cf_covjson
from xpublish_edr.geometry.area import select_by_area
from xpublish_edr.geometry.bbox import select_by_bbox
from xpublish_edr.geometry.common import project_dataset, project_geometry
from xpublish_edr.geometry.parse import parse_area_body, parse_position_body
from xpublish_edr.geometry.position import select_by_position
from xpublish_edr.logger import logger
from xpublish_edr.metadata import collection_metadata
from xpublish_edr.query import EDRAreaQuery, EDRCubeQuery, EDRPositionQuery


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
        router = APIRouter(tags=["OGC EDR"])

        @router.get("/collections/{collection_id}/position", summary="OGC EDR Position endpoint")
        def get_position(collection_id: str, request: Request, query: Annotated[EDRPositionQuery, Query()],):
            """Stub for OGC EDR position endpoint"""
            dataset = deps.dataset(collection_id)
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

        return router
    
    @hookimpl
    def ogc_collection_dataqueries(self, collection_id: str, ds: xr.Dataset):
        """Register data queries for OGC collection metadata"""
        return {
            "position": {
                "link": {
                    "href": f"/collections/{collection_id}/position",
                    "variables": {},
                    "rel": "alternate",
                    "type": "application/geo+json",
                    "hreflang": "en",
                    "title": "OGC EDR Position Query Endpoint",
                    "length": 0,
                    "templated": True
                }
            }
        }

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

        def _run_position_query(
            dataset: xr.Dataset,
            query: EDRPositionQuery,
            geometry: shapely.Geometry,
            query_params: dict,
        ):
            """Shared select/project/format pipeline for GET and POST position queries."""
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
                projected_geometry = project_geometry(ds, query.crs, geometry)
                ds = select_by_position(ds, projected_geometry, query.method)
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
                        "Ensure that dataset has valid CF metadatad has 1D coordinates."
                    ),
                )

            logger.debug(f"Dataset filtered by position ({geometry}): {ds}")

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

        @router.api_route(
            "/position",
            methods=["GET", "POST"],
            summary="Position query",
        )
        def position(
            request: Request,
            query: Annotated[EDRPositionQuery, Query()],
            body: Annotated[bytes, Depends(_raw_body)],
            dataset: xr.Dataset = Depends(deps.dataset),
        ):
            """
            Returns vectorized position data for one or more points.

            For GET, points are passed as WKT via the `coords` query parameter.

            For POST, points are submitted in the request body as either CSV
            (Content-Type: text/csv) with x/y, lon/lat, or longitude/latitude
            columns, or GeoJSON (Content-Type: application/geo+json) as a Point,
            MultiPoint, Feature, FeatureCollection, or GeometryCollection.

            All other selection parameters (datetime, z, parameter-name, crs, f,
            method) are passed as query string parameters in both cases.

            This endpoint is intentionally a synchronous (`def`) handler so that
            Starlette runs it in the threadpool. The select/project/format
            pipeline is CPU-bound and blocking; running it on the event loop
            would starve other requests (e.g. ``/health``). The request body is
            injected via FastAPI's ``Body`` so we never need ``await`` here.
            """
            if request.method == "POST":
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
            else:
                if query.coords is None:
                    raise HTTPException(
                        status_code=422,
                        detail="coords query parameter is required for GET; "
                        "use POST /position to submit points in the request body",
                    )
                try:
                    geometry = query.geometry
                except GEOSException:
                    raise HTTPException(
                        status_code=422,
                        detail="Could not parse coordinates to geometry, "
                        "check the format of the 'coords' query parameter",
                    )

            return _run_position_query(
                dataset,
                query,
                geometry,
                dict(request.query_params),
            )

        def _run_area_query(
            dataset: xr.Dataset,
            query: EDRAreaQuery,
            geometry: shapely.Geometry,
            query_params: dict,
        ):
            """Shared select/project/format pipeline for GET and POST area queries."""
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
                projected_geometry = project_geometry(ds, query.crs, geometry)
                ds = select_by_area(ds, projected_geometry)
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
                ds = project_dataset(ds, query.crs)
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

        @router.api_route(
            "/area",
            methods=["GET", "POST"],
            summary="Area query",
        )
        def area(
            request: Request,
            query: Annotated[EDRAreaQuery, Query()],
            body: Annotated[bytes, Depends(_raw_body)],
            dataset: xr.Dataset = Depends(deps.dataset),
        ):
            """
            Returns vectorized area data for a Polygon or MultiPolygon.

            For GET, the polygon is passed as WKT via the `coords` query parameter.

            For POST, the polygon is submitted in the request body as either
            GeoJSON (Content-Type: application/geo+json) -- Polygon, MultiPolygon,
            Feature, FeatureCollection, or GeometryCollection -- or raw WKT
            (Content-Type: application/wkt or text/plain).

            All other selection parameters (datetime, z, parameter-name, crs, f,
            method) are passed as query string parameters in both cases.

            Synchronous (`def`) for the same reason as ``position``: the query
            pipeline is blocking CPU work and must run in the threadpool rather
            than on the event loop. The body is injected via ``Body`` so no
            ``await`` is needed.
            """
            if request.method == "POST":
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
            else:
                if query.coords is None:
                    raise HTTPException(
                        status_code=422,
                        detail="coords query parameter is required for GET; "
                        "use POST /area to submit the polygon in the request body",
                    )
                try:
                    geometry = query.geometry
                except GEOSException:
                    raise HTTPException(
                        status_code=422,
                        detail="Could not parse coordinates to geometry, "
                        "check the format of the 'coords' query parameter",
                    )

            return _run_area_query(
                dataset,
                query,
                geometry,
                dict(request.query_params),
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
            try:
                ds = query.select(dataset, dict(request.query_params))
            except ValueError as e:
                logger.error(f"Error selecting from query while selecting by cube: {e}")
                raise HTTPException(
                    status_code=404,
                    detail=f"Error selecting from query: {e.args[0]}",
                )

            logger.debug(f"Dataset filtered by query params {ds}")

            try:
                ds = select_by_bbox(ds, query.project_bbox(ds))
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
                ds = project_dataset(ds, query.crs)
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

        return router
