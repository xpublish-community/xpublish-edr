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
from xpublish_edr.geometry.common import DEFAULT_CRS, dataset_crs, project_dataset
from xpublish_edr.geometry.position import select_by_position
from xpublish_edr.logger import logger
from xpublish_edr.query import EDRQuery, edr_query


def output_formats():
    """
    Return response format functions from registered
    `xpublish_edr_position_formats` entry_points
    """
    formats = {}

    entry_points = importlib.metadata.entry_points()
    for entry_point in entry_points.get("xpublish_edr_position_formats", []):
        formats[entry_point.name] = entry_point.load()

    return formats


def variable_description(variable: xr.DataArray):
    """
    Return CF version of EDR Parameter metadata for a given xarray variable
    """
    name = variable.attrs.get("name", None)
    standard_name = variable.attrs.get("standard_name", name if name else "")
    label = standard_name if not name else name
    long_name = variable.attrs.get("long_name", "")
    units = variable.attrs.get("units", "")
    return {
        "type": "Parameter",
        "description": long_name,
        "unit": {
            "label": units,
        },
        "observedProperty": {
            "label": label,
            "standard_name": standard_name,
            "long_name": long_name,
        },
    }


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
            id = dataset.attrs.get("_xpublish_id", "unknown")
            title = dataset.attrs.get("title", "unknown")
            description = dataset.attrs.get("description", "no description")

            crs = dataset_crs(dataset)

            available_output_formats = list(output_formats().keys())

            ds_cf = dataset.cf
            axes = ds_cf.axes

            # We will use the dataset's CRS as the default CRS for the extents,
            # but override when it makes sense.
            extent_crs = crs

            if len(axes["X"]) > 1:
                if "latitude" and "longitude" in ds_cf:
                    min_lon = float(ds_cf["longitude"].min().values)
                    max_lon = float(ds_cf["longitude"].max().values)
                    min_lat = float(ds_cf["latitude"].min().values)
                    max_lat = float(ds_cf["latitude"].max().values)

                    # When we are explicitly using latitude and longitude, we should use WGS84
                    extent_crs = DEFAULT_CRS
                else:
                    raise HTTPException(
                        status_code=404,
                        detail="Dataset does not have EDR compliant metadata: Multiple X axes found",
                    )
            else:
                min_lon = float(ds_cf["X"].min().values)
                max_lon = float(ds_cf["X"].max().values)
                min_lat = float(ds_cf["Y"].min().values)
                max_lat = float(ds_cf["Y"].max().values)

            extents: dict = {
                "spatial": {
                    "bbox": [
                        [
                            min_lon,
                            min_lat,
                            max_lon,
                            max_lat,
                        ],
                    ],
                    "crs": extent_crs.to_string(),
                },
            }

            if "T" in ds_cf:
                time_min = ds_cf["T"].min().dt.strftime("%Y-%m-%dT%H:%M:%S").values
                time_max = ds_cf["T"].max().dt.strftime("%Y-%m-%dT%H:%M:%S").values

                extents["temporal"] = {
                    "interval": [
                        str(time_min),
                        str(time_max),
                    ],
                    "values": [
                        f"{time_min}/{time_max}",
                    ],
                    "trs": 'TIMECRS["DateTime",TDATUM["Gregorian Calendar"],CS[TemporalDateTime,1],AXIS["Time (T)",unspecified]]',  # noqa
                }

            if "Z" in ds_cf:
                units = ds_cf["Z"].attrs.get("units", "unknown")
                positive = ds_cf["Z"].attrs.get("positive", "up")
                elevations = ds_cf["Z"].values
                min_z = elevations.min()
                max_z = elevations.max()
                elevation_values = ",".join([str(e) for e in elevations])

                extents["vertical"] = {
                    "interval": [
                        min_z,
                        max_z,
                    ],
                    "values": elevation_values,
                    "vrs": f"VERTCRS[VERT_CS['unknown'],AXIS['Z',{positive}],UNIT[{units},1]]",  # noqa
                    "positive": positive,
                    "units": units,
                }

            parameters = {
                k: variable_description(v)
                for k, v in dataset.variables.items()
                if "axis" not in v.attrs
            }

            crs_details = [
                {
                    "crs": crs.to_string(),
                    "wkt": crs.to_wkt(),
                },
            ]

            # 4326 is always available
            if crs != DEFAULT_CRS:
                crs_details.append(
                    {
                        "crs": DEFAULT_CRS.to_string(),
                        "wkt": DEFAULT_CRS.to_wkt(),
                    },
                )

            return {
                "id": id,
                "title": title,
                "description": description,
                "links": [],
                "extent": extents,
                "data_queries": {
                    "position": {
                        "href": "/edr/position?coords={coords}",
                        "hreflang": "en",
                        "rel": "data",
                        "templated": True,
                        "variables": {
                            "title": "Position query",
                            "description": "Returns position data based on WKT `POINT(lon lat)` or `MULTIPOINT(lon lat, ...)` coordinates",  # noqa
                            "query_type": "position",
                            "coords": {
                                "type": "string",
                                "description": "WKT `POINT(lon lat)` or `MULTIPOINT(lon lat, ...)` coordinates",  # noqa
                                "required": True,
                            },
                            "output_format": available_output_formats,
                            "default_output_format": "cf_covjson",
                            "crs_details": crs_details,
                        },
                    },
                    "area": {
                        "href": "/edr/area",
                        "hreflang": "en",
                        "rel": "data",
                        "templated": True,
                        "variables": {
                            "title": "Area query",
                            "description": "Returns data in a polygon based on WKT `POLYGON(lon lat, ...)` coordinates",  # noqa
                            "query_type": "position",
                            "coords": {
                                "type": "string",
                                "description": "WKT `POLYGON(lon lat, ...)` coordinates",
                                "required": True,
                            },
                            "output_format": available_output_formats,
                            "default_output_format": "cf_covjson",
                            "crs_details": crs_details,
                        },
                    },
                },
                "crs": [crs.to_string()],
                "output_formats": available_output_formats,
                "parameter_names": parameters,
            }

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
                ds = select_by_position(ds, query.project_geometry(ds), query.method)
            except KeyError:
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
                logger.error(f"Error projecting dataset: {e}")
                raise HTTPException(
                    status_code=404,
                    detail="Error projecting dataset",
                )

            logger.debug(f"Dataset projected to {query.crs}: {ds}")

            if query.format:
                try:
                    format_fn = output_formats()[query.format]
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
                ds = select_by_area(ds, query.project_geometry(ds))
            except KeyError:
                raise HTTPException(
                    status_code=404,
                    detail="Dataset does not have CF Convention compliant metadata",
                )

            logger.debug(f"Dataset filtered by polygon {query.geometry.boundary}: {ds}")

            try:
                ds = project_dataset(ds, query.crs)
            except Exception as e:
                logger.error(f"Error projecting dataset: {e}")
                raise HTTPException(
                    status_code=404,
                    detail="Error projecting dataset",
                )

            logger.debug(f"Dataset projected to {query.crs}: {ds}")

            if query.format:
                try:
                    format_fn = output_formats()[query.format]
                except KeyError:
                    raise HTTPException(
                        404,
                        f"{query.format} is not a valid format for EDR position queries. "
                        "Get `./formats` for valid formats",
                    )

                return format_fn(ds)

            return to_cf_covjson(ds)

        return router
