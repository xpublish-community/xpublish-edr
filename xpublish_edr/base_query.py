"""
OGC EDR Query param parsing
"""

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import xarray as xr
from fastapi import HTTPException
from pydantic import BaseModel, Field
from shapely import Geometry, wkt
from shapely.errors import GEOSException

from xpublish_edr.format import area_formats, cube_formats, position_formats
from xpublish_edr.formats.to_covjson import to_cf_covjson
from xpublish_edr.geometry.common import (
    PreparedSpatialGrid,
    prepare_spatial_grid,
    project_dataset,
    project_geometry,
)
from xpublish_edr.logger import logger
from xpublish_edr.utils import _load_dataset


class BaseEDRQuery(BaseModel):
    """
    Base class for EDR queries
    """

    format: str | None = Field(
        None,
        title="Response format",
        description="The format of the response. Default is CoverageJSON. "
        f"Valid position formats: {', '.join(position_formats().keys())}, "
        f"valid area formats: {', '.join(area_formats().keys())}, "
        f"valid cube formats: {', '.join(cube_formats().keys())}",
        validation_alias="f",
    )
    z: str | None = Field(
        None,
        title="Elevation",
        description="Elevation for the query",
        alias="z",
    )
    datetime: str | None = Field(
        None,
        title="Datetime",
        description="Datetime for the query",
    )
    parameters: str | None = Field(
        None,
        title="Parameters",
        description="Parameters for the query",
        validation_alias="parameter-name",
    )
    crs: str = Field(
        "EPSG:4326",
        title="Coordinate Reference System",
        description="Coordinate Reference System for the query. Default is EPSG:4326",
    )
    method: Literal["nearest", "linear"] = Field(
        "nearest",
        title="Method",
        description="Method for the query",
    )

    def _require_indexed_axis(self, ds: xr.Dataset, axis: str) -> None:
        """Raise ValueError if cf axis ``axis`` (e.g. "T", "Z") is absent or unindexed."""
        try:
            coord_names = ds.cf.axes[axis]
        except KeyError:
            coord_names = []
        if not any(name in ds.indexes for name in coord_names):
            raise ValueError(
                f"Cannot select on {axis} axis via cf_xarray: "
                f"no indexed {axis} coordinate found. "
                f"The {axis} coordinate may not be indexed. "
                f"Indexed dimensions available for direct selection: {list(ds.indexes.keys())}",
            )

    def select(self, ds: xr.Dataset, query_params: dict) -> xr.Dataset:
        """Select data from a dataset based on the query"""
        if self.z:
            try:
                z_value = float(self.z)
            except ValueError as e:
                raise ValueError(f"Invalid z value {self.z!r}: {e}") from e
            self._require_indexed_axis(ds, "Z")
            if self.method == "nearest":
                ds = ds.cf.sel(Z=[z_value], method=self.method)
            else:
                ds = ds.cf.interp(Z=[z_value], method=self.method)

        if self.datetime:
            datetimes = self.datetime.split("/")
            if len(datetimes) > 2:
                raise ValueError(f"Invalid datetimes submitted - {datetimes}")
            try:
                parsed_datetimes = [pd.Timestamp(d) for d in datetimes]
            except ValueError as e:
                logger.error("Error with datetime", exc_info=True)
                raise ValueError(f"Invalid datetime ({e})") from e
            self._require_indexed_axis(ds, "T")
            if len(parsed_datetimes) == 1:
                if self.method == "nearest":
                    ds = ds.cf.sel(T=parsed_datetimes, method=self.method)
                else:
                    ds = ds.cf.interp(T=parsed_datetimes, method=self.method)
            else:
                ds = ds.cf.sel(T=slice(parsed_datetimes[0], parsed_datetimes[1]))

        if self.parameters:
            try:
                ds = ds[self.parameters.split(",")]
            except KeyError as e:
                raise ValueError(f"Invalid variable: {e}") from e

        query_param_keys = list(query_params.keys())
        for query_param in query_param_keys:
            if query_param in edr_query_params:
                del query_params[query_param]

        sel_params = {}
        sliced_sel_params = {}
        for key, value in query_params.items():
            # String dimensions are not sliced but they cannot be interpolated so
            # we select them directly using equality
            if ds[key].dtype.type is np.str_:
                sliced_sel_params[key] = value
                continue

            split_value = [float(v) if v.isnumeric() else v for v in value.split("/")]
            if len(split_value) == 1:
                sel_params[key] = [split_value[0]]
            elif len(split_value) == 2:
                sliced_sel_params[key] = slice(split_value[0], split_value[1])
            else:
                raise ValueError(f"Too many values for selecting {key}")

        # We separate the slice selection from the single value selection in order to take
        # advantage of selection method which breaks when mixing the two
        if len(sliced_sel_params) > 0:
            ds = ds.sel(sliced_sel_params)

        if self.method == "nearest":
            ds = ds.sel(sel_params, method=self.method)
        else:
            # Interpolation may not be supported for all possible selection
            # parameters, so we provide a fallback to xarray's nearest selection
            try:
                ds = ds.interp(sel_params, method=self.method)
            except Exception as e:
                logger.warning(f"Interpolation failed: {e}, falling back to selection")
                ds = ds.sel(sel_params, method="nearest")

        return ds

    def run_query(
        self,
        dataset: xr.Dataset,
        query_params: dict,
        geometry: Geometry | None = None,
    ):
        """Select, spatially filter, project, and format an EDR query.

        This is the shared pipeline for every geometry query. The query-type
        specific parts -- the spatial selection, the format registry, and the
        label used in messages -- are provided by :meth:`spatial_select`,
        :meth:`formats`, and :meth:`query_label` on the subclasses.

        ``geometry`` is the point(s)/polygon to query (parsed from ``coords`` on
        GET or the request body on POST); cube queries ignore it and use their
        ``bbox`` field instead.
        """
        try:
            ds = self.select(dataset, query_params)
        except ValueError as e:
            logger.error(f"Error selecting from query for {self.query_label()} query: {e}")
            raise HTTPException(
                status_code=404,
                detail=f"Error selecting from query: {e.args[0]}",
            )

        logger.debug(f"Dataset filtered by query params {ds}")

        grid = prepare_spatial_grid(ds, require_regular=True)
        ds = self.spatial_select(grid, geometry)

        logger.debug(f"Dataset filtered spatially: {ds}")

        try:
            ds = project_dataset(ds, self.crs, grid.spatial_ref)
        except Exception as e:
            logger.error(f"Error projecting dataset for {self.query_label()} query: {e}")
            raise HTTPException(
                status_code=404,
                detail="Error projecting dataset",
            )

        logger.debug(f"Dataset projected to {self.crs}: {ds}")

        ds = _load_dataset(ds)

        logger.debug("Dataset loaded")

        if self.format:
            try:
                format_fn = self.formats()[self.format]
            except KeyError as e:
                label = self.query_label()
                logger.error(f"Error getting format function for {label} query: {e}")
                raise HTTPException(
                    404,
                    f"{self.format} is not a valid format for EDR {label} queries. "
                    f"Get `./{label}/formats` for valid formats",
                )

            return format_fn(ds)

        return to_cf_covjson(ds)

    def spatial_select(
        self,
        grid: PreparedSpatialGrid,
        geometry: Geometry | None = None,
    ) -> xr.Dataset:
        """Spatially filter the prepared grid for this query type.

        Implemented by subclasses: position/area project ``geometry`` and select
        by it; cube projects its ``bbox`` field. Implementations own their own
        selection-error to ``HTTPException`` mapping.
        """
        raise NotImplementedError

    def formats(self) -> dict:
        """Return the response-format registry for this query type."""
        raise NotImplementedError

    def query_label(self) -> str:
        """Return the query-type label (``position``/``area``/``cube``)."""
        raise NotImplementedError


def load_wkt(value: str) -> Geometry:
    """Parse WKT, normalizing any parse failure to ``GEOSException``.

    ``shapely.wkt.loads`` raises ``GEOSException`` for malformed WKT, but other
    error types leak through for some inputs (e.g. ``UnicodeDecodeError`` for a
    string GEOS cannot encode). Callers catch ``GEOSException`` to return a 422,
    so collapse everything else into it rather than letting it escape as a 500.
    """
    try:
        return wkt.loads(value)
    except GEOSException:
        raise
    except Exception as e:
        raise GEOSException(f"Invalid WKT coordinates: {e}") from e


class WKTGeometryQuery:
    """Mixin providing WKT ``coords`` geometry parsing for GET query endpoints.

    Position and area GET queries take their geometry from a required ``coords``
    query parameter (WKT); this parses and projects that geometry. Concrete
    subclasses declare the ``coords`` field itself, since its title/description
    differ per query type (Point vs Polygon). Intentionally not part of
    :class:`BaseEDRQuery`: POST queries read the geometry from the request body
    and cube queries take their extent from ``bbox``.
    """

    if TYPE_CHECKING:
        # Supplied at runtime by the concrete query class (``coords``) and
        # :class:`BaseEDRQuery` (``crs``); declared here only for type checkers.
        coords: str
        crs: str

    @property
    def geometry(self) -> Geometry:
        """Shapely geometry parsed from the WKT ``coords`` parameter."""
        return load_wkt(self.coords)

    def project_geometry(self, ds: xr.Dataset) -> Geometry:
        """Project the query geometry to the dataset's CRS."""
        return project_geometry(ds, self.crs, self.geometry)


edr_query_params = {
    "coords",
    "bbox",
    "z",
    "datetime",
    "parameter-name",
    "crs",
    "f",
    "method",
}
