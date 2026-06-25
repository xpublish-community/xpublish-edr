"""
OGC EDR Query param parsing
"""

from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import BaseModel, Field
from shapely import Geometry, wkt
from shapely.errors import GEOSException

from xpublish_edr.format import area_formats, cube_formats, position_formats
from xpublish_edr.logger import logger


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
