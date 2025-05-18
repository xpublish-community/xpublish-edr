"""
OGC EDR Query param parsing
"""

from typing import Literal, Optional

import xarray as xr
from pydantic import BaseModel, Field
from shapely import Geometry, wkt

from xpublish_edr.geometry.common import project_bbox, project_geometry
from xpublish_edr.logger import logger


class BaseEDRQuery(BaseModel):
    """
    Base class for EDR queries
    """

    format: Optional[str] = None
    z: Optional[str] = None
    datetime: Optional[str] = None
    parameters: Optional[str] = None
    crs: str = Field(
        "EPSG:4326",
        title="Coordinate Reference System",
        description="Coordinate Reference System for the query. Default is EPSG:4326",
    )
    method: Literal["nearest", "linear"] = "nearest"

    def select(self, ds: xr.Dataset, query_params: dict) -> xr.Dataset:
        """Select data from a dataset based on the query"""
        if self.z:
            if self.method == "nearest":
                ds = ds.cf.sel(Z=[self.z], method=self.method)
            else:
                ds = ds.cf.interp(Z=[self.z], method=self.method)

        if self.datetime:
            try:
                datetimes = self.datetime.split("/")
                if len(datetimes) == 1:
                    if self.method == "nearest":
                        ds = ds.cf.sel(T=datetimes, method=self.method)
                    else:
                        ds = ds.cf.interp(T=datetimes, method=self.method)
                elif len(datetimes) == 2:
                    ds = ds.cf.sel(T=slice(datetimes[0], datetimes[1]))
                else:
                    raise ValueError(
                        f"Invalid datetimes submitted - {datetimes}",
                    )
            except ValueError as e:
                logger.error("Error with datetime", exc_info=True)
                raise ValueError(f"Invalid datetime ({e})") from e

        if self.parameters:
            try:
                ds = ds.cf[self.parameters.split(",")]
            except KeyError as e:
                raise ValueError(f"Invalid variable: {e}") from e

        query_param_keys = list(query_params.keys())
        for query_param in query_param_keys:
            if query_param in edr_query_params:
                del query_params[query_param]

        sel_params = {}
        sliced_sel_params = {}
        for key, value in query_params.items():
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


class EDRPositionQuery(BaseEDRQuery):
    """
    Capture query parameters for EDR position queries
    """

    coords: Geometry = Field(
        ...,
        title="Points in WKT format",
        description="Well Known Text coordinates",
        validator=lambda v: wkt.loads(v),
    )

    @property
    def geometry(self) -> Geometry:
        """Shapely point from WKT query params"""
        return self.coords

    def project_geometry(self, ds: xr.Dataset) -> Geometry:
        """Project the geometry to the dataset's CRS"""
        return project_geometry(ds, self.crs, self.geometry)


class EDRAreaQuery(BaseEDRQuery):
    """
    Capture query parameters for EDR area queries
    """

    coords: Geometry = Field(
        ...,
        title="Polygon in WKT format",
        description="Well Known Text coordinates",
    )

    @property
    def geometry(self) -> Geometry:
        """Shapely polygon from WKT query params"""
        return self.coords

    def project_geometry(self, ds: xr.Dataset) -> Geometry:
        """Project the geometry to the dataset's CRS"""
        return project_geometry(ds, self.crs, self.geometry)


class EDRCubeQuery(BaseEDRQuery):
    """
    Capture query parameters for EDR cube queries
    """

    bbox: tuple[float, float, float, float] = Field(
        ...,
        title="Bounding box in minx, miny, maxx, maxy format",
        description="Bounding box for the query",
        validator=lambda v: tuple(float(v.strip()) for v in v.split(",")),
    )

    def project_bbox(self, ds: xr.Dataset) -> tuple[float, float, float, float]:
        """Project the bbox to the dataset's CRS"""
        return project_bbox(ds, self.crs, self.bbox)


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
