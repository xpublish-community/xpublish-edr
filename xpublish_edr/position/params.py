"""Query parameter definitions for EDR position queries."""

import xarray as xr
from pydantic import ConfigDict, Field, field_validator
from shapely import Geometry

from xpublish_edr.base_query import BaseEDRQuery, load_wkt
from xpublish_edr.format import position_formats
from xpublish_edr.geometry.common import project_geometry


class EDRPositionQueryPost(BaseEDRQuery):
    """Position query selection parameters with no ``coords``.

    On POST the point(s) are read from the request body, so ``coords`` is not a
    query parameter; this carries only the shared selection parameters and the
    position format validation. The GET/dataset variants add ``coords`` on top.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("format", mode="before")
    def validate_format(cls, v):
        """Validate the format is a valid position format"""
        if v not in position_formats().keys():
            raise ValueError(f"Invalid format: {v}")
        return v


class EDRPositionQuery(EDRPositionQueryPost):
    """
    Capture query parameters for EDR position queries
    """

    coords: str | None = Field(
        None,
        title="Point(s) in WKT format",
        description="Well Known Text coordinates for the point(s) to query. "
        "Required for GET; for POST the points are read from the request body.",
    )

    @property
    def geometry(self) -> Geometry:
        """Shapely point from WKT query params"""
        if self.coords is None:
            raise ValueError("coords query parameter is required")
        return load_wkt(self.coords)

    def project_geometry(self, ds: xr.Dataset) -> Geometry:
        """Project the geometry to the dataset's CRS"""
        return project_geometry(ds, self.crs, self.geometry)


class EDRPositionQueryGet(BaseEDRQuery):
    """Position query for GET-only endpoints, where ``coords`` is mandatory.

    The shared :class:`EDRPositionQuery` keeps ``coords`` optional because the
    GET+POST dataset route reads the points from the request body on POST. The
    OGC ``/collections/{id}/position`` endpoint is GET-only, and the OGC EDR
    schema (and the CITE suite) require its ``coords`` parameter to be declared
    ``required: true``, so this subclass redeclares it as mandatory.
    """

    coords: str = Field(
        ...,
        title="Point(s) in WKT format",
        description="Well Known Text coordinates for the point(s) to query.",
    )

    @field_validator("format", mode="before")
    def validate_format(cls, v):
        """Validate the format is a valid position format"""
        if v not in position_formats().keys():
            raise ValueError(f"Invalid format: {v}")
        return v

    @property
    def geometry(self) -> Geometry:
        """Shapely point from WKT query params"""
        return load_wkt(self.coords)

    def project_geometry(self, ds: xr.Dataset) -> Geometry:
        """Project the geometry to the dataset's CRS"""
        return project_geometry(ds, self.crs, self.geometry)
