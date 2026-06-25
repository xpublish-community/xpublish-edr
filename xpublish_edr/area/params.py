"""Query parameter models for EDR area queries"""

import xarray as xr
from pydantic import ConfigDict, Field, field_validator
from shapely import Geometry

from xpublish_edr.base_query import BaseEDRQuery, load_wkt
from xpublish_edr.format import area_formats
from xpublish_edr.geometry.common import project_geometry


class EDRAreaQueryPost(BaseEDRQuery):
    """Area query selection parameters with no ``coords``.

    On POST the polygon is read from the request body, so ``coords`` is not a
    query parameter; this carries only the shared selection parameters and the
    area format validation. The GET/dataset variants add ``coords`` on top.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("format", mode="before")
    def validate_format(cls, v):
        """Validate the format is a valid area format"""
        if v not in area_formats().keys():
            raise ValueError(f"Invalid format: {v}")
        return v


class EDRAreaQuery(EDRAreaQueryPost):
    """
    Capture query parameters for EDR area queries
    """

    coords: str | None = Field(
        None,
        title="Polygon in WKT format",
        description="Well Known Text coordinates. "
        "Required for GET; for POST the polygon is read from the request body.",
    )

    @property
    def geometry(self) -> Geometry:
        """Shapely polygon from WKT query params"""
        if self.coords is None:
            raise ValueError("coords query parameter is required")
        return load_wkt(self.coords)

    def project_geometry(self, ds: xr.Dataset) -> Geometry:
        """Project the geometry to the dataset's CRS"""
        return project_geometry(ds, self.crs, self.geometry)


class EDRAreaQueryGet(BaseEDRQuery):
    """Area query for GET-only endpoints, where ``coords`` is mandatory.

    See :class:`EDRPositionQueryGet`; the OGC ``/collections/{id}/area``
    endpoint is GET-only and its ``coords`` parameter must be ``required: true``.
    """

    coords: str = Field(
        ...,
        title="Polygon in WKT format",
        description="Well Known Text coordinates for the polygon to query.",
    )

    @field_validator("format", mode="before")
    def validate_format(cls, v):
        """Validate the format is a valid area format"""
        if v not in area_formats().keys():
            raise ValueError(f"Invalid format: {v}")
        return v

    @property
    def geometry(self) -> Geometry:
        """Shapely polygon from WKT query params"""
        return load_wkt(self.coords)

    def project_geometry(self, ds: xr.Dataset) -> Geometry:
        """Project the geometry to the dataset's CRS"""
        return project_geometry(ds, self.crs, self.geometry)
