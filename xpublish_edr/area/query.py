"""Query parameter models for EDR area queries"""

import xarray as xr
from fastapi import HTTPException
from pydantic import ConfigDict, Field, field_validator
from shapely import Geometry
from shapely.errors import GEOSException

from xpublish_edr.base_query import BaseEDRQuery, WKTGeometryQuery
from xpublish_edr.format import area_formats
from xpublish_edr.geometry.common import (
    PreparedSpatialGrid,
    project_geometry,
)
from xpublish_edr.logger import logger

from .geom import select_by_area


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

    def formats(self) -> dict:
        """Return the registered area response formats."""
        return area_formats()

    def query_label(self) -> str:
        """Return the query label for this query type."""
        return "area"

    def spatial_select(
        self,
        grid: PreparedSpatialGrid,
        geometry: Geometry | None = None,
    ) -> xr.Dataset:
        """Project the query polygon and select the data within it."""
        try:
            projected_geometry = project_geometry(
                grid.ds,
                self.crs,
                geometry,
                grid.spatial_ref,
            )
            return select_by_area(grid.ds, projected_geometry, grid.spatial_ref)
        except GEOSException as e:
            logger.error(
                f"Error parsing coordinates to geometry while selecting by area: {e}",
            )
            raise HTTPException(
                status_code=422,
                detail="Could not parse coordinates to geometry, "
                "check the format of the 'coords' query parameter",
            )
        except KeyError as e:
            logger.error(f"Error selecting by area: {e}")
            raise HTTPException(
                status_code=404,
                detail="Dataset does not have CF Convention compliant metadata",
            )


class EDRAreaQueryGet(WKTGeometryQuery, EDRAreaQueryPost):
    """Area query for GET-only endpoints, where ``coords`` is mandatory.

    See :class:`EDRPositionQueryGet`; the OGC ``/collections/{id}/area``
    endpoint is GET-only and its ``coords`` parameter must be ``required: true``.
    Geometry parsing/projection comes from :class:`WKTGeometryQuery`.
    """

    coords: str = Field(
        ...,
        title="Polygon in WKT format",
        description="Well Known Text coordinates for the polygon to query.",
    )
