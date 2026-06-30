"""Query parameter definitions for EDR position queries."""

import xarray as xr
from fastapi import HTTPException
from pydantic import ConfigDict, Field, field_validator
from shapely import Geometry
from shapely.errors import GEOSException

from xpublish_edr.base_query import BaseEDRQuery, WKTGeometryQuery
from xpublish_edr.format import position_formats
from xpublish_edr.geometry.common import (
    PreparedSpatialGrid,
    project_geometry,
)
from xpublish_edr.logger import logger

from .geom import select_by_position


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

    def formats(self) -> dict:
        """Return the registered position response formats."""
        return position_formats()

    def query_label(self) -> str:
        """Return the query label for this query type."""
        return "position"

    def spatial_select(
        self,
        grid: PreparedSpatialGrid,
        geometry: Geometry | None = None,
    ) -> xr.Dataset:
        """Project the query point(s) and select the nearest/interpolated data."""
        try:
            projected_geometry = project_geometry(
                grid.ds,
                self.crs,
                geometry,
                grid.spatial_ref,
            )
            return select_by_position(
                grid.ds,
                projected_geometry,
                self.method,
                grid.spatial_ref,
            )
        except GEOSException as e:
            logger.error(
                f"Error parsing coordinates to geometry while selecting by position: {e}",
            )
            raise HTTPException(
                status_code=422,
                detail="Could not parse coordinates to geometry, "
                "check the format of the 'coords' query parameter",
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


class EDRPositionQueryGet(WKTGeometryQuery, EDRPositionQueryPost):
    """Position query for GET-only endpoints, where ``coords`` is mandatory.

    :class:`EDRPositionQueryPost` keeps ``coords`` off the model entirely because
    the POST routes read the points from the request body. The GET routes are
    coords-driven, and the OGC EDR schema (and the CITE suite) require the
    ``coords`` parameter to be declared ``required: true``, so this subclass adds
    it as a mandatory field. Geometry parsing/projection comes from
    :class:`WKTGeometryQuery`.
    """

    coords: str = Field(
        ...,
        title="Point(s) in WKT format",
        description="Well Known Text coordinates for the point(s) to query.",
    )
