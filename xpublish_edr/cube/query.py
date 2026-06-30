"""Query parameter model for EDR cube queries"""

import xarray as xr
from fastapi import HTTPException
from pydantic import Field, field_validator
from shapely import Geometry

from xpublish_edr.base_query import BaseEDRQuery
from xpublish_edr.format import cube_formats
from xpublish_edr.geometry.bbox import select_by_bbox
from xpublish_edr.geometry.common import (
    PreparedSpatialGrid,
    project_bbox,
)
from xpublish_edr.logger import logger


class EDRCubeQuery(BaseEDRQuery):
    """
    Capture query parameters for EDR cube queries
    """

    bbox: tuple[float, float, float, float] = Field(
        ...,
        title="Bounding box in minx, miny, maxx, maxy",
        description="Bounding box for the query",
    )

    @field_validator("format", mode="before")
    def validate_format(cls, v):
        """Validate the format is a valid cube format"""
        if v not in cube_formats().keys():
            raise ValueError(f"Invalid format: {v}")
        return v

    @field_validator("bbox", mode="before")
    def validate_bbox(cls, v):
        """Validate the bbox is a tuple of 4 floats"""
        if isinstance(v, str):
            return tuple(float(v.strip()) for v in v.split(","))
        if isinstance(v, list):
            return tuple(float(v.strip()) for v in v[0].split(","))
        raise ValueError(f"Invalid bbox: {v}")

    def project_bbox(self, ds: xr.Dataset) -> tuple[float, float, float, float]:
        """Project the bbox to the dataset's CRS"""
        return project_bbox(ds, self.crs, self.bbox)

    def formats(self) -> dict:
        """Return the registered cube response formats."""
        return cube_formats()

    def query_label(self) -> str:
        """Return the query label for this query type."""
        return "cube"

    def spatial_select(
        self,
        grid: PreparedSpatialGrid,
        geometry: Geometry | None = None,
    ) -> xr.Dataset:
        """Project the query bbox and select the data within it.

        ``geometry`` is unused: cube queries take their extent from the ``bbox``
        field rather than a WKT/body geometry.
        """
        try:
            bbox = project_bbox(grid.ds, self.crs, self.bbox, grid.spatial_ref)
            return select_by_bbox(grid.ds, bbox, grid.spatial_ref)
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
