"""
OGC EDR Query param parsing
"""
from typing import Optional

from fastapi import Query
from pydantic import BaseModel, Field
from shapely import wkt


class EDRQuery(BaseModel):
    """
    Capture query parameters for EDR position queries
    """

    coords: str = Field(
        ...,
        title="Point in WKT format",
        description="Well Known Text coordinates",
    )
    z: Optional[str] = None
    datetime: Optional[str] = None
    parameters: Optional[str] = None
    crs: Optional[str] = None
    format: Optional[str] = None

    @property
    def point(self):
        """Shapely point from WKT query params"""
        return wkt.loads(self.coords)


def edr_query(
    coords: str = Query(
        ...,
        title="Point in WKT format",
        description="Well Known Text coordinates",
    ),
    z: Optional[str] = Query(
        None,
        title="Z axis",
        description="Height or depth of query",
    ),
    datetime: Optional[str] = Query(
        None,
        title="Datetime or datetime range",
        description=(
            "Query by a single ISO time or a range of ISO times. "
            "To query by a range, split the times with a slash"
        ),
    ),
    parameters: Optional[str] = Query(
        None,
        alias="parameter-name",
        description="xarray variables to query",
    ),
    crs: Optional[str] = Query(
        None,
        deprecated=True,
        description="CRS is not yet implemented",
    ),
    f: Optional[str] = Query(
        None,
        title="Response format",
        description=(
            "Data is returned as a CoverageJSON by default. "
            "Get `/formats` to discover what other formats are accessible"
        ),
    ),
):
    """Extract EDR query params from request query strings"""
    return EDRQuery(
        coords=coords,
        z=z,
        datetime=datetime,
        parameters=parameters,
        crs=crs,
        format=f,
    )


edr_query_params = {"coords", "z", "datetime", "parameter-name", "crs", "f"}
