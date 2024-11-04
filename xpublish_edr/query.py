"""
OGC EDR Query param parsing
"""

from typing import Optional

import xarray as xr
from fastapi import Query
from pydantic import BaseModel, Field
from shapely import wkt

from xpublish_edr.logging import logger


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
    def geometry(self):
        """Shapely point from WKT query params"""
        return wkt.loads(self.coords)

    def select(self, ds: xr.Dataset, query_params: dict) -> xr.Dataset:
        """Select data from a dataset based on the query"""
        if self.z:
            ds = ds.cf.sel(Z=self.z, method="nearest")

        if self.datetime:
            try:
                datetimes = self.datetime.split("/")
                if len(datetimes) == 1:
                    ds = ds.cf.sel(T=datetimes[0], method="nearest")
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

        # TODO: allow for controlling selection method
        method: Optional[str] = "nearest"

        for key, value in query_params.items():
            split_value = value.split("/")
            if len(split_value) == 1:
                continue
            elif len(split_value) == 2:
                query_params[key] = slice(split_value[0], split_value[1])
                method = None
            else:
                raise ValueError(f"Too many values for selecting {key}")

        ds = ds.sel(query_params, method=method)
        return ds


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
