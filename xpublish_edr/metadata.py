from typing import Literal, Optional

import pandas as pd
import pyproj
import xarray as xr
from pydantic import BaseModel, Field

from xpublish_edr.geometry.common import (
    DEFAULT_CRS,
    dataset_crs,
    project_dataset,
    spatial_bounds,
)


class CRSDetails(BaseModel):
    """OGC EDR CRS metadata

    https://docs.ogc.org/is/19-086r6/19-086r6.html#_7124ec17-6401-4eb7-ba1d-8ec329b7e677
    """

    crs: str
    wkt: str


class VariablesMetadata(BaseModel):
    """OGC EDR Variables metadata

    https://docs.ogc.org/is/19-086r6/19-086r6.html#_1b54f97a-e1dc-4920-b8b4-e4981554138d
    """

    title: Optional[str] = None
    description: Optional[str] = None
    query_type: Optional[str] = None
    coords: Optional[dict] = None
    output_formats: Optional[list[str]] = None
    default_output_format: Optional[str] = None
    crs_details: Optional[list[CRSDetails]] = None


class Link(BaseModel):
    """OGC EDR Link metadata

    https://docs.ogc.org/is/19-086r6/19-086r6.html#_ea77762b-89c0-4704-b845-748efc66e597
    """

    href: str
    rel: str
    type_: Optional[str] = Field(None, serialization_alias="type")
    hreflang: Optional[str] = None
    title: Optional[str] = None
    length: Optional[int] = None
    templated: Optional[bool] = None
    variables: Optional[VariablesMetadata] = None


class SpatialExtent(BaseModel):
    """OGC EDR Spatial Extent metadata

    https://docs.ogc.org/is/19-086r6/19-086r6.html#_0afff399-4d8e-4a9b-961b-cab841d23cc1
    """

    bbox: list[list[float]]
    crs: str


class TemporalExtent(BaseModel):
    """OGC EDR Temporal Extent metadata

    https://docs.ogc.org/is/19-086r6/19-086r6.html#_8f4c9f38-bc6a-4b98-8fd9-772e42d60ab2
    """

    interval: list[str]
    values: list[str]
    trs: str


class VerticalExtent(BaseModel):
    """OGC EDR Vertical Extent metadata

    https://docs.ogc.org/is/19-086r6/19-086r6.html#_52bf970b-315a-4a09-8b92-51757b584a62
    """

    interval: list[float]
    values: list[float]
    vrs: str


class Extent(BaseModel):
    """OGC EDR Extent metadata

    https://docs.ogc.org/is/19-086r6/19-086r6.html#_2a2d533f-6efe-48df-8056-2eca9deb848f
    """

    spatial: SpatialExtent
    temporal: Optional[TemporalExtent] = None
    vertical: Optional[VerticalExtent] = None


class EDRQueryMetadata(BaseModel):
    """OGC EDR Query metadata

    https://docs.ogc.org/is/19-086r6/19-086r6.html#_9a6620ce-6093-4b1b-8f68-2e2c04a13746
    """

    link: Link


class DataQueries(BaseModel):
    """OGC EDR Data Queries metadata

    https://docs.ogc.org/is/19-086r6/19-086r6.html#_df2c080b-949c-40c3-ad14-d20228270c2d
    """

    position: Optional[EDRQueryMetadata] = None
    radius: Optional[EDRQueryMetadata] = None
    area: Optional[EDRQueryMetadata] = None
    cube: Optional[EDRQueryMetadata] = None
    trajectory: Optional[EDRQueryMetadata] = None
    corridor: Optional[EDRQueryMetadata] = None
    item: Optional[EDRQueryMetadata] = None
    location: Optional[EDRQueryMetadata] = None


class SymbolMetadata(BaseModel):
    """OGC EDR Symbol metadata

    https://docs.ogc.org/is/19-086r6/19-086r6.html#_3e50c10c-85bd-46d9-8e09-1c5fffffb055
    """

    title: Optional[str] = None
    description: Optional[str] = None
    value: Optional[str] = None
    type_: Optional[str] = Field(None, serialization_alias="type")


class UnitMetadata(BaseModel):
    """OGC EDR Unit metadata

    https://docs.ogc.org/is/19-086r6/19-086r6.html#_5378d779-6a38-4607-9051-6f12c3d3107b
    """

    label: str
    symbol: SymbolMetadata


class MeasurementType(BaseModel):
    """OGC EDR Measurement Type metadata

    https://docs.ogc.org/is/19-086r6/19-086r6.html#_c81181d6-fd09-454e-9c00-a3bb3b21d592
    """

    method: str
    duration: str


class ObservedProperty(BaseModel):
    """OGC EDR Observed Property metadata

    https://docs.ogc.org/is/19-086r6/19-086r6.html#_7e053ab4-5cde-4a5c-a8be-acc6495f9eb5
    """

    id: Optional[str] = None
    label: str
    description: Optional[str] = None


class Parameter(BaseModel):
    """OGC EDR Parameter metadata

    https://docs.ogc.org/is/19-086r6/19-086r6.html#_da400aef-f6ee-4d08-b36c-2f535d581d53
    """

    id: Optional[str] = None
    type_: Literal["Parameter"] = Field("Parameter", serialization_alias="type")
    label: Optional[str] = None
    description: Optional[str] = None
    data_type: Optional[str] = Field(None, serialization_alias="data-type")
    unit: Optional[UnitMetadata] = None
    observed_property: ObservedProperty = Field(
        ...,
        serialization_alias="observedProperty",
    )
    extent: Optional[Extent] = None
    measurement_type: Optional[MeasurementType] = Field(
        None,
        serialization_alias="measurementType",
    )


class Collection(BaseModel):
    """OGC EDR Collection metadata

    https://docs.ogc.org/is/19-086r6/19-086r6.html#_b6db449c-4ca7-4117-9bf4-241984cef569
    """

    links: list[Link]
    id: str
    title: str
    description: str
    keywords: list[str]
    extent: Extent
    data_queries: DataQueries
    crs: list[str]
    output_formats: list[str]
    parameter_names: dict[str, Parameter]


def crs_details(crs: pyproj.CRS) -> CRSDetails:
    """
    Return CF version of EDR CRS metadata
    """
    return CRSDetails(crs=crs.to_string(), wkt=crs.to_wkt())


def unit(unit: str) -> UnitMetadata:
    """
    Return CF version of EDR Unit metadata
    """
    return UnitMetadata(
        label=unit,
        symbol=SymbolMetadata(
            value=unit,
            type="unit",
        ),
    )


def parameter(da: xr.DataArray) -> Parameter:
    """
    Return CF version of EDR Parameter metadata for a given xarray variable
    """
    name = da.attrs.get("name", None)
    standard_name = da.attrs.get("standard_name", name if name else "")
    observed_property = ObservedProperty(
        label=standard_name,
        description=da.attrs.get("long_name", ""),
    )
    return Parameter(
        label=standard_name,
        type_="Parameter",
        description=da.attrs.get("long_name", ""),
        data_type=da.dtype.name,
        unit=unit(da.attrs.get("units", "")),
        observed_property=observed_property,
    )


def spatial_extent(ds: xr.Dataset, crs: pyproj.CRS) -> SpatialExtent:
    """Extract the spatial extent from the dataset into collection metadata specific format"""
    bounds = spatial_bounds(ds)

    return SpatialExtent(
        bbox=[bounds],
        crs=crs.to_string(),
    )


def temporal_extent(ds: xr.Dataset) -> Optional[TemporalExtent]:
    """Extract the temporal extent from the dataset into collection metadata specific format"""
    if "T" not in ds.cf:
        return None

    t = pd.to_datetime(ds.cf["T"])
    time_min = t.min().strftime("%Y-%m-%dT%H:%M:%S")
    time_max = t.max().strftime("%Y-%m-%dT%H:%M:%S")
    return TemporalExtent(
        interval=[str(time_min), str(time_max)],
        values=[f"{time_min}/{time_max}"],
        trs='TIMECRS["DateTime",TDATUM["Gregorian Calendar"],CS[TemporalDateTime,1],AXIS["Time (T)",unspecified]]',  # noqa
    )


def vertical_extent(ds: xr.Dataset) -> Optional[VerticalExtent]:
    """Extract the vertical extent from the dataset into collection metadata specific format"""
    if "Z" not in ds.cf:
        return None

    z = ds.cf["Z"]
    elevations = z.values
    units = z.attrs.get("units", "unknown")
    positive = z.attrs.get("positive", "up")
    min_z = elevations.min()
    max_z = elevations.max()

    return VerticalExtent(
        interval=[min_z, max_z],
        values=elevations,
        vrs=f"VERTCRS[VERT_CS['unknown'],AXIS['Z',{positive}],UNIT[{units},1]]",  # noqa
    )


def extent(ds: xr.Dataset, crs: pyproj.CRS) -> Extent:
    """
    Extract the extent from the dataset into collection metadata specific format
    """
    spatial = spatial_extent(ds, crs)
    temporal = temporal_extent(ds)
    vertical = vertical_extent(ds)

    return Extent(
        spatial=spatial,
        temporal=temporal,
        vertical=vertical,
    )


def extract_parameters(ds: xr.Dataset) -> dict[str, Parameter]:
    """
    Extract the parameters from the dataset into collection metadata specific format
    """
    return {k: parameter(v) for k, v in ds.variables.items() if "axis" not in v.attrs}


def position_query_description(
    output_formats: list[str],
    crs_details: list[CRSDetails],
) -> EDRQueryMetadata:
    """
    Return CF version of EDR Position Query metadata
    """
    return EDRQueryMetadata(
        link=Link(
            href="/edr/position?coords={coords}",
            hreflang="en",
            rel="data",
            templated=True,
            variables=VariablesMetadata(
                title="Position query",
                description="Returns position data based on WKT `POINT(lon lat)` or `MULTIPOINT(lon lat, ...)` coordinates",  # noqa
                query_type="position",
                coords={
                    "type": "string",
                    "description": "WKT `POINT(lon lat)` or `MULTIPOINT(lon lat, ...)` coordinates",  # noqa
                    "required": True,
                },
                output_formats=output_formats,
                default_output_format="cf_covjson",
                crs_details=crs_details,
            ),
        ),
    )


def area_query_description(
    output_formats: list[str],
    crs_details: list[CRSDetails],
) -> EDRQueryMetadata:
    """
    Return CF version of EDR Area Query metadata
    """
    return EDRQueryMetadata(
        link=Link(
            href="/edr/area?coords={coords}",
            hreflang="en",
            rel="data",
            templated=True,
            variables=VariablesMetadata(
                title="Area query",
                description="Returns data in a polygon based on WKT `POLYGON(lon lat, ...)` coordinates",  # noqa
                query_type="area",
                coords={
                    "type": "string",
                    "description": "WKT `POLYGON(lon lat, ...)` coordinates",
                    "required": True,
                },
                output_formats=output_formats,
                default_output_format="cf_covjson",
                crs_details=crs_details,
            ),
        ),
    )


def cube_query_description(
    output_formats: list[str],
    crs_details: list[CRSDetails],
) -> EDRQueryMetadata:
    """
    Return CF version of EDR Cube Query metadata
    """
    return EDRQueryMetadata(
        link=Link(
            href="/edr/cube?bbox={bbox}",
            hreflang="en",
            rel="data",
            templated=True,
            variables=VariablesMetadata(
                title="Cube query",
                description="Returns data in a cube based on a bounding box, with optional elevation",
                query_type="cube",
                bbox={
                    "type": "string",
                    "description": "Bounding box in the format `min_x,min_y,max_x,max_y`",
                    "required": True,
                },
                output_formats=output_formats,
                default_output_format="cf_covjson",
                crs_details=crs_details,
            ),
        ),
    )


def collection_metadata(ds: xr.Dataset, output_formats: list[str]) -> Collection:
    """
    Returns the collection metadata for the dataset
    There is no nested hierarchy in our router right now, so instead we return the metadata
    for the current dataset as the a single collection. See the spec for more information:
    https://docs.ogc.org/is/19-086r6/19-086r6.html#_162817c2-ccd7-43c9-b1ea-ad3aea1b4d6b
    """
    id = ds.attrs.get("_xpublish_id", "unknown")
    title = ds.attrs.get("title", "unknown")
    description = ds.attrs.get("description", "no description")

    crs = dataset_crs(ds)

    # We will use the dataset's CRS as the default CRS, but use 4326 for the extents
    # since it is always available
    projected_ds = project_dataset(ds, DEFAULT_CRS)

    extents = extent(projected_ds, crs)

    parameters = extract_parameters(ds)

    supported_crs = [
        crs_details(crs),
    ]

    # 4326 is always available
    if crs != DEFAULT_CRS:
        supported_crs.append(
            crs_details(DEFAULT_CRS),
        )

    return Collection(
        links=[],
        id=id,
        title=title,
        description=description,
        keywords=[],
        extent=extents,
        data_queries=DataQueries(
            position=position_query_description(output_formats, supported_crs),
            area=area_query_description(output_formats, supported_crs),
            cube=cube_query_description(output_formats, supported_crs),
        ),
        crs=[crs.to_string()],
        output_formats=output_formats,
        parameter_names=parameters,
    )
