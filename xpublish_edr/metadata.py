from typing import Optional

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

    title: Optional[str]
    description: Optional[str]
    query_type: Optional[str]
    coords: Optional[dict]
    output_formats: Optional[list[str]]
    default_output_format: Optional[str]
    crs_details: Optional[list[CRSDetails]]


class Link(BaseModel):
    """OGC EDR Link metadata

    https://docs.ogc.org/is/19-086r6/19-086r6.html#_ea77762b-89c0-4704-b845-748efc66e597
    """

    href: str
    rel: str
    type_: Optional[str] = Field(None, serialization_alias="type")
    hreflang: Optional[str]
    title: Optional[str]
    length: Optional[int]
    templated: Optional[bool]
    variables: Optional[VariablesMetadata]


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
    temporal: Optional[TemporalExtent]
    vertical: Optional[VerticalExtent]


class EDRQueryMetadata(BaseModel):
    """OGC EDR Query metadata

    https://docs.ogc.org/is/19-086r6/19-086r6.html#_9a6620ce-6093-4b1b-8f68-2e2c04a13746
    """

    link: Link


class DataQueries(BaseModel):
    """OGC EDR Data Queries metadata

    https://docs.ogc.org/is/19-086r6/19-086r6.html#_df2c080b-949c-40c3-ad14-d20228270c2d
    """

    position: Optional[EDRQueryMetadata]
    radius: Optional[EDRQueryMetadata]
    area: Optional[EDRQueryMetadata]
    cube: Optional[EDRQueryMetadata]
    trajectory: Optional[EDRQueryMetadata]
    corridor: Optional[EDRQueryMetadata]
    item: Optional[EDRQueryMetadata]
    location: Optional[EDRQueryMetadata]


class SymbolMetadata(BaseModel):
    """OGC EDR Symbol metadata

    https://docs.ogc.org/is/19-086r6/19-086r6.html#_3e50c10c-85bd-46d9-8e09-1c5fffffb055
    """

    title: Optional[str]
    description: Optional[str]
    value: Optional[str]
    type: Optional[str]


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

    id: Optional[str]
    label: str
    description: Optional[str]


class Parameter(BaseModel):
    """OGC EDR Parameter metadata

    https://docs.ogc.org/is/19-086r6/19-086r6.html#_da400aef-f6ee-4d08-b36c-2f535d581d53
    """

    id: Optional[str]
    type_: str = Field(..., serialization_alias="type")
    label: Optional[str]
    description: Optional[str]
    data_type: Optional[str] = Field(None, serialization_alias="data-type")
    unit: Optional[UnitMetadata]
    observed_property: ObservedProperty = Field(
        ...,
        serialization_alias="observedProperty",
    )
    extent: Optional[Extent]
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
    extent: dict
    data_queries: dict
    crs: list[str]
    output_formats: list[str]
    parameter_names: dict


def crs_description(crs: pyproj.CRS) -> dict:
    """
    Return CF version of EDR CRS metadata
    """
    return {
        "crs": crs.to_string(),
        "wkt": crs.to_wkt(),
    }


def variable_description(da: xr.DataArray) -> dict:
    """
    Return CF version of EDR Parameter metadata for a given xarray variable
    """
    name = da.attrs.get("name", None)
    standard_name = da.attrs.get("standard_name", name if name else "")
    label = standard_name if not name else name
    long_name = da.attrs.get("long_name", "")
    units = da.attrs.get("units", "")
    return {
        "type": "Parameter",
        "description": long_name,
        "unit": {
            "label": units,
        },
        "observedProperty": {
            "label": label,
            "standard_name": standard_name,
            "long_name": long_name,
        },
    }


def extract_parameters(ds: xr.Dataset) -> dict:
    """
    Extract the parameters from the dataset into collection metadata specific format
    """
    return {
        k: variable_description(v)
        for k, v in ds.variables.items()
        if "axis" not in v.attrs
    }


def spatial_extent_description(ds: xr.Dataset, crs: pyproj.CRS) -> dict:
    """
    Extract the spatial extent from the dataset into collection metadata specific format
    """
    # We will use the dataset's CRS as the default CRS, but use 4326 for the extents
    # since it is always available
    bounds = spatial_bounds(ds)

    return {
        "bbox": [bounds],
        "crs": crs.to_string(),
    }


def temporal_extent_description(ds: xr.Dataset) -> dict:
    """
    Extract the temporal extent from the dataset into collection metadata specific format
    """
    time_min = ds["T"].min().dt.strftime("%Y-%m-%dT%H:%M:%S").values
    time_max = ds["T"].max().dt.strftime("%Y-%m-%dT%H:%M:%S").values
    return {
        "interval": [
            str(time_min),
            str(time_max),
        ],
        "values": [
            f"{time_min}/{time_max}",
        ],
        # TODO: parse `ds.cf["time"].dt.calendar`
        "trs": 'TIMECRS["DateTime",TDATUM["Gregorian Calendar"],CS[TemporalDateTime,1],AXIS["Time (T)",unspecified]]',  # noqa
    }


def vertical_extent_description(ds: xr.Dataset) -> dict:
    """
    Extract the vertical extent from the dataset into collection metadata specific format
    """
    elevations = ds.cf["Z"].values
    units = ds.cf["Z"].attrs.get("units", "unknown")
    positive = ds.cf["Z"].attrs.get("positive", "up")
    min_z = elevations.min()
    max_z = elevations.max()
    elevation_values = ",".join([str(e) for e in elevations])

    return {
        "interval": [
            min_z,
            max_z,
        ],
        "values": elevation_values,
        "vrs": f"VERTCRS[VERT_CS['unknown'],AXIS['Z',{positive}],UNIT[{units},1]]",  # noqa
        "positive": positive,
        "units": units,
    }


def position_query_description(
    output_formats: list[str],
    crs_details: list[dict],
) -> dict:
    """
    Return CF version of EDR Position Query metadata
    """
    return {
        "href": "/edr/position",
        "hreflang": "en",
        "rel": "data",
        "templated": True,
        "variables": {
            "title": "Position query",
            "description": "Returns position data based on WKT `POINT(lon lat)` or `MULTIPOINT(lon lat, ...)` coordinates",  # noqa
            "query_type": "position",
            "coords": {
                "type": "string",
                "description": "WKT `POINT(lon lat)` or `MULTIPOINT(lon lat, ...)` coordinates",  # noqa
                "required": True,
            },
            "output_format": output_formats,
            "default_output_format": "cf_covjson",
            "crs_details": crs_details,
        },
    }


def area_query_description(output_formats: list[str], crs_details: list[dict]) -> dict:
    """
    Return CF version of EDR Area Query metadata
    """
    return {
        "href": "/edr/area?coords={coords}",
        "hreflang": "en",
        "rel": "data",
        "templated": True,
        "variables": {
            "title": "Area query",
            "description": "Returns data in a polygon based on WKT `POLYGON(lon lat, ...)` coordinates",  # noqa
            "query_type": "position",
            "coords": {
                "type": "string",
                "description": "WKT `POLYGON(lon lat, ...)` coordinates",
                "required": True,
            },
            "output_format": output_formats,
            "default_output_format": "cf_covjson",
            "crs_details": crs_details,
        },
    }


def collection_metadata(ds: xr.Dataset, output_formats: list[str]) -> dict:
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

    ds_cf = ds.cf

    # We will use the dataset's CRS as the default CRS, but use 4326 for the extents
    # since it is always available
    projected_ds = project_dataset(ds, DEFAULT_CRS)

    extents: dict = {
        "spatial": spatial_extent_description(projected_ds, DEFAULT_CRS),
    }

    if "T" in ds_cf:
        extents["temporal"] = temporal_extent_description(ds_cf)

    if "Z" in ds_cf:
        extents["vertical"] = vertical_extent_description(ds_cf)

    parameters = extract_parameters(ds)

    crs_details = [
        crs_description(crs),
    ]

    # 4326 is always available
    if crs != DEFAULT_CRS:
        crs_details.append(
            crs_description(DEFAULT_CRS),
        )

    return {
        "id": id,
        "title": title,
        "description": description,
        "links": [],
        "extent": extents,
        "data_queries": {
            "position": position_query_description(output_formats, crs_details),
            "area": area_query_description(output_formats, crs_details),
        },
        "crs": [crs.to_string()],
        "output_formats": output_formats,
        "parameter_names": parameters,
    }
