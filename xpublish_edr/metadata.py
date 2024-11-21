import pyproj
import xarray as xr

from xpublish_edr.geometry.common import (
    DEFAULT_CRS,
    dataset_crs,
    project_dataset,
    spatial_bounds,
)


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
    output_formats: list[str], crs_details: list[dict]
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
