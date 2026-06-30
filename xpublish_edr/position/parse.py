"""Dispatch position body parsing based on Content-Type."""

import shapely

from xpublish_edr.geometry.parse import (
    JSON_MEDIA_TYPES,
    _media_type,
    parse_csv_points,
    parse_geojson_points,
)


def parse_body(
    body: bytes,
    content_type: str | None,
) -> shapely.Point | shapely.MultiPoint:
    """Dispatch body parsing based on Content-Type.

    Supported types:
      - text/csv
      - application/geo+json, application/json
    """
    media_type = _media_type(content_type)
    if media_type == "text/csv":
        return parse_csv_points(body)
    if media_type in JSON_MEDIA_TYPES:
        return parse_geojson_points(body)
    raise ValueError(
        f"Unsupported Content-Type {content_type!r}. Use text/csv or application/geo+json.",
    )
