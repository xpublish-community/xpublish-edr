"""Dispatch area body parsing based on Content-Type."""

import shapely

from xpublish_edr.geometry.parse import (
    JSON_MEDIA_TYPES,
    WKT_MEDIA_TYPES,
    _media_type,
    parse_geojson_polygons,
    parse_wkt_polygons,
)


def parse_body(
    body: bytes,
    content_type: str | None,
) -> shapely.Polygon | shapely.MultiPolygon:
    """Dispatch area body parsing based on Content-Type.

    Supported types:
      - application/geo+json, application/json: Polygon / MultiPolygon /
        Feature / FeatureCollection / GeometryCollection
      - application/wkt, text/wkt, text/plain: raw WKT Polygon or MultiPolygon
    """
    media_type = _media_type(content_type)
    if media_type in JSON_MEDIA_TYPES:
        return parse_geojson_polygons(body)
    if media_type in WKT_MEDIA_TYPES:
        return parse_wkt_polygons(body)
    raise ValueError(
        f"Unsupported Content-Type {content_type!r}. Use application/geo+json or application/wkt.",
    )
