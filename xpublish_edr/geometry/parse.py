"""
Parse request bodies (CSV, GeoJSON) into shapely geometries for position queries.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Iterable

import shapely

CSV_X_ALIASES = ("x", "lon", "longitude")
CSV_Y_ALIASES = ("y", "lat", "latitude")


def parse_position_body(
    body: bytes,
    content_type: str | None,
) -> shapely.Point | shapely.MultiPoint:
    """Dispatch body parsing based on Content-Type.

    Supported types:
      - text/csv
      - application/geo+json, application/json
    """
    media_type = (content_type or "").split(";", 1)[0].strip().lower()
    if media_type == "text/csv":
        return parse_csv_points(body)
    if media_type in ("application/geo+json", "application/json", ""):
        return parse_geojson_points(body)
    raise ValueError(
        f"Unsupported Content-Type {content_type!r}. "
        "Use text/csv or application/geo+json.",
    )


def parse_csv_points(body: bytes) -> shapely.Point | shapely.MultiPoint:
    """Parse a CSV body with x/y (or lon/lat) columns into a (Multi)Point."""
    text = body.decode("utf-8")
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        raise ValueError("CSV body is empty")

    normalized = [h.strip().lower() for h in header]
    x_idx = _find_column(normalized, CSV_X_ALIASES)
    y_idx = _find_column(normalized, CSV_Y_ALIASES)
    if x_idx is None or y_idx is None:
        raise ValueError(
            "CSV must include x/y, lon/lat, or longitude/latitude columns. "
            f"Got columns: {header}",
        )

    coords: list[tuple[float, float]] = []
    for line_no, row in enumerate(reader, start=2):
        if not row or all(not cell.strip() for cell in row):
            continue
        try:
            coords.append((float(row[x_idx]), float(row[y_idx])))
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid coordinate on CSV row {line_no}: {e}")

    return _points_to_geometry(coords)


def parse_geojson_points(body: bytes) -> shapely.Point | shapely.MultiPoint:
    """Parse a GeoJSON body into a (Multi)Point.

    Accepts a Point or MultiPoint geometry, a Feature wrapping one,
    or a FeatureCollection of Point/MultiPoint features.
    """
    try:
        obj = json.loads(body)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid GeoJSON: {e}")

    coords: list[tuple[float, float]] = []
    _collect_point_coords(obj, coords)
    if not coords:
        raise ValueError("GeoJSON body contained no Point or MultiPoint geometries")
    return _points_to_geometry(coords)


def _collect_point_coords(
    obj: object,
    out: list[tuple[float, float]],
) -> None:
    """Recursively collect (x, y) pairs from a GeoJSON object into ``out``."""
    if not isinstance(obj, dict):
        raise ValueError("GeoJSON must be an object")
    obj_type = obj.get("type")
    if obj_type == "FeatureCollection":
        for feature in obj.get("features", []):
            _collect_point_coords(feature, out)
    elif obj_type == "Feature":
        geom = obj.get("geometry")
        if geom is not None:
            _collect_point_coords(geom, out)
    elif obj_type == "GeometryCollection":
        for geom in obj.get("geometries", []):
            _collect_point_coords(geom, out)
    elif obj_type == "Point":
        out.append(_coord_pair(obj.get("coordinates")))
    elif obj_type == "MultiPoint":
        for c in obj.get("coordinates", []):
            out.append(_coord_pair(c))
    else:
        raise ValueError(
            f"Unsupported GeoJSON type {obj_type!r}; expected Point, MultiPoint, "
            "Feature, FeatureCollection, or GeometryCollection",
        )


def _coord_pair(c: object) -> tuple[float, float]:
    """Validate a GeoJSON coordinate sequence and return it as an (x, y) tuple."""
    if not isinstance(c, (list, tuple)) or len(c) < 2:
        raise ValueError(f"Invalid GeoJSON coordinate: {c!r}")
    return float(c[0]), float(c[1])


def _find_column(header: list[str], aliases: Iterable[str]) -> int | None:
    """Return the index of the first alias present in ``header``, or None."""
    for alias in aliases:
        if alias in header:
            return header.index(alias)
    return None


def _points_to_geometry(
    coords: list[tuple[float, float]],
) -> shapely.Point | shapely.MultiPoint:
    """Build a Point if there is exactly one coord, otherwise a MultiPoint."""
    if not coords:
        raise ValueError("No coordinates provided")
    if len(coords) == 1:
        return shapely.Point(coords[0])
    return shapely.MultiPoint(coords)
