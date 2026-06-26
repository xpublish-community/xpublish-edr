"""Unit tests for the request-body geometry parsers in ``geometry/parse.py``.

These exercise the parsers directly (rather than through the app) to cover the
malformed-input error branches that the router-level POST tests don't reach.
"""

import json

import pytest
import shapely

from xpublish_edr.geometry.parse import (
    parse_csv_points,
    parse_geojson_points,
    parse_geojson_polygons,
    parse_wkt_polygons,
)


class TestParseCsvPoints:
    def test_single_point(self):
        geom = parse_csv_points(b"lon,lat\n202,43\n")
        assert isinstance(geom, shapely.Point)
        assert (geom.x, geom.y) == (202.0, 43.0)

    def test_multipoint_with_aliases(self):
        geom = parse_csv_points(b"longitude,latitude\n1,2\n3,4\n")
        assert isinstance(geom, shapely.MultiPoint)
        assert len(geom.geoms) == 2

    def test_xy_aliases_and_blank_rows_skipped(self):
        geom = parse_csv_points(b"x,y\n1,2\n\n   \n3,4\n")
        assert isinstance(geom, shapely.MultiPoint)
        assert len(geom.geoms) == 2

    def test_empty_body(self):
        with pytest.raises(ValueError, match="CSV body is empty"):
            parse_csv_points(b"")

    def test_missing_columns(self):
        with pytest.raises(ValueError, match="must include x/y"):
            parse_csv_points(b"foo,bar\n1,2\n")

    def test_invalid_coordinate(self):
        with pytest.raises(ValueError, match="Invalid coordinate on CSV row 2"):
            parse_csv_points(b"lon,lat\nnotanumber,43\n")


class TestParseGeojsonPoints:
    def test_point(self):
        body = json.dumps({"type": "Point", "coordinates": [202, 43]}).encode()
        geom = parse_geojson_points(body)
        assert isinstance(geom, shapely.Point)

    def test_multipoint(self):
        body = json.dumps({"type": "MultiPoint", "coordinates": [[1, 2], [3, 4]]}).encode()
        geom = parse_geojson_points(body)
        assert isinstance(geom, shapely.MultiPoint)

    def test_feature_collection_with_geometry_collection(self):
        body = json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "GeometryCollection",
                            "geometries": [{"type": "Point", "coordinates": [1, 2]}],
                        },
                    },
                ],
            },
        ).encode()
        geom = parse_geojson_points(body)
        assert isinstance(geom, shapely.Point)

    def test_feature_with_null_geometry_is_skipped(self):
        body = json.dumps({"type": "Feature", "geometry": None}).encode()
        with pytest.raises(ValueError, match="no Point or MultiPoint"):
            parse_geojson_points(body)

    def test_invalid_json(self):
        with pytest.raises(ValueError, match="Invalid GeoJSON"):
            parse_geojson_points(b"{not json")

    def test_non_object(self):
        with pytest.raises(ValueError, match="must be an object"):
            parse_geojson_points(b"[1, 2]")

    def test_no_point_geometries(self):
        body = json.dumps({"type": "FeatureCollection", "features": []}).encode()
        with pytest.raises(ValueError, match="no Point or MultiPoint"):
            parse_geojson_points(body)

    def test_unsupported_type(self):
        body = json.dumps({"type": "LineString", "coordinates": [[1, 2], [3, 4]]}).encode()
        with pytest.raises(ValueError, match="Unsupported GeoJSON type"):
            parse_geojson_points(body)

    def test_invalid_coordinate_pair(self):
        body = json.dumps({"type": "Point", "coordinates": [1]}).encode()
        with pytest.raises(ValueError, match="Invalid GeoJSON coordinate"):
            parse_geojson_points(body)


class TestParseGeojsonPolygons:
    SHELL = [[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]

    def test_polygon(self):
        body = json.dumps({"type": "Polygon", "coordinates": [self.SHELL]}).encode()
        geom = parse_geojson_polygons(body)
        assert isinstance(geom, shapely.Polygon)

    def test_polygon_with_hole(self):
        hole = [[0.2, 0.2], [0.2, 0.8], [0.8, 0.8], [0.8, 0.2], [0.2, 0.2]]
        body = json.dumps({"type": "Polygon", "coordinates": [self.SHELL, hole]}).encode()
        geom = parse_geojson_polygons(body)
        assert isinstance(geom, shapely.Polygon)
        assert len(geom.interiors) == 1

    def test_multipolygon_merges(self):
        other = [[[2, 2], [2, 3], [3, 3], [3, 2], [2, 2]]]
        body = json.dumps(
            {"type": "MultiPolygon", "coordinates": [[self.SHELL], other]},
        ).encode()
        geom = parse_geojson_polygons(body)
        assert isinstance(geom, shapely.MultiPolygon)
        assert len(geom.geoms) == 2

    def test_invalid_json(self):
        with pytest.raises(ValueError, match="Invalid GeoJSON"):
            parse_geojson_polygons(b"{not json")

    def test_no_polygons(self):
        body = json.dumps({"type": "Point", "coordinates": [1, 2]}).encode()
        with pytest.raises(ValueError, match="Unsupported GeoJSON type"):
            parse_geojson_polygons(body)

    def test_empty_feature_collection(self):
        body = json.dumps({"type": "FeatureCollection", "features": []}).encode()
        with pytest.raises(ValueError, match="no Polygon or MultiPolygon"):
            parse_geojson_polygons(body)

    def test_invalid_rings(self):
        body = json.dumps({"type": "Polygon", "coordinates": []}).encode()
        with pytest.raises(ValueError, match="Invalid Polygon coordinates"):
            parse_geojson_polygons(body)


class TestParseWktPolygons:
    def test_polygon(self):
        geom = parse_wkt_polygons(b"POLYGON((0 0, 0 1, 1 1, 1 0, 0 0))")
        assert isinstance(geom, shapely.Polygon)

    def test_multipolygon(self):
        geom = parse_wkt_polygons(
            b"MULTIPOLYGON(((0 0, 0 1, 1 1, 1 0, 0 0)), ((2 2, 2 3, 3 3, 3 2, 2 2)))",
        )
        assert isinstance(geom, shapely.MultiPolygon)

    def test_empty_body(self):
        with pytest.raises(ValueError, match="WKT body is empty"):
            parse_wkt_polygons(b"   ")

    def test_invalid_wkt(self):
        with pytest.raises(ValueError, match="Invalid WKT"):
            parse_wkt_polygons(b"POLYGON(garbage)")

    def test_non_polygon_geometry(self):
        with pytest.raises(ValueError, match="must be a Polygon or MultiPolygon"):
            parse_wkt_polygons(b"POINT(1 2)")
