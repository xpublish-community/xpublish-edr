"""Integration compliance suite for xpublish-edr composed with xpublish-ogc-core.

This tests both the OGC core plugin serves the landing page, conformance, and collection endpoints, and the EDR plugin contributes its conformance classes, collection metadata, and data queries through the OGC hookspecs.
"""

import cf_xarray  # noqa: F401
import pytest
import xpublish
from fastapi.testclient import TestClient
from xpublish_ogc_core.plugin import (
    OGC_API_COMMON_CONFORMANCE_CLASSES,
    OgcCorePlugin,
)
from xpublish_ogc_core.testing import validate_response

from xpublish_edr.plugin import EDR_CONFORMANCE_CLASSES, CfEdrPlugin


@pytest.fixture(scope="module")
def cf_air_dataset():
    from cf_xarray.datasets import airds

    return airds


@pytest.fixture(scope="module")
def ogc_app(cf_air_dataset):
    rest = xpublish.Rest(
        {"air": cf_air_dataset},
        plugins={
            "ogc": OgcCorePlugin(),
            "edr": CfEdrPlugin(),
        },
    )

    return rest.app


@pytest.fixture(scope="module")
def client(ogc_app):
    return TestClient(ogc_app)


def test_landing_page(client):
    response = client.get("/")

    assert response.status_code == 200, "Response did not return successfully"

    data = response.json()
    validate_response("landingPage", data)

    rels = {link["rel"] for link in data["links"]}
    for rel in ("self", "service-desc", "service-doc", "conformance", "data"):
        assert rel in rels, f"Landing page should include a {rel!r} link"


def test_conformance(client):
    response = client.get("/conformance")

    assert response.status_code == 200, "Response did not return successfully"

    data = response.json()
    validate_response("confClasses", data)

    for conformance_class in OGC_API_COMMON_CONFORMANCE_CLASSES:
        assert conformance_class in data["conformsTo"], (
            f"OGC API Common class {conformance_class} should be declared"
        )

    for conformance_class in EDR_CONFORMANCE_CLASSES:
        assert conformance_class in data["conformsTo"], (
            f"EDR class {conformance_class} should be declared"
        )


def test_collections(client):
    response = client.get("/collections")

    assert response.status_code == 200, "Response did not return successfully"

    data = response.json()
    validate_response("collections", data)

    collection_ids = [collection["id"] for collection in data["collections"]]
    assert collection_ids == ["air"]


def test_collection(client):
    response = client.get("/collections/air")

    assert response.status_code == 200, "Response did not return successfully"

    data = response.json()
    validate_response("collection", data)

    assert data["id"] == "air"

    # EDR collection metadata contributed via ogc_collection_metadata
    assert data["extent"]["spatial"]["bbox"] == [[200.0, 15.0, 322.5, 75.0]]
    assert data["extent"]["temporal"]["interval"] == [
        ["2013-01-01T00:00:00", "2013-01-01T18:00:00"],
    ]
    assert "air" in data["parameter_names"]
    assert data["crs"] == ["EPSG:4326"]
    assert "cf_covjson" in data["output_formats"]

    # every supported EDR geometry query is described via ogc_collection_dataqueries,
    # with the relative hrefs made absolute by ogc-core
    for query_type in ("position", "area", "cube"):
        link = data["data_queries"][query_type]["link"]
        assert link["href"].startswith(
            f"http://testserver/collections/air/{query_type}",
        )
        assert link["variables"]["query_type"] == query_type


def test_unknown_collection_returns_ogc_exception(client):
    response = client.get("/collections/not-a-collection")

    assert response.status_code == 404

    data = response.json()
    validate_response("exception", data)


def test_position_query(client):
    response = client.get("/collections/air/position?coords=POINT(204 44)&f=cf_covjson")

    assert response.status_code == 200, "Response did not return successfully"
    assert "json" in response.headers["content-type"]

    data = response.json()
    assert data["type"] == "Coverage", "Response should be a CoverageJSON Coverage"
    assert "air" in data["ranges"], "Response should include the air parameter"


def test_area_query(client):
    coords = "POLYGON((200 40, 200 50, 210 50, 210 40, 200 40))"
    response = client.get(f"/collections/air/area?coords={coords}&f=cf_covjson")

    assert response.status_code == 200, "Response did not return successfully"

    data = response.json()
    assert data["type"] == "Coverage", "Response should be a CoverageJSON Coverage"


def test_cube_query(client):
    response = client.get("/collections/air/cube?bbox=200,40,210,50&f=cf_covjson")

    assert response.status_code == 200, "Response did not return successfully"

    data = response.json()
    assert data["type"] == "Coverage", "Response should be a CoverageJSON Coverage"


def test_invalid_query_returns_ogc_exception(client):
    """A request validation error returns an OGC exception body, not FastAPI's
    default ``{"detail": [...]}``.

    The official OGC EDR schema requires error responses to be OGC exception
    objects (with a string ``code`` member). FastAPI rejects the empty ``f``
    here with a 422 before the handler runs; OGCExceptionRoute reshapes it.
    """
    response = client.get("/collections/air/position?coords=POINT(204 44)&f=")

    assert response.status_code == 422
    data = response.json()
    assert data["code"] == "422", "OGC exception requires a string `code` member"
    validate_response("exception", data)


def test_malformed_coords_returns_422_not_500(client):
    """Coords that WKT parsing cannot handle return a 422, not a 500.

    ``shapely.wkt.loads`` raises non-``GEOSException`` errors for some inputs
    (e.g. ``UnicodeDecodeError`` for a string GEOS cannot encode); these must
    be caught and surfaced as an OGC exception rather than an unhandled 500.
    """
    # "%C0" is an invalid UTF-8 byte; shapely raises UnicodeDecodeError on it
    response = client.get("/collections/air/position?coords=%C0")

    assert response.status_code == 422
    data = response.json()
    assert data["code"] == "422"
    validate_response("exception", data)


def test_position_post_query(client):
    """The collection-level position route accepts POST with points in the body.

    Without a POST handler, the GET-only route would return 405 (with a body
    that violates the OGC exception schema); the points are submitted in the
    body instead of via the (GET-only) required ``coords`` query parameter.
    """
    response = client.post(
        "/collections/air/position?f=cf_covjson",
        content="lon,lat\n202,43\n205,45\n",
        headers={"content-type": "text/csv"},
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["type"] == "Coverage"
    assert "air" in data["ranges"]


def test_area_post_query(client):
    """The collection-level area route accepts POST with a polygon in the body."""
    polygon = "POLYGON((200 40, 200 50, 210 50, 210 40, 200 40))"
    response = client.post(
        "/collections/air/area?f=cf_covjson",
        content=polygon,
        headers={"content-type": "text/plain"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["type"] == "Coverage"


@pytest.mark.parametrize("query_type", ["position", "area"])
def test_coords_parameter_is_required(client, query_type):
    """The OGC EDR ``coords`` query parameter must be declared ``required``.

    These endpoints are GET-only (POST submits geometry in the body lives on
    the dataset router), so coords is mandatory; the CITE suite's
    ``{position,area}CoordsParameterDefinition`` tests assert ``required: true``.
    """
    schema = client.get("/openapi.json").json()
    operation = schema["paths"][f"/collections/{{collection_id}}/{query_type}"]["get"]
    coords = next(p for p in operation["parameters"] if p["name"] == "coords")

    assert coords["required"] is True

    # and at runtime omitting coords is a (well-shaped) 422, not a 200
    response = client.get(f"/collections/air/{query_type}")
    assert response.status_code == 422
    assert response.json()["code"] == "422"


@pytest.mark.parametrize("query_type", ["position", "area"])
def test_post_openapi_omits_coords_parameter(client, query_type):
    """POST position/area read geometry from the request body, not query params."""
    schema = client.get("/openapi.json").json()
    operation = schema["paths"][f"/collections/{{collection_id}}/{query_type}"]["post"]

    assert all(param["name"] != "coords" for param in operation.get("parameters", []))
