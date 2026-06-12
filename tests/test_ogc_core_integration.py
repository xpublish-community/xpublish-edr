"""Integration compliance suite for xpublish-edr composed with xpublish-ogc-core.

This suite lives here because this repo composes both plugins: the OGC core
plugin serves the landing page, conformance, and collection endpoints, and the
EDR plugin contributes its conformance classes, collection metadata, and data
queries through the OGC hookspecs.

Responses are validated against the official OGC schemas vendored by
xpublish-ogc-core, which is not yet on PyPI, so the suite is skipped when it
isn't importable (in the xpublish-dev workspace both plugins are editable
installs, so it always runs there).
"""

import pytest

pytest.importorskip("xpublish_ogc_core")

import cf_xarray  # noqa: F401
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
