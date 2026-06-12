"""Schemathesis fuzz of the composed xpublish-ogc-core + xpublish-edr app.

Cases are generated from the app's own OpenAPI description and validated
against it, scoped to the OGC API paths.
"""

import pytest

pytest.importorskip("xpublish_ogc_core")
schemathesis = pytest.importorskip("schemathesis")

import cf_xarray  # noqa: F401
import xpublish
from schemathesis.specs.openapi.checks import positive_data_acceptance
from xpublish_ogc_core import testing
from xpublish_ogc_core.plugin import OgcCorePlugin

from xpublish_edr.plugin import CfEdrPlugin


def build_app():
    from cf_xarray.datasets import airds

    rest = xpublish.Rest(
        {"air": airds},
        plugins={
            "ogc": OgcCorePlugin(),
            "edr": CfEdrPlugin(),
        },
    )
    return rest.app


plugin_schema = schemathesis.openapi.from_asgi("/openapi.json", build_app()).include(
    path_regex=r"^/(collections|conformance|$)",
)


ogc_schema = (
    testing.bundled_schema(with_app=build_app())
    .exclude(path_regex=r"^/collections/\{collectionId\}/items")
    .exclude(path_regex=r"^/collections/\{collectionId\}/instances")
    .exclude(path_regex=r"^/collections/\{collectionId\}/locations")
    .exclude(path_regex=r"^/collections/\{collectionId\}/radius")
    .exclude(path_regex=r"^/collections/\{collectionId\}/trajectory")
    .exclude(path_regex=r"^/collections/\{collectionId\}/corridor")
)


@schemathesis.pytest.parametrize(
    plugin=plugin_schema,
    ogc=ogc_schema,
)
def test_schema(case):
    # the EDR query parameters (WKT coords, comma separated bbox) are looser
    # than their OpenAPI parameter schemas can express, so 422 rejections of
    # schema-compliant inputs are expected
    case.call_and_validate(excluded_checks=[positive_data_acceptance])
