import pytest
import xpublish
from fastapi.testclient import TestClient
from xpublish_ogc_core.plugin import OgcCorePlugin

from xpublish_edr.plugin import CfEdrPlugin


def build_ogc_app():
    """Compose the xpublish-ogc-core + xpublish-edr app with the CF air dataset.

    Shared by the ``ogc_app`` fixture and the schemathesis tests, which build
    their schemas at module-collection time and so cannot use the fixture.
    """
    from cf_xarray.datasets import airds

    rest = xpublish.Rest(
        {"air": airds},
        plugins={
            "ogc": OgcCorePlugin(),
            "edr": CfEdrPlugin(),
        },
    )

    return rest.app


@pytest.fixture(scope="module")
def ogc_app():
    return build_ogc_app()


@pytest.fixture(scope="module")
def client(ogc_app):
    return TestClient(ogc_app)
