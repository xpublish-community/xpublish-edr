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


@pytest.fixture(scope="session")
def cf_air_dataset():
    from cf_xarray.datasets import airds

    # Create a float16 version of the air variable
    airds["air_float16"] = airds["air"].astype("float16")

    return airds


@pytest.fixture(scope="session")
def cf_temp_dataset():
    from cf_xarray.datasets import rotds

    return rotds


@pytest.fixture(scope="session")
def cf_xpublish(cf_air_dataset, cf_temp_dataset):
    rest = xpublish.Rest(
        {"air": cf_air_dataset, "temp": cf_temp_dataset},
        plugins={"edr": CfEdrPlugin()},
    )

    return rest


@pytest.fixture(scope="session")
def cf_client(cf_xpublish):
    app = cf_xpublish.app
    client = TestClient(app)

    return client
