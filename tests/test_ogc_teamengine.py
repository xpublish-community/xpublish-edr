"""Official OGC CITE compliance suite, run with TeamEngine in Docker.

The composed xpublish-ogc-core + xpublish-edr app is served over HTTP and
tested by the ets-ogcapi-edr10 executable test suite running in the official
OGC Docker image. Requires Docker (the image is pulled on first use) and
to not be running on Windows; skipped otherwise. Deselect with `-m "not cite"`.
"""

import platform

import pytest
from xpublish_ogc_core import teamengine

pytestmark = [
    pytest.mark.cite,
    pytest.mark.skipif(
        platform.system() == "Windows",
        reason="TeamEngine does not run on Windows Containers",
    ),
    pytest.mark.skipif(
        not teamengine.docker_available(),
        reason="requires the docker CLI and a running daemon",
    ),
]

ETS_IMAGE = "ogccite/ets-ogcapi-edr10:1.3-teamengine-6.0.0-RC2"
SUITE = "ogcapi-edr10"

KNOWN_FAILURES = {
    # FastAPI generates an OpenAPI 3.1 document, while the suite validates it
    # against OpenAPI 3.0 and rejects the `"type": "null"` members that
    # pydantic emits for optional fields
    "ApiDefinition.apiDefinitionValidation",
    # FastAPI does not emit the `style: form` member on query parameter
    # definitions (it is the OpenAPI default for query parameters, but the
    # suite requires it to be explicit)
    "AreaCollections.areaDateTimeParameterDefinition",
    "PositionCollections.positionDateTimeParameterDefinition",
    "CubeCollections.cubeDateTimeParameterDefinition",
}


def test_edr_cite_suite(subtests, ogc_app):
    with (
        teamengine.serve_app(ogc_app) as app_url,
        teamengine.teamengine_container(ETS_IMAGE) as engine_url,
    ):
        result = teamengine.run_suite(
            engine_url,
            SUITE,
            {
                "iut": app_url,
                "apiDefinition": f"{app_url}/openapi.json",
            },
        )

    teamengine.report_subtests(
        result,
        subtests,
        known_failures=KNOWN_FAILURES,
        expected_passed=28,
    )
