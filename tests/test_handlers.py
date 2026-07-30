"""Tests for the shared query-handler error branches.

The three geometry handlers (``position``/``area``/``cube``) wrap
``query.select`` and turn a selection ``ValueError`` into a 404. An invalid
``parameter-name`` is the natural way to trip that branch through the app.
"""

import pytest

PARAMS = {
    "position": "coords=POINT(200 45)",
    "area": "coords=POLYGON((201 41, 201 49, 209 49, 209 41, 201 41))",
    "cube": "bbox=200,40,210,50",
}


@pytest.mark.parametrize("query_type", ["position", "area", "cube"])
def test_invalid_parameter_name_returns_404(cf_client, query_type):
    response = cf_client.get(
        f"/datasets/air/edr/{query_type}?{PARAMS[query_type]}&parameter-name=nonexistent",
    )
    assert response.status_code == 404, response.text
    assert "Invalid variable" in response.text


@pytest.mark.parametrize("query_type", ["position", "area", "cube"])
def test_too_many_datetimes_returns_404(cf_client, query_type):
    response = cf_client.get(
        f"/datasets/air/edr/{query_type}?{PARAMS[query_type]}"
        "&datetime=2013-01-01/2013-01-02/2013-01-03",
    )
    assert response.status_code == 404, response.text
