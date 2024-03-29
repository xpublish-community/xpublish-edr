import pytest
import xpublish
from fastapi.testclient import TestClient

from xpublish_edr import CfEdrPlugin


@pytest.fixture(scope="session")
def cf_dataset():
    from cf_xarray.datasets import airds

    return airds


@pytest.fixture(scope="session")
def cf_xpublish(cf_dataset):
    rest = xpublish.Rest({"air": cf_dataset}, plugins={"edr": CfEdrPlugin()})

    return rest


@pytest.fixture(scope="session")
def cf_client(cf_xpublish):
    app = cf_xpublish.app
    client = TestClient(app)

    return client


def test_cf_position_formats(cf_client):
    response = cf_client.get("/edr/position/formats")

    assert response.status_code == 200, "Response did not return successfully"

    data = response.json()

    assert "cf_covjson" in data, "cf_covjson is not a valid format"
    assert "nc" in data, "nc is not a valid format"
    assert "csv" in data, "csv is not a valid format"


def test_cf_position_query(cf_client, cf_dataset):
    x = 204
    y = 44
    response = cf_client.get(f"/datasets/air/edr/position?coords=POINT({x} {y})")

    assert response.status_code == 200, "Response did not return successfully"

    data = response.json()

    for key in ("type", "domain", "parameters", "ranges"):
        assert key in data, f"Key {key} is not a top level key in the CovJSON response"

    axes = data["domain"]["axes"]

    assert axes["x"] == {"values": [205.0]}, "Did not select nearby x coordinate"
    assert axes["y"] == {"values": [45.0]}, "Did not select a nearby y coordinate"

    assert (
        len(axes["t"]["values"]) == 4
    ), "There should be a time value for each time step"

    air_param = data["parameters"]["air"]

    assert (
        air_param["unit"]["label"]["en"] == cf_dataset["air"].attrs["units"]
    ), "DataArray units should be set as parameter units"
    assert (
        air_param["observedProperty"]["id"] == cf_dataset["air"].attrs["standard_name"]
    ), "DataArray standard_name should be set as the observed property id"
    assert (
        air_param["observedProperty"]["label"]["en"]
        == cf_dataset["air"].attrs["long_name"]
    ), "DataArray long_name should be set as parameter observed property"
    assert (
        air_param["description"]["en"] == cf_dataset["air"].attrs["long_name"]
    ), "DataArray long_name should be set as parameter description"

    air_range = data["ranges"]["air"]

    assert air_range["type"] == "NdArray", "Response range should be a NdArray"
    assert air_range["dataType"] == "float", "Air dataType should be floats"
    assert air_range["axisNames"] == ["t"], "Time should be the only remaining axes"
    assert len(air_range["shape"]) == 1, "There should only one axes"
    assert air_range["shape"][0] == len(axes["t"]["values"]), "The shape of the "
    assert (
        len(air_range["values"]) == 4
    ), "There should be 4 values, one for each time step"


def test_cf_position_csv(cf_client):
    x = 204
    y = 44
    response = cf_client.get(f"/datasets/air/edr/position?coords=POINT({x} {y})&f=csv")

    assert response.status_code == 200, "Response did not return successfully"
    assert (
        "text/csv" in response.headers["content-type"]
    ), "The content type should be set as a CSV"
    assert (
        "attachment" in response.headers["content-disposition"]
    ), "The response should be set as an attachment to trigger download"
    assert (
        "position.csv" in response.headers["content-disposition"]
    ), "The file name should be position.csv"

    csv_data = [
        line.split(",") for line in response.content.decode("utf-8").splitlines()
    ]

    assert (
        len(csv_data) == 5
    ), "There should be 4 data rows (one for each time step), and one header row"
    for key in ("time", "lat", "lon", "air", "cell_area"):
        assert key in csv_data[0], f"column {key} should be in the header"


def test_cf_position_nc(cf_client):
    x = 204
    y = 44
    response = cf_client.get(f"/datasets/air/edr/position?coords=POINT({x} {y})&f=nc")

    assert response.status_code == 200, "Response did not return successfully"
    assert (
        "application/netcdf" in response.headers["content-type"]
    ), "The content type should be set as a NetCDF"
    assert (
        "attachment" in response.headers["content-disposition"]
    ), "The response should be set as an attachment to trigger download"
    assert (
        "position.nc" in response.headers["content-disposition"]
    ), "The file name should be position.nc"


def test_percent_encoded_cf_position_nc(cf_client):
    x = 204
    y = 44
    response = cf_client.get(f"/datasets/air/edr/position?coords=POINT({x}%20{y})&f=nc")

    assert response.status_code == 200, "Response did not return successfully"
    assert (
        "application/netcdf" in response.headers["content-type"]
    ), "The content type should be set as a NetCDF"
    assert (
        "attachment" in response.headers["content-disposition"]
    ), "The response should be set as an attachment to trigger download"
    assert (
        "position.nc" in response.headers["content-disposition"]
    ), "The file name should be position.nc"
