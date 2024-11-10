import numpy.testing as npt
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


def test_cf_area_formats(cf_client):
    response = cf_client.get("/edr/area/formats")

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
    response = cf_client.get(
        f"/datasets/air/edr/position?coords=POINT({x} {y})&f=csv&parameter-name=air",
    )

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


def test_cf_position_csv_interpolate(cf_client):
    x = 204
    y = 44
    response = cf_client.get(
        f"/datasets/air/edr/position?coords=POINT({x} {y})&f=csv&method=linear",
    )

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

    lon_index = csv_data[0].index("lon")
    lons = [float(row[lon_index]) for row in csv_data[1:]]
    (
        npt.assert_array_equal(
            lons,
            [204.0, 204.0, 204.0, 204.0],
        ),
        "Longitude should be interpolated as 204.0",
    )

    lat_index = csv_data[0].index("lat")
    lats = [float(row[lat_index]) for row in csv_data[1:]]
    (
        npt.assert_array_almost_equal(
            lats,
            [44.0, 44.0, 44.0, 44.0],
        ),
        "Latitude should be interpolated as 44.0",
    )


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


def test_cf_position_geojson(cf_client):
    x = 204
    y = 44
    response = cf_client.get(
        f"/datasets/air/edr/position?coords=POINT({x} {y})&f=geojson&parameter-name=air",
    )

    assert response.status_code == 200, "Response did not return successfully"
    assert (
        "application/json" in response.headers["content-type"]
    ), "The content type should be set as a JSON"

    data = response.json()

    assert "type" in data, "GeoJSON response should have a type key"

    features = data["features"]
    assert len(features) == 4, "There should be 4 features, one for each time step"

    first_feature = features[0]
    assert "geometry" in first_feature, "Each feature should have a geometry key"
    assert "properties" in first_feature, "Each feature should have a properties key"
    assert (
        "time" in first_feature["properties"]
    ), "Each feature should have a time property"
    assert (
        "air" in first_feature["properties"]
    ), "Each feature should have an air property"

    assert first_feature["geometry"]["type"] == "Point", "Geometry should be a Point"
    assert first_feature["geometry"]["coordinates"] == [
        205.0,
        45.0,
    ], "Geometry should be at the requested point"

    assert (
        first_feature["properties"]["time"] == "2013-01-01T00:00:00"
    ), "Time should be set in isoformat"
    assert first_feature["properties"]["air"] == 280.2, "Air should be set"


def test_cf_multiple_position(cf_client):
    points = "MULTIPOINT((202 43),(205 45))"
    response = cf_client.get(f"/datasets/air/edr/position?coords={points}")

    assert response.status_code == 200, "Response did not return successfully"
    data = response.json()

    for key in ("type", "domain", "parameters", "ranges"):
        assert key in data, f"Key {key} is not a top level key in the CovJSON response"

    axes = data["domain"]["axes"]

    assert axes["x"] == {
        "values": [202.5, 205.0],
    }, "Did not select nearby x coordinates within the polygon"
    assert axes["y"] == {
        "values": [42.5, 45.0],
    }, "Did not select a nearby y coordinates within the polygon"

    assert (
        len(axes["t"]["values"]) == 4
    ), "There should be a time value for each time step"

    air_range = data["ranges"]["air"]

    assert air_range["type"] == "NdArray", "Response range should be a NdArray"
    assert air_range["dataType"] == "float", "Air dataType should be floats"
    assert air_range["axisNames"] == [
        "t",
        "pts",
    ], "Time should be the only remaining axes"
    assert len(air_range["shape"]) == 2, "There should be 2 axes"
    assert air_range["shape"][0] == len(axes["t"]["values"]), "The shape of the "
    assert air_range["shape"][1] == len(
        axes["x"]["values"],
    ), "The shape of the pts axis"
    assert (
        len(air_range["values"]) == 8
    ), "There should be 8 values, 2 for each time step"


def test_cf_multiple_position_csv(cf_client):
    points = "MULTIPOINT((202 43),(205 45))"
    response = cf_client.get(f"/datasets/air/edr/position?coords={points}&f=csv")

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
        len(csv_data) == 9
    ), "There should be 4 data rows (one for each time step), and one header row"
    for key in ("time", "lat", "lon", "air", "cell_area"):
        assert key in csv_data[0], f"column {key} should be in the header"


def test_cf_area_query(cf_client, cf_dataset):
    coords = "POLYGON((201 41, 201 49, 209 49, 209 41, 201 41))"
    response = cf_client.get(f"/datasets/air/edr/area?coords={coords}&f=cf_covjson")

    assert response.status_code == 200, "Response did not return successfully"
    data = response.json()

    for key in ("type", "domain", "parameters", "ranges"):
        assert key in data, f"Key {key} is not a top level key in the CovJSON response"

    axes = data["domain"]["axes"]

    assert axes["x"] == {
        "values": [202.5, 205.0, 207.5, 202.5, 205.0, 207.5, 202.5, 205.0, 207.5],
    }, "Did not select nearby x coordinates within the polygon"
    assert axes["y"] == {
        "values": [47.5, 47.5, 47.5, 45.0, 45.0, 45.0, 42.5, 42.5, 42.5],
    }, "Did not select a nearby y coordinates within the polygon"

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
    print(air_range)

    assert air_range["type"] == "NdArray", "Response range should be a NdArray"
    assert air_range["dataType"] == "float", "Air dataType should be floats"
    assert air_range["axisNames"] == [
        "t",
        "pts",
    ], "Time should be the only remaining axes"
    assert len(air_range["shape"]) == 2, "There should be 2 axes"
    assert air_range["shape"][0] == len(axes["t"]["values"]), "The shape of the "
    assert air_range["shape"][1] == len(
        axes["x"]["values"],
    ), "The shape of the pts axis"
    assert (
        len(air_range["values"]) == 36
    ), "There should be 26 values, 9 for each time step"


def test_cf_area_csv_query(cf_client, cf_dataset):
    coords = "POLYGON((201 41, 201 49, 209 49, 209 41, 201 41))"
    response = cf_client.get(f"/datasets/air/edr/area?coords={coords}&f=csv")

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

    assert len(csv_data) == 37, "There should be 37 data rows, and one header row"
    for key in ("time", "lat", "lon", "air", "cell_area"):
        assert key in csv_data[0], f"column {key} should be in the header"


def test_cf_area_geojson_query(cf_client, cf_dataset):
    coords = "POLYGON((201 41, 201 49, 209 49, 209 41, 201 41))"
    response = cf_client.get(f"/datasets/air/edr/area?coords={coords}&f=geojson")

    assert response.status_code == 200, "Response did not return successfully"
    assert (
        "application/json" in response.headers["content-type"]
    ), "The content type should be set as a JSON"

    data = response.json()

    assert "type" in data, "GeoJSON response should have a type key"
    assert "features" in data, "GeoJSON response should have a features key"

    features = data["features"]

    assert len(features) == 36, "There should be 36 data points"


def test_cf_area_nc_query(cf_client, cf_dataset):
    coords = "POLYGON((201 41, 201 49, 209 49, 209 41, 201 41))"
    response = cf_client.get(f"/datasets/air/edr/area?coords={coords}&f=nc")

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
