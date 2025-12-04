from io import BytesIO

import cf_xarray  # noqa: F401
import numpy.testing as npt
import pandas as pd
import pytest
import xpublish
from fastapi.testclient import TestClient

from xpublish_edr import CfEdrPlugin


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


def test_cf_position_formats(cf_client):
    response = cf_client.get("/edr/position/formats")

    assert response.status_code == 200, "Response did not return successfully"

    data = response.json()

    assert "cf_covjson" in data, "cf_covjson is not reported as a valid format"
    assert "nc" in data, "nc is not reported as a valid format"
    assert "csv" in data, "csv is not reported as a valid format"
    assert "parquet" in data, "parquet is not reported as a valid format"
    assert "geojson" in data, "geojson is not reported as a valid format"
    assert (
        "geotiff" not in data
    ), "geotiff is reported as a valid format for position queries, but it is not"


def test_cf_area_formats(cf_client):
    response = cf_client.get("/edr/area/formats")

    assert response.status_code == 200, "Response did not return successfully"

    data = response.json()

    assert "cf_covjson" in data, "cf_covjson is not reported as a valid format"
    assert "nc" in data, "nc is not reported as a valid format"
    assert "csv" in data, "csv is not reported as a valid format"
    assert "geojson" in data, "geojson is not reported as a valid format"
    assert (
        "geotiff" not in data
    ), "geotiff is reported as a valid format for area queries, but it is not"


def test_cf_cube_formats(cf_client):
    response = cf_client.get("/edr/cube/formats")

    assert response.status_code == 200, "Response did not return successfully"

    data = response.json()

    assert "cf_covjson" in data, "cf_covjson is not reported as a valid format"
    assert "nc" in data, "nc is not reported as a valid format"
    assert "csv" in data, "csv is not reported as a valid format"
    assert "geojson" in data, "geojson is not reported as a valid format"
    assert "geotiff" in data, "geotiff is not reported as a valid format"


def test_cf_metadata_query(cf_client):
    response = cf_client.get("/datasets/air/edr/")
    assert response.status_code == 200, "Response did not return successfully"
    data = response.json()

    assert data["id"] == "air", "The id should be air"
    assert data["title"] == "4x daily NMC reanalysis (1948)", "The title is incorrect"
    assert (
        data["description"]
        == "Data is from NMC initialized reanalysis\n(4x/day).  These are the 0.9950 sigma level values."
    ), "The description is incorrect"
    assert data["crs"] == ["EPSG:4326"], "The crs is incorrect"

    # Top-level formats should reflect formats common to all query types
    assert set(data["output_formats"]) == {
        "cf_covjson",
        "nc",
        "netcdf4",
        "nc4",
        "netcdf",
        "csv",
        "geojson",
        "parquet",
    }, "The output formats are incorrect"

    # Per-query advertised formats should match supported sets
    pos_formats = set(
        data["data_queries"]["position"]["link"]["variables"]["output_formats"],
    )
    area_formats = set(
        data["data_queries"]["area"]["link"]["variables"]["output_formats"],
    )
    cube_formats = set(
        data["data_queries"]["cube"]["link"]["variables"]["output_formats"],
    )

    for fmts in (pos_formats, area_formats):
        assert (
            "geotiff" not in fmts
        ), "geotiff should not be advertised for position/area"
        for f in (
            "cf_covjson",
            "nc",
            "netcdf",
            "nc4",
            "netcdf4",
            "csv",
            "geojson",
            "parquet",
        ):
            assert f in fmts, f"{f} should be advertised for position/area"

    assert "geotiff" in cube_formats, "geotiff should be advertised for cube"
    for f in (
        "cf_covjson",
        "nc",
        "netcdf",
        "nc4",
        "netcdf4",
        "csv",
        "geojson",
        "parquet",
    ):
        assert f in cube_formats, f"{f} should be advertised for cube"

    assert (
        "position" in data["data_queries"] and "area" in data["data_queries"]
    ), "The data queries are incorrect"

    assert (
        "temporal" and "spatial" in data["extent"]
    ), "Temporal and spatial extents should be present in extent"
    assert (
        "vertical" not in data["extent"]
    ), "Vertical extent should not be present in extent"

    assert data["extent"]["temporal"]["interval"] == [
        "2013-01-01T00:00:00",
        "2013-01-01T18:00:00",
    ], "Temporal interval is incorrect"
    assert (
        data["extent"]["temporal"]["values"][0]
        == "2013-01-01T00:00:00/2013-01-01T18:00:00"
    ), "Temporal values are incorrect"

    assert data["extent"]["spatial"]["bbox"] == [
        [200.0, 15.0, 322.5, 75.0],
    ], "Spatial bbox is incorrect"
    assert data["extent"]["spatial"]["crs"] == "EPSG:4326", "Spatial CRS is incorrect"

    assert "air" in data["parameter_names"], "Air parameter should be present"
    assert "lat" not in data["parameter_names"], "lat should not be present"
    assert "lon" not in data["parameter_names"], "lon should not be present"


def test_cf_metadata_rotated_lat_lon(cf_client):
    response = cf_client.get("/datasets/temp/edr/")
    assert response.status_code == 200, "Response did not return successfully"
    data = response.json()

    # We want to verify that the extents are in the correct crs and that both the rotated
    # crs and the lat lng crs are present
    assert data["extent"]["spatial"]["crs"] != "EPSG:4326", "Spatial CRS is incorrect"
    npt.assert_allclose(
        data["extent"]["spatial"]["bbox"],
        [[17.935, 21.615, 18.155, 21.835]],
        atol=1e-6,
    )


def test_cf_metadata_query_temp_smoke_test(cf_client):
    response = cf_client.get("/datasets/temp/edr/")
    assert response.status_code == 200, "Response did not return successfully"
    data = response.json()

    assert data["id"] == "temp", "The id should be temp"
    for key in (
        "title",
        "description",
        "crs",
        "extent",
        "output_formats",
        "data_queries",
    ):
        assert key in data, f"Key {key} is not a top level key in the metadata response"


def test_cf_position_query_invalid_coords(cf_client):
    response = cf_client.get("/datasets/air/edr/position?coords=(71, 41)")
    assert response.status_code == 422, "Response should have returned a 422"
    assert "Could not parse coordinates to geometry" in response.json()["detail"]


def test_cf_position_query(cf_client, cf_air_dataset, cf_temp_dataset):
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
        air_param["unit"]["label"]["en"] == cf_air_dataset["air"].attrs["units"]
    ), "DataArray units should be set as parameter units"
    assert (
        air_param["observedProperty"]["id"]
        == cf_air_dataset["air"].attrs["standard_name"]
    ), "DataArray standard_name should be set as the observed property id"
    assert (
        air_param["observedProperty"]["label"]["en"]
        == cf_air_dataset["air"].attrs["long_name"]
    ), "DataArray long_name should be set as parameter observed property"
    assert (
        air_param["description"]["en"] == cf_air_dataset["air"].attrs["long_name"]
    ), "DataArray long_name should be set as parameter description"

    air_range = data["ranges"]["air"]

    assert air_range["type"] == "NdArray", "Response range should be a NdArray"
    assert air_range["dataType"] == "float", "Air dataType should be floats"
    assert air_range["axisNames"] == ["t", "y", "x"], "All dimensions should persist"
    assert air_range["shape"] == [4, 1, 1], "The shape of the array should be 4x1x1"
    assert (
        len(air_range["values"]) == 4
    ), "There should be 4 values, one for each time step"

    # Test with a dataset containing data in a different coordinate system
    x = 64.59063409
    y = 66.66454929
    response = cf_client.get(f"/datasets/temp/edr/position?coords=POINT({x} {y})")
    assert response.status_code == 200, "Response did not return successfully"

    data = response.json()
    for key in ("type", "domain", "parameters", "ranges"):
        assert key in data, f"Key {key} is not a top level key in the CovJSON response"

    axes = data["domain"]["axes"]

    (
        npt.assert_array_almost_equal(
            axes["x"]["values"],
            [[x]],
        ),
        "Did not select nearby x coordinate",
    )
    (
        npt.assert_array_almost_equal(
            axes["y"]["values"],
            [[y]],
        ),
        "Did not select a nearby y coordinate",
    )

    temp_range = data["ranges"]["temp"]
    assert temp_range["type"] == "NdArray", "Response range should be a NdArray"
    assert temp_range["dataType"] == "float", "Air dataType should be floats"
    assert temp_range["axisNames"] == ["rlat", "rlon"], "All dimensions should persist"
    assert temp_range["shape"] == [1, 1], "The shape of the array should be 1x1"
    assert len(temp_range["values"]) == 1, "There should be 1 value selected"


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
        "data.csv" in response.headers["content-disposition"]
    ), "The file name should be data.csv"

    csv_data = [
        line.split(",") for line in response.content.decode("utf-8").splitlines()
    ]

    assert (
        len(csv_data) == 5
    ), "There should be 4 data rows (one for each time step), and one header row"
    for key in ("time", "lat", "lon", "air", "cell_area"):
        assert key in csv_data[0], f"column {key} should be in the header"

    # single time step test
    response = cf_client.get(
        f"/datasets/air/edr/position?coords=POINT({x} {y})&f=csv&parameter-name=air&datetime=2013-01-01T00:00:00",  # noqa
    )

    assert response.status_code == 200, "Response did not return successfully"
    assert (
        "text/csv" in response.headers["content-type"]
    ), "The content type should be set as a CSV"
    assert (
        "attachment" in response.headers["content-disposition"]
    ), "The response should be set as an attachment to trigger download"
    assert (
        "data.csv" in response.headers["content-disposition"]
    ), "The file name should be data.csv"

    csv_data = [
        line.split(",") for line in response.content.decode("utf-8").splitlines()
    ]

    assert (
        len(csv_data) == 2
    ), "There should be 2 data rows, one data and one header row"
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
        "data.csv" in response.headers["content-disposition"]
    ), "The file name should be data.csv"

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


def test_cf_position_parquet(cf_client) -> None:
    x = 204
    y = 44
    response = cf_client.get(
        f"/datasets/air/edr/position?coords=POINT({x} {y})&f=parquet&parameter-name=air",
    )

    assert response.status_code == 200, "Response did not return successfully"
    assert (
        "application/parquet" in response.headers["content-type"]
    ), "The content type should be set as a Parquet"
    assert (
        "attachment" in response.headers["content-disposition"]
    ), "The response should be set as an attachment to trigger download"
    assert (
        "data.parquet" in response.headers["content-disposition"]
    ), "The file name should be data.parquet"

    df = pd.read_parquet(BytesIO(response.content))

    assert (
        len(df) == 4
    ), "There should be 4 data rows (one for each time step), and one header row"
    assert set(df.reset_index().columns) == {
        "time",
        "lat",
        "lon",
        "air",
        "cell_area",
        "spatial_ref",
    }

    # single time step test
    response = cf_client.get(
        f"/datasets/air/edr/position?coords=POINT({x} {y})&f=parquet&parameter-name=air&datetime=2013-01-01T00:00:00",  # noqa
    )

    assert response.status_code == 200, "Response did not return successfully"
    assert (
        "application/parquet" in response.headers["content-type"]
    ), "The content type should be set as a Parquet"
    assert (
        "attachment" in response.headers["content-disposition"]
    ), "The response should be set as an attachment to trigger download"
    assert (
        "data.parquet" in response.headers["content-disposition"]
    ), "The file name should be data.parquet"

    df = pd.read_parquet(BytesIO(response.content))

    assert len(df) == 1, "There should be 2 data rows, one data and one header row"
    assert set(df.reset_index().columns) == {
        "time",
        "lat",
        "lon",
        "air",
        "cell_area",
        "spatial_ref",
    }


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
        "data.nc" in response.headers["content-disposition"]
    ), "The file name should be data.nc"


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
        "data.nc" in response.headers["content-disposition"]
    ), "The file name should be data.nc"


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
        "data.csv" in response.headers["content-disposition"]
    ), "The file name should be data.csv"

    csv_data = [
        line.split(",") for line in response.content.decode("utf-8").splitlines()
    ]

    assert (
        len(csv_data) == 9
    ), "There should be 4 data rows (one for each time step), and one header row"
    for key in ("time", "lat", "lon", "air", "cell_area"):
        assert key in csv_data[0], f"column {key} should be in the header"


def test_cf_area_query(cf_client, cf_air_dataset):
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
        air_param["unit"]["label"]["en"] == cf_air_dataset["air"].attrs["units"]
    ), "DataArray units should be set as parameter units"
    assert (
        air_param["observedProperty"]["id"]
        == cf_air_dataset["air"].attrs["standard_name"]
    ), "DataArray standard_name should be set as the observed property id"
    assert (
        air_param["observedProperty"]["label"]["en"]
        == cf_air_dataset["air"].attrs["long_name"]
    ), "DataArray long_name should be set as parameter observed property"
    assert (
        air_param["description"]["en"] == cf_air_dataset["air"].attrs["long_name"]
    ), "DataArray long_name should be set as parameter description"

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
        len(air_range["values"]) == 36
    ), "There should be 26 values, 9 for each time step"


def test_cf_area_csv_query(cf_client, cf_air_dataset):
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
        "data.csv" in response.headers["content-disposition"]
    ), "The file name should be data.csv"

    csv_data = [
        line.split(",") for line in response.content.decode("utf-8").splitlines()
    ]

    assert len(csv_data) == 37, "There should be 37 data rows, and one header row"
    for key in ("time", "lat", "lon", "air", "cell_area"):
        assert key in csv_data[0], f"column {key} should be in the header"


def test_cf_area_geojson_query(cf_client, cf_air_dataset):
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


def test_cf_area_nc_query(cf_client, cf_air_dataset):
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
        "data.nc" in response.headers["content-disposition"]
    ), "The file name should be data.nc"


def test_cf_cube_query_covjson(cf_client, cf_air_dataset):
    bbox = "200,40,210,50"
    response = cf_client.get(f"/datasets/air/edr/cube?bbox={bbox}")

    assert response.status_code == 200, "Response did not return successfully"
    assert (
        "application/json" in response.headers["content-type"]
    ), "The content type should be set as a JSON"

    data = response.json()

    for key in ("type", "domain", "parameters", "ranges"):
        assert key in data, f"Key {key} is not a top level key in the CovJSON response"

    axes = data["domain"]["axes"]

    assert axes["x"]["values"] == [
        200,
        202.5,
        205,
        207.5,
        210,
    ], "X coordinates are incorrect"
    assert axes["y"]["values"] == [
        50,
        47.5,
        45,
        42.5,
        40,
    ], "Y coordinates are incorrect"
    assert (
        len(axes["t"]["values"]) == 4
    ), "There should be a time value for each time step"

    air_param = data["parameters"]["air"]

    assert (
        air_param["unit"]["label"]["en"] == cf_air_dataset["air"].attrs["units"]
    ), "DataArray units should be set as parameter units"
    assert (
        air_param["observedProperty"]["id"]
        == cf_air_dataset["air"].attrs["standard_name"]
    ), "DataArray standard_name should be set as the observed property id"
    assert (
        air_param["observedProperty"]["label"]["en"]
        == cf_air_dataset["air"].attrs["long_name"]
    ), "DataArray long_name should be set as parameter observed property"
    assert (
        air_param["description"]["en"] == cf_air_dataset["air"].attrs["long_name"]
    ), "DataArray long_name should be set as parameter description"

    air_range = data["ranges"]["air"]

    assert air_range["type"] == "NdArray", "Response range should be a NdArray"
    assert air_range["dataType"] == "float", "Air dataType should be floats"
    assert air_range["axisNames"] == ["t", "y", "x"], "All dimensions should persist"
    assert air_range["shape"] == [4, 5, 5], "The shape of the array should be 4x5x5"
    assert (
        len(air_range["values"]) == 100
    ), "There should be 100 values (4 time steps * 5 lat points * 5 lon points)"


def test_cf_cube_query_geojson(cf_client, cf_air_dataset):
    bbox = "200,40,210,50"
    response = cf_client.get(f"/datasets/air/edr/cube?bbox={bbox}&f=geojson")

    assert response.status_code == 200, "Response did not return successfully"
    assert (
        "application/json" in response.headers["content-type"]
    ), "The content type should be set as a JSON"

    data = response.json()

    assert "type" in data, "GeoJSON response should have a type key"
    assert "features" in data, "GeoJSON response should have a features key"

    features = data["features"]

    assert (
        len(features) == 100
    ), "There should be 100 data points (4 time steps * 5 lat points * 5 lon points)"


def test_cf_cube_query_nc(cf_client, cf_air_dataset):
    bbox = "200,40,210,50"
    response = cf_client.get(f"/datasets/air/edr/cube?bbox={bbox}&f=nc")

    assert response.status_code == 200, "Response did not return successfully"
    assert (
        "application/netcdf" in response.headers["content-type"]
    ), "The content type should be set as a NetCDF"
    assert (
        "attachment" in response.headers["content-disposition"]
    ), "The response should be set as an attachment to trigger download"
    assert (
        "data.nc" in response.headers["content-disposition"]
    ), "The file name should be data.nc"


def test_cf_cube_query_csv(cf_client, cf_air_dataset):
    bbox = "200,40,210,50"
    response = cf_client.get(f"/datasets/air/edr/cube?bbox={bbox}&f=csv")

    assert response.status_code == 200, "Response did not return successfully"
    assert (
        "text/csv" in response.headers["content-type"]
    ), "The content type should be set as a CSV"
    assert (
        "attachment" in response.headers["content-disposition"]
    ), "The response should be set as an attachment to trigger download"
    assert (
        "data.csv" in response.headers["content-disposition"]
    ), "The file name should be data.csv"

    csv_data = [
        line.split(",") for line in response.content.decode("utf-8").splitlines()
    ]

    assert len(csv_data) == 101, "There should be 100 data rows and one header row"
    for key in ("time", "lat", "lon", "air"):
        assert key in csv_data[0], f"column {key} should be in the header"


@pytest.mark.parametrize("parameter", ["air", "air_float16"])
def test_cf_cube_query_geotiff_latlng_grid(cf_client, cf_air_dataset, parameter):
    import io

    import rioxarray

    bbox = "200,40,210,50"

    # Test with multiple time steps
    response = cf_client.get(
        f"/datasets/air/edr/cube?bbox={bbox}&parameter-name={parameter}&f=geotiff",
    )
    assert response.status_code == 200, "Response should have returned a 200"
    assert (
        "image/tiff" in response.headers["content-type"]
    ), "The content type should be set as a TIFF"
    assert (
        "attachment" in response.headers["content-disposition"]
    ), "The response should be set as an attachment to trigger download"
    assert (
        "data.tiff" in response.headers["content-disposition"]
    ), "The file name should be data.tiff"

    # Read the GeoTIFF back in from the response content
    da = rioxarray.open_rasterio(io.BytesIO(response.content))
    assert da.band.shape == (
        4,
    ), "GeoTIFF should have 4 time steps represented as bands"
    assert da.x.shape == (5,), "GeoTIFF should have 5 x coordinates"
    assert da.y.shape == (5,), "GeoTIFF should have 5 y coordinates"
    assert da.shape == (
        4,
        5,
        5,
    ), "GeoTIFF should have 4 time steps, 5 x coordinates, and 5 y coordinates"

    with open("test_cf_cube_query_geotiff_latlng_grid.tiff", "wb") as f:
        f.write(response.content)

    # Test with a single time step
    response = cf_client.get(
        f"/datasets/air/edr/cube?bbox={bbox}&parameter-name=air&f=geotiff&time=2013-01-01T00:00:00",
    )
    assert response.status_code == 200, "Response did not return successfully"
    assert (
        "image/tiff" in response.headers["content-type"]
    ), "The content type should be set as a TIFF"
    assert (
        "attachment" in response.headers["content-disposition"]
    ), "The response should be set as an attachment to trigger download"
    assert (
        "data.tiff" in response.headers["content-disposition"]
    ), "The file name should be data.tiff"

    # Read the GeoTIFF back in from the response content
    da = rioxarray.open_rasterio(io.BytesIO(response.content))
    assert da.band.shape == (1,), "GeoTIFF should have 1 time step represented as bands"
    assert da.x.shape == (5,), "GeoTIFF should have 5 x coordinates"
    assert da.y.shape == (5,), "GeoTIFF should have 5 y coordinates"
    assert da.shape == (
        1,
        5,
        5,
    ), "GeoTIFF should have 1 time step, 5 x coordinates, and 5 y coordinates"


def test_cf_generic_extents_band_and_step():
    import numpy as np
    import xarray as xr

    # Build coords: lat/lon for CF spatial axes, plus band (int) and step (timedelta)
    lat = xr.DataArray(
        [10.0, 11.0],
        dims=("lat",),
        attrs={"axis": "Y", "standard_name": "latitude", "units": "degrees_north"},
    )
    lon = xr.DataArray(
        [20.0, 21.5, 23.0],
        dims=("lon",),
        attrs={"axis": "X", "standard_name": "longitude", "units": "degrees_east"},
    )

    band = xr.DataArray([1, 2], dims=("band",))
    step = xr.DataArray(pd.to_timedelta([0, "6h", "12h"]), dims=("step",))

    data = xr.DataArray(
        np.arange(3 * 2 * 2 * 3).reshape(3, 2, 2, 3).astype(float),
        dims=("step", "band", "lat", "lon"),
        coords={"step": step, "band": band, "lat": lat, "lon": lon},
        name="var",
        attrs={"standard_name": "test_var", "long_name": "Test variable", "units": "1"},
    )

    # Add a large generic dimension (>100) to test compression
    member = xr.DataArray(np.arange(150), dims=("member",))
    big = xr.DataArray(
        np.zeros((150, 2, 3), dtype=float),
        dims=("member", "lat", "lon"),
        coords={"member": member, "lat": lat, "lon": lon},
        name="big",
        attrs={"standard_name": "big_var", "long_name": "Big variable", "units": "1"},
    )

    # Add a problematic dimension whose string conversion fails to ensure it is skipped
    class _BadRepr:
        def __str__(self):  # type: ignore[no-redef]
            raise ValueError("boom")

        def __repr__(self):  # type: ignore[no-redef]
            raise ValueError("boom")

    bad = xr.DataArray(np.array([_BadRepr(), _BadRepr()], dtype=object), dims=("bad",))

    ds = xr.Dataset({"var": data, "big": big})
    ds = ds.assign_coords(bad=bad)
    ds.attrs["_xpublish_id"] = "custom"

    # Stand up app with plugin
    rest = xpublish.Rest({"custom": ds}, plugins={"edr": CfEdrPlugin()})
    client = TestClient(rest.app)

    # Request collection metadata
    r = client.get("/datasets/custom/edr/")
    assert r.status_code == 200
    meta = r.json()

    assert "extent" in meta
    assert "spatial" in meta["extent"], "spatial extent should be present"
    ext = meta["extent"]

    # failing dimension should be skipped and not crash
    assert "bad" not in ext

    # band: integer values and interval
    assert "band" in ext
    assert ext["band"]["values"] == [1, 2]
    assert ext["band"]["interval"] == [1, 2]

    # step: timedelta values as ISO 8601 durations and proper interval
    expected_steps = [pd.Timedelta(x).isoformat() for x in [0, "6h", "12h"]]
    assert ext["step"]["values"] == expected_steps
    assert ext["step"]["interval"] == [expected_steps[0], expected_steps[-1]]

    # Now run a position query selecting a single step and band
    lon_pt = 21.5
    lat_pt = 11.0
    response = client.get(
        f"/datasets/custom/edr/position?coords=POINT({lon_pt} {lat_pt})&parameter-name=var&step=6h&band=2&f=csv",  # noqa
    )
    assert response.status_code == 200, "Position query should return successfully"
    assert "text/csv" in response.headers["content-type"], "Should return CSV"

    csv_data = [
        line.split(",") for line in response.content.decode("utf-8").splitlines()
    ]
    assert len(csv_data) == 2, "Single row expected with one step and one band"

    header = csv_data[0]
    row = csv_data[1]

    # Locate indices for assertions
    step_idx = header.index("step") if "step" in header else None
    band_idx = header.index("band") if "band" in header else None
    lon_idx = header.index("lon") if "lon" in header else None
    lat_idx = header.index("lat") if "lat" in header else None
    var_idx = header.index("var")

    # Check coordinates and selections
    if step_idx is not None:
        assert row[step_idx] == str(pd.Timedelta("6h"))
    if band_idx is not None:
        assert int(float(row[band_idx])) == 2
    if lon_idx is not None:
        assert float(row[lon_idx]) == lon_pt
    if lat_idx is not None:
        assert float(row[lat_idx]) == lat_pt

    # Data value should match the expected location
    assert float(row[var_idx]) == 22.0

    # parameter_names includes extents
    assert set(meta["parameter_names"]["var"]["extent"]) == {"spatial", "band", "step"}
    assert set(meta["parameter_names"]["big"]["extent"].keys()) == {"spatial", "member"}
