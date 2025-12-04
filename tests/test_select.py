import cf_xarray  # noqa
import numpy as np
import numpy.testing as npt
import pandas as pd
import pyproj
import pytest
import xarray as xr
import xarray.testing as xrt
from shapely import MultiPoint, Point, from_wkt

from xpublish_edr.geometry.area import select_by_area
from xpublish_edr.geometry.bbox import select_by_bbox
from xpublish_edr.geometry.common import project_dataset
from xpublish_edr.geometry.position import select_by_position
from xpublish_edr.query import EDRAreaQuery, EDRCubeQuery, EDRPositionQuery


@pytest.fixture(scope="function")
def regular_xy_dataset():
    """Loads a sample dataset with regular X and Y coordinates"""
    return xr.tutorial.load_dataset("air_temperature")


@pytest.fixture(scope="function")
def projected_xy_dataset():
    """Loads a sample dataset with projected X and Y coordinates"""
    from cf_xarray.datasets import rotds

    return rotds


@pytest.fixture(scope="function")
def no_grid_mapping_projected_dataset():
    """Loads a sample dataset with projected X and Y coordinates"""
    ds = xr.Dataset(
        # no grid_mapping attribute on data var
        {
            "foo": (("y", "x"), np.arange(6).reshape(3, 2)),
            "spatial_ref": ((), 0, pyproj.CRS.from_epsg(3035).to_cf()),
        },
        coords={
            "x": ("x", [3, 4], {"axis": "X"}),
            "y": ("y", [7, 8, 9], {"axis": "Y"}),
        },
    )
    return ds


@pytest.fixture(scope="function")
def regular_xy_dataset_with_string_dim():
    """Loads a sample dataset with regular X and Y coordinates and a custom string dimension"""
    ds = xr.tutorial.load_dataset("air_temperature")

    # Add a new dimension for statistics
    ds = ds.assign_coords(stat=["none", "random"])

    # Add the stat dimension to the air variable
    air_data = ds["air"].values
    air_data = np.expand_dims(air_data, axis=0)  # Add stat dimension
    air_data = np.repeat(air_data, 2, axis=0)  # Duplicate for second stat value

    # Multiply second stat value by random values between 0.8 and 1.2
    random_factors = np.random.uniform(0.8, 1.2, size=air_data[1].shape)
    air_data[1] = air_data[1] * random_factors

    ds["air"] = xr.DataArray(
        air_data,
        dims=["stat", "time", "lat", "lon"],
        coords={"stat": ds.stat, "time": ds.time, "lat": ds.lat, "lon": ds.lon},
    )

    return ds


def test_no_grid_mapping_projected_dataset(no_grid_mapping_projected_dataset):
    ds = no_grid_mapping_projected_dataset
    transformer = pyproj.Transformer.from_crs(3035, 4326, always_xy=True)
    lon, lat = transformer.transform(3, 8)
    query = EDRPositionQuery(
        coords=f"POINT({lon} {lat})",
        crs="epsg:4326",
        parameters="air",
    )
    geom = query.project_geometry(ds)
    actual = select_by_position(ds, geom)
    expected = ds.isel(x=[0], y=[1])
    xr.testing.assert_identical(actual, expected)

    # same point but in native EPSG:3035
    query = EDRPositionQuery(
        coords=f"POINT(3 8)",
        crs="epsg:3035",
        parameters="air",
    )
    geom = query.project_geometry(ds)
    actual = select_by_position(ds, geom)
    xr.testing.assert_identical(actual, expected)


def test_select_query(regular_xy_dataset):
    query = EDRPositionQuery(
        coords="POINT(200 45)",
        datetime="2013-01-01T06:00:00",
        parameters="air,time",
    )
    query_params = {}

    ds = query.select(regular_xy_dataset, query_params)

    assert ds is not None, "Dataset was not returned"
    assert "air" in ds, "Dataset does not contain the air variable"
    assert "lat" in ds, "Dataset does not contain the lat variable"
    assert "lon" in ds, "Dataset does not contain the lon variable"
    assert "time" in ds, "Dataset does not contain the time variable"

    assert ds["time"] == pd.to_datetime(
        "2013-01-01T06:00:00",
    ), "Dataset shape is incorrect"
    assert ds["air"].shape == (1, 25, 53), "Dataset shape is incorrect"

    query = EDRPositionQuery(
        coords="POINT(200 45)",
        datetime="2013-01-01T06:00:00/2013-01-01T12:00:00",
        parameters="air,time",
        method="nearest",
    )

    ds = query.select(regular_xy_dataset, query_params)
    (
        npt.assert_array_equal(
            ds["time"],
            np.array(
                ["2013-01-01T06:00:00.000000000", "2013-01-01T12:00:00.000000000"],
                dtype="datetime64[ns]",
            ),
        ),
        "Dataset shape is incorrect",
    )
    assert ds["air"].shape == (2, 25, 53), "Dataset shape is incorrect"

    query = EDRPositionQuery(
        coords="POINT(203 46)",
        datetime="2013-01-01T08:00:00",
        parameters="air,time",
        method="linear",
    )

    ds = query.select(regular_xy_dataset, query_params)
    (
        npt.assert_array_equal(
            ds["time"],
            np.array(
                ["2013-01-01T08:00:00"],
                dtype="datetime64[ns]",
            ),
        ),
        "Time is incorrect",
    )

    custom_dim_ds = xr.Dataset(
        coords={
            "lat": np.arange(45, 47),
            "lon": np.arange(200, 202),
            "elevation": np.arange(100, 105),
            "step": pd.timedelta_range("0 days", periods=72, freq="1H"),
        },
        data_vars={
            "air": (("lat", "lon", "elevation", "step"), np.random.rand(2, 2, 5, 72)),
        },
    )

    query = EDRPositionQuery(
        coords="POINT(201 46)",
        parameters="air",
        method="linear",
    )
    ds = query.select(custom_dim_ds, {"step": "0 hours/10 hours", "elevation": "101"})
    assert ds["air"].shape == (2, 2, 1, 11), "Dataset shape is incorrect"
    npt.assert_array_equal(
        ds["step"],
        pd.timedelta_range("0 days", periods=11, freq="1H"),
    )
    npt.assert_equal(ds["elevation"].values, 101)

    ds = query.select(custom_dim_ds, {"step": "1 hours", "elevation": "101/103"})
    assert ds["air"].shape == (2, 2, 3, 1), "Dataset shape is incorrect"
    npt.assert_array_equal(
        ds["step"],
        pd.timedelta_range("1 hours", periods=1, freq="1H"),
    )
    npt.assert_equal(ds["elevation"].values, np.array([101, 102, 103]))


def test_select_query_error(regular_xy_dataset):
    query = EDRPositionQuery(
        coords="POINT(200 45)",
        datetime="2013-01-01T06:00:00",
        parameters="water",
    )
    query_params = {"foo": "bar"}

    with pytest.raises(KeyError):
        query.select(regular_xy_dataset, query_params)

    query = EDRPositionQuery(
        coords="POINT(200 45)",
        datetime="2013-01-0 06:00",
        parameters="air",
    )

    with pytest.raises(TypeError):
        query.select(regular_xy_dataset, {})

    query = EDRPositionQuery(
        coords="POINT(200 45)",
        datetime="2013-01-01T06:00:00",
        parameters="air",
        z="100",
    )

    with pytest.raises(KeyError):
        query.select(regular_xy_dataset, {})

    with pytest.raises(ValueError):
        query = EDRPositionQuery(
            coords="POINT(200 45)",
            datetime="2013-01-01T06:00:00",
            parameters="air",
            z="100",
            method="foo",
        )


def test_select_position_regular_xy(regular_xy_dataset):
    point = Point((204, 44))
    ds = select_by_position(regular_xy_dataset, point)

    assert ds is not None, "Dataset was not returned"
    assert "air" in ds, "Dataset does not contain the air variable"
    assert "lat" in ds, "Dataset does not contain the lat variable"
    assert "lon" in ds, "Dataset does not contain the lon variable"

    assert ds["air"].shape == (2920, 1, 1), "Dataset shape is incorrect"
    npt.assert_array_equal(ds["lat"], 45.0), "Latitude is incorrect"
    npt.assert_array_equal(ds["lon"], 205.0), "Longitude is incorrect"
    npt.assert_approx_equal(ds["air"][0], 280.2), "Temperature is incorrect"
    npt.assert_approx_equal(ds["air"][-1], 279.19), "Temperature is incorrect"


def test_select_position_projected_xy(projected_xy_dataset):
    query = EDRPositionQuery(
        coords="POINT(64.59063409 66.66454929)",
        crs="EPSG:4326",
    )

    projected_point = query.project_geometry(projected_xy_dataset)
    npt.assert_approx_equal(projected_point.x, 18.045), "Longitude is incorrect"
    npt.assert_approx_equal(projected_point.y, 21.725), "Latitude is incorrect"

    ds = select_by_position(projected_xy_dataset, projected_point)
    xrt.assert_identical(
        ds,
        projected_xy_dataset.sel(rlon=[18.045], rlat=[21.725], method="nearest"),
    )

    projected_ds = project_dataset(ds, query.crs)
    (
        npt.assert_approx_equal(projected_ds.cf["X"].values, 64.59063409),
        "Longitude is incorrect",
    )
    (
        npt.assert_approx_equal(projected_ds.cf["Y"].values, 66.66454929),
        "Latitude is incorrect",
    )
    (
        npt.assert_approx_equal(
            projected_ds.temp.values,
            projected_xy_dataset.sel(
                rlon=[18.045],
                rlat=[21.725],
                method="nearest",
            ).temp.values,
        ),
        "Temperature is incorrect",
    )


def test_select_position_regular_xy_interpolate(regular_xy_dataset):
    point = Point((204, 44))
    ds = select_by_position(regular_xy_dataset, point, method="linear")

    assert ds is not None, "Dataset was not returned"
    assert "air" in ds, "Dataset does not contain the air variable"
    assert "lat" in ds, "Dataset does not contain the lat variable"
    assert "lon" in ds, "Dataset does not contain the lon variable"

    assert ds["air"].shape == (2920, 1, 1), "Dataset shape is incorrect"
    npt.assert_array_equal(ds["lat"], 44.0), "Latitude is incorrect"
    npt.assert_array_equal(ds["lon"], 204.0), "Longitude is incorrect"
    npt.assert_approx_equal(ds["air"][0], 281.376), "Temperature is incorrect"
    npt.assert_approx_equal(ds["air"][-1], 279.87), "Temperature is incorrect"


def test_select_position_regular_xy_multi(regular_xy_dataset):
    points = MultiPoint([(202, 45), (205, 48)])
    ds = select_by_position(regular_xy_dataset, points)

    assert ds is not None, "Dataset was not returned"
    assert "air" in ds, "Dataset does not contain the air variable"
    assert "lat" in ds, "Dataset does not contain the lat variable"
    assert "lon" in ds, "Dataset does not contain the lon variable"

    npt.assert_array_equal(ds["lat"], [45.0, 47.5]), "Latitude is incorrect"
    npt.assert_array_equal(ds["lon"], [202.5, 205.0]), "Longitude is incorrect"
    (
        npt.assert_array_equal(
            ds["air"].isel(time=2).values,
            [279.1, 278.6],
        ),
        "Temperature is incorrect",
    )


def test_select_position_projected_xy_multi(projected_xy_dataset):
    query = EDRPositionQuery(
        coords="MULTIPOINT(64.3 66.6, 64.6 66.5)",
        crs="EPSG:4326",
        method="linear",
    )

    projected_points = query.project_geometry(projected_xy_dataset)
    ds = select_by_position(projected_xy_dataset, projected_points, method="linear")
    projected_ds = project_dataset(ds, query.crs)
    assert "temp" in projected_ds, "Dataset does not contain the temp variable"
    assert "rlon" not in projected_ds, "Dataset does not contain the rlon variable"
    assert "rlat" not in projected_ds, "Dataset does not contain the rlat variable"
    (
        npt.assert_array_almost_equal(projected_ds.longitude, [64.3, 64.6]),
        "Longitude is incorrect",
    )
    (
        npt.assert_array_almost_equal(projected_ds.latitude, [66.6, 66.5]),
        "Latitude is incorrect",
    )
    (
        npt.assert_array_almost_equal(
            ds.temp,
            projected_ds.temp,
        ),
        "Temperature is incorrect",
    )


def test_select_position_regular_xy_multi_interpolate(regular_xy_dataset):
    points = MultiPoint([(202, 45), (205, 48)])
    ds = select_by_position(regular_xy_dataset, points, method="linear")

    assert ds is not None, "Dataset was not returned"
    assert "air" in ds, "Dataset does not contain the air variable"
    assert "lat" in ds, "Dataset does not contain the lat variable"
    assert "lon" in ds, "Dataset does not contain the lon variable"

    npt.assert_array_equal(ds["lat"], [45.0, 48.0]), "Latitude is incorrect"
    npt.assert_array_equal(ds["lon"], [202.0, 205.0]), "Longitude is incorrect"
    (
        npt.assert_array_almost_equal(
            ds["air"].isel(time=2).values,
            [279.0, 278.2],
        ),
        "Temperature is incorrect",
    )


def test_select_area_regular_xy(regular_xy_dataset):
    polygon = Point(204, 44).buffer(5)
    ds = select_by_area(regular_xy_dataset, polygon)

    assert ds is not None, "Dataset was not returned"
    assert "air" in ds, "Dataset does not contain the air variable"
    assert "lat" in ds, "Dataset does not contain the lat variable"
    assert "lon" in ds, "Dataset does not contain the lon variable"

    assert ds["air"].shape == (2920, 13), "Dataset shape is incorrect"
    assert ds["lat"].shape == (13,), "Latitude shape is incorrect"
    assert ds["lon"].shape == (13,), "Longitude shape is incorrect"

    (
        npt.assert_array_equal(np.unique(ds["lat"]), [40.0, 42.5, 45.0, 47.5]),
        "Latitude is incorrect",
    )
    (
        npt.assert_array_equal(np.unique(ds["lon"]), [200.0, 202.5, 205.0, 207.5]),
        "Longitude is incorrect",
    )
    (
        npt.assert_array_almost_equal(
            ds["air"].isel(time=0),
            np.array(
                [
                    279.0,
                    279.0,
                    278.9,
                    280.0,
                    280.7,
                    280.2,
                    279.6,
                    282.79,
                    283.2,
                    282.6,
                    281.9,
                    284.9,
                    284.2,
                ],
            ),
        ),
        "Temperature is incorrect",
    )


def test_select_area_projected_xy(projected_xy_dataset):
    query = EDRAreaQuery(
        coords="POLYGON((64.3 66.82, 64.5 66.82, 64.5 66.6, 64.3 66.6, 64.3 66.82))",
        crs="EPSG:4326",
    )

    projected_area = query.project_geometry(projected_xy_dataset)
    ds = select_by_area(projected_xy_dataset, projected_area)
    projected_ds = project_dataset(ds, query.crs)

    assert projected_ds is not None, "Dataset was not returned"
    assert "temp" in projected_ds, "Dataset does not contain the air variable"
    assert "latitude" in projected_ds, "Dataset does not contain the latitude variable"
    assert (
        "longitude" in projected_ds
    ), "Dataset does not contain the longitude variable"

    assert projected_ds.longitude.shape[0] == 1, "Longitude shape is incorrect"
    assert projected_ds.latitude.shape[0] == 1, "Latitude shape is incorrect"
    assert projected_ds.temp.shape[0] == 1, "Temperature shape is incorrect"


def test_select_area_regular_xy_boundary(regular_xy_dataset):
    polygon = from_wkt("POLYGON((200 40, 200 50, 210 50, 210 40, 200 40))").buffer(
        0.0001,
    )
    ds = select_by_area(regular_xy_dataset, polygon)

    assert ds["lat"].min() == 40.0, "Latitude is incorrect"
    assert ds["lat"].max() == 50.0, "Latitude is incorrect"
    assert ds["lon"].min() == 200.0, "Longitude is incorrect"
    assert ds["lon"].max() == 210.0, "Longitude is incorrect"


def test_select_cube_regular_xy(regular_xy_dataset):
    query = EDRCubeQuery(
        bbox="200,40,210,50",
        crs="EPSG:4326",
    )

    bbox = query.project_bbox(regular_xy_dataset)
    ds = select_by_bbox(regular_xy_dataset, bbox)

    assert ds["lat"].min() == 40.0, "Latitude is incorrect"
    assert ds["lat"].max() == 50.0, "Latitude is incorrect"
    assert ds["lon"].min() == 200.0, "Longitude is incorrect"
    assert ds["lon"].max() == 210.0, "Longitude is incorrect"


def test_select_string_dim(regular_xy_dataset_with_string_dim):
    query = EDRPositionQuery(
        coords="POINT(200 45)",
        datetime="2013-01-01T06:00:00",
        parameters="air",
    )

    ds = query.select(
        regular_xy_dataset_with_string_dim,
        {
            "stat": "none",
        },
    )
    assert ds["air"].shape == (1, 25, 53), "Dataset shape is incorrect"
