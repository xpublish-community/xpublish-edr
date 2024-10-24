import cf_xarray  # noqa
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
import xarray as xr
from shapely import Point

from xpublish_edr.query import EDRQuery
from xpublish_edr.select import select_area, select_postition, select_query


@pytest.fixture(scope="function")
def regular_xy_dataset():
    """Loads a sample dataset with regular X and Y coordinates"""
    return xr.tutorial.load_dataset("air_temperature")


def test_select_query(regular_xy_dataset):
    query = EDRQuery(
        coords="POINT(200 45)",
        datetime="2013-01-01T06:00:00",
        parameters="air,time",
    )
    query_params = {}

    ds = select_query(regular_xy_dataset, query, query_params)

    assert ds is not None, "Dataset was not returned"
    assert "air" in ds, "Dataset does not contain the air variable"
    assert "lat" in ds, "Dataset does not contain the lat variable"
    assert "lon" in ds, "Dataset does not contain the lon variable"
    assert "time" in ds, "Dataset does not contain the time variable"

    assert ds["time"] == pd.to_datetime(
        "2013-01-01T06:00:00",
    ), "Dataset shape is incorrect"
    assert ds["air"].shape == (25, 53), "Dataset shape is incorrect"


def test_select_position_regular_xy(regular_xy_dataset):
    point = Point((204, 44))
    ds = select_postition(regular_xy_dataset, point)

    assert ds is not None, "Dataset was not returned"
    assert "air" in ds, "Dataset does not contain the air variable"
    assert "lat" in ds, "Dataset does not contain the lat variable"
    assert "lon" in ds, "Dataset does not contain the lon variable"

    assert ds["air"].shape == ds["time"].shape, "Dataset shape is incorrect"
    npt.assert_array_equal(ds["lat"], 45.0), "Latitude is incorrect"
    npt.assert_array_equal(ds["lon"], 205.0), "Longitude is incorrect"
    npt.assert_approx_equal(ds["air"][0], 280.2), "Temperature is incorrect"
    npt.assert_approx_equal(ds["air"][-1], 279.19), "Temperature is incorrect"


def test_select_area_regular_xy(regular_xy_dataset):
    polygon = Point(204, 44).buffer(5)
    ds = select_area(regular_xy_dataset, polygon)

    assert ds is not None, "Dataset was not returned"
    assert "air" in ds, "Dataset does not contain the air variable"
    assert "lat" in ds, "Dataset does not contain the lat variable"
    assert "lon" in ds, "Dataset does not contain the lon variable"

    assert ds["air"].shape == (2920, 4, 4), "Dataset shape is incorrect"
    assert ds["lat"].shape == (4,), "Latitude shape is incorrect"
    assert ds["lon"].shape == (4,), "Longitude shape is incorrect"

    npt.assert_array_equal(ds["lat"], [47.5, 45.0, 42.5, 40.0]), "Latitude is incorrect"
    (
        npt.assert_array_equal(ds["lon"], [200.0, 202.5, 205.0, 207.5]),
        "Longitude is incorrect",
    )
    (
        npt.assert_array_almost_equal(
            ds["air"][0],
            np.array(
                [
                    [np.nan, 279.0, 279.0, 278.9],
                    [280.0, 280.7, 280.2, 279.6],
                    [282.79, 283.2, 282.6, 281.9],
                    [np.nan, 284.9, 284.2, np.nan],
                ],
            ),
        ),
        "Temperature is incorrect",
    )
