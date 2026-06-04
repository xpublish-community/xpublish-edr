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
from xpublish_edr.geometry.common import (
    dataset_crs,
    dataset_spatial_ref,
    is_regular_xy_coords,
    project_dataset,
    with_spatial_coords,
)
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
        coords="POINT(3 8)",
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
            "step": pd.timedelta_range("0 days", periods=72, freq="1h"),
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
        pd.timedelta_range("0 days", periods=11, freq="1h"),
    )
    npt.assert_equal(ds["elevation"].values, 101)

    ds = query.select(custom_dim_ds, {"step": "1 hours", "elevation": "101/103"})
    assert ds["air"].shape == (2, 2, 3, 1), "Dataset shape is incorrect"
    npt.assert_array_equal(
        ds["step"],
        pd.timedelta_range("1 hours", periods=1, freq="1h"),
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

    with pytest.raises(ValueError, match="Invalid datetime"):
        query.select(regular_xy_dataset, {})

    query = EDRPositionQuery(
        coords="POINT(200 45)",
        datetime="2013-01-01T06:00:00",
        parameters="air",
        z="100",
    )

    with pytest.raises(ValueError, match="Cannot select on Z axis via cf_xarray"):
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
    npt.assert_approx_equal(
        ds["air"].isel(time=0).values.item(),
        280.2,
    ), "Temperature is incorrect"
    npt.assert_approx_equal(
        ds["air"].isel(time=-1).values.item(),
        279.19,
    ), "Temperature is incorrect"


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
        npt.assert_approx_equal(projected_ds.cf["X"].values.item(), 64.59063409),
        "Longitude is incorrect",
    )
    (
        npt.assert_approx_equal(projected_ds.cf["Y"].values.item(), 66.66454929),
        "Latitude is incorrect",
    )
    npt.assert_array_almost_equal(
        projected_ds.temp.values,
        projected_xy_dataset.sel(
            rlon=[18.045],
            rlat=[21.725],
            method="nearest",
        ).temp.values,
    ), "Temperature is incorrect"


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
    npt.assert_approx_equal(
        ds["air"].isel(time=0).values.item(),
        281.376,
    ), "Temperature is incorrect"
    npt.assert_approx_equal(
        ds["air"].isel(time=-1).values.item(),
        279.87,
    ), "Temperature is incorrect"


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


@pytest.fixture(scope="function")
def dataset_with_non_indexed_axes():
    """Creates a dataset with non-indexed CF axis coordinates (like GFS forecast)"""
    init_times = pd.date_range("2024-01-01", periods=4, freq="6h")
    lead_times = pd.to_timedelta([0, 1, 2, 3], unit="h")
    levels = [1000, 850, 500]

    ds = xr.Dataset(
        coords={
            "init_time": init_times,
            "lead_time": lead_times,
            "level": levels,
            "lat": np.arange(40, 45, dtype=float),
            "lon": np.arange(200, 205, dtype=float),
        },
        data_vars={
            "temperature": (
                ("init_time", "lead_time", "level", "lat", "lon"),
                np.random.rand(4, 4, 3, 5, 5),
            ),
        },
    )
    # Add CF attributes for lat/lon
    ds.lat.attrs["axis"] = "Y"
    ds.lon.attrs["axis"] = "X"

    # Add non-indexed 2D coordinates with CF attributes
    ds = ds.assign_coords(valid_time=ds.init_time + ds.lead_time)
    ds.valid_time.attrs["axis"] = "T"
    ds.valid_time.attrs["standard_name"] = "time"

    # Add non-indexed Z coordinate
    ds = ds.assign_coords(altitude=("level", [10000, 8500, 5000]))
    ds.altitude.attrs["axis"] = "Z"
    ds.altitude.attrs["positive"] = "up"

    return ds


def test_temporal_extent_skips_non_indexed(dataset_with_non_indexed_axes):
    """Temporal extent should be None when T axis is not indexed"""
    from xpublish_edr.metadata import temporal_extent

    extent = temporal_extent(dataset_with_non_indexed_axes)
    assert (
        extent is None
    ), "Should not report temporal extent for non-indexed T coordinate"


def test_vertical_extent_skips_non_indexed(dataset_with_non_indexed_axes):
    """Vertical extent should be None when Z axis is not indexed"""
    from xpublish_edr.metadata import vertical_extent

    extent = vertical_extent(dataset_with_non_indexed_axes)
    assert (
        extent is None
    ), "Should not report vertical extent for non-indexed Z coordinate"


def test_generic_extents_includes_indexed_dims_from_non_indexed_axes(
    dataset_with_non_indexed_axes,
):
    """init_time, lead_time, and level should appear because T and Z axes are not indexed"""
    from xpublish_edr.metadata import generic_extents

    extents = generic_extents(dataset_with_non_indexed_axes)
    assert extents is not None
    assert "init_time" in extents, "init_time should be in generic extents"
    assert "lead_time" in extents, "lead_time should be in generic extents"
    assert "level" in extents, "level should be in generic extents"


def test_generic_extents_excludes_non_indexed_dims():
    """Dimensions without indexes should not appear in generic extents"""
    from xpublish_edr.metadata import generic_extents

    # Create dataset with a dimension that has no index
    ds = xr.Dataset(
        data_vars={
            "data": (("x", "y", "ensemble"), np.random.rand(5, 5, 10)),
        },
        coords={
            "x": np.arange(5, dtype=float),
            "y": np.arange(5, dtype=float),
            # "ensemble" has no coordinate, so no index
        },
    )
    ds.x.attrs["axis"] = "X"
    ds.y.attrs["axis"] = "Y"

    extents = generic_extents(ds)
    # ensemble should not be in extents because it has no index
    assert extents is None or "ensemble" not in extents


def test_datetime_query_error_non_indexed(dataset_with_non_indexed_axes):
    """Datetime queries should raise clear error for non-indexed T coordinate"""
    query = EDRPositionQuery(
        coords="POINT(202 42)",
        datetime="2024-01-01T06:00:00",
        parameters="temperature",
    )
    with pytest.raises(ValueError, match="Cannot select on T axis via cf_xarray"):
        query.select(dataset_with_non_indexed_axes, {})


def test_z_query_error_non_indexed(dataset_with_non_indexed_axes):
    """Z queries should raise clear error for non-indexed Z coordinate"""
    query = EDRPositionQuery(
        coords="POINT(202 42)",
        z="8500",
        parameters="temperature",
    )
    with pytest.raises(ValueError, match="Cannot select on Z axis via cf_xarray"):
        query.select(dataset_with_non_indexed_axes, {})


@pytest.fixture(scope="function")
def geozarr_proj_code_dataset():
    """A GeoZarr dataset declaring its CRS/coords via the proj:/spatial: conventions.

    Uses ``proj:code`` for the CRS and ``spatial:dimensions`` ([Y, X] order) for
    the coordinate identification. The 1D coordinate variables carry *no* CF
    axis/standard_name attributes, so detection must come from the conventions.
    """
    foo = np.arange(12).reshape(4, 3).astype(float)
    return xr.Dataset(
        {"foo": (("y", "x"), foo)},
        coords={
            "x": ("x", [0.0, 1000.0, 2000.0]),
            "y": ("y", [0.0, 1000.0, 2000.0, 3000.0]),
        },
        attrs={"proj:code": "EPSG:3857", "spatial:dimensions": ["y", "x"]},
    )


@pytest.fixture(scope="function")
def geozarr_proj_wkt2_dataset():
    """A GeoZarr dataset declaring its CRS via the proj:wkt2 attribute."""
    foo = np.arange(6).reshape(3, 2).astype(float)
    return xr.Dataset(
        {"foo": (("y", "x"), foo)},
        coords={
            "x": ("x", [0.0, 1000.0]),
            "y": ("y", [0.0, 1000.0, 2000.0]),
        },
        attrs={
            "proj:wkt2": pyproj.CRS.from_epsg(27700).to_wkt(),
            "spatial:dimensions": ["y", "x"],
        },
    )


@pytest.fixture(scope="function")
def multiple_grid_mapping_dataset():
    """A dataset with multiple CF grid mappings (GeoZarr alternate CRS pattern).

    Native grid is EPSG:27700 with 1D indexed x/y; an alternate EPSG:4326 grid
    mapping references 2D longitude/latitude. The resolver must pick the native
    1D grid (the old ``dataset_crs`` raised on multiple grid mappings).
    """
    return xr.Dataset(
        {
            "foo": (
                ("y", "x"),
                np.arange(6).reshape(2, 3).astype(float),
                {"grid_mapping": "spatial_ref: x y crs_4326: longitude latitude"},
            ),
        },
        coords={
            "x": (
                "x",
                [400000.0, 401000.0, 402000.0],
                {"axis": "X", "standard_name": "projection_x_coordinate"},
            ),
            "y": (
                "y",
                [100000.0, 101000.0],
                {"axis": "Y", "standard_name": "projection_y_coordinate"},
            ),
            "longitude": (("y", "x"), np.zeros((2, 3)), {"standard_name": "longitude"}),
            "latitude": (("y", "x"), np.zeros((2, 3)), {"standard_name": "latitude"}),
            "spatial_ref": ((), 0, pyproj.CRS.from_epsg(27700).to_cf()),
            "crs_4326": ((), 0, pyproj.CRS.from_epsg(4326).to_cf()),
        },
    )


def test_geozarr_proj_code_spatial_ref(geozarr_proj_code_dataset):
    """proj:code + spatial:dimensions resolve to the right CRS and X/Y names."""
    sr = dataset_spatial_ref(geozarr_proj_code_dataset)
    assert sr.crs == pyproj.CRS.from_epsg(3857)
    assert (sr.X, sr.Y) == ("x", "y")


def test_geozarr_proj_wkt2_spatial_ref(geozarr_proj_wkt2_dataset):
    """proj:wkt2 resolves to the right CRS."""
    sr = dataset_spatial_ref(geozarr_proj_wkt2_dataset)
    assert sr.crs == pyproj.CRS.from_epsg(27700)
    assert (sr.X, sr.Y) == ("x", "y")


def test_geozarr_position_native_crs(geozarr_proj_code_dataset):
    """A position query in the dataset's native CRS selects the nearest cell."""
    query = EDRPositionQuery(coords="POINT(1000 2000)", crs="EPSG:3857")
    geom = query.project_geometry(geozarr_proj_code_dataset)
    ds = select_by_position(geozarr_proj_code_dataset, geom)
    npt.assert_array_equal(ds["x"], 1000.0)
    npt.assert_array_equal(ds["y"], 2000.0)
    # foo = arange(12).reshape(4, 3) -> value at y-index 2, x-index 1 == 7
    npt.assert_array_equal(ds["foo"].values.ravel(), [7.0])


def test_geozarr_position_reprojected(geozarr_proj_code_dataset):
    """A position query in EPSG:4326 is projected into the native CRS and back."""
    transformer = pyproj.Transformer.from_crs(3857, 4326, always_xy=True)
    lon, lat = transformer.transform(1000.0, 2000.0)
    query = EDRPositionQuery(coords=f"POINT({lon} {lat})", crs="EPSG:4326")
    geom = query.project_geometry(geozarr_proj_code_dataset)
    ds = select_by_position(geozarr_proj_code_dataset, geom)
    npt.assert_array_equal(ds["foo"].values.ravel(), [7.0])

    projected = project_dataset(ds, query.crs)
    # Output carries CF longitude/latitude after projection to a geographic CRS
    npt.assert_approx_equal(projected.cf["X"].values.item(), lon, significant=5)
    npt.assert_approx_equal(projected.cf["Y"].values.item(), lat, significant=5)


def test_geozarr_cube_native_crs(geozarr_proj_code_dataset):
    """A cube/bbox query subsets a GeoZarr dataset by its native coordinates."""
    ds = select_by_bbox(geozarr_proj_code_dataset, (0.0, 0.0, 1000.0, 2000.0))
    npt.assert_array_equal(ds["x"].values, [0.0, 1000.0])
    npt.assert_array_equal(ds["y"].values, [0.0, 1000.0, 2000.0])


def test_geozarr_area_native_crs(geozarr_proj_code_dataset):
    """An area query masks a GeoZarr dataset to points within a polygon."""
    polygon = from_wkt("POLYGON((0 0, 1000 0, 1000 1000, 0 1000, 0 0))")
    ds = select_by_area(geozarr_proj_code_dataset, polygon)
    # The 2x2 block of cells with x in {0, 1000} and y in {0, 1000}
    assert ds.sizes["pts"] == 4
    assert set(np.unique(ds["x"].values)) == {0.0, 1000.0}
    assert set(np.unique(ds["y"].values)) == {0.0, 1000.0}


def test_multiple_grid_mappings_pick_native(multiple_grid_mapping_dataset):
    """Datasets with multiple grid mappings resolve to the native 1D grid."""
    # Previously dataset_crs raised "Multiple grid mappings found".
    assert dataset_crs(multiple_grid_mapping_dataset) == pyproj.CRS.from_epsg(27700)
    sr = dataset_spatial_ref(multiple_grid_mapping_dataset)
    assert sr.crs == pyproj.CRS.from_epsg(27700)
    assert (sr.X, sr.Y) == ("x", "y")

    query = EDRPositionQuery(coords="POINT(401000 100000)", crs="EPSG:27700")
    geom = query.project_geometry(multiple_grid_mapping_dataset)
    ds = select_by_position(multiple_grid_mapping_dataset, geom)
    npt.assert_array_equal(ds["x"], 401000.0)
    npt.assert_array_equal(ds["y"], 100000.0)


@pytest.fixture(scope="function")
def geotransform_affine_dataset():
    """An affine (raster) dataset: a CF/GDAL ``GeoTransform`` and no coordinate arrays.

    rasterix materializes the x/y coordinates from the transform. The transform
    ``"0 1000 0 3000 0 -1000"`` gives 1km pixels with origin (0, 3000) and a
    north-up (descending y) orientation, i.e. pixel centers x=[500, 1500, 2500]
    and y=[2500, 1500, 500, -500].
    """
    return xr.Dataset(
        {
            "foo": (
                ("y", "x"),
                np.arange(12).reshape(4, 3).astype(float),
                {"grid_mapping": "spatial_ref"},
            ),
        },
        coords={
            "spatial_ref": (
                (),
                0,
                {
                    **pyproj.CRS.from_epsg(3857).to_cf(),
                    "GeoTransform": "0 1000 0 3000 0 -1000",
                },
            ),
        },
    )


@pytest.fixture(scope="function")
def geozarr_spatial_transform_dataset():
    """An affine GeoZarr dataset using ``proj:`` + ``spatial:transform`` (no coords).

    ``spatial:transform`` is rasterio/Affine order ``[a, b, c, d, e, f]``;
    here it matches the GeoTransform fixture (1km pixels, north-up).
    """
    return xr.Dataset(
        {"foo": (("y", "x"), np.arange(12).reshape(4, 3).astype(float))},
        attrs={
            "proj:code": "EPSG:3857",
            "spatial:dimensions": ["y", "x"],
            "spatial:transform": [1000.0, 0.0, 0.0, 0.0, -1000.0, 3000.0],
        },
    )


def test_geotransform_affine_materializes_coords(geotransform_affine_dataset):
    """rasterix materializes 1D coordinates from a CF GeoTransform (plain index)."""
    ds = geotransform_affine_dataset
    assert "x" not in ds.coords and "y" not in ds.coords  # affine-only to start
    materialized = with_spatial_coords(ds)
    npt.assert_array_equal(materialized["x"].values, [500.0, 1500.0, 2500.0])
    npt.assert_array_equal(materialized["y"].values, [2500.0, 1500.0, 500.0, -500.0])
    assert is_regular_xy_coords(materialized)
    # The original dataset is not mutated (GeoTransform preserved, no new coords)
    assert "GeoTransform" in ds["spatial_ref"].attrs
    assert "x" not in ds.coords


def test_geotransform_affine_position_and_cube(geotransform_affine_dataset):
    """Position and cube selection work on a GeoTransform-only dataset."""
    ds = geotransform_affine_dataset
    assert dataset_crs(ds) == pyproj.CRS.from_epsg(3857)
    point = Point((1500, 1500))  # center cell -> foo == 4
    sel = select_by_position(ds, point)
    npt.assert_array_equal(sel["foo"].values.ravel(), [4.0])

    cube = select_by_bbox(ds, (0.0, 0.0, 1600.0, 1600.0))
    npt.assert_array_equal(cube["x"].values, [500.0, 1500.0])


def test_geozarr_spatial_transform_affine(geozarr_spatial_transform_dataset):
    """proj: + spatial:transform (no coords) resolve and select correctly."""
    ds = geozarr_spatial_transform_dataset
    sr = dataset_spatial_ref(ds)
    assert sr.crs == pyproj.CRS.from_epsg(3857)
    assert (sr.X, sr.Y) == ("x", "y")

    materialized = with_spatial_coords(ds)
    npt.assert_array_equal(materialized["x"].values, [500.0, 1500.0, 2500.0])
    assert is_regular_xy_coords(materialized)
    # Original untouched (still affine-only)
    assert "x" not in ds.coords

    sel = select_by_position(ds, Point((1500, 1500)))
    npt.assert_array_equal(sel["foo"].values.ravel(), [4.0])
