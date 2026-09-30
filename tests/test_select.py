import cf_xarray  # noqa
import numpy as np
import numpy.testing as npt
import pandas as pd
import pyproj
import pytest
import xarray as xr
import xarray.testing as xrt
import xpublish
from fastapi.testclient import TestClient
from shapely import MultiPoint, Point, from_wkt

from xpublish_edr.area.geom import select_by_area
from xpublish_edr.geometry.bbox import select_by_bbox
from xpublish_edr.geometry.common import (
    GridKind,
    dataset_spatial_ref,
    is_regular_xy_coords,
    prepare_spatial_grid,
    project_dataset,
    with_spatial_coords,
)
from xpublish_edr.plugin import CfEdrPlugin
from xpublish_edr.position.geom import select_by_position
from xpublish_edr.area.query import EDRAreaQueryGet
from xpublish_edr.cube.query import EDRCubeQuery
from xpublish_edr.position.query import EDRPositionQueryGet


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


def test_select_query(regular_xy_dataset):
    query = EDRPositionQueryGet(
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

    query = EDRPositionQueryGet(
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

    query = EDRPositionQueryGet(
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

    query = EDRPositionQueryGet(
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
    query = EDRPositionQueryGet(
        coords="POINT(200 45)",
        datetime="2013-01-01T06:00:00",
        parameters="water",
    )
    query_params = {"foo": "bar"}

    with pytest.raises(KeyError):
        query.select(regular_xy_dataset, query_params)

    query = EDRPositionQueryGet(
        coords="POINT(200 45)",
        datetime="2013-01-0 06:00",
        parameters="air",
    )

    with pytest.raises(ValueError, match="Invalid datetime"):
        query.select(regular_xy_dataset, {})

    query = EDRPositionQueryGet(
        coords="POINT(200 45)",
        datetime="2013-01-01T06:00:00",
        parameters="air",
        z="100",
    )

    with pytest.raises(ValueError, match="Cannot select on Z axis via cf_xarray"):
        query.select(regular_xy_dataset, {})

    with pytest.raises(ValueError):
        query = EDRPositionQueryGet(
            coords="POINT(200 45)",
            datetime="2013-01-01T06:00:00",
            parameters="air",
            z="100",
            method="foo",
        )


def test_select_invalid_z_value(regular_xy_dataset):
    """A non-numeric ``z`` is rejected before any axis selection."""
    query = EDRPositionQueryGet(coords="POINT(200 45)", z="not-a-number")
    with pytest.raises(ValueError, match="Invalid z value"):
        query.select(regular_xy_dataset, {})


def test_select_too_many_slice_values(regular_xy_dataset):
    """A query param with more than two ``/``-separated values is rejected."""
    query = EDRPositionQueryGet(coords="POINT(200 45)")
    with pytest.raises(ValueError, match="Too many values for selecting"):
        query.select(regular_xy_dataset, {"lat": "40/45/50"})


@pytest.mark.parametrize(
    "coords",
    [
        "not wkt at all",  # GEOSException (ParseException)
        "\udce0",  # surrogate shapely cannot encode -> UnicodeDecodeError/Error
    ],
    ids=["unparsable", "unencodable"],
)
def test_geometry_invalid_coords_raise_geos_exception(coords):
    """Any coords WKT parse failure surfaces as GEOSException.

    Route handlers catch GEOSException to return a 422; if shapely raised a
    different error type (e.g. UnicodeDecodeError) it would escape as a 500.
    """
    from shapely.errors import GEOSException

    query = EDRPositionQueryGet(coords=coords)
    with pytest.raises(GEOSException):
        _ = query.geometry


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
    (
        npt.assert_approx_equal(
            ds["air"].isel(time=0).values.item(),
            280.2,
        ),
        "Temperature is incorrect",
    )
    (
        npt.assert_approx_equal(
            ds["air"].isel(time=-1).values.item(),
            279.19,
        ),
        "Temperature is incorrect",
    )


def test_select_position_projected_xy(projected_xy_dataset):
    query = EDRPositionQueryGet(
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
    (
        npt.assert_array_almost_equal(
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
    (
        npt.assert_approx_equal(
            ds["air"].isel(time=0).values.item(),
            281.376,
        ),
        "Temperature is incorrect",
    )
    (
        npt.assert_approx_equal(
            ds["air"].isel(time=-1).values.item(),
            279.87,
        ),
        "Temperature is incorrect",
    )


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
    query = EDRPositionQueryGet(
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
    query = EDRAreaQueryGet(
        coords="POLYGON((64.3 66.82, 64.5 66.82, 64.5 66.6, 64.3 66.6, 64.3 66.82))",
        crs="EPSG:4326",
    )

    projected_area = query.project_geometry(projected_xy_dataset)
    ds = select_by_area(projected_xy_dataset, projected_area)
    projected_ds = project_dataset(ds, query.crs)

    assert projected_ds is not None, "Dataset was not returned"
    assert "temp" in projected_ds, "Dataset does not contain the air variable"
    assert "latitude" in projected_ds, "Dataset does not contain the latitude variable"
    assert "longitude" in projected_ds, "Dataset does not contain the longitude variable"

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
    query = EDRPositionQueryGet(
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


@pytest.mark.skipif(
    not hasattr(np.dtypes, "StringDType"),
    reason="variable-width StringDType requires numpy >= 2.0",
)
def test_select_vlen_string_dim(regular_xy_dataset_with_string_dim):
    """Variable-width string coords (numpy 2 StringDType, produced by zarr v3
    string arrays) must be equality-selected, not nearest-selected"""
    ds = regular_xy_dataset_with_string_dim
    ds = ds.assign_coords(stat=ds["stat"].astype(np.dtypes.StringDType()))

    query = EDRPositionQueryGet(
        coords="POINT(200 45)",
        datetime="2013-01-01T06:00:00",
        parameters="air",
    )

    ds = query.select(
        ds,
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
    assert extent is None, "Should not report temporal extent for non-indexed T coordinate"


def test_vertical_extent_skips_non_indexed(dataset_with_non_indexed_axes):
    """Vertical extent should be None when Z axis is not indexed"""
    from xpublish_edr.metadata import vertical_extent

    extent = vertical_extent(dataset_with_non_indexed_axes)
    assert extent is None, "Should not report vertical extent for non-indexed Z coordinate"


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
    query = EDRPositionQueryGet(
        coords="POINT(202 42)",
        datetime="2024-01-01T06:00:00",
        parameters="temperature",
    )
    with pytest.raises(ValueError, match="Cannot select on T axis via cf_xarray"):
        query.select(dataset_with_non_indexed_axes, {})


def test_z_query_error_non_indexed(dataset_with_non_indexed_axes):
    """Z queries should raise clear error for non-indexed Z coordinate"""
    query = EDRPositionQueryGet(
        coords="POINT(202 42)",
        z="8500",
        parameters="temperature",
    )
    with pytest.raises(ValueError, match="Cannot select on Z axis via cf_xarray"):
        query.select(dataset_with_non_indexed_axes, {})


@pytest.fixture(scope="function")
def dataset_with_ambiguous_vertical_coords():
    """Mimics FVCOM's ambiguous, non-indexed vertical (Z) coordinates.

    ``siglay`` and ``siglev`` are both 2D sigma coordinates sharing
    ``standard_name="ocean_sigma_coordinate"``, so ``ds.cf["Z"]`` (and
    ``ds.cf.sel``/``ds.cf.interp``) raise cf_xarray's "multiple variables"
    ``KeyError`` for the "Z" key. Both are 2D (sigma dim, node), so neither
    is a dimension coordinate/index either way -- this reproduces the real
    FVCOM 500 in ``collection_metadata``.
    """
    times = pd.date_range("2024-01-01", periods=3, freq="h")
    n_node = 4

    return xr.Dataset(
        coords={
            "time": ("time", times, {"standard_name": "time", "long_name": "time"}),
            "lat": (
                "node",
                np.linspace(43.0, 44.0, n_node),
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
            "lon": (
                "node",
                np.linspace(-70.0, -69.0, n_node),
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
            "siglay": (
                ("siglay", "node"),
                np.linspace(-1 / 6, -5 / 6, 2 * n_node).reshape(2, n_node),
                {"standard_name": "ocean_sigma_coordinate", "positive": "up"},
            ),
            "siglev": (
                ("siglev", "node"),
                np.linspace(0, -1, 3 * n_node).reshape(3, n_node),
                {"standard_name": "ocean_sigma_coordinate", "positive": "up"},
            ),
        },
        data_vars={
            "temp": (
                ("time", "siglay", "node"),
                np.random.rand(len(times), 2, n_node),
            ),
        },
    )


def test_vertical_extent_none_for_ambiguous_non_indexed_z(dataset_with_ambiguous_vertical_coords):
    """Vertical extent is None when the Z axis has multiple non-indexed candidates"""
    from xpublish_edr.metadata import vertical_extent

    assert vertical_extent(dataset_with_ambiguous_vertical_coords) is None


def test_temporal_extent_for_ambiguous_vertical_dataset(dataset_with_ambiguous_vertical_coords):
    """Temporal extent still resolves normally when only Z is ambiguous"""
    from xpublish_edr.metadata import temporal_extent

    extent = temporal_extent(dataset_with_ambiguous_vertical_coords)
    assert extent is not None
    assert extent.interval == [["2024-01-01T00:00:00", "2024-01-01T02:00:00"]]


def test_cf_axis_is_indexed_false_for_ambiguous_z(dataset_with_ambiguous_vertical_coords):
    """cf_axis_is_indexed reports False rather than raising for an ambiguous axis"""
    from xpublish_edr.metadata import cf_axis_is_indexed

    assert cf_axis_is_indexed(dataset_with_ambiguous_vertical_coords, "Z") is False


def test_collection_metadata_for_ambiguous_vertical_dataset(dataset_with_ambiguous_vertical_coords):
    """collection_metadata succeeds (no 500) for an FVCOM-like dataset with two Z candidates"""
    from xpublish_edr.metadata import collection_metadata

    metadata = collection_metadata(
        dataset_with_ambiguous_vertical_coords,
        position_output_formats=["cf_covjson"],
        area_output_formats=["cf_covjson"],
        cube_output_formats=["cf_covjson"],
    )
    assert metadata.extent.vertical is None
    assert metadata.extent.temporal is not None


@pytest.fixture(scope="function")
def dataset_with_one_indexed_of_two_z_coords():
    """Two Z candidates, but only ``depth`` is actually indexed.

    ``sigma(depth, x)`` is a second ``ocean_sigma_coordinate`` Z candidate; it
    is 2D so it is never indexable itself, but it still makes cf_xarray treat
    "Z" as ambiguous. ``depth`` is a 1D dimension coordinate and is indexed.
    """
    depths = np.array([0.0, 10.0, 20.0])
    xs = np.array([0.0, 1.0, 2.0, 3.0])
    ys = np.array([0.0, 1.0])

    return xr.Dataset(
        coords={
            "depth": ("depth", depths, {"axis": "Z", "units": "m", "positive": "down"}),
            "x": ("x", xs, {"axis": "X"}),
            "y": ("y", ys, {"axis": "Y"}),
            "sigma": (
                ("depth", "x"),
                np.tile(np.linspace(0, -1, len(depths)), (len(xs), 1)).T,
                {"standard_name": "ocean_sigma_coordinate", "positive": "up"},
            ),
        },
        data_vars={
            "temp": (("y", "depth", "x"), np.random.rand(len(ys), len(depths), len(xs))),
        },
    )


def test_indexed_cf_axis_resolves_the_single_indexed_candidate(
    dataset_with_one_indexed_of_two_z_coords,
):
    """indexed_cf_axis resolves Z to depth when sigma is a second, non-indexed candidate"""
    from xpublish_edr.metadata import indexed_cf_axis

    coord = indexed_cf_axis(dataset_with_one_indexed_of_two_z_coords, "Z")
    assert coord is not None
    assert coord.name == "depth"


def test_vertical_extent_uses_the_indexed_candidate(dataset_with_one_indexed_of_two_z_coords):
    """vertical_extent reports depth's range rather than raising on the ambiguous Z axis"""
    from xpublish_edr.metadata import vertical_extent

    extent = vertical_extent(dataset_with_one_indexed_of_two_z_coords)
    assert extent is not None
    assert extent.interval == [["0.0", "20.0"]]


def test_select_z_with_one_indexed_of_two_z_candidates(dataset_with_one_indexed_of_two_z_coords):
    """select() resolves Z to depth by name instead of raising cf_xarray's ambiguous-Z error"""
    query = EDRPositionQueryGet(
        coords="POINT(1 0)",
        z="10.0",
        parameters="temp",
    )
    ds = query.select(dataset_with_one_indexed_of_two_z_coords, {})
    assert ds["depth"].values.tolist() == [10.0]


@pytest.fixture(scope="function")
def geozarr_proj_code_dataset():
    """A GeoZarr dataset declaring CRS/coords via the proj:/spatial: conventions.

    Uses ``proj:code`` for the CRS and ``spatial:dimensions`` ([Y, X] order) for
    coordinate identification; the 1D x/y coordinates carry *no* CF
    axis/standard_name attributes, so detection must come from the conventions.
    """
    return xr.Dataset(
        {"foo": (("y", "x"), np.arange(12).reshape(4, 3).astype(float))},
        coords={
            "x": ("x", [0.0, 1000.0, 2000.0]),
            "y": ("y", [0.0, 1000.0, 2000.0, 3000.0]),
        },
        attrs={"proj:code": "EPSG:3857", "spatial:dimensions": ["y", "x"]},
    )


@pytest.fixture(scope="function")
def geozarr_proj_wkt2_dataset(geozarr_proj_code_dataset):
    """GeoZarr CRS via proj:wkt2 instead of proj:code."""
    ds = geozarr_proj_code_dataset.copy()
    del ds.attrs["proj:code"]
    ds.attrs["proj:wkt2"] = pyproj.CRS.from_epsg(27700).to_wkt()
    return ds


@pytest.fixture(scope="function")
def multiple_grid_mappings_dataset():
    """Native 1D projected grid plus alternate 2D geographic grid mapping."""
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


@pytest.fixture(scope="function")
def geotransform_affine_dataset():
    """An affine (raster) dataset: a CF/GDAL ``GeoTransform`` and no coordinate arrays."""
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
def geozarr_spatial_transform_affine_dataset():
    """GeoZarr affine transform with no explicit x/y coordinates."""
    return xr.Dataset(
        {"foo": (("y", "x"), np.arange(12).reshape(4, 3).astype(float))},
        attrs={
            "proj:code": "EPSG:3857",
            "spatial:dimensions": ["y", "x"],
            "spatial:transform": [1000.0, 0.0, 0.0, 0.0, -1000.0, 3000.0],
        },
    )


@pytest.fixture(
    params=[
        pytest.param(
            {
                "fixture": "no_grid_mapping_projected_dataset",
                "crs": pyproj.CRS.from_epsg(3035),
                "xy": ("x", "y"),
                "point": (3.0, 8.0),
                "value": 2,
                "roundtrip_crs": "EPSG:4326",
            },
            id="legacy-spatial-ref",
        ),
        pytest.param(
            {
                "fixture": "geozarr_proj_code_dataset",
                "crs": pyproj.CRS.from_epsg(3857),
                "xy": ("x", "y"),
                "point": (1000.0, 2000.0),
                "value": 7.0,
                "roundtrip_crs": "EPSG:4326",
            },
            id="geozarr-proj-code",
        ),
        pytest.param(
            {
                "fixture": "geozarr_proj_wkt2_dataset",
                "crs": pyproj.CRS.from_epsg(27700),
                "xy": ("x", "y"),
                "point": (1000.0, 2000.0),
                "value": 7.0,
            },
            id="geozarr-proj-wkt2",
        ),
        pytest.param(
            {
                "fixture": "multiple_grid_mappings_dataset",
                "crs": pyproj.CRS.from_epsg(27700),
                "xy": ("x", "y"),
                "point": (401000.0, 101000.0),
                "value": 4.0,
            },
            id="multiple-grid-mappings",
        ),
        pytest.param(
            {
                "fixture": "geotransform_affine_dataset",
                "crs": pyproj.CRS.from_epsg(3857),
                "xy": ("x", "y"),
                "point": (1500.0, 1500.0),
                "value": 4.0,
                "materialized_x": [500.0, 1500.0, 2500.0],
                "original_missing_coords": ("x", "y"),
            },
            id="cf-geotransform-affine",
        ),
        pytest.param(
            {
                "fixture": "geozarr_spatial_transform_affine_dataset",
                "crs": pyproj.CRS.from_epsg(3857),
                "xy": ("x", "y"),
                "point": (1500.0, 1500.0),
                "value": 4.0,
                "materialized_x": [500.0, 1500.0, 2500.0],
                "original_missing_coords": ("x", "y"),
            },
            id="geozarr-spatial-transform-affine",
        ),
    ],
)
def spatial_selection_case(request):
    """Dataset-specific inputs for the shared spatial resolution/selection contract."""
    case = request.param.copy()
    case["ds"] = request.getfixturevalue(case.pop("fixture"))
    return case


def test_spatial_resolution_and_position_selection(spatial_selection_case):
    """CRS/X/Y resolution and point selection work across supported grid metadata."""
    case = spatial_selection_case
    ds = case["ds"]

    sr = dataset_spatial_ref(ds)
    assert sr.crs == case["crs"]
    assert (sr.X, sr.Y) == case["xy"]

    materialized = with_spatial_coords(ds, sr)
    assert is_regular_xy_coords(materialized, sr)
    if "materialized_x" in case:
        npt.assert_array_equal(materialized[sr.X].values, case["materialized_x"])
    for coord in case.get("original_missing_coords", ()):
        assert coord not in ds.coords

    selected = select_by_position(ds, Point(case["point"]), spatial_ref=sr)
    npt.assert_array_equal(selected["foo"].values.ravel(), [case["value"]])

    if "roundtrip_crs" not in case:
        return

    x, y = pyproj.Transformer.from_crs(
        case["crs"],
        case["roundtrip_crs"],
        always_xy=True,
    ).transform(*case["point"])
    query = EDRPositionQueryGet(coords=f"POINT({x} {y})", crs=case["roundtrip_crs"])
    selected = select_by_position(ds, query.project_geometry(ds), spatial_ref=sr)
    npt.assert_array_equal(selected["foo"].values.ravel(), [case["value"]])

    projected = project_dataset(selected, query.crs, sr)
    npt.assert_approx_equal(projected.cf["X"].values.item(), x, significant=5)
    npt.assert_approx_equal(projected.cf["Y"].values.item(), y, significant=5)


@pytest.fixture(scope="function")
def two_grid_dataset():
    """A regular grid carrying a second, unrelated longitude/latitude pair.

    Neither pair declares a CF ``axis`` attribute, so cf_xarray sees two
    longitude and two latitude candidates on the full dataset and cannot pick
    one. ``parameter-name=air`` drops the second grid, which makes the filtered
    dataset unambiguous -- so spatial metadata for a non-mesh dataset has to be
    resolved from the filtered dataset rather than from the source.
    """
    lon = np.linspace(-70.0, -69.0, 4)
    lat = np.linspace(43.0, 44.0, 3)
    lon2 = np.linspace(-70.0, -69.0, 5)
    lat2 = np.linspace(43.0, 44.0, 6)
    time = pd.date_range("2024-01-01", periods=2, freq="h")
    longitude = {"standard_name": "longitude", "units": "degrees_east"}
    latitude = {"standard_name": "latitude", "units": "degrees_north"}

    return xr.Dataset(
        {
            "air": (("time", "lat", "lon"), np.arange(24.0).reshape(2, 3, 4), {"units": "K"}),
            "sst": (("time", "lat2", "lon2"), np.arange(60.0).reshape(2, 6, 5), {"units": "K"}),
        },
        coords={
            "lon": ("lon", lon, longitude),
            "lat": ("lat", lat, latitude),
            "lon2": ("lon2", lon2, longitude),
            "lat2": ("lat2", lat2, latitude),
            "time": ("time", time),
        },
    )


def test_prepare_spatial_grid_resolves_from_the_filtered_dataset(two_grid_dataset):
    """Without a mesh, spatial metadata comes from the filtered dataset."""
    prepared = prepare_spatial_grid(
        two_grid_dataset[["air"]],
        source=two_grid_dataset,
        require_selectable=True,
    )

    assert prepared.spatial_ref.X == "lon"
    assert prepared.spatial_ref.Y == "lat"
    assert prepared.spatial_ref.mesh is None
    assert prepared.kind is GridKind.REGULAR


def test_position_query_on_a_dataset_with_two_grids(two_grid_dataset):
    """End to end, a position query on the filtered parameter still works."""
    rest = xpublish.Rest({"two": two_grid_dataset}, plugins={"edr": CfEdrPlugin()})
    client = TestClient(rest.app)

    response = client.get(
        "/datasets/two/edr/position?parameter-name=air&coords=POINT(-69.5 43.5)",
    )
    assert response.status_code == 200, response.text
    assert set(response.json()["ranges"]) == {"air"}
