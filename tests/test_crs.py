"""Tests for CRS resolution and coordinate materialization in ``geometry/common.py``.

These cover convention paths (GeoZarr ``proj:wkt2`` and ``spatial:transform``)
and the 2D-coordinate guard that the CF/air-temperature fixtures don't exercise.
"""

import numpy as np
import pyproj
import pytest
import xarray as xr

from xpublish_edr.geometry.common import (
    dataset_crs,
    project_dataset,
    with_spatial_coords,
)


def test_proj_wkt2_crs_resolution():
    """A GeoZarr dataset describing its CRS via ``proj:wkt2`` resolves correctly."""
    wkt2 = pyproj.CRS.from_epsg(3857).to_wkt()
    ds = xr.Dataset(
        {"foo": (("y", "x"), np.arange(12.0).reshape(4, 3), {"standard_name": "air_temperature"})},
        coords={"x": ("x", [0.0, 1000.0, 2000.0]), "y": ("y", [0.0, 1000.0, 2000.0, 3000.0])},
        attrs={"proj:wkt2": wkt2, "spatial:dimensions": ["y", "x"]},
    )
    assert dataset_crs(ds).to_epsg() == 3857


def test_spatial_transform_materializes_coordinates():
    """``spatial:transform`` (affine, no coordinate vars) materializes 1D x/y coords."""
    ds = xr.Dataset(
        {"foo": (("y", "x"), np.arange(12.0).reshape(4, 3), {"standard_name": "air_temperature"})},
        attrs={
            "proj:code": "EPSG:3857",
            "spatial:dimensions": ["y", "x"],
            "spatial:transform": [1000.0, 0.0, 0.0, 0.0, -1000.0, 0.0],
        },
    )
    out = with_spatial_coords(ds)
    assert "x" in out.coords and "y" in out.coords
    assert out["x"].ndim == 1 and out["y"].ndim == 1
    # Pixel centers for a 1000 m pixel starting at the origin.
    assert out["x"].values[0] == 500.0


def test_project_dataset_rejects_2d_coordinates():
    """Projecting a curvilinear (2D lon/lat) grid to a different CRS is unsupported."""
    lon = np.array([[10.0, 11.0, 12.0], [10.0, 11.0, 12.0]])
    lat = np.array([[40.0, 40.0, 40.0], [41.0, 41.0, 41.0]])
    ds = xr.Dataset(
        {"foo": (("y", "x"), np.arange(6.0).reshape(2, 3))},
        coords={
            "lon": (("y", "x"), lon, {"standard_name": "longitude", "units": "degrees_east"}),
            "lat": (("y", "x"), lat, {"standard_name": "latitude", "units": "degrees_north"}),
        },
    )
    with pytest.raises(NotImplementedError, match="Only 1D coordinates are supported"):
        project_dataset(ds, "EPSG:3857")
