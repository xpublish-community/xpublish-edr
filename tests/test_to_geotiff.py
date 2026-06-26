"""Unit tests for the GeoTIFF dimensionality guards in ``formats/to_geotiff.py``.

The success path is covered end-to-end in ``test_cf_router.py``; these cover the
400 error branches for datasets GeoTIFF export can't represent.
"""

import numpy as np
import pytest
import xarray as xr
from fastapi import HTTPException

from xpublish_edr.formats.to_geotiff import to_geotiff


def test_no_data_variables():
    ds = xr.Dataset(coords={"x": [1, 2], "y": [3, 4]})
    with pytest.raises(HTTPException) as exc:
        to_geotiff(ds)
    assert exc.value.status_code == 400
    assert "No variables" in exc.value.detail


def test_single_variable_too_many_dims():
    ds = xr.Dataset(
        {"air": (("a", "b", "c", "d"), np.zeros((2, 2, 2, 2)))},
    )
    with pytest.raises(HTTPException) as exc:
        to_geotiff(ds)
    assert exc.value.status_code == 400
    assert "up to 3 dimensions" in exc.value.detail


def test_multiple_variables_too_many_dims():
    ds = xr.Dataset(
        {
            "air": (("a", "b", "c"), np.zeros((2, 2, 2))),
            "temp": (("a", "b", "c"), np.zeros((2, 2, 2))),
        },
    )
    with pytest.raises(HTTPException) as exc:
        to_geotiff(ds)
    assert exc.value.status_code == 400
    assert "up to 2 dimensions" in exc.value.detail
