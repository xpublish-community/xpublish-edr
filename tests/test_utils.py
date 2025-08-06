import numpy as np
import xarray as xr

from xpublish_edr.utils import to_compat_da_dtype


def test_to_compat_da_dtype():
    # Test float16 conversion to float32
    da_float16 = xr.DataArray([1.0, 2.0, 3.0]).astype(np.float16)
    result = to_compat_da_dtype(da_float16)
    assert result.dtype == np.float32

    # Test float32 remains unchanged
    da_float32 = xr.DataArray([1.0, 2.0, 3.0]).astype(np.float32)
    result = to_compat_da_dtype(da_float32)
    assert result.dtype == np.float32

    # Test int types remain unchanged
    da_int = xr.DataArray([1, 2, 3]).astype(np.int32)
    result = to_compat_da_dtype(da_int)
    assert result.dtype == np.int32
