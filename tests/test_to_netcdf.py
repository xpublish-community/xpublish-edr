import numpy as np
import pytest

from xpublish_edr.formats.to_netcdf import to_compat_nc_attr


@pytest.mark.parametrize(
    "value, expected",
    [
        (True, np.int8(1)),
        (np.bool_(False), np.int8(0)),
        (np.str_("abc"), "abc"),
        (np.array(["ab", "c"]), ["ab", "c"]),
        (np.array(["ab", "c"], dtype=object), ["ab", "c"]),
        (np.array([True, False]), np.array([1, 0], dtype=np.int8)),
        ([True, False], np.array([1, 0], dtype=np.int8)),
        (["ab", "c"], ["ab", "c"]),
        (1.5, 1.5),
        (np.array([1, 2]), np.array([1, 2])),
    ],
)
def test_to_compat_nc_attr(value, expected):
    result = to_compat_nc_attr(value)

    assert type(result) is type(expected)
    if isinstance(expected, np.ndarray):
        assert result.dtype == expected.dtype
        np.testing.assert_array_equal(result, expected)
    else:
        assert result == expected
