import shapely
import xarray as xr

from xpublish_edr.geometry.common import is_regular_xy_coords


def select_by_postition(ds: xr.Dataset, point: shapely.Point) -> xr.Dataset:
    """
    Return a dataset with the position nearest to the given coordinates
    """
    if not is_regular_xy_coords(ds):
        # TODO: Handle 2D coordinates
        raise NotImplementedError("Only 1D coordinates are supported")

    return _select_by_position_regular_xy_grid(ds, point)


def _select_by_position_regular_xy_grid(
    ds: xr.Dataset,
    point: shapely.Point,
) -> xr.Dataset:
    """
    Return a dataset with the position nearest to the given coordinates
    """
    # Find the nearest X and Y coordinates to the point
    return ds.cf.sel(X=point.x, Y=point.y, method="nearest")
