"""
Handle selection and formatting for cube queries
"""

import xarray as xr

from xpublish_edr.geometry.common import GridKind, SpatialRef, prepare_spatial_grid
from xpublish_edr.geometry.ugrid import IndexedGrid


def select_by_bbox(
    ds: xr.Dataset,
    bbox: tuple[float, float, float, float],
    spatial_ref: SpatialRef | None = None,
    grid: IndexedGrid | None = None,
) -> xr.Dataset:
    """
    Return a dataset with the data within the given bbox

    Assumes that the dataset is in the same CRS as the bbox. ``grid`` is an
    already built mesh index, passed through only so that an unstructured
    dataset is not re-indexed on the way to the error below.
    """
    prepared = prepare_spatial_grid(
        ds,
        spatial_ref=spatial_ref,
        require_selectable=True,
        grid=grid,
    )
    if prepared.kind is not GridKind.REGULAR:
        raise NotImplementedError("Cube queries require a regular X/Y grid")
    ds = prepared.ds
    X, Y = prepared.spatial_ref.X, prepared.spatial_ref.Y
    indexes = ds.indexes
    if indexes[X].is_monotonic_increasing:
        x_slice = slice(bbox[0], bbox[2])
    else:
        x_slice = slice(bbox[2], bbox[0])
    if indexes[Y].is_monotonic_increasing:
        y_slice = slice(bbox[1], bbox[3])
    else:
        y_slice = slice(bbox[3], bbox[1])
    return ds.sel({X: x_slice, Y: y_slice})
