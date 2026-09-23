"""
Handle selection and formatting for cube queries
"""

import xarray as xr

from xpublish_edr.geometry.common import (
    GridKind,
    PreparedSpatialGrid,
    SpatialRef,
    prepare_spatial_grid,
)


def select_prepared_bbox(
    prepared: PreparedSpatialGrid,
    bbox: tuple[float, float, float, float],
) -> xr.Dataset:
    """
    Return a dataset with the data within the given bbox

    Assumes that the dataset is in the same CRS as the bbox. A bbox selection
    never needs a mesh index: an unstructured grid is simply not selectable
    this way, so this is a standalone guard, not one the query pipeline
    reaches (it rejects an unstructured grid earlier). Use
    :func:`select_by_bbox` to select directly from a plain dataset.
    """
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


def select_by_bbox(
    ds: xr.Dataset,
    bbox: tuple[float, float, float, float],
    spatial_ref: SpatialRef | None = None,
) -> xr.Dataset:
    """
    Prepare ``ds`` and return the data within the given bbox

    Convenience entry point for callers (e.g. tests) that have not already
    prepared the grid.
    """
    prepared = prepare_spatial_grid(ds, spatial_ref=spatial_ref, require_selectable=True)
    return select_prepared_bbox(prepared, bbox)
