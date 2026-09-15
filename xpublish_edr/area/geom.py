"""
Handle selection and formatting for area queries
"""

import numpy as np
import shapely
import xarray as xr

from xpublish_edr.geometry.common import (
    VECTORIZED_DIM,
    GridKind,
    SpatialRef,
    prepare_spatial_grid,
    selection_targets,
)
from xpublish_edr.geometry.ugrid import IndexedGrid


def select_by_area(
    ds: xr.Dataset,
    polygon: shapely.Polygon,
    spatial_ref: SpatialRef | None = None,
    grid: IndexedGrid | None = None,
) -> xr.Dataset:
    """
    Return a dataset with the area within the given polygon

    ``grid`` is the mesh index for an unstructured dataset
    """
    prepared = prepare_spatial_grid(
        ds,
        spatial_ref=spatial_ref,
        require_selectable=True,
        grid=grid,
    )

    if prepared.kind is GridKind.UNSTRUCTURED:
        if prepared.grid is None:
            raise ValueError("Unstructured grid index was not built")
        return _select_area_unstructured(
            prepared.ds,
            polygon,
            prepared.grid,
            prepared.spatial_ref,
        )

    return _select_area_regular_xy_grid(
        prepared.ds,
        polygon,
        prepared.spatial_ref.X,
        prepared.spatial_ref.Y,
    )


def _select_area_unstructured(
    ds: xr.Dataset,
    polygon: shapely.Polygon,
    grid: IndexedGrid,
    spatial_ref: SpatialRef,
) -> xr.Dataset:
    """
    Return a dataset with the mesh nodes or faces within the given polygon

    The polygon arrives in the dataset's CRS and is projected into the grid's
    index plane before testing containment. Node-located parameters are
    tested against the mesh nodes; face-located parameters against xugrid's
    computed face centroids (not the UGRID ``face_coordinates`` variables).
    A dataset with both node- and face-located parameters selected is
    rejected, since the two locations would produce ``pts`` of different
    lengths.

    Like the regular-grid path, a polygon that crosses the antimeridian is
    not split and is matched as given.
    """
    mesh = spatial_ref.mesh
    if mesh is None:  # pragma: no cover - guarded by the grid kind
        raise ValueError("Unstructured selection requires a UGRID mesh")

    targets = selection_targets(ds, mesh)
    if not targets:
        raise ValueError("No mesh-located variables selected")
    if len(targets) > 1:
        raise ValueError("Area queries cannot mix node- and face-located parameters")
    (target,) = targets

    polygon_index = grid.project_geometry(polygon)
    xy = grid.node_xy if target.location == "node" else grid.face_xy

    minx, miny, maxx, maxy = polygon_index.bounds
    mask = (xy[:, 0] >= minx) & (xy[:, 0] <= maxx) & (xy[:, 1] >= miny) & (xy[:, 1] <= maxy)
    cand = np.flatnonzero(mask)
    inside = shapely.intersects_xy(polygon_index, xy[cand, 0], xy[cand, 1])
    idx = cand[inside]

    return ds.isel({target.dim: xr.Variable(VECTORIZED_DIM, idx)})


def _select_area_regular_xy_grid(
    ds: xr.Dataset,
    polygon: shapely.Polygon,
    X: str,
    Y: str,
) -> xr.Dataset:
    """
    Return a dataset with the area within the given polygon
    """
    # To minimize performance impact, we first subset the dataset to the bounding box of the polygon
    minx, miny, maxx, maxy = polygon.bounds
    indexes = ds.indexes
    if indexes[X].is_monotonic_increasing:
        x_sel = slice(minx, maxx)
    else:
        x_sel = slice(maxx, minx)
    if indexes[Y].is_monotonic_increasing:
        y_sel = slice(miny, maxy)
    else:
        y_sel = slice(maxy, miny)
    ds = ds.sel({X: x_sel, Y: y_sel})

    # For a regular grid, we can create a meshgrid of the X and Y coordinates to create a spatial mask
    pts = np.meshgrid(ds[X], ds[Y])

    # Create a mask of the points within the polygon
    mask = shapely.intersects_xy(polygon, pts[0], pts[1])

    # Find the x and y indices that have any points within the polygon
    y_inds, x_inds = np.nonzero(mask)
    x_isel = xr.Variable(data=x_inds, dims=VECTORIZED_DIM)
    y_isel = xr.Variable(data=y_inds, dims=VECTORIZED_DIM)

    # Apply the mask and vectorize to a 1d collection of points
    return ds.isel({X: x_isel, Y: y_isel})
