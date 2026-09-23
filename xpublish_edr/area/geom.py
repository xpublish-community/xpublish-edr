"""
Handle selection and formatting for area queries
"""

import dataclasses

import numpy as np
import shapely
import xarray as xr

from xpublish_edr.geometry.common import (
    VECTORIZED_DIM,
    GridKind,
    PreparedSpatialGrid,
    SpatialRef,
    ensure_xy_coords,
    prepare_spatial_grid,
    selection_targets,
)
from xpublish_edr.geometry.ugrid import MeshIndex, MeshInfo, get_mesh_index, variable_location


def select_prepared_area(
    prepared: PreparedSpatialGrid,
    polygon: shapely.Polygon,
) -> xr.Dataset:
    """
    Return a dataset with the area within the given polygon

    ``prepared`` must already carry a built mesh index (``prepared.mesh_index``)
    when its grid is unstructured; the query pipeline builds it once, before
    calling this. Use :func:`select_by_area` to select directly from a plain
    dataset.
    """
    if prepared.kind is GridKind.UNSTRUCTURED:
        return _select_area_unstructured(
            prepared.ds,
            polygon,
            prepared.mesh_index,
            prepared.spatial_ref,
        )

    return _select_area_regular_xy_grid(
        prepared.ds,
        polygon,
        prepared.spatial_ref.X,
        prepared.spatial_ref.Y,
    )


def select_by_area(
    ds: xr.Dataset,
    polygon: shapely.Polygon,
    spatial_ref: SpatialRef | None = None,
) -> xr.Dataset:
    """
    Prepare ``ds`` and return the area within the given polygon

    Convenience entry point for callers (e.g. tests) that have not already
    prepared the grid: builds an uncached mesh index when ``ds`` is
    unstructured. The query pipeline instead prepares once and calls
    :func:`select_prepared_area` directly with the index it already built.
    """
    prepared = prepare_spatial_grid(ds, spatial_ref=spatial_ref, require_selectable=True)
    if prepared.kind is GridKind.UNSTRUCTURED:
        prepared = dataclasses.replace(
            prepared,
            mesh_index=get_mesh_index(ds, prepared.spatial_ref),
        )
    return select_prepared_area(prepared, polygon)


def _mixed_location_message(ds: xr.Dataset, mesh: MeshInfo) -> str:
    """Build the error naming the node- and face-located parameters to choose from.

    An area query returns a single point set, so the caller has to restrict
    ``parameter-name`` to one mesh location (nodes or faces).
    Listing what is available on each makes it easier for users to recover from an error.

    The mesh's structural variables (the connectivities, which live on the face
    dimension) are not parameters, so they are only named when a location has
    nothing else.
    """
    parameters: dict[str, list[str]] = {"node": [], "face": []}
    structural: dict[str, list[str]] = {"node": [], "face": []}
    for name in ds.data_vars:
        location = variable_location(ds[name], mesh)
        if location is None:
            continue
        group = structural if str(name) in mesh.structural_vars else parameters
        group[location].append(str(name))

    node, face = (parameters[loc] or structural[loc] for loc in ("node", "face"))
    return (
        "Area queries select either node- or face-located parameters; "
        f"use parameter-name to choose from node: {', '.join(node)} "
        f"or face: {', '.join(face)}"
    )


def _select_area_unstructured(
    ds: xr.Dataset,
    polygon: shapely.Polygon,
    mesh_index: MeshIndex,
    spatial_ref: SpatialRef,
) -> xr.Dataset:
    """
    Return a dataset with the mesh nodes or faces within the given polygon

    The polygon arrives in the dataset's CRS and is projected into the mesh
    index's index plane before testing containment. Node-located parameters
    are tested against the mesh nodes; face-located parameters against
    xugrid's computed face centroids (not the UGRID ``face_coordinates``
    variables). A dataset with both node- and face-located parameters selected
    is rejected (with a message naming the parameters on each location), since
    the two locations would produce ``pts`` of different lengths.

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
        raise ValueError(_mixed_location_message(ds, mesh))
    (target,) = targets

    polygon_index = mesh_index.project_geometry(polygon)
    xy = mesh_index.node_xy if target.location == "node" else mesh_index.face_xy

    minx, miny, maxx, maxy = polygon_index.bounds
    mask = (xy[:, 0] >= minx) & (xy[:, 0] <= maxx) & (xy[:, 1] >= miny) & (xy[:, 1] <= maxy)
    cand = np.flatnonzero(mask)
    inside = shapely.intersects_xy(polygon_index, xy[cand, 0], xy[cand, 1])
    idx = cand[inside]

    selected = ds.isel({target.dim: xr.Variable(VECTORIZED_DIM, idx)})
    return ensure_xy_coords(selected, target, mesh_index, idx)


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
