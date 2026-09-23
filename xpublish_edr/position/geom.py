"""Handle selection and formatting for position queries"""

from __future__ import annotations

import dataclasses
from typing import Literal

import numpy as np
import shapely
import xarray as xr

from xpublish_edr.geometry.common import (
    VECTORIZED_DIM,
    GridKind,
    PreparedSpatialGrid,
    SelectionTarget,
    SpatialRef,
    ensure_xy_coords,
    prepare_spatial_grid,
    selection_targets,
)
from xpublish_edr.geometry.ugrid import (
    MeshIndex,
    MeshInfo,
    MeshSelectionError,
    get_mesh_index,
    variable_location,
)
from xpublish_edr.logger import logger

# Temporary dimension holding the three vertices of each containing triangle.
VERTEX_DIM = "_vertex"


def select_prepared_position(
    prepared: PreparedSpatialGrid,
    point: shapely.Point | shapely.MultiPoint,
    method: Literal["nearest", "linear"] = "nearest",
) -> xr.Dataset:
    """
    Return a dataset with the position nearest to the given coordinates

    ``prepared`` must already carry a built mesh index (``prepared.mesh_index``)
    when its grid is unstructured; the query pipeline builds it once, before
    calling this, so it is not rebuilt per selector. Use
    :func:`select_by_position` to select directly from a plain dataset.
    """
    ds = prepared.ds

    if prepared.kind is GridKind.UNSTRUCTURED:
        return _select_by_position_unstructured(
            ds,
            point,
            prepared.mesh_index,
            prepared.spatial_ref,
            method,
        )

    X, Y = prepared.spatial_ref.X, prepared.spatial_ref.Y

    if isinstance(point, shapely.Point):
        return _select_by_position_regular_xy_grid(ds, point, X, Y, method)
    elif isinstance(point, shapely.MultiPoint):
        return _select_by_multiple_positions_regular_xy_grid(ds, point, X, Y, method)
    else:
        raise ValueError(
            f"Invalid point type {point.geom_type}, must be Point or MultiPoint",
        )


def select_by_position(
    ds: xr.Dataset,
    point: shapely.Point | shapely.MultiPoint,
    method: Literal["nearest", "linear"] = "nearest",
    spatial_ref: SpatialRef | None = None,
) -> xr.Dataset:
    """
    Prepare ``ds`` and return the position nearest to the given coordinates

    Convenience entry point for callers (e.g. tests) that have not already
    prepared the grid: builds an uncached mesh index when ``ds`` is
    unstructured. The query pipeline instead prepares once and calls
    :func:`select_prepared_position` directly with the index it already built.
    """
    prepared = prepare_spatial_grid(ds, spatial_ref=spatial_ref, require_selectable=True)
    if prepared.kind is GridKind.UNSTRUCTURED:
        prepared = dataclasses.replace(
            prepared,
            mesh_index=get_mesh_index(ds, prepared.spatial_ref),
        )
    return select_prepared_position(prepared, point, method)


def _select_by_position_regular_xy_grid(
    ds: xr.Dataset,
    point: shapely.Point,
    X: str,
    Y: str,
    method: Literal["nearest", "linear"] = "nearest",
) -> xr.Dataset:
    """
    Return a dataset with the position nearest to the given coordinates
    """
    # Find the nearest X and Y coordinates to the point
    if method == "nearest":
        return ds.sel({X: [point.x], Y: [point.y]}, method=method)
    else:
        return ds.interp({X: [point.x], Y: [point.y]}, method=method)


def _select_by_multiple_positions_regular_xy_grid(
    ds: xr.Dataset,
    points: shapely.MultiPoint,
    X: str,
    Y: str,
    method: Literal["nearest", "linear"] = "nearest",
) -> xr.Dataset:
    """
    Return a dataset with the positions nearest to the given coordinates
    """
    # Find the nearest X and Y coordinates to the point using vectorized indexing
    x, y = np.array(list(zip(*[(point.x, point.y) for point in points.geoms])))

    # When using vectorized indexing with interp, we need to persist the attributes explicitly
    sel_x = xr.Variable(data=x, dims=VECTORIZED_DIM, attrs=ds[X].attrs)
    sel_y = xr.Variable(data=y, dims=VECTORIZED_DIM, attrs=ds[Y].attrs)
    if method == "nearest":
        return ds.sel({X: sel_x, Y: sel_y}, method=method)
    else:
        return ds.interp({X: sel_x, Y: sel_y}, method=method)


def _points_xy(point: shapely.Point | shapely.MultiPoint) -> np.ndarray:
    """Return the query point(s) as an ``(n, 2)`` array in the dataset's CRS."""
    if isinstance(point, shapely.Point):
        return np.array([[point.x, point.y]], dtype="float64")
    if isinstance(point, shapely.MultiPoint):
        return np.array([[p.x, p.y] for p in point.geoms], dtype="float64")
    raise ValueError(
        f"Invalid point type {point.geom_type}, must be Point or MultiPoint",
    )


def _face_indices(mesh_index: MeshIndex, xy: np.ndarray) -> np.ndarray:
    """Containing face for each point, falling back to the nearest centroid outside."""
    faces = np.asarray(mesh_index.containing_faces(xy)).copy()
    outside = faces < 0
    if outside.any():
        faces[outside] = mesh_index.nearest_faces(xy[outside])
    return faces


def _select_faces(
    ds: xr.Dataset,
    target: SelectionTarget,
    mesh_index: MeshIndex,
    xy: np.ndarray,
) -> xr.Dataset:
    """Select the containing (or nearest) face for each point."""
    faces = _face_indices(mesh_index, xy)
    selected = ds.isel({target.dim: xr.Variable(VECTORIZED_DIM, faces)})
    return ensure_xy_coords(selected, target, mesh_index, faces)


def _barycentric_with_fallback(
    mesh_index: MeshIndex,
    xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Barycentric face/weights per point, falling back to nearest node outside the mesh.

    Points that land outside the mesh get the nearest face and a weight of one
    on that face's vertex closest to the point, so linear selection degrades to
    nearest rather than returning NaN.
    """
    faces, weights = mesh_index.barycentric(xy)
    faces = np.asarray(faces).copy()
    outside = faces < 0
    if not outside.any():
        return faces, weights

    logger.warning(
        f"{int(outside.sum())} of {faces.size} query point(s) fell outside the mesh; "
        "falling back to the nearest node",
    )
    rows = np.nonzero(outside)[0]
    fallback_faces = np.asarray(mesh_index.nearest_faces(xy[rows]))
    faces[rows] = fallback_faces

    connectivity = np.asarray(mesh_index.ugrid.face_node_connectivity)
    vertex_xy = mesh_index.node_xy[connectivity[fallback_faces]]
    distances = ((vertex_xy - xy[rows][:, None, :]) ** 2).sum(axis=-1)
    weights[rows] = 0.0
    weights[rows, distances.argmin(axis=1)] = 1.0
    return faces, weights


def _interpolate_nodes(
    ds: xr.Dataset,
    target: SelectionTarget,
    mesh_index: MeshIndex,
    pts: np.ndarray,
    xy: np.ndarray,
) -> xr.Dataset:
    """Barycentrically interpolate node-located variables onto the query points."""
    faces, weights = _barycentric_with_fallback(mesh_index, xy)
    vertices = np.asarray(mesh_index.ugrid.face_node_connectivity)[faces]

    # The node coordinates may have been data variables that ``parameter-name``
    # dropped, in which case the mesh index still remembers how they were described.
    node_attrs = mesh_index.node_coord_attrs
    x_attrs = dict(ds[target.X].attrs) if target.X in ds.variables else dict(node_attrs[0])
    y_attrs = dict(ds[target.Y].attrs) if target.Y in ds.variables else dict(node_attrs[1])

    # Coordinates on the node dimension cannot be meaningfully interpolated;
    # X/Y are replaced by the query point below and the rest are dropped.
    node_coords = [name for name in ds.coords if target.dim in ds[name].dims]
    tri = ds.drop_vars(node_coords).isel(
        {target.dim: xr.Variable((VECTORIZED_DIM, VERTEX_DIM), vertices)},
    )
    weight_var = xr.Variable((VECTORIZED_DIM, VERTEX_DIM), weights)

    data_vars = {}
    for name, da in tri.data_vars.items():
        if VERTEX_DIM not in da.dims:
            data_vars[name] = da
        elif da.dtype.kind in "fiu":
            with xr.set_options(keep_attrs=True):
                data_vars[name] = (da * weight_var).sum(VERTEX_DIM, skipna=False)
        else:
            data_vars[name] = da.isel({VERTEX_DIM: 0}, drop=True)

    out = xr.Dataset(data_vars, attrs=dict(ds.attrs))
    return out.assign_coords(
        {
            target.X: xr.Variable(VECTORIZED_DIM, pts[:, 0], attrs=x_attrs),
            target.Y: xr.Variable(VECTORIZED_DIM, pts[:, 1], attrs=y_attrs),
        },
    )


def _split_by_location(
    ds: xr.Dataset,
    face_target: SelectionTarget,
    mesh: MeshInfo,
) -> tuple[list[str], list[str]]:
    """Split the data variable names into node-located and face-located groups.

    Variables on neither mesh dimension travel with the node group so they are
    passed through exactly once.
    """
    node_names: list[str] = []
    face_names: list[str] = []
    for name in ds.data_vars:
        if face_target.dim in ds[name].dims and variable_location(ds[name], mesh) == "face":
            face_names.append(str(name))
        else:
            node_names.append(str(name))
    return node_names, face_names


def _select_by_position_unstructured(
    ds: xr.Dataset,
    point: shapely.Point | shapely.MultiPoint,
    mesh_index: MeshIndex,
    spatial_ref: SpatialRef,
    method: Literal["nearest", "linear"] = "nearest",
) -> xr.Dataset:
    """Select the query point(s) from a UGRID mesh dataset.

    The point coordinates arrive in the dataset's CRS and are projected into
    the mesh index's index plane before any lookup. Every mesh dimension in
    the result is collapsed to the vectorized ``pts`` dimension, so a single
    point yields ``pts`` of length one, matching the multipoint/area output
    shape.
    """
    mesh = spatial_ref.mesh
    if mesh is None:  # pragma: no cover - guarded by the grid kind
        raise ValueError("Unstructured selection requires a UGRID mesh")

    pts = _points_xy(point)
    xy = mesh_index.project(pts[:, 0], pts[:, 1])

    targets = selection_targets(ds, mesh)
    if not targets:
        raise MeshSelectionError("No mesh-located variables selected")

    if method == "nearest":
        indexers = {}
        selected_indices: list[tuple[SelectionTarget, np.ndarray]] = []
        for target in targets:
            if target.location == "node":
                indices = np.asarray(mesh_index.nearest_nodes(xy))
            else:
                indices = _face_indices(mesh_index, xy)
            indexers[target.dim] = xr.Variable(VECTORIZED_DIM, indices)
            selected_indices.append((target, indices))
        selected = ds.isel(indexers)
        for target, indices in selected_indices:
            selected = ensure_xy_coords(selected, target, mesh_index, indices)
        return selected

    node_target = next((t for t in targets if t.location == "node"), None)
    face_target = next((t for t in targets if t.location == "face"), None)

    if face_target is not None:
        logger.info(
            "Face-located variables are not interpolated; using the value of the containing face",
        )

    if node_target is None:
        return _select_faces(ds, face_target, mesh_index, xy)
    if face_target is None:
        return _interpolate_nodes(ds, node_target, mesh_index, pts, xy)

    node_names, face_names = _split_by_location(ds, face_target, mesh)
    node_part = _interpolate_nodes(ds[node_names], node_target, mesh_index, pts, xy)
    face_part = _select_faces(ds[face_names], face_target, mesh_index, xy)
    shared = {face_target.X, face_target.Y} & {node_target.X, node_target.Y}
    if shared:
        # Without UGRID face coordinates both targets report under the node
        # coordinate names; the node part's query points are the result's X/Y.
        face_part = face_part.drop_vars([n for n in shared if n in face_part.variables])
    merged = xr.merge([node_part, face_part], combine_attrs="override")
    merged.attrs = dict(ds.attrs)
    return merged
