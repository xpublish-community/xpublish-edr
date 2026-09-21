"""Handle selection and formatting for position queries"""

from __future__ import annotations

from typing import Literal

import numpy as np
import shapely
import xarray as xr

from xpublish_edr.geometry.common import (
    VECTORIZED_DIM,
    GridKind,
    SelectionTarget,
    SpatialRef,
    ensure_xy_coords,
    prepare_spatial_grid,
    selection_targets,
)
from xpublish_edr.geometry.ugrid import IndexedGrid, MeshInfo, variable_location
from xpublish_edr.logger import logger

# Temporary dimension holding the three vertices of each containing triangle.
VERTEX_DIM = "_vertex"


def select_by_position(
    ds: xr.Dataset,
    point: shapely.Point | shapely.MultiPoint,
    method: Literal["nearest", "linear"] = "nearest",
    spatial_ref: SpatialRef | None = None,
    grid: IndexedGrid | None = None,
) -> xr.Dataset:
    """
    Return a dataset with the position nearest to the given coordinates

    ``grid`` is the mesh index for an unstructured dataset, already built (and
    cached) by the caller; it is passed through so it is not rebuilt here.
    """
    prepared = prepare_spatial_grid(
        ds,
        spatial_ref=spatial_ref,
        require_selectable=True,
        grid=grid,
    )
    ds = prepared.ds

    if prepared.kind is GridKind.UNSTRUCTURED:
        if prepared.grid is None:
            raise ValueError("Unstructured grid index was not built")
        return _select_by_position_unstructured(
            ds,
            point,
            prepared.grid,
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


def _face_indices(grid: IndexedGrid, xy: np.ndarray) -> np.ndarray:
    """Containing face for each point, falling back to the nearest centroid outside."""
    faces = np.asarray(grid.containing_faces(xy)).copy()
    outside = faces < 0
    if outside.any():
        faces[outside] = grid.nearest_faces(xy[outside])
    return faces


def _select_faces(
    ds: xr.Dataset,
    target: SelectionTarget,
    grid: IndexedGrid,
    xy: np.ndarray,
) -> xr.Dataset:
    """Select the containing (or nearest) face for each point."""
    faces = _face_indices(grid, xy)
    selected = ds.isel({target.dim: xr.Variable(VECTORIZED_DIM, faces)})
    return ensure_xy_coords(selected, target, grid, faces)


def _barycentric_with_fallback(
    grid: IndexedGrid,
    xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Barycentric face/weights per point, falling back to nearest node outside the mesh.

    Points that land outside the mesh get the nearest face and a weight of one
    on that face's vertex closest to the point, so linear selection degrades to
    nearest rather than returning NaN.
    """
    faces, weights = grid.barycentric(xy)
    faces = np.asarray(faces).copy()
    outside = faces < 0
    if not outside.any():
        return faces, weights

    logger.warning(
        f"{int(outside.sum())} of {faces.size} query point(s) fell outside the mesh; "
        "falling back to the nearest node",
    )
    rows = np.nonzero(outside)[0]
    fallback_faces = np.asarray(grid.nearest_faces(xy[rows]))
    faces[rows] = fallback_faces

    connectivity = np.asarray(grid.grid.face_node_connectivity)
    vertex_xy = grid.node_xy[connectivity[fallback_faces]]
    distances = ((vertex_xy - xy[rows][:, None, :]) ** 2).sum(axis=-1)
    weights[rows] = 0.0
    weights[rows, distances.argmin(axis=1)] = 1.0
    return faces, weights


def _interpolate_nodes(
    ds: xr.Dataset,
    target: SelectionTarget,
    grid: IndexedGrid,
    pts: np.ndarray,
    xy: np.ndarray,
) -> xr.Dataset:
    """Barycentrically interpolate node-located variables onto the query points."""
    faces, weights = _barycentric_with_fallback(grid, xy)
    vertices = np.asarray(grid.grid.face_node_connectivity)[faces]

    # The node coordinates may have been data variables that ``parameter-name``
    # dropped, in which case the grid still remembers how they were described.
    node_attrs = grid.node_coord_attrs
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
    grid: IndexedGrid,
    spatial_ref: SpatialRef,
    method: Literal["nearest", "linear"] = "nearest",
) -> xr.Dataset:
    """Select the query point(s) from a UGRID mesh dataset.

    The point coordinates arrive in the dataset's CRS and are projected into the
    grid's index plane before any lookup. Every mesh dimension in the result is
    collapsed to the vectorized ``pts`` dimension, so a single point yields
    ``pts`` of length one, matching the multipoint/area output shape.
    """
    mesh = spatial_ref.mesh
    if mesh is None:  # pragma: no cover - guarded by the grid kind
        raise ValueError("Unstructured selection requires a UGRID mesh")

    pts = _points_xy(point)
    xy = grid.project(pts[:, 0], pts[:, 1])

    targets = selection_targets(ds, mesh)
    if not targets:
        raise ValueError("No mesh-located variables selected")

    if method == "nearest":
        indexers = {}
        selected_indices: list[tuple[SelectionTarget, np.ndarray]] = []
        for target in targets:
            if target.location == "node":
                indices = np.asarray(grid.nearest_nodes(xy))
            else:
                indices = _face_indices(grid, xy)
            indexers[target.dim] = xr.Variable(VECTORIZED_DIM, indices)
            selected_indices.append((target, indices))
        selected = ds.isel(indexers)
        for target, indices in selected_indices:
            selected = ensure_xy_coords(selected, target, grid, indices)
        return selected

    node_target = next((t for t in targets if t.location == "node"), None)
    face_target = next((t for t in targets if t.location == "face"), None)

    if face_target is not None:
        logger.info(
            "Face-located variables are not interpolated; using the value of the containing face",
        )

    if node_target is None:
        return _select_faces(ds, face_target, grid, xy)
    if face_target is None:
        return _interpolate_nodes(ds, node_target, grid, pts, xy)

    node_names, face_names = _split_by_location(ds, face_target, mesh)
    node_part = _interpolate_nodes(ds[node_names], node_target, grid, pts, xy)
    face_part = _select_faces(ds[face_names], face_target, grid, xy)
    shared = {face_target.X, face_target.Y} & {node_target.X, node_target.Y}
    if shared:
        # Without UGRID face coordinates both targets report under the node
        # coordinate names; the node part's query points are the result's X/Y.
        face_part = face_part.drop_vars([n for n in shared if n in face_part.variables])
    merged = xr.merge([node_part, face_part], combine_attrs="override")
    merged.attrs = dict(ds.attrs)
    return merged
