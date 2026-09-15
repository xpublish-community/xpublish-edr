"""UGRID (unstructured mesh) detection helpers.

Mesh detection depends only on cf_xarray, so that collection metadata and
error messages work even when the optional ``xpublish-edr[ugrid]`` extra
(xugrid + numba-celltree) is not installed. Anything that actually indexes or queries the mesh goes through
:func:`_require_xugrid`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cf_xarray  # noqa: F401  (registers the ``.cf`` dataset accessor)
import xarray as xr

from xpublish_edr.logger import logger

UGRID_SUPPORT_MESSAGE = "Dataset uses a UGRID mesh; install xpublish-edr[ugrid] to query it"

#: UGRID reference attributes that ``decode_coords="all"`` may relocate into
#: ``.encoding`` but that xugrid (and our own parsing) only reads from ``.attrs``.
UGRID_TOPOLOGY_ATTRS = (
    "cf_role",
    "topology_dimension",
    "node_coordinates",
    "face_coordinates",
    "face_node_connectivity",
    "face_dimension",
)

#: Number of vertices per face we support (triangular meshes only).
TRIANGLE_VERTICES = 3


class UgridSupportUnavailable(ImportError):
    """Raised when a UGRID mesh needs to be queried but xugrid is not installed."""


def _require_xugrid():
    """Import and return ``xugrid``, or raise :class:`UgridSupportUnavailable`."""
    try:
        import xugrid
    except ImportError as e:
        raise UgridSupportUnavailable(UGRID_SUPPORT_MESSAGE) from e
    return xugrid


def _is_topology_var(var: xr.DataArray | xr.Variable) -> bool:
    """Whether a variable declares ``cf_role = "mesh_topology"`` in attrs or encoding."""
    return (
        var.attrs.get("cf_role") == "mesh_topology"
        or var.encoding.get("cf_role") == "mesh_topology"
    )


def restore_ugrid_attrs(ds: xr.Dataset) -> xr.Dataset:
    """Copy UGRID reference attributes from ``.encoding`` back into ``.attrs``.

    ``xarray.open_dataset(..., decode_coords="all")`` moves ``node_coordinates``
    and friends out of ``.attrs`` and into ``.encoding``. UGRID readers only look
    at ``.attrs``, so mirror the keys back. The dataset is shallow copied and no
    data is touched.
    """
    topologies = [name for name, var in ds.variables.items() if _is_topology_var(var)]
    if not topologies:
        return ds

    out = ds.copy(deep=False)
    for name in topologies:
        topology = out[name]
        restored = {
            key: topology.encoding[key]
            for key in UGRID_TOPOLOGY_ATTRS
            if key not in topology.attrs and key in topology.encoding
        }
        if restored:
            topology.attrs = {**topology.attrs, **restored}

        connectivity = topology.attrs.get("face_node_connectivity")
        if connectivity is not None and connectivity in out.variables:
            conn = out[connectivity]
            if "start_index" not in conn.attrs and "start_index" in conn.encoding:
                conn.attrs = {**conn.attrs, "start_index": conn.encoding["start_index"]}
    return out


@dataclass(frozen=True)
class MeshInfo:
    """Resolved UGRID-1.0 topology metadata for a 2D triangular mesh."""

    topology: str
    node_dim: str
    face_dim: str
    vertex_dim: str
    node_coordinates: tuple[str, str]
    face_coordinates: tuple[str, str] | None
    face_node_connectivity: str
    start_index: int
    structural_vars: frozenset[str]


def _coordinate_pair(ds: xr.Dataset, value) -> tuple[str, str] | None:
    """Parse a whitespace separated UGRID coordinate attribute into a name pair."""
    if not isinstance(value, str):
        return None
    names = value.split()
    if len(names) != 2:
        return None
    if not all(name in ds.variables for name in names):
        return None
    return str(names[0]), str(names[1])


def _node_dim(ds: xr.Dataset, coordinates: tuple[str, str]) -> str | None:
    """Return the shared 1D dimension of the node coordinate pair, if there is one."""
    dims = {ds[name].dims for name in coordinates}
    if len(dims) != 1:
        return None
    (only,) = dims
    if len(only) != 1:
        return None
    return str(only[0])


def _face_and_vertex_dims(conn: xr.DataArray, declared_face_dim) -> tuple[str, str] | None:
    """Resolve ``(face_dim, vertex_dim)`` for a face-node connectivity variable.

    Honors an explicit ``face_dimension`` when it names one of the connectivity
    dimensions; otherwise infers FVCOM's vertex-first ``(three, nele)`` layout.
    """
    dims = [str(d) for d in conn.dims]
    if len(dims) != 2:
        return None

    if isinstance(declared_face_dim, str) and declared_face_dim in dims:
        face_axis = dims.index(declared_face_dim)
    elif conn.shape[0] == TRIANGLE_VERTICES and conn.shape[1] != TRIANGLE_VERTICES:
        face_axis = 1
    else:
        face_axis = 0

    face_dim = dims[face_axis]
    vertex_dim = dims[1 - face_axis]
    if conn.sizes[vertex_dim] != TRIANGLE_VERTICES:
        return None
    return face_dim, vertex_dim


def _structural_vars(ds: xr.Dataset, topology: str, vertex_dim: str) -> frozenset[str]:
    """Collect the variables that describe the mesh rather than the data on it."""
    structural = {topology}
    topology_attrs = ds[topology].attrs
    for key, value in topology_attrs.items():
        if key.endswith("_connectivity") and isinstance(value, str) and value in ds.variables:
            structural.add(str(value))
    for name, var in ds.variables.items():
        cf_role = var.attrs.get("cf_role")
        if isinstance(cf_role, str) and cf_role.endswith("_connectivity"):
            structural.add(str(name))
    for name in ds.data_vars:
        if vertex_dim in ds[name].dims:
            structural.add(str(name))
    return frozenset(structural)


def detect_mesh(ds: xr.Dataset) -> MeshInfo | None:
    """Detect a 2D triangular UGRID mesh in the dataset.

    Returns ``None`` when the dataset carries no usable mesh topology. Only
    cf_xarray is used, so this works without the optional ``ugrid`` extra.
    """
    ds = restore_ugrid_attrs(ds)

    try:
        topologies = list(ds.cf.cf_roles.get("mesh_topology", []))
    except Exception as e:
        logger.debug(f"Could not inspect cf_roles for UGRID topologies: {e}")
        return None
    if not topologies:
        return None

    topology = str(topologies[0])
    attrs = ds[topology].attrs

    try:
        topology_dimension = int(attrs.get("topology_dimension", 0))
    except (TypeError, ValueError):
        return None
    if topology_dimension != 2:
        return None

    node_coordinates = _coordinate_pair(ds, attrs.get("node_coordinates"))
    if node_coordinates is None:
        return None
    node_dim = _node_dim(ds, node_coordinates)
    if node_dim is None:
        return None

    face_coordinates = _coordinate_pair(ds, attrs.get("face_coordinates"))

    connectivity = attrs.get("face_node_connectivity")
    if not isinstance(connectivity, str) or connectivity not in ds.variables:
        return None
    conn = ds[connectivity]

    dims = _face_and_vertex_dims(conn, attrs.get("face_dimension"))
    if dims is None:
        return None
    face_dim, vertex_dim = dims

    try:
        start_index = int(conn.attrs.get("start_index", 0))
    except (TypeError, ValueError):
        start_index = 0

    return MeshInfo(
        topology=topology,
        node_dim=node_dim,
        face_dim=face_dim,
        vertex_dim=vertex_dim,
        node_coordinates=node_coordinates,
        face_coordinates=face_coordinates,
        face_node_connectivity=str(connectivity),
        start_index=start_index,
        structural_vars=_structural_vars(ds, topology, vertex_dim),
    )


def variable_location(
    var: xr.DataArray,
    mesh: MeshInfo,
) -> Literal["node", "face"] | None:
    """Return whether a variable lives on mesh nodes or faces.

    Prefers the UGRID ``location`` attribute and falls back to the variable's
    dimensions. The ``mesh`` attribute is deliberately never consulted: FVCOM
    output routinely names a mesh variable that does not exist.
    """
    location = var.attrs.get("location")
    if location in ("node", "face"):
        return location
    if mesh.node_dim in var.dims:
        return "node"
    if mesh.face_dim in var.dims:
        return "face"
    return None
