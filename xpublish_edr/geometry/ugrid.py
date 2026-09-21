"""UGRID (unstructured mesh) detection helpers.

Mesh detection depends only on cf_xarray, so that collection metadata and
error messages work even when the optional ``xpublish-edr[ugrid]`` extra
(xugrid + numba-celltree) is not installed. Anything that actually indexes or queries the mesh goes through
:func:`_require_xugrid`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import cf_xarray  # noqa: F401  (registers the ``.cf`` dataset accessor)
import numpy as np
import pyproj
import shapely
import xarray as xr
from xpublish.utils.api import DATASET_ID_ATTR_KEY

from xpublish_edr.geometry.proj import transformer_from_crs
from xpublish_edr.logger import logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import cachey

    from xpublish_edr.geometry.common import SpatialRef

UGRID_SUPPORT_MESSAGE = "Dataset uses a UGRID mesh; install xpublish-edr[ugrid] to query it"

# UGRID reference attributes that ``decode_coords="all"`` may relocate into
# ``.encoding`` but that xugrid (and our own parsing) only reads from ``.attrs``.
UGRID_TOPOLOGY_ATTRS = (
    "cf_role",
    "topology_dimension",
    "node_coordinates",
    "face_coordinates",
    "face_node_connectivity",
    "face_dimension",
)

# Number of vertices per face we support (triangular meshes only).
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
    #: The connectivity's declared ``start_index``, or ``None`` when it declares
    #: none; see :func:`resolve_start_index`, which infers it from the values.
    start_index: int | None
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


def _parse_topology(ds: xr.Dataset, topology: str) -> MeshInfo | None:
    """Parse one mesh topology variable, or ``None`` if it is not a 2D triangular mesh.

    No data is read: the connectivity's ``start_index`` is taken from its attrs
    and left as ``None`` when it declares none.
    """
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

    declared = conn.attrs.get("start_index")
    try:
        start_index = None if declared is None else int(declared)
    except (TypeError, ValueError):
        start_index = None

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


def detect_mesh(ds: xr.Dataset) -> MeshInfo | None:
    """Detect a 2D triangular UGRID mesh in the dataset.

    A UGRID file may declare several topologies (a 1D network alongside a 2D
    mesh, say), and the usable one is not necessarily listed first, so every
    ``mesh_topology`` variable is tried and the first that parses is returned.
    ``None`` means none of them describes a 2D triangular mesh.

    Only cf_xarray is used, so this works without the optional ``ugrid`` extra,
    and no data is loaded -- it runs on every metadata request.
    """
    ds = restore_ugrid_attrs(ds)

    try:
        topologies = [str(name) for name in ds.cf.cf_roles.get("mesh_topology", [])]
    except Exception as e:
        logger.debug(f"Could not inspect cf_roles for UGRID topologies: {e}")
        return None

    for topology in topologies:
        mesh = _parse_topology(ds, topology)
        if mesh is None:
            continue
        if len(topologies) > 1:
            logger.debug(
                f"Using UGRID topology {topology!r} of {topologies} "
                "(the first that describes a 2D triangular mesh)",
            )
        return mesh
    return None


def resolve_start_index(
    conn_values: np.ndarray,
    n_node: int,
    declared: int | None,
) -> int:
    """Resolve the base index of a face-node connectivity array.

    UGRID lets a connectivity variable declare ``start_index``, but real FVCOM
    output routinely ships a 1-based ``nv`` without it. When nothing is
    declared, infer: a connectivity whose (non fill) values run from ``1`` to
    ``n_node`` can only be 1-based, since a 0-based one would address a node
    that does not exist.
    """
    if declared is not None:
        return declared

    values = conn_values[conn_values >= 0]
    if values.size and int(values.min()) == 1 and int(values.max()) == n_node:
        logger.info(
            "UGRID face_node_connectivity declares no start_index; "
            f"values run 1..{n_node}, so it is read as 1-based",
        )
        return 1
    return 0


def _topology_for_xugrid(ds: xr.Dataset, mesh: MeshInfo) -> tuple[xr.Dataset, int]:
    """Return a dataset xugrid can parse the mesh from, plus the start index.

    xugrid reads the topology strictly from attributes, so fill in what UGRID
    allows a file to omit: without ``face_dimension`` it raises on a vertex
    first connectivity that also has face coordinates (and silently builds a
    three-face mesh when it does not), and without ``start_index`` it takes a
    1-based connectivity at face value and addresses nodes past the end of the
    mesh.

    The connectivity is loaded once here and handed to xugrid as an in-memory
    array, so a dask backed dataset is not read twice. Only a shallow copy is
    mutated; the caller's dataset (and its attrs) are left alone.
    """
    out = restore_ugrid_attrs(ds)
    if out is ds:
        out = ds.copy(deep=False)

    topology = out[mesh.topology]
    if "face_dimension" not in topology.attrs:
        topology.attrs = {**topology.attrs, "face_dimension": mesh.face_dim}

    name = mesh.face_node_connectivity
    values = np.asarray(ds[name].values)
    start_index = resolve_start_index(values, int(ds.sizes[mesh.node_dim]), mesh.start_index)

    connectivity = out[name].copy(data=values)
    connectivity.attrs = {**connectivity.attrs, "start_index": start_index}
    out[name] = connectivity
    return out, start_index


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


@dataclass
class IndexedGrid:
    """A xugrid ``Ugrid2d`` mesh with its spatial indexes built and ready to query.

    The mesh geometry is held in ``index_crs``, which is not necessarily the
    dataset's own ``crs`` (see :func:`_index_crs_for`). Every method takes (or
    returns) coordinates in ``index_crs``; use :meth:`project` or
    :meth:`project_geometry` to get there from dataset coordinates.
    """

    grid: Any  # xugrid.Ugrid2d
    mesh: MeshInfo
    crs: pyproj.CRS
    index_crs: pyproj.CRS
    to_index: pyproj.Transformer
    nbytes: int
    build_seconds: float

    @property
    def node_xy(self) -> np.ndarray:
        """Mesh node coordinates in the index CRS, shaped ``(n_node, 2)``."""
        return np.column_stack([self.grid.node_x, self.grid.node_y])

    @property
    def face_xy(self) -> np.ndarray:
        """Mesh face centroids in the index CRS, shaped ``(n_face, 2)``.

        xugrid computes these from the connectivity rather than reading the
        UGRID ``face_coordinates`` variables.
        """
        return np.column_stack([self.grid.face_x, self.grid.face_y])

    def project(self, x, y) -> np.ndarray:
        """Project dataset-CRS coordinate arrays into the index plane."""
        index_x, index_y = self.to_index.transform(np.asarray(x), np.asarray(y))
        return np.column_stack([index_x, index_y])

    def project_geometry(self, geometry: shapely.Geometry) -> shapely.Geometry:
        """Project a dataset-CRS geometry into the index plane."""

        def _transform(coords: np.ndarray) -> np.ndarray:
            """Vectorized callback for shapely.transform."""
            x, y = self.to_index.transform(coords[:, 0], coords[:, 1])
            return np.column_stack([x, y])

        return shapely.transform(geometry, _transform)

    def nearest_nodes(self, xy: np.ndarray) -> np.ndarray:
        """Index of the nearest mesh node for each point (KDTree, index plane)."""
        return self.grid.node_kdtree.query(xy, workers=-1)[1]

    def containing_faces(self, xy: np.ndarray) -> np.ndarray:
        """Index of the face containing each point, or ``-1`` outside the mesh."""
        return self.grid.locate_points(xy)

    def nearest_faces(self, xy: np.ndarray) -> np.ndarray:
        """Index of the face whose centroid is nearest to each point."""
        return self.grid.locate_nearest_face(xy)

    def barycentric(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Containing face and barycentric vertex weights for each point.

        Points outside the mesh get a face index of ``-1`` and zero weights.
        """
        faces, weights = self.grid.compute_barycentric_weights(xy)
        faces = np.asarray(faces)
        weights = np.asarray(weights, dtype="float64").copy()
        weights[faces < 0] = 0.0
        return faces, weights


def _index_crs_for(
    grid_bounds: tuple[float, float, float, float],
    crs: pyproj.CRS,
) -> pyproj.CRS:
    """Pick the CRS the mesh geometry should be indexed in.

    xugrid's KDTrees and cell tree are planar: they measure distance in whatever
    units ``node_x``/``node_y`` hold. In degrees that is wrong by the cosine of
    the latitude (in the Gulf of Maine a degree of longitude is only ~0.73 of
    a degree of latitude) so "nearest" in degree space can pick the wrong node.

    For a geographic CRS whose mesh spans less than 180 degrees of longitude we
    therefore index in a local azimuthal equidistant projection centred on the
    mesh, which makes those distances true metres. Projected CRSs (already
    metric) and near-global geographic meshes (where no single local projection
    helps) are indexed as they are.
    """
    if not crs.is_geographic:
        return crs

    min_x, min_y, max_x, max_y = grid_bounds
    if not np.isfinite([min_x, min_y, max_x, max_y]).all():
        return crs
    if (max_x - min_x) >= 180.0:
        return crs

    lon_0 = (min_x + max_x) / 2.0
    lat_0 = (min_y + max_y) / 2.0
    return pyproj.CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat_0} +lon_0={lon_0} {_ellipsoid_proj4(crs)} +units=m +no_defs",
    )


def _ellipsoid_proj4(crs: pyproj.CRS) -> str:
    """Return proj4 ellipsoid parameters for ``crs``, defaulting to WGS84."""
    try:
        ellipsoid = crs.ellipsoid
        semi_major = ellipsoid.semi_major_metre
        inverse_flattening = ellipsoid.inverse_flattening
    except Exception:  # pragma: no cover - exotic/undefined ellipsoids
        return "+datum=WGS84"
    if not semi_major or not inverse_flattening:
        return "+datum=WGS84"
    return f"+a={semi_major} +rf={inverse_flattening}"


def build_grid(ds: xr.Dataset, mesh: MeshInfo, crs: pyproj.CRS) -> IndexedGrid:
    """Build (and eagerly index) a xugrid mesh for the dataset.

    Node coordinates and connectivity are read into numpy here, so a dask backed
    dataset is loaded once rather than per request. The cell tree and the node
    and face KDTrees are touched while building so that their construction (and
    numba's JIT compilation) is paid once, at build time, and cached on the
    ``Ugrid2d`` instance.
    """
    xugrid = _require_xugrid()

    started = time.perf_counter()
    topology_ds, start_index = _topology_for_xugrid(ds, mesh)
    grid = xugrid.Ugrid2d.from_dataset(topology_ds, topology=mesh.topology)

    if int(np.asarray(grid.face_node_connectivity).max()) >= grid.n_node:
        raise ValueError(
            "UGRID connectivity references nodes outside the mesh "
            f"({mesh.face_node_connectivity} read with start_index={start_index}, "
            f"{grid.n_node} nodes)",
        )

    node_x = np.asarray(grid.node_x)
    node_y = np.asarray(grid.node_y)
    bounds = (
        float(node_x.min()),
        float(node_y.min()),
        float(node_x.max()),
        float(node_y.max()),
    )
    index_crs = _index_crs_for(bounds, crs)
    if index_crs != crs:
        grid.set_crs(crs, allow_override=True)
        # ``to_crs`` returns a reprojected copy, but tolerate an in place variant
        projected = grid.to_crs(index_crs)
        if projected is not None:
            grid = projected

    for index in ("celltree", "node_kdtree", "face_kdtree"):
        index_started = time.perf_counter()
        getattr(grid, index)
        logger.debug(f"Built UGRID {index} in {time.perf_counter() - index_started:.3f}s")

    connectivity = np.asarray(grid.face_node_connectivity)
    # The celltree and KDTrees roughly triple the footprint of the raw geometry
    nbytes = 3 * int(node_x.nbytes + node_y.nbytes + connectivity.nbytes)

    build_seconds = time.perf_counter() - started
    logger.info(
        f"Built unstructured grid index for {mesh.topology} "
        f"({grid.n_node} nodes, {grid.n_face} faces) in {build_seconds:.3f}s",
    )

    return IndexedGrid(
        grid=grid,
        mesh=mesh,
        crs=crs,
        index_crs=index_crs,
        to_index=transformer_from_crs(crs_from=crs, crs_to=index_crs),
        nbytes=nbytes,
        build_seconds=build_seconds,
    )


# Cache keys we have already warned about, so the warning is logged once.
_CACHE_WARNED_KEYS: set[str] = set()


def get_indexed_grid(
    ds: xr.Dataset,
    spatial_ref: SpatialRef,
    cache: cachey.Cache | None = None,
) -> IndexedGrid:
    """Return the dataset's :class:`IndexedGrid`, via xpublish's cache if possible.

    Building the index is expensive (the NECOFS mesh is on the order of 20 MB),
    so it is stored in the application cache keyed on the dataset id and the
    mesh sizes. Without a cache, or without an ``_xpublish_id`` to key on, the
    grid is rebuilt per call.
    """
    mesh = spatial_ref.mesh
    if mesh is None:
        raise ValueError("Cannot build an unstructured grid: no UGRID mesh was detected")

    dataset_id = ds.attrs.get(DATASET_ID_ATTR_KEY)
    if cache is None or dataset_id is None:
        logger.debug(
            "Building an uncached unstructured grid "
            f"(cache={cache is not None}, {DATASET_ID_ATTR_KEY}={dataset_id!r})",
        )
        return build_grid(ds, mesh, spatial_ref.crs)

    key = (
        f"{dataset_id}/edr/ugrid/{mesh.topology}"
        f"/{ds.sizes.get(mesh.node_dim, 0)}/{ds.sizes.get(mesh.face_dim, 0)}"
    )
    grid = cache.get(key)
    if grid is not None:
        return grid

    grid = build_grid(ds, mesh, spatial_ref.crs)
    cache.put(key, grid, cost=max(grid.build_seconds, 1.0), nbytes=grid.nbytes)
    if cache.get(key) is None and key not in _CACHE_WARNED_KEYS:
        _CACHE_WARNED_KEYS.add(key)
        logger.warning(
            f"UGRID grid for {dataset_id} ({grid.nbytes / 1e6:.0f} MB) does not fit in "
            "the xpublish cache; raise cache_kws={'available_bytes': ...} on "
            "xpublish.Rest to avoid rebuilding it per request",
        )
    return grid
