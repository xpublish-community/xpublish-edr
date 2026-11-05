"""UGRID (unstructured mesh) detection helpers.

Mesh detection depends only on cf_xarray, so that collection metadata and
error messages work even when the optional ``xpublish-edr[ugrid]`` extra
(xugrid + numba-celltree) is not installed. Anything that actually indexes
or queries the mesh goes through :func:`_require_xugrid`.
"""

from __future__ import annotations

import itertools
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

# Name given to the mesh topology synthesized for raw FVCOM output, which
# carries no ``mesh_topology`` variable of its own. It never names a variable in
# the source dataset; :func:`_synthetic_topology_name` keeps it unique.
FVCOM_TOPOLOGY_NAME = "fvcom_mesh_topology"

# ``long_name`` FVCOM gives its face-node connectivity, the only description of
# ``nv`` in files that predate (or ignore) the UGRID conventions.
FVCOM_CONNECTIVITY_LONG_NAME = "nodes surrounding element"

# The variable name FVCOM always uses for its face-node connectivity.
FVCOM_CONNECTIVITY_NAME = "nv"


class UgridSupportUnavailable(ImportError):
    """Raised when a UGRID mesh needs to be queried but xugrid is not installed."""


class InvalidMeshError(ValueError):
    """Raised when a detected UGRID mesh's own metadata is internally inconsistent.

    Distinguishes a broken dataset (the mesh :func:`build_grid` was asked to
    build cannot actually be built, e.g. its connectivity references nodes
    that do not exist) from a bad *request*; callers map it to a 500 rather
    than the 404s used for request-level selection errors.
    """


class MeshSelectionError(ValueError):
    """Raised for a user-facing mesh *selection* problem on an otherwise valid mesh.

    Covers cases like a ``parameter-name`` filter that leaves no mesh-located
    variables, or one that mixes node- and face-located parameters in an area
    query. Distinct from :class:`InvalidMeshError` (a broken dataset) and from
    a bare ``ValueError`` (which may not be a request-level problem at all);
    callers map it to a 404.
    """


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
    # The connectivity's declared ``start_index``, or ``None`` when it declares
    # none; see :func:`resolve_start_index`, which infers it from the values.
    start_index: int | None
    structural_vars: frozenset[str]
    # Whether ``topology`` names a variable in the dataset (``False``, the
    # UGRID case) or one this package synthesizes for raw FVCOM output that
    # declares no ``mesh_topology`` variable at all (``True``).
    synthetic: bool = False


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


def _structural_vars(
    ds: xr.Dataset,
    vertex_dim: str,
    topology: str | None = None,
    connectivity: str | None = None,
) -> frozenset[str]:
    """Collect the variables that describe the mesh rather than the data on it.

    ``topology`` is omitted for a synthesized FVCOM mesh, whose topology is not
    a variable of the dataset; ``connectivity`` then names the face-node
    connectivity directly, since no topology variable points at it.
    """
    structural: set[str] = set()
    if topology is not None and topology in ds.variables:
        structural.add(topology)
        for key, value in ds[topology].attrs.items():
            if key.endswith("_connectivity") and isinstance(value, str) and value in ds.variables:
                structural.add(str(value))
    if connectivity is not None and connectivity in ds.variables:
        structural.add(connectivity)
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
        structural_vars=_structural_vars(ds, vertex_dim, topology=topology),
    )


def _synthetic_topology_name(ds: xr.Dataset) -> str:
    """A topology name that does not collide with anything already in ``ds``."""
    name = FVCOM_TOPOLOGY_NAME
    suffix = 1
    while name in ds.variables or name in ds.dims:
        name = f"{FVCOM_TOPOLOGY_NAME}_{suffix}"
        suffix += 1
    return name


def _is_fvcom_source(ds: xr.Dataset) -> bool:
    """Whether the dataset's global ``source`` attribute names FVCOM."""
    return str(ds.attrs.get("source", "")).lower().startswith("fvcom")


def _fvcom_connectivity_candidates(ds: xr.Dataset) -> list[str]:
    """2D integer variables that look like an FVCOM face-node connectivity.

    Raw FVCOM output describes ``nv`` with nothing but a ``long_name``, so both
    the conventional name and that description are accepted.
    """
    candidates = []
    for name, var in ds.variables.items():
        if var.ndim != 2 or not np.issubdtype(var.dtype, np.integer):
            continue
        long_name = var.attrs.get("long_name")
        matches_long_name = (
            isinstance(long_name, str) and long_name.strip().lower() == FVCOM_CONNECTIVITY_LONG_NAME
        )
        if str(name) == FVCOM_CONNECTIVITY_NAME or matches_long_name:
            candidates.append(str(name))
    return candidates


def _fvcom_coordinate_pairs(
    ds: xr.Dataset,
    face_dim: str,
    vertex_dim: str,
) -> tuple[tuple[str, str] | None, str | None, tuple[str, str] | None]:
    """Find the node and face longitude/latitude pairs of a raw FVCOM dataset.

    Returns ``(node_coordinates, node_dim, face_coordinates)``. FVCOM carries
    two CF longitude/latitude pairs (``lon``/``lat`` on the nodes and
    ``lonc``/``latc`` on the elements) distinguishable only by the dimension
    they share.
    """
    try:
        coordinates = ds.cf.coordinates
    except Exception as e:  # pragma: no cover - cf_xarray parsing failure
        logger.debug(f"Could not inspect cf coordinates for an FVCOM mesh: {e}")
        return None, None, None

    lons = [str(n) for n in coordinates.get("longitude", []) if n in ds.variables]
    lats = [str(n) for n in coordinates.get("latitude", []) if n in ds.variables]

    node_coordinates: tuple[str, str] | None = None
    node_dim: str | None = None
    face_coordinates: tuple[str, str] | None = None
    for lon, lat in itertools.product(lons, lats):
        pair = (lon, lat)
        dim = _node_dim(ds, pair)
        if dim is None or dim == vertex_dim:
            continue
        if dim == face_dim:
            if face_coordinates is None:
                face_coordinates = pair
        elif node_coordinates is None:
            node_coordinates, node_dim = pair, dim
    return node_coordinates, node_dim, face_coordinates


def _detect_fvcom(ds: xr.Dataset) -> MeshInfo | None:
    """Recognize raw FVCOM output that declares no UGRID topology variable.

    FVCOM has always written its mesh as a bare ``nv(three, nele)`` connectivity
    plus nodal and elemental longitude/latitude, and files from before the model
    adopted the UGRID conventions (and plenty written since) carry no
    ``mesh_topology`` variable, no ``cf_role`` attributes and no ``location`` or
    ``mesh`` attributes on the data variables. There is still enough to build
    the mesh, so synthesize the topology those files leave out.

    To stay conservative, the dataset must either announce itself as FVCOM in
    its global ``source`` attribute (what xpublish-wms keys off) or name its
    connectivity ``nv``.
    """
    is_fvcom = _is_fvcom_source(ds)
    for connectivity in _fvcom_connectivity_candidates(ds):
        if not is_fvcom and connectivity != FVCOM_CONNECTIVITY_NAME:
            continue

        dims = _face_and_vertex_dims(ds[connectivity], None)
        if dims is None:
            continue
        face_dim, vertex_dim = dims

        node_coordinates, node_dim, face_coordinates = _fvcom_coordinate_pairs(
            ds,
            face_dim,
            vertex_dim,
        )
        if node_coordinates is None or node_dim is None:
            continue

        topology = _synthetic_topology_name(ds)
        logger.info(
            f"Recognized an FVCOM mesh with no UGRID topology variable: "
            f"{connectivity}({vertex_dim}, {face_dim}) over "
            f"{node_coordinates[0]}/{node_coordinates[1]}({node_dim}); "
            f"synthesizing the topology as {topology!r}",
        )
        return MeshInfo(
            topology=topology,
            node_dim=node_dim,
            face_dim=face_dim,
            vertex_dim=vertex_dim,
            node_coordinates=node_coordinates,
            face_coordinates=face_coordinates,
            face_node_connectivity=connectivity,
            # Raw FVCOM never declares one; the values decide (1-based in
            # practice). See :func:`resolve_start_index`.
            start_index=None,
            structural_vars=_structural_vars(ds, vertex_dim, connectivity=connectivity),
            synthetic=True,
        )
    return None


def detect_mesh(ds: xr.Dataset) -> MeshInfo | None:
    """Detect a 2D triangular UGRID mesh in the dataset.

    A UGRID file may declare several topologies (a 1D network alongside a 2D
    mesh, say), and the usable one is not necessarily listed first, so every
    ``mesh_topology`` variable is tried and the first that parses is returned.
    When none does, raw FVCOM output (which declares no topology variable at
    all) is recognized from its connectivity and coordinates
    (:func:`_detect_fvcom`). ``None`` means neither path found a mesh.

    Only cf_xarray is used, so this works without the optional ``ugrid`` extra.
    """
    ds = restore_ugrid_attrs(ds)

    try:
        topologies = [str(name) for name in ds.cf.cf_roles.get("mesh_topology", [])]
    except Exception as e:
        logger.debug(f"Could not inspect cf_roles for UGRID topologies: {e}")
        topologies = []

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

    return _detect_fvcom(ds)


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


def _synthetic_topology_attrs(mesh: MeshInfo) -> dict[str, Any]:
    """The UGRID topology attributes describing a mesh that declares none."""
    attrs: dict[str, Any] = {
        "cf_role": "mesh_topology",
        "topology_dimension": 2,
        "node_coordinates": " ".join(mesh.node_coordinates),
        "face_node_connectivity": mesh.face_node_connectivity,
        "face_dimension": mesh.face_dim,
    }
    if mesh.face_coordinates is not None:
        attrs["face_coordinates"] = " ".join(mesh.face_coordinates)
    return attrs


def _topology_for_xugrid(ds: xr.Dataset, mesh: MeshInfo) -> tuple[xr.Dataset, int]:
    """Return a dataset xugrid can parse the mesh from, plus the start index.

    xugrid reads the topology strictly from attributes, so fill in what UGRID
    allows a file to omit: without ``face_dimension`` it raises on a vertex
    first connectivity that also has face coordinates (and silently builds a
    three-face mesh when it does not), and without ``start_index`` it takes a
    1-based connectivity at face value and addresses nodes past the end of the
    mesh.

    A raw FVCOM mesh has no topology variable at all
    (:func:`_detect_fvcom`); the whole thing is written out here, from the
    :class:`MeshInfo` that detection resolved.

    The connectivity is loaded once here and handed to xugrid as an in-memory
    array, so a dask backed dataset is not read twice. Only a shallow copy is
    mutated; the caller's dataset (and its attrs) are left alone.
    """
    out = restore_ugrid_attrs(ds)
    if out is ds:
        out = ds.copy(deep=False)

    if mesh.topology not in out.variables:
        out[mesh.topology] = xr.DataArray(
            np.int32(0),
            attrs=_synthetic_topology_attrs(mesh),
        )

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
class MeshIndex:
    """A xugrid ``Ugrid2d`` mesh with its spatial indexes built and ready to query.

    The mesh geometry is held in ``index_crs``, which is not necessarily the
    dataset's own ``crs`` (see :func:`_index_crs_for`). Every method takes (or
    returns) coordinates in ``index_crs``; use :meth:`project` or
    :meth:`project_geometry` to get there from dataset coordinates.

    The derived geometry, the node/face coordinate arrays in both the index
    plane (``node_xy``, ``face_xy``) and the dataset CRS (``node_xy_crs``,
    ``face_xy_crs``), is materialized once, in :func:`build_grid`, rather than
    recomputed per access.
    """

    ugrid: Any  # xugrid.Ugrid2d
    mesh: MeshInfo
    crs: pyproj.CRS
    index_crs: pyproj.CRS
    to_index: pyproj.Transformer
    # Node coordinates in the index CRS, shaped ``(n_node, 2)``.
    node_xy: np.ndarray
    # Face centroids in the index CRS, shaped ``(n_face, 2)``. xugrid computes
    # these from the connectivity, not from UGRID ``face_coordinates``.
    face_xy: np.ndarray
    # Node coordinates in the dataset CRS, shaped ``(n_node, 2)``.
    node_xy_crs: np.ndarray
    # Face centroids in the dataset CRS, shaped ``(n_face, 2)``.
    face_xy_crs: np.ndarray
    # Attrs of the two node coordinate variables in the source dataset.
    node_coord_attrs: tuple[dict, dict]
    # Attrs of the two face coordinate variables, or the node attrs when the
    # mesh declares no ``face_coordinates``.
    face_coord_attrs: tuple[dict, dict]
    nbytes: int
    build_seconds: float

    def xy_for(
        self,
        location: Literal["node", "face"],
        idx: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Dataset-CRS X and Y for the given node or face indices."""
        xy = self.node_xy_crs if location == "node" else self.face_xy_crs
        selected = xy[np.asarray(idx)]
        return selected[:, 0], selected[:, 1]

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
        return self.ugrid.node_kdtree.query(xy, workers=-1)[1]

    def containing_faces(self, xy: np.ndarray) -> np.ndarray:
        """Index of the face containing each point, or ``-1`` outside the mesh."""
        return self.ugrid.locate_points(xy)

    def nearest_faces(self, xy: np.ndarray) -> np.ndarray:
        """Index of the face whose centroid is nearest to each point."""
        return self.ugrid.locate_nearest_face(xy)

    def barycentric(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Containing face and barycentric vertex weights for each point.

        Points outside the mesh get a face index of ``-1`` and zero weights.
        """
        faces, weights = self.ugrid.compute_barycentric_weights(xy)
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


def _coord_attrs(ds: xr.Dataset, names: tuple[str, str] | None) -> tuple[dict, dict] | None:
    """Return the attrs of a coordinate name pair, or ``None`` if either is absent."""
    if names is None:
        return None
    if not all(name in ds.variables for name in names):
        return None
    return dict(ds[names[0]].attrs), dict(ds[names[1]].attrs)


def build_grid(ds: xr.Dataset, mesh: MeshInfo, crs: pyproj.CRS) -> MeshIndex:
    """Build (and eagerly index) a xugrid mesh for the dataset.

    Node coordinates and connectivity are read into numpy here, so a dask backed
    dataset is loaded once rather than per request. The cell tree and the node
    and face KDTrees are touched while building so that their construction (and
    numba's JIT compilation) is paid once, at build time, and cached on the
    ``Ugrid2d`` instance.

    The node and face coordinate arrays are materialized here too, in both the
    index plane and the dataset CRS, so that every selection can report where a
    node or face actually is without recomputing them per request; see
    :class:`MeshIndex`.
    """
    xugrid = _require_xugrid()

    started = time.perf_counter()
    topology_ds, start_index = _topology_for_xugrid(ds, mesh)
    grid = xugrid.Ugrid2d.from_dataset(topology_ds, topology=mesh.topology)

    if int(np.asarray(grid.face_node_connectivity).max()) >= grid.n_node:
        raise InvalidMeshError(
            "UGRID connectivity references nodes outside the mesh "
            f"({mesh.face_node_connectivity} read with start_index={start_index}, "
            f"{grid.n_node} nodes)",
        )

    node_x = np.asarray(grid.node_x)
    node_y = np.asarray(grid.node_y)
    # Captured before any reprojection: these are the dataset's own coordinates
    node_xy_crs = np.column_stack([node_x, node_y]).astype("float64", copy=False)
    face_xy_crs = node_xy_crs[np.asarray(grid.face_node_connectivity)].mean(axis=1)

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
    node_xy = np.column_stack([grid.node_x, grid.node_y]).astype("float64", copy=False)
    face_xy = np.column_stack([grid.face_x, grid.face_y]).astype("float64", copy=False)

    node_coord_attrs = _coord_attrs(ds, mesh.node_coordinates) or ({}, {})
    face_coord_attrs = _coord_attrs(ds, mesh.face_coordinates) or node_coord_attrs

    # The celltree and KDTrees roughly triple the footprint of the raw geometry
    nbytes = 3 * int(node_x.nbytes + node_y.nbytes + connectivity.nbytes) + int(
        node_xy.nbytes + face_xy.nbytes + node_xy_crs.nbytes + face_xy_crs.nbytes,
    )

    build_seconds = time.perf_counter() - started
    logger.info(
        f"Built unstructured grid index for {mesh.topology} "
        f"({grid.n_node} nodes, {grid.n_face} faces) in {build_seconds:.3f}s",
    )

    return MeshIndex(
        ugrid=grid,
        mesh=mesh,
        crs=crs,
        index_crs=index_crs,
        to_index=transformer_from_crs(crs_from=crs, crs_to=index_crs),
        node_xy=node_xy,
        face_xy=face_xy,
        node_xy_crs=node_xy_crs,
        face_xy_crs=face_xy_crs,
        node_coord_attrs=node_coord_attrs,
        face_coord_attrs=face_coord_attrs,
        nbytes=nbytes,
        build_seconds=build_seconds,
    )


# Cache keys we have already warned about, so the warning is logged once.
_CACHE_WARNED_KEYS: set[str] = set()


def get_mesh_index(
    ds: xr.Dataset,
    spatial_ref: SpatialRef,
    cache: cachey.Cache | None = None,
) -> MeshIndex:
    """Return the dataset's :class:`MeshIndex`, via xpublish's cache if possible.

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
    mesh_index = cache.get(key)
    if mesh_index is not None:
        return mesh_index

    mesh_index = build_grid(ds, mesh, spatial_ref.crs)
    cache.put(key, mesh_index, cost=max(mesh_index.build_seconds, 1.0), nbytes=mesh_index.nbytes)
    if cache.get(key) is None and key not in _CACHE_WARNED_KEYS:
        _CACHE_WARNED_KEYS.add(key)
        logger.warning(
            f"UGRID grid for {dataset_id} ({mesh_index.nbytes / 1e6:.0f} MB) does not fit in "
            "the xpublish cache; raise cache_kws={'available_bytes': ...} on "
            "xpublish.Rest to avoid rebuilding it per request",
        )
    return mesh_index
