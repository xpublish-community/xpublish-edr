"""Tests for UGRID (FVCOM-style unstructured mesh) detection and X/Y resolution."""

import dataclasses
import logging
import sys

import cachey
import numpy as np
import pandas as pd
import pyproj
import pytest
import shapely
import xarray as xr
from xpublish.utils.api import DATASET_ID_ATTR_KEY

from xpublish_edr.geometry import ugrid as ugrid_module
from xpublish_edr.geometry.common import (
    GridKind,
    dataset_spatial_ref,
    grid_kind,
    prepare_spatial_grid,
)
from xpublish_edr.geometry.ugrid import (
    IndexedGrid,
    UgridSupportUnavailable,
    _index_crs_for,
    _require_xugrid,
    build_grid,
    detect_mesh,
    get_indexed_grid,
    restore_ugrid_attrs,
    variable_location,
)
from xpublish_edr.metadata import collection_metadata

UGRID_KEYS = (
    "cf_role",
    "topology_dimension",
    "node_coordinates",
    "face_coordinates",
    "face_node_connectivity",
    "face_dimension",
)


def _triangles(side: int) -> np.ndarray:
    """Split a ``side x side`` node lattice into ``2 * (side - 1) ** 2`` triangles."""
    tris = []
    for i in range(side - 1):
        for j in range(side - 1):
            n0 = i * side + j
            n1 = i * side + j + 1
            n2 = (i + 1) * side + j
            n3 = (i + 1) * side + j + 1
            tris.append((n0, n1, n3))
            tris.append((n0, n3, n2))
    return np.array(tris, dtype="int32")


def make_fvcom_dataset(
    *,
    side: int = 5,
    start_index: int = 1,
    face_dimension: bool = True,
    dask: bool = False,
    attrs_in_encoding: bool = False,
) -> xr.Dataset:
    """Build a small FVCOM-flavored UGRID-1.0 dataset.

    The mesh is a ``side x side`` lattice of nodes over lon -70..-69 / lat 43..44,
    split into triangles. ``nv`` is transposed (vertex dimension first) and
    optionally 1-based, exactly like real FVCOM output.
    """
    lon_1d = np.linspace(-70.0, -69.0, side)
    lat_1d = np.linspace(43.0, 44.0, side)
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
    lon = lon_2d.ravel()
    lat = lat_2d.ravel()

    tris = _triangles(side)
    n_faces = tris.shape[0]
    lonc = lon[tris].mean(axis=1)
    latc = lat[tris].mean(axis=1)
    nv = (tris + start_index).T.astype("int32")

    time = pd.date_range("2024-01-01", periods=4, freq="h")
    t_index = np.arange(time.size, dtype="float32")

    siglay = np.repeat(np.linspace(-1.0 / 6.0, -5.0 / 6.0, 3)[:, None], lon.size, axis=1)

    zeta = (2.0 * lon + 3.0 * lat + 1.0)[None, :] + 0.1 * t_index[:, None]
    u = np.arange(n_faces, dtype="float32")[None, :] + 0.1 * t_index[:, None]

    topology_attrs = {
        "cf_role": "mesh_topology",
        "topology_dimension": 2,
        "node_coordinates": "lon lat",
        "face_coordinates": "lonc latc",
        "face_node_connectivity": "nv",
    }
    if face_dimension:
        topology_attrs["face_dimension"] = "nele"

    ds = xr.Dataset(
        data_vars={
            "nv": (
                ("three", "nele"),
                nv,
                {"cf_role": "face_node_connectivity", "start_index": start_index},
            ),
            "nbe": (("three", "nele"), np.zeros((3, n_faces), dtype="int32")),
            "mesh_topology": ((), np.int32(0), topology_attrs),
            "zeta": (
                ("time", "node"),
                zeta.astype("float32"),
                {
                    "mesh": "mesh_topology",
                    "location": "node",
                    "standard_name": "sea_surface_height",
                    "units": "m",
                },
            ),
            "u": (
                ("time", "nele"),
                u.astype("float32"),
                {"mesh": "fvcom_mesh", "location": "face", "units": "m s-1"},
            ),
            "h": (("node",), (10.0 + lat).astype("float32"), {"location": "node"}),
        },
        coords={
            "lon": (
                ("node",),
                lon,
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
            "lat": (
                ("node",),
                lat,
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
            "x": (("node",), lon * 1e5, {"units": "meters"}),
            "y": (("node",), lat * 1e5, {"units": "meters"}),
            "lonc": (
                ("nele",),
                lonc,
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
            "latc": (
                ("nele",),
                latc,
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
            "siglay": (
                ("siglay", "node"),
                siglay,
                {"standard_name": "ocean_sigma_coordinate", "positive": "up"},
            ),
            "time": (("time",), time),
        },
        attrs={"Conventions": "CF-1.11, UGRID-1.0", "source": "FVCOM_4.4.1"},
    )

    if dask:
        ds = ds.chunk({"node": 7, "nele": 9, "time": 2})

    if attrs_in_encoding:
        topology = ds["mesh_topology"]
        moved = {k: topology.attrs.pop(k) for k in UGRID_KEYS if k in topology.attrs}
        topology.encoding.update(moved)
        ds["nv"].encoding["start_index"] = ds["nv"].attrs.pop("start_index")

    return ds


def make_quad_mesh_dataset() -> xr.Dataset:
    """Build a UGRID dataset whose faces have four vertices (unsupported)."""
    lon = np.array([0.0, 1.0, 0.0, 1.0])
    lat = np.array([0.0, 0.0, 1.0, 1.0])
    return xr.Dataset(
        data_vars={
            "quad_conn": (
                ("nface", "nmax_face"),
                np.array([[0, 1, 3, 2]], dtype="int32"),
                {"cf_role": "face_node_connectivity", "start_index": 0},
            ),
            "mesh": (
                (),
                np.int32(0),
                {
                    "cf_role": "mesh_topology",
                    "topology_dimension": 2,
                    "node_coordinates": "lon lat",
                    "face_node_connectivity": "quad_conn",
                    "face_dimension": "nface",
                },
            ),
        },
        coords={
            "lon": (("nnode",), lon, {"standard_name": "longitude", "units": "degrees_east"}),
            "lat": (("nnode",), lat, {"standard_name": "latitude", "units": "degrees_north"}),
        },
    )


def make_curvilinear_dataset() -> xr.Dataset:
    """Build the 2D (curvilinear) lon/lat dataset used in ``tests/test_crs.py``."""
    lon = np.array([[10.0, 11.0, 12.0], [10.0, 11.0, 12.0]])
    lat = np.array([[40.0, 40.0, 40.0], [41.0, 41.0, 41.0]])
    return xr.Dataset(
        {"foo": (("y", "x"), np.arange(6.0).reshape(2, 3))},
        coords={
            "lon": (("y", "x"), lon, {"standard_name": "longitude", "units": "degrees_east"}),
            "lat": (("y", "x"), lat, {"standard_name": "latitude", "units": "degrees_north"}),
        },
    )


def make_triangle_dataset(lon: np.ndarray, lat: np.ndarray) -> xr.Dataset:
    """Build a minimal UGRID dataset with a single triangular face."""
    return xr.Dataset(
        data_vars={
            "nv": (
                ("three", "nele"),
                np.array([[0], [1], [2]], dtype="int32"),
                {"cf_role": "face_node_connectivity", "start_index": 0},
            ),
            "mesh_topology": (
                (),
                np.int32(0),
                {
                    "cf_role": "mesh_topology",
                    "topology_dimension": 2,
                    "node_coordinates": "lon lat",
                    "face_node_connectivity": "nv",
                    "face_dimension": "nele",
                },
            ),
            "zeta": (("node",), np.arange(lon.size, dtype="float32"), {"location": "node"}),
        },
        coords={
            "lon": (
                ("node",),
                lon,
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
            "lat": (
                ("node",),
                lat,
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
        },
        attrs={"Conventions": "CF-1.11, UGRID-1.0"},
    )


@pytest.fixture
def fvcom_dataset() -> xr.Dataset:
    """A default 5x5-node FVCOM-style UGRID dataset."""
    return make_fvcom_dataset()


@pytest.fixture
def fvcom_grid(fvcom_dataset) -> IndexedGrid:
    """An :class:`IndexedGrid` built from the default FVCOM fixture."""
    spatial_ref = dataset_spatial_ref(fvcom_dataset)
    assert spatial_ref.mesh is not None
    return build_grid(fvcom_dataset, spatial_ref.mesh, spatial_ref.crs)


def test_restore_ugrid_attrs_from_encoding():
    """UGRID keys stashed in ``.encoding`` are copied back into ``.attrs``."""
    ds = make_fvcom_dataset(attrs_in_encoding=True)
    assert "node_coordinates" not in ds["mesh_topology"].attrs
    assert "start_index" not in ds["nv"].attrs

    restored = restore_ugrid_attrs(ds)

    assert restored["mesh_topology"].attrs["cf_role"] == "mesh_topology"
    assert restored["mesh_topology"].attrs["node_coordinates"] == "lon lat"
    assert restored["mesh_topology"].attrs["face_coordinates"] == "lonc latc"
    assert restored["mesh_topology"].attrs["face_node_connectivity"] == "nv"
    assert restored["mesh_topology"].attrs["face_dimension"] == "nele"
    assert restored["mesh_topology"].attrs["topology_dimension"] == 2
    assert restored["nv"].attrs["start_index"] == 1
    # The source dataset is untouched
    assert "node_coordinates" not in ds["mesh_topology"].attrs
    # Data is shared, not copied
    np.testing.assert_array_equal(restored["zeta"].values, ds["zeta"].values)


def test_restore_ugrid_attrs_noop_when_attrs_present(fvcom_dataset):
    """A dataset that already carries the UGRID attrs is returned unchanged."""
    restored = restore_ugrid_attrs(fvcom_dataset)
    xr.testing.assert_identical(restored, fvcom_dataset)


def test_detect_mesh(fvcom_dataset):
    """A well-formed FVCOM dataset yields the expected ``MeshInfo``."""
    mesh = detect_mesh(fvcom_dataset)
    assert mesh is not None
    assert mesh.topology == "mesh_topology"
    assert mesh.node_dim == "node"
    assert mesh.face_dim == "nele"
    assert mesh.vertex_dim == "three"
    assert mesh.node_coordinates == ("lon", "lat")
    assert mesh.face_coordinates == ("lonc", "latc")
    assert mesh.face_node_connectivity == "nv"
    assert mesh.start_index == 1
    assert mesh.structural_vars == frozenset({"mesh_topology", "nv", "nbe"})


def test_detect_mesh_zero_based_start_index():
    """``start_index`` defaults to (and is read as) 0 when the mesh is 0-based."""
    mesh = detect_mesh(make_fvcom_dataset(start_index=0))
    assert mesh is not None
    assert mesh.start_index == 0


def test_detect_mesh_infers_fvcom_layout_without_face_dimension():
    """Without ``face_dimension`` the vertex-first FVCOM layout is inferred."""
    mesh = detect_mesh(make_fvcom_dataset(face_dimension=False))
    assert mesh is not None
    assert mesh.face_dim == "nele"
    assert mesh.vertex_dim == "three"


def test_detect_mesh_from_encoding_only():
    """A ``decode_coords="all"`` style dataset is still detected."""
    mesh = detect_mesh(make_fvcom_dataset(attrs_in_encoding=True))
    assert mesh is not None
    assert mesh.node_coordinates == ("lon", "lat")
    assert mesh.start_index == 1


def test_detect_mesh_dask_backed():
    """Detection works on a chunked dataset."""
    mesh = detect_mesh(make_fvcom_dataset(dask=True))
    assert mesh is not None
    assert mesh.face_dim == "nele"


def test_detect_mesh_returns_none_for_regular_grid():
    """A regular lon/lat dataset has no mesh."""
    from cf_xarray.datasets import airds

    assert detect_mesh(airds) is None


def test_detect_mesh_returns_none_for_quad_mesh():
    """Only triangular meshes are supported."""
    assert detect_mesh(make_quad_mesh_dataset()) is None


def test_variable_location(fvcom_dataset):
    """Variable locations come from ``location`` attrs or dims, never ``mesh``."""
    mesh = detect_mesh(fvcom_dataset)
    assert mesh is not None
    assert variable_location(fvcom_dataset["zeta"], mesh) == "node"
    # ``u`` names a mesh that does not exist; location/dims still resolve it
    assert variable_location(fvcom_dataset["u"], mesh) == "face"

    no_location = fvcom_dataset["h"].copy()
    no_location.attrs = {}
    assert variable_location(no_location, mesh) == "node"

    assert variable_location(fvcom_dataset["time"], mesh) is None


def test_dataset_spatial_ref_uses_mesh_node_coordinates(fvcom_dataset):
    """X/Y resolve to the mesh node coordinates despite ambiguous lonc/latc."""
    spatial_ref = dataset_spatial_ref(fvcom_dataset)
    assert spatial_ref.X == "lon"
    assert spatial_ref.Y == "lat"
    assert spatial_ref.mesh is not None
    assert spatial_ref.mesh.topology == "mesh_topology"
    assert spatial_ref.crs.to_epsg() == 4326


def test_grid_kind_unstructured(fvcom_dataset):
    """A UGRID mesh dataset reports an unstructured grid."""
    spatial_ref = dataset_spatial_ref(fvcom_dataset)
    assert grid_kind(fvcom_dataset, spatial_ref) is GridKind.UNSTRUCTURED


def test_grid_kind_regular():
    """A regular 1D lon/lat dataset reports a regular grid."""
    from cf_xarray.datasets import airds

    spatial_ref = dataset_spatial_ref(airds)
    assert grid_kind(airds, spatial_ref) is GridKind.REGULAR


def test_grid_kind_none_for_curvilinear():
    """2D coordinates are neither regular nor unstructured."""
    ds = make_curvilinear_dataset()
    spatial_ref = dataset_spatial_ref(ds)
    assert grid_kind(ds, spatial_ref) is None


def test_prepare_spatial_grid_rejects_curvilinear():
    """``require_selectable`` still raises the historical error for 2D coords."""
    ds = make_curvilinear_dataset()
    with pytest.raises(NotImplementedError, match="Only 1D coordinates are supported"):
        prepare_spatial_grid(ds, require_selectable=True)


def test_prepare_spatial_grid_unstructured(fvcom_dataset):
    """An unstructured dataset passes the selectable gate and reports its kind."""
    grid = prepare_spatial_grid(fvcom_dataset, require_selectable=True)
    assert grid.kind is GridKind.UNSTRUCTURED
    assert grid.spatial_ref.mesh is not None


def test_prepare_spatial_grid_uses_source_dataset(fvcom_dataset):
    """Mesh metadata is taken from the unfiltered source dataset."""
    filtered = fvcom_dataset[["zeta"]]
    assert "mesh_topology" not in filtered.variables
    assert "nv" not in filtered.variables

    grid = prepare_spatial_grid(filtered, source=fvcom_dataset, require_selectable=True)
    assert grid.kind is GridKind.UNSTRUCTURED
    assert grid.spatial_ref.X == "lon"
    assert grid.spatial_ref.Y == "lat"
    assert grid.spatial_ref.mesh is not None


def test_collection_metadata_for_mesh(fvcom_dataset):
    """Collection metadata uses node bounds and hides structural variables."""
    metadata = collection_metadata(
        fvcom_dataset,
        position_output_formats=["cf_covjson"],
        area_output_formats=["cf_covjson"],
        cube_output_formats=["cf_covjson"],
    )

    bbox = metadata.extent.spatial.bbox[0]
    assert bbox == pytest.approx(
        [
            float(fvcom_dataset["lon"].min()),
            float(fvcom_dataset["lat"].min()),
            float(fvcom_dataset["lon"].max()),
            float(fvcom_dataset["lat"].max()),
        ],
    )

    names = set(metadata.parameter_names)
    assert {"zeta", "u"} <= names
    assert names.isdisjoint({"nv", "nbe", "mesh_topology"})


def test_require_xugrid_returns_module():
    """The dev environment installs the ``ugrid`` extra."""
    import xugrid

    assert _require_xugrid() is xugrid


def test_require_xugrid_missing_extra(monkeypatch):
    """A missing ``xugrid`` surfaces as ``UgridSupportUnavailable``."""
    monkeypatch.setitem(sys.modules, "xugrid", None)
    with pytest.raises(UgridSupportUnavailable, match=r"install xpublish-edr\[ugrid\]"):
        _require_xugrid()


def test_build_grid_parses_fvcom_layout(fvcom_dataset, fvcom_grid):
    """xugrid parses the 1-based, vertex-first ``nv(three, nele)`` connectivity."""
    side = 5
    grid = fvcom_grid.grid
    assert grid.n_face == 2 * (side - 1) ** 2
    assert grid.n_node == side * side
    connectivity = grid.face_node_connectivity
    assert connectivity.shape == (grid.n_face, 3)
    assert connectivity.min() == 0
    assert connectivity.max() == grid.n_node - 1

    assert fvcom_grid.mesh.topology == "mesh_topology"
    assert fvcom_grid.nbytes > 0
    assert fvcom_grid.build_seconds > 0

    assert fvcom_grid.node_xy.shape == (grid.n_node, 2)
    assert fvcom_grid.face_xy.shape == (grid.n_face, 2)


def test_build_grid_indexes_in_local_aeqd(fvcom_dataset, fvcom_grid):
    """A regional geographic mesh is indexed in a local azimuthal equidistant plane."""
    assert fvcom_grid.crs.to_epsg() == 4326
    assert not fvcom_grid.index_crs.is_geographic
    assert fvcom_grid.index_crs.coordinate_operation.method_name == "Azimuthal Equidistant"

    lon0 = float(fvcom_dataset["lon"].values.mean())
    lat0 = float(fvcom_dataset["lat"].values.mean())
    centre = fvcom_grid.project(np.array([lon0]), np.array([lat0]))
    assert centre.shape == (1, 2)
    np.testing.assert_allclose(centre[0], [0.0, 0.0], atol=1.0)

    # The mesh itself is held in the index plane, i.e. metres, not degrees
    assert np.abs(fvcom_grid.node_xy).max() > 1e3


def test_index_crs_for_projected_crs_is_identity():
    """A projected dataset CRS is indexed as-is."""
    crs = pyproj.CRS.from_epsg(3857)
    assert _index_crs_for((-1e6, -1e6, 1e6, 1e6), crs) == crs


def test_index_crs_for_global_mesh_is_identity():
    """A near-global geographic mesh keeps its own CRS."""
    crs = pyproj.CRS.from_epsg(4326)
    assert _index_crs_for((-180.0, -80.0, 180.0, 80.0), crs) == crs


def test_nearest_nodes(fvcom_grid):
    """Nearest node lookup returns the node itself for slightly offset points."""
    expected = np.array([0, 3, 7, 12, 24])
    xy = fvcom_grid.node_xy[expected] + np.array([1.0, -1.0])
    np.testing.assert_array_equal(fvcom_grid.nearest_nodes(xy), expected)


def test_containing_and_nearest_faces(fvcom_grid):
    """Faces are located inside the mesh and fall back to nearest outside it."""
    inside = fvcom_grid.face_xy[[0, 5, 31]]
    np.testing.assert_array_equal(fvcom_grid.containing_faces(inside), [0, 5, 31])

    outside = fvcom_grid.project(np.array([-80.0]), np.array([30.0]))
    assert fvcom_grid.containing_faces(outside)[0] == -1

    nearest = fvcom_grid.nearest_faces(outside)
    assert 0 <= nearest[0] < fvcom_grid.grid.n_face


def test_barycentric_weights(fvcom_grid):
    """Barycentric weights are convex inside the mesh and flagged outside."""
    inside = fvcom_grid.face_xy[[0, 5, 31]]
    outside = fvcom_grid.project(np.array([-80.0]), np.array([30.0]))
    xy = np.vstack([inside, outside])

    faces, weights = fvcom_grid.barycentric(xy)
    np.testing.assert_array_equal(faces[:3], [0, 5, 31])
    assert faces[3] == -1

    assert (weights[:3] >= -1e-9).all()
    np.testing.assert_allclose(weights[:3].sum(axis=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(weights[3], 0.0, atol=1e-12)

    connectivity = fvcom_grid.grid.face_node_connectivity
    for i in range(3):
        vertices = fvcom_grid.node_xy[connectivity[faces[i]]]
        recovered = (vertices * weights[i][:, None]).sum(axis=0)
        np.testing.assert_allclose(recovered, xy[i], atol=1e-6)


def test_project_geometry(fvcom_grid):
    """Query geometry is projected into the index plane with the same transformer."""
    point = shapely.Point(-69.5, 43.5)
    projected = fvcom_grid.project_geometry(point)
    expected = fvcom_grid.project(np.array([-69.5]), np.array([43.5]))[0]
    np.testing.assert_allclose([projected.x, projected.y], expected)

    polygon = shapely.box(-70.0, 43.0, -69.0, 44.0)
    projected_polygon = fvcom_grid.project_geometry(polygon)
    assert projected_polygon.area > 1e8  # square metres, not square degrees


def test_nearest_nodes_uses_metric_distance():
    """Nearest node is measured in metres, not degrees.

    At 43 degrees north 0.5 degrees of longitude is ~40.7 km while 0.4 degrees
    of latitude is ~44.5 km, so a naive lookup in degree space picks the wrong
    node.
    """
    lon = np.array([-68.5, -69.0, -69.5])
    lat = np.array([43.0, 43.4, 42.9])
    ds = make_triangle_dataset(lon, lat)
    spatial_ref = dataset_spatial_ref(ds)
    assert spatial_ref.mesh is not None
    grid = build_grid(ds, spatial_ref.mesh, spatial_ref.crs)

    query_lon, query_lat = -69.0, 43.0
    degree_distance = np.hypot(lon - query_lon, lat - query_lat)
    assert int(degree_distance.argmin()) == 1

    xy = grid.project(np.array([query_lon]), np.array([query_lat]))
    (node,) = grid.nearest_nodes(xy)
    np.testing.assert_allclose(
        grid.node_xy[node],
        grid.project(lon[:1], lat[:1])[0],
        atol=1e-6,
    )


def test_build_grid_dask_backed():
    """A chunked dataset builds the same grid as the in-memory one."""
    ds = make_fvcom_dataset(dask=True)
    spatial_ref = dataset_spatial_ref(ds)
    assert spatial_ref.mesh is not None
    grid = build_grid(ds, spatial_ref.mesh, spatial_ref.crs)

    reference = build_grid(make_fvcom_dataset(), spatial_ref.mesh, spatial_ref.crs)
    assert isinstance(grid.node_xy, np.ndarray)
    np.testing.assert_allclose(grid.node_xy, reference.node_xy)
    np.testing.assert_array_equal(
        grid.grid.face_node_connectivity,
        reference.grid.face_node_connectivity,
    )


def test_get_indexed_grid_caches_by_dataset_id(fvcom_dataset):
    """The same dataset id reuses the cached grid; a different id builds a new one."""
    cache = cachey.Cache(available_bytes=1e9)
    spatial_ref = dataset_spatial_ref(fvcom_dataset)

    fvcom_dataset.attrs[DATASET_ID_ATTR_KEY] = "fvcom"
    first = get_indexed_grid(fvcom_dataset, spatial_ref, cache)
    second = get_indexed_grid(fvcom_dataset, spatial_ref, cache)
    assert first is second

    other = fvcom_dataset.copy()
    other.attrs[DATASET_ID_ATTR_KEY] = "fvcom-other"
    assert get_indexed_grid(other, spatial_ref, cache) is not first


def test_get_indexed_grid_without_dataset_id(fvcom_dataset):
    """Without a dataset id (or a cache) the grid is rebuilt per call."""
    cache = cachey.Cache(available_bytes=1e9)
    spatial_ref = dataset_spatial_ref(fvcom_dataset)

    assert DATASET_ID_ATTR_KEY not in fvcom_dataset.attrs
    assert get_indexed_grid(fvcom_dataset, spatial_ref, cache) is not get_indexed_grid(
        fvcom_dataset,
        spatial_ref,
        cache,
    )

    fvcom_dataset.attrs[DATASET_ID_ATTR_KEY] = "fvcom"
    assert get_indexed_grid(fvcom_dataset, spatial_ref, None) is not get_indexed_grid(
        fvcom_dataset,
        spatial_ref,
        None,
    )


def test_get_indexed_grid_warns_when_cache_is_too_small(fvcom_dataset, caplog, monkeypatch):
    """A grid larger than the cache is still returned, with a one-off warning."""
    spatial_ref = dataset_spatial_ref(fvcom_dataset)
    fvcom_dataset.attrs[DATASET_ID_ATTR_KEY] = "fvcom-big"
    huge = dataclasses.replace(
        build_grid(fvcom_dataset, spatial_ref.mesh, spatial_ref.crs),
        nbytes=int(20e6),
    )
    monkeypatch.setattr(ugrid_module, "build_grid", lambda *a, **kw: huge)
    monkeypatch.setattr(ugrid_module, "_CACHE_WARNED_KEYS", set())

    # xpublish's default cache is 1 MB
    cache = cachey.Cache(available_bytes=1e6)
    with caplog.at_level(logging.WARNING, logger="cf_edr"):
        assert get_indexed_grid(fvcom_dataset, spatial_ref, cache) is huge
        assert get_indexed_grid(fvcom_dataset, spatial_ref, cache) is huge

    warnings = [r for r in caplog.records if "available_bytes" in r.message]
    assert len(warnings) == 1
    assert "fvcom-big" in warnings[0].message


def test_prepare_spatial_grid_builds_indexed_grid(fvcom_dataset):
    """A selectable unstructured grid carries a built (and cached) index."""
    cache = cachey.Cache(available_bytes=1e9)
    fvcom_dataset.attrs[DATASET_ID_ATTR_KEY] = "fvcom"
    filtered = fvcom_dataset[["zeta"]]

    prepared = prepare_spatial_grid(
        filtered,
        source=fvcom_dataset,
        require_selectable=True,
        cache=cache,
    )
    assert prepared.kind is GridKind.UNSTRUCTURED
    assert isinstance(prepared.grid, IndexedGrid)
    assert prepared.grid.grid.n_face == 32

    again = prepare_spatial_grid(
        filtered,
        source=fvcom_dataset,
        require_selectable=True,
        cache=cache,
    )
    assert again.grid is prepared.grid


def test_prepare_spatial_grid_metadata_path_needs_no_xugrid(fvcom_dataset, monkeypatch):
    """Without ``require_selectable`` no index is built, so xugrid is not needed."""
    monkeypatch.setitem(sys.modules, "xugrid", None)
    prepared = prepare_spatial_grid(fvcom_dataset)
    assert prepared.kind is GridKind.UNSTRUCTURED
    assert prepared.grid is None


def test_prepare_spatial_grid_requires_xugrid(fvcom_dataset, monkeypatch):
    """Selecting on a mesh without xugrid raises ``UgridSupportUnavailable``."""
    monkeypatch.setitem(sys.modules, "xugrid", None)
    with pytest.raises(UgridSupportUnavailable, match=r"install xpublish-edr\[ugrid\]"):
        prepare_spatial_grid(fvcom_dataset, require_selectable=True)
