"""Tests for UGRID (FVCOM-style unstructured mesh) detection and X/Y resolution."""

import dataclasses
import logging
import sys

import cachey
import numpy as np
import pyproj
import pytest
import shapely
import xarray as xr
from conftest import make_fvcom_dataset
from xpublish.utils.api import DATASET_ID_ATTR_KEY

from xpublish_edr.area.geom import select_by_area
from xpublish_edr.area.query import EDRAreaQueryGet
from xpublish_edr.geometry import ugrid as ugrid_module
from xpublish_edr.geometry.common import (
    GridKind,
    PreparedSpatialGrid,
    dataset_spatial_ref,
    finalize_unstructured_selection,
    grid_kind,
    prepare_spatial_grid,
    project_dataset,
    project_geometry,
    selected_spatial_ref,
    selection_targets,
)
from xpublish_edr.geometry.ugrid import (
    IndexedGrid,
    UgridSupportUnavailable,
    _index_crs_for,
    _require_xugrid,
    _topology_for_xugrid,
    build_grid,
    detect_mesh,
    get_indexed_grid,
    resolve_start_index,
    restore_ugrid_attrs,
    variable_location,
)
from xpublish_edr.metadata import collection_metadata
from xpublish_edr.position import geom as position_geom
from xpublish_edr.position.geom import select_by_position
from xpublish_edr.position.query import EDRPositionQueryGet


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


def make_leading_1d_topology_dataset() -> xr.Dataset:
    """The FVCOM fixture with a 1D network topology listed before the 2D mesh.

    UGRID allows several topologies in one file; only the 2D triangular one is
    usable here, and it is not necessarily the first one listed. cf_xarray sorts
    ``cf_roles`` by variable name, hence ``edge_network``.
    """
    base = make_fvcom_dataset()
    n_node = base.sizes["node"]
    edges = np.column_stack(
        [np.arange(n_node - 1, dtype="int32"), np.arange(1, n_node, dtype="int32")],
    )
    return xr.Dataset(
        data_vars={
            "edge_network": (
                (),
                np.int32(0),
                {
                    "cf_role": "mesh_topology",
                    "topology_dimension": 1,
                    "node_coordinates": "lon lat",
                    "edge_node_connectivity": "edges",
                },
            ),
            "edges": (
                ("nedges", "two"),
                edges,
                {"cf_role": "edge_node_connectivity", "start_index": 0},
            ),
            **{name: base[name] for name in base.data_vars},
        },
        coords={name: base[name] for name in base.coords},
        attrs=dict(base.attrs),
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


def test_detect_mesh_skips_non_2d_topology(caplog):
    """A 1D topology listed first does not hide the 2D mesh behind it."""
    ds = make_leading_1d_topology_dataset()
    topologies = list(ds.cf.cf_roles["mesh_topology"])
    assert topologies[0] == "edge_network"

    with caplog.at_level(logging.DEBUG, logger="cf_edr"):
        mesh = detect_mesh(ds)

    assert mesh is not None
    assert mesh.topology == "mesh_topology"
    assert mesh.face_dim == "nele"
    assert any("mesh_topology" in record.message for record in caplog.records)


def test_detect_mesh_start_index_is_none_when_undeclared():
    """An undeclared ``start_index`` is reported as unknown, not assumed to be 0."""
    mesh = detect_mesh(make_fvcom_dataset(start_index=None))
    assert mesh is not None
    assert mesh.start_index is None


@pytest.mark.parametrize(
    ("values", "n_node", "declared", "expected"),
    [
        pytest.param([[1, 2, 3]], 3, 0, 0, id="declared-zero-wins"),
        pytest.param([[0, 1, 2]], 3, 1, 1, id="declared-one-wins"),
        pytest.param([[1, 2, 3], [2, 3, 1]], 3, None, 1, id="inferred-one-based"),
        pytest.param([[0, 1, 2], [1, 2, 0]], 3, None, 0, id="inferred-zero-based"),
        pytest.param([[1, 2, 3], [1, 2, -1]], 3, None, 1, id="inferred-one-based-with-fill"),
        pytest.param([[0, 1, 2], [0, 1, -1]], 3, None, 0, id="inferred-zero-based-with-fill"),
    ],
)
def test_resolve_start_index(values, n_node, declared, expected):
    """The declared ``start_index`` wins; otherwise it is inferred from the values."""
    assert resolve_start_index(np.array(values, dtype="int32"), n_node, declared) == expected


def test_resolve_start_index_logs_when_inferred(caplog):
    """Inferring a 1-based connectivity is noted in the log."""
    with caplog.at_level(logging.INFO, logger="cf_edr"):
        assert resolve_start_index(np.array([[1, 2, 3]], dtype="int32"), 3, None) == 1
    assert any("1-based" in record.message for record in caplog.records)


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


def grid_for(ds: xr.Dataset) -> IndexedGrid:
    """Resolve the mesh and build its index, the way the query pipeline does."""
    spatial_ref = dataset_spatial_ref(ds)
    assert spatial_ref.mesh is not None
    return build_grid(ds, spatial_ref.mesh, spatial_ref.crs)


def test_build_grid_without_face_dimension():
    """A mesh with no ``face_dimension`` attr still gets all of its faces."""
    ds = make_fvcom_dataset(face_dimension=False)
    assert "face_dimension" not in ds["mesh_topology"].attrs

    grid = grid_for(ds).grid
    assert grid.n_face == 32
    assert grid.n_node == 25
    connectivity = grid.face_node_connectivity
    assert connectivity.min() == 0
    assert connectivity.max() == grid.n_node - 1


def test_build_grid_without_face_dimension_or_face_coordinates():
    """Neither ``face_dimension`` nor ``face_coordinates`` are required."""
    ds = make_fvcom_dataset(face_dimension=False).drop_vars(["lonc", "latc"])
    ds["mesh_topology"].attrs.pop("face_coordinates")

    grid = grid_for(ds).grid
    assert grid.n_face == 32
    assert grid.face_node_connectivity.max() == grid.n_node - 1


def test_build_grid_infers_one_based_connectivity(fvcom_dataset):
    """A 1-based ``nv`` with no ``start_index`` attr is still read correctly."""
    ds = make_fvcom_dataset(start_index=None)
    assert "start_index" not in ds["nv"].attrs
    assert int(ds["nv"].values.min()) == 1

    indexed = grid_for(ds)
    grid = indexed.grid
    assert grid.face_node_connectivity.min() == 0
    assert grid.face_node_connectivity.max() == grid.n_node - 1

    node = 12
    xy = indexed.project(ds["lon"].values[[node]], ds["lat"].values[[node]])
    np.testing.assert_array_equal(indexed.nearest_nodes(xy), [node])
    np.testing.assert_allclose(indexed.node_xy[node], xy[0], atol=1e-6)


def test_build_grid_infers_one_based_connectivity_without_face_dimension():
    """The inferred start index and the inferred face dimension combine."""
    ds = make_fvcom_dataset(start_index=None, face_dimension=False)
    grid = grid_for(ds).grid
    assert grid.n_face == 32
    assert grid.face_node_connectivity.max() == grid.n_node - 1


def test_build_grid_rejects_out_of_range_connectivity():
    """A connectivity that points past the last node is an error, not a silent mesh."""
    ds = make_fvcom_dataset(start_index=None)
    # Lie about the start index so the 1-based values are read as 0-based
    ds["nv"].attrs["start_index"] = 0

    with pytest.raises(ValueError, match="outside the mesh"):
        grid_for(ds)


def test_topology_for_xugrid_does_not_mutate_the_source(fvcom_dataset):
    """Preparing the topology for xugrid leaves the caller's dataset alone."""
    ds = make_fvcom_dataset(start_index=None, face_dimension=False, dask=True)
    mesh = detect_mesh(ds)
    assert mesh is not None

    out, start_index = _topology_for_xugrid(ds, mesh)

    assert start_index == 1
    assert out["mesh_topology"].attrs["face_dimension"] == "nele"
    assert out["nv"].attrs["start_index"] == 1
    # The connectivity is loaded once, here, rather than again inside xugrid
    assert isinstance(out["nv"].data, np.ndarray)

    assert "face_dimension" not in ds["mesh_topology"].attrs
    assert "start_index" not in ds["nv"].attrs


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


def select_position(
    ds: xr.Dataset,
    source: xr.Dataset,
    point: shapely.Geometry,
    method: str = "nearest",
) -> xr.Dataset:
    """Run the full prepare + select pipeline the way ``run_query`` does."""
    prepared = prepare_spatial_grid(ds, source=source, require_selectable=True)
    return select_by_position(
        prepared.ds,
        point,
        method,
        prepared.spatial_ref,
        grid=prepared.grid,
    )


def zeta_at(lon, lat, n_times: int = 4) -> np.ndarray:
    """The fixture's node field ``2*lon + 3*lat + 1 + 0.1*t``, shaped ``(time, pts)``."""
    lon = np.atleast_1d(np.asarray(lon, dtype="float64"))
    lat = np.atleast_1d(np.asarray(lat, dtype="float64"))
    t_index = np.arange(n_times, dtype="float64")
    return (2.0 * lon + 3.0 * lat + 1.0)[None, :] + 0.1 * t_index[:, None]


def test_selection_targets_follow_the_parameter_filter(fvcom_dataset):
    """Which mesh dimensions are selectable depends on the filtered variables."""
    mesh = detect_mesh(fvcom_dataset)
    assert mesh is not None

    (node_target,) = selection_targets(fvcom_dataset[["zeta"]], mesh)
    assert node_target == ("lon", "lat", "node", "node")

    (face_target,) = selection_targets(fvcom_dataset[["u"]], mesh)
    assert face_target == ("lonc", "latc", "nele", "face")

    both = selection_targets(fvcom_dataset[["zeta", "u"]], mesh)
    assert [t.location for t in both] == ["node", "face"]

    assert selection_targets(fvcom_dataset[["zeta"]].isel(node=0), mesh) == []


def test_selection_targets_require_face_coordinates(fvcom_dataset):
    """A face selection without UGRID ``face_coordinates`` is an error, not a crash."""
    ds = fvcom_dataset.drop_vars(["lonc", "latc"])
    ds["mesh_topology"].attrs.pop("face_coordinates")
    mesh = detect_mesh(ds)
    assert mesh is not None
    assert mesh.face_coordinates is None

    with pytest.raises(ValueError, match="Face-located variables need face coordinates"):
        selection_targets(ds[["u"]], mesh)


def test_select_by_position_nearest_single_point(fvcom_dataset):
    """A single point snaps to the nearest node and yields ``pts`` of length one."""
    ds = select_position(
        fvcom_dataset[["zeta"]],
        fvcom_dataset,
        shapely.Point(-69.5 + 1e-4, 43.5 + 1e-4),
    )

    assert ds["zeta"].dims == ("time", "pts")
    assert ds.sizes["pts"] == 1
    assert "node" not in ds.dims
    np.testing.assert_allclose(ds["lon"].values, [-69.5])
    np.testing.assert_allclose(ds["lat"].values, [43.5])
    # Node 12 of the 5x5 lattice
    np.testing.assert_allclose(
        ds["zeta"].values,
        fvcom_dataset["zeta"].isel(node=12).values[:, None],
    )


def test_select_by_position_nearest_multipoint(fvcom_dataset):
    """A MULTIPOINT keeps the requested order along ``pts``."""
    points = shapely.MultiPoint(
        [(-70.0 + 1e-4, 43.0 + 1e-4), (-69.5, 43.5), (-69.0 - 1e-4, 44.0 - 1e-4)],
    )
    ds = select_position(fvcom_dataset[["zeta"]], fvcom_dataset, points)

    assert ds.sizes["pts"] == 3
    np.testing.assert_allclose(ds["lon"].values, [-70.0, -69.5, -69.0])
    np.testing.assert_allclose(ds["lat"].values, [43.0, 43.5, 44.0])
    np.testing.assert_allclose(
        ds["zeta"].values,
        fvcom_dataset["zeta"].isel(node=[0, 12, 24]).values,
    )


def test_select_by_position_nearest_face_variable(fvcom_dataset, fvcom_grid):
    """Face variables use the containing face, or the nearest centroid outside."""
    face = 17
    lonc = float(fvcom_dataset["lonc"].values[face])
    latc = float(fvcom_dataset["latc"].values[face])

    ds = select_position(fvcom_dataset[["u"]], fvcom_dataset, shapely.Point(lonc, latc))
    assert ds["u"].dims == ("time", "pts")
    assert "nele" not in ds.dims
    np.testing.assert_allclose(ds["lonc"].values, [lonc])
    np.testing.assert_allclose(ds["latc"].values, [latc])
    np.testing.assert_allclose(
        ds["u"].values,
        fvcom_dataset["u"].isel(nele=face).values[:, None],
    )

    outside = shapely.Point(-80.0, 30.0)
    (expected,) = fvcom_grid.nearest_faces(fvcom_grid.project([-80.0], [30.0]))
    ds = select_position(fvcom_dataset[["u"]], fvcom_dataset, outside)
    np.testing.assert_allclose(
        ds["u"].values,
        fvcom_dataset["u"].isel(nele=int(expected)).values[:, None],
    )


def test_select_by_position_nearest_mixed_locations(fvcom_dataset):
    """A node+face request collapses both mesh dimensions onto the same ``pts``."""
    points = shapely.MultiPoint([(-69.5, 43.5), (-69.9, 43.1)])
    ds = select_position(fvcom_dataset[["zeta", "u"]], fvcom_dataset, points)

    assert "node" not in ds.dims
    assert "nele" not in ds.dims
    assert ds.sizes["pts"] == 2
    assert ds["zeta"].dims == ("time", "pts")
    assert ds["u"].dims == ("time", "pts")
    assert ds["lon"].dims == ("pts",)
    assert ds["lonc"].dims == ("pts",)


def test_select_by_position_linear_recovers_the_planar_field(fvcom_dataset, fvcom_grid):
    """Barycentric interpolation reproduces the fixture's linear node field."""
    lonc = fvcom_dataset["lonc"].values
    latc = fvcom_dataset["latc"].values
    points = [
        (float(lonc[0]), float(latc[0])),
        (float(lonc[17]), float(latc[17])),
        # Midpoint of the interior edge between nodes 6 and 12
        (-69.625, 43.375),
    ]
    ds = select_position(
        fvcom_dataset[["zeta"]],
        fvcom_dataset,
        shapely.MultiPoint(points),
        method="linear",
    )

    lon = np.array([p[0] for p in points])
    lat = np.array([p[1] for p in points])
    np.testing.assert_allclose(ds["lon"].values, lon)
    np.testing.assert_allclose(ds["lat"].values, lat)
    assert ds["zeta"].dims == ("time", "pts")

    # Exact against the barycentric combination of the enclosing nodes ...
    xy = fvcom_grid.project(lon, lat)
    faces, weights = fvcom_grid.barycentric(xy)
    assert (faces >= 0).all()
    vertices = fvcom_grid.grid.face_node_connectivity[faces]
    expected = np.einsum(
        "tpv,pv->tp",
        fvcom_dataset["zeta"].values[:, vertices],
        weights,
    )
    np.testing.assert_allclose(ds["zeta"].values, expected, atol=1e-9)

    # ... and close to the analytic field, up to the local aeqd distortion
    np.testing.assert_allclose(ds["zeta"].values, zeta_at(lon, lat), atol=1e-3)


def test_select_by_position_linear_outside_falls_back_to_nearest(fvcom_dataset, caplog):
    """Outside the mesh, linear degrades to the nearest node and warns."""
    with caplog.at_level(logging.WARNING, logger="cf_edr"):
        ds = select_position(
            fvcom_dataset[["zeta"]],
            fvcom_dataset,
            shapely.Point(-80.0, 30.0),
            method="linear",
        )

    # Node 0 (-70, 43) is the corner nearest to the query point
    np.testing.assert_allclose(
        ds["zeta"].values,
        fvcom_dataset["zeta"].isel(node=0).values[:, None],
    )
    # The query point is echoed back, not the node's position
    np.testing.assert_allclose(ds["lon"].values, [-80.0])
    np.testing.assert_allclose(ds["lat"].values, [30.0])

    assert any("outside the mesh" in record.message for record in caplog.records)


def test_select_by_position_linear_face_variable_is_piecewise_constant(fvcom_dataset, caplog):
    """Face variables are not interpolated; the containing face value is used."""
    face = 11
    lonc = float(fvcom_dataset["lonc"].values[face])
    latc = float(fvcom_dataset["latc"].values[face])

    with caplog.at_level(logging.INFO, logger="cf_edr"):
        ds = select_position(
            fvcom_dataset[["u"]],
            fvcom_dataset,
            shapely.Point(lonc, latc),
            method="linear",
        )

    np.testing.assert_allclose(
        ds["u"].values,
        fvcom_dataset["u"].isel(nele=face).values[:, None],
    )
    assert any("not interpolated" in record.message for record in caplog.records)


def test_select_by_position_linear_mixed_locations(fvcom_dataset):
    """A mixed request interpolates node data and isels face data onto one ``pts``."""
    points = [(-69.6, 43.4), (-69.2, 43.8)]
    ds = select_position(
        fvcom_dataset[["zeta", "u"]],
        fvcom_dataset,
        shapely.MultiPoint(points),
        method="linear",
    )

    lon = np.array([p[0] for p in points])
    lat = np.array([p[1] for p in points])
    assert ds.sizes["pts"] == 2
    assert ds["zeta"].dims == ("time", "pts")
    assert ds["u"].dims == ("time", "pts")
    np.testing.assert_allclose(ds["lon"].values, lon)
    np.testing.assert_allclose(ds["zeta"].values, zeta_at(lon, lat), atol=1e-3)
    # Face data stays on the face centroids, and stays piecewise constant
    assert set(np.asarray(ds["u"].values).ravel().tolist()) <= set(
        np.asarray(fvcom_dataset["u"].values).ravel().tolist(),
    )
    assert ds["lonc"].dims == ("pts",)


def test_select_by_position_unstructured_requires_mesh_variables(fvcom_dataset):
    """Selecting a dataset with no mesh-located variables is a client error."""
    prepared = prepare_spatial_grid(
        fvcom_dataset[["zeta"]],
        source=fvcom_dataset,
        require_selectable=True,
    )
    with pytest.raises(ValueError, match="No mesh-located variables selected"):
        select_by_position(
            prepared.ds.isel(node=0),
            shapely.Point(-69.5, 43.5),
            "nearest",
            prepared.spatial_ref,
            grid=prepared.grid,
        )


def test_select_by_position_unstructured_requires_a_built_grid(fvcom_dataset, monkeypatch):
    """The unstructured path refuses to run without an index."""
    prepared = PreparedSpatialGrid(
        ds=fvcom_dataset[["zeta"]],
        spatial_ref=dataset_spatial_ref(fvcom_dataset),
        kind=GridKind.UNSTRUCTURED,
        grid=None,
    )
    monkeypatch.setattr(position_geom, "prepare_spatial_grid", lambda *a, **kw: prepared)
    with pytest.raises(ValueError, match="Unstructured grid index was not built"):
        select_by_position(prepared.ds, shapely.Point(-69.5, 43.5))


def test_select_by_position_unstructured_rejects_polygons(fvcom_dataset):
    """Only Point/MultiPoint geometries are valid for a position query."""
    prepared = prepare_spatial_grid(
        fvcom_dataset[["zeta"]],
        source=fvcom_dataset,
        require_selectable=True,
    )
    with pytest.raises(ValueError, match="must be Point or MultiPoint"):
        select_by_position(
            prepared.ds,
            shapely.box(-70, 43, -69, 44),
            "nearest",
            prepared.spatial_ref,
            grid=prepared.grid,
        )


def test_finalize_unstructured_selection(fvcom_dataset):
    """Finalizing drops mesh scaffolding and tags the surviving X/Y axes."""
    spatial_ref = dataset_spatial_ref(fvcom_dataset)
    selected = select_position(fvcom_dataset, fvcom_dataset, shapely.Point(-69.5, 43.5))
    assert {"x", "y", "siglay", "lonc", "latc"} <= set(selected.coords)
    assert {"nv", "nbe", "mesh_topology"} <= set(selected.variables)

    ds = finalize_unstructured_selection(selected, spatial_ref, {"zeta"})

    assert set(ds.variables).isdisjoint({"nv", "nbe", "mesh_topology"})
    assert set(ds.coords).isdisjoint({"x", "y", "siglay", "lonc", "latc"})
    assert ds["lon"].attrs["axis"] == "X"
    assert ds["lat"].attrs["axis"] == "Y"
    assert ds.cf.axes["X"] == ["lon"]
    assert ds.cf.axes["Y"] == ["lat"]
    assert "zeta" in ds


def test_finalize_unstructured_selection_keeps_requested_structural_var(fvcom_dataset):
    """``parameter-name=nv`` keeps the connectivity variable it asked for."""
    spatial_ref = dataset_spatial_ref(fvcom_dataset)
    selected = select_position(fvcom_dataset[["nv"]], fvcom_dataset, shapely.Point(-69.5, 43.5))

    ds = finalize_unstructured_selection(selected, spatial_ref, {"nv"})
    assert "nv" in ds

    dropped = finalize_unstructured_selection(selected, spatial_ref, {"zeta"})
    assert "nv" not in dropped


def test_selected_spatial_ref_falls_back_to_face_coordinates(fvcom_dataset):
    """A face-only selection reports the face coordinate pair as its X/Y."""
    spatial_ref = dataset_spatial_ref(fvcom_dataset)
    assert (spatial_ref.X, spatial_ref.Y) == ("lon", "lat")

    node_only = select_position(fvcom_dataset[["zeta"]], fvcom_dataset, shapely.Point(-69.5, 43.5))
    assert selected_spatial_ref(node_only, spatial_ref) is spatial_ref

    face_only = select_position(fvcom_dataset[["u"]], fvcom_dataset, shapely.Point(-69.5, 43.5))
    effective = selected_spatial_ref(face_only, spatial_ref)
    assert (effective.X, effective.Y) == ("lonc", "latc")
    assert effective.crs == spatial_ref.crs
    assert effective.mesh is spatial_ref.mesh


def test_project_dataset_unstructured_same_crs(fvcom_dataset):
    """Without reprojection the X/Y axis attributes survive onto ``pts``."""
    spatial_ref = dataset_spatial_ref(fvcom_dataset)
    selected = select_position(fvcom_dataset[["zeta"]], fvcom_dataset, shapely.Point(-69.5, 43.5))
    ds = finalize_unstructured_selection(selected, spatial_ref, {"zeta"})

    projected = project_dataset(ds, "EPSG:4326", selected_spatial_ref(ds, spatial_ref))
    assert projected.sizes["pts"] == 1
    assert projected["lon"].attrs["axis"] == "X"
    assert projected["lat"].attrs["axis"] == "Y"
    assert projected.cf.axes["X"] == ["lon"]
    assert projected.cf.axes["Y"] == ["lat"]


def test_project_dataset_unstructured_reprojection(fvcom_dataset):
    """Reprojecting a ``pts`` result swaps lon/lat for projected coordinates."""
    spatial_ref = dataset_spatial_ref(fvcom_dataset)
    points = shapely.MultiPoint([(-69.5, 43.5), (-69.25, 43.75)])
    selected = select_position(fvcom_dataset[["zeta"]], fvcom_dataset, points)
    ds = finalize_unstructured_selection(selected, spatial_ref, {"zeta"})

    projected = project_dataset(ds, "EPSG:3857", selected_spatial_ref(ds, spatial_ref))

    assert projected.sizes["pts"] == 2
    assert "lon" not in projected.variables
    assert "lat" not in projected.variables
    assert projected["projection_x_coordinate"].dims == ("pts",)
    assert projected["projection_y_coordinate"].dims == ("pts",)
    assert projected.cf.axes["X"] == ["projection_x_coordinate"]
    assert projected.cf.axes["Y"] == ["projection_y_coordinate"]

    expected_x, expected_y = pyproj.Transformer.from_crs(
        4326,
        3857,
        always_xy=True,
    ).transform([-69.5, -69.25], [43.5, 43.75])
    np.testing.assert_allclose(projected["projection_x_coordinate"].values, expected_x)
    np.testing.assert_allclose(projected["projection_y_coordinate"].values, expected_y)


def test_project_dataset_unstructured_face_selection(fvcom_dataset):
    """A face-only selection projects on its own coordinate pair."""
    spatial_ref = dataset_spatial_ref(fvcom_dataset)
    selected = select_position(fvcom_dataset[["u"]], fvcom_dataset, shapely.Point(-69.5, 43.5))
    ds = finalize_unstructured_selection(selected, spatial_ref, {"u"})

    projected = project_dataset(ds, "EPSG:3857", selected_spatial_ref(ds, spatial_ref))
    assert "lonc" not in projected.variables
    assert projected.cf.axes["X"] == ["projection_x_coordinate"]


@pytest.mark.parametrize("method", ["nearest", "linear"])
def test_run_query_position_unstructured(method):
    """An end to end position query returns a ``(t, pts)`` CoverageJSON coverage."""
    ds = make_fvcom_dataset()
    ds.attrs[DATASET_ID_ATTR_KEY] = "fvcom"

    query = EDRPositionQueryGet.model_validate(
        {"coords": "POINT(-69.5 43.5)", "parameter-name": "zeta", "method": method},
    )
    covjson = query.run_query(ds, {}, query.geometry, cache=cachey.Cache(1e9))

    assert covjson["type"] == "Coverage"
    assert set(covjson["ranges"]) == {"zeta"}
    assert covjson["ranges"]["zeta"]["axisNames"] == ["t", "pts"]
    assert list(covjson["ranges"]["zeta"]["shape"]) == [4, 1]
    assert {"x", "y", "t"} <= set(covjson["domain"]["axes"])
    assert covjson["domain"]["axes"]["x"]["values"] == [-69.5]
    assert covjson["domain"]["axes"]["y"]["values"] == [43.5]
    np.testing.assert_allclose(
        covjson["ranges"]["zeta"]["values"],
        zeta_at(-69.5, 43.5).ravel(),
        atol=1e-5,
    )


def test_run_query_position_unstructured_face_parameter():
    """A face-located parameter round trips through the pipeline."""
    ds = make_fvcom_dataset()
    ds.attrs[DATASET_ID_ATTR_KEY] = "fvcom"
    face = 17
    lonc = float(ds["lonc"].values[face])
    latc = float(ds["latc"].values[face])

    query = EDRPositionQueryGet.model_validate(
        {"coords": f"POINT({lonc} {latc})", "parameter-name": "u"},
    )
    covjson = query.run_query(ds, {}, query.geometry, cache=cachey.Cache(1e9))

    assert set(covjson["ranges"]) == {"u"}
    assert covjson["ranges"]["u"]["axisNames"] == ["t", "pts"]
    assert covjson["domain"]["axes"]["x"]["values"] == pytest.approx([lonc])
    np.testing.assert_allclose(
        covjson["ranges"]["u"]["values"],
        ds["u"].isel(nele=face).values,
    )


def test_run_query_position_unstructured_multipoint_reprojected():
    """A MULTIPOINT query in EPSG:3857 selects the same nodes and reprojects back."""
    ds = make_fvcom_dataset()
    ds.attrs[DATASET_ID_ATTR_KEY] = "fvcom"

    to_3857 = pyproj.Transformer.from_crs(4326, 3857, always_xy=True)
    (x0, x1), (y0, y1) = to_3857.transform([-69.5, -69.25], [43.5, 43.75])

    query = EDRPositionQueryGet.model_validate(
        {
            "coords": f"MULTIPOINT({x0} {y0}, {x1} {y1})",
            "parameter-name": "zeta",
            "crs": "EPSG:3857",
        },
    )
    covjson = query.run_query(ds, {}, query.geometry, cache=cachey.Cache(1e9))

    assert covjson["ranges"]["zeta"]["axisNames"] == ["t", "pts"]
    assert list(covjson["ranges"]["zeta"]["shape"]) == [4, 2]
    np.testing.assert_allclose(covjson["domain"]["axes"]["x"]["values"], [x0, x1])
    np.testing.assert_allclose(covjson["domain"]["axes"]["y"]["values"], [y0, y1])


# A box entirely interior to the 5x5 lattice, away from any node/centroid lon/lat value.
AREA_POLYGON_WKT = "POLYGON((-69.8 43.2, -69.8 43.8, -69.2 43.8, -69.2 43.2, -69.8 43.2))"
AREA_POLYGON = shapely.from_wkt(AREA_POLYGON_WKT)


def select_area(
    ds: xr.Dataset,
    source: xr.Dataset,
    polygon: shapely.Geometry,
) -> xr.Dataset:
    """Run the full prepare + select pipeline the way ``run_query`` does."""
    prepared = prepare_spatial_grid(ds, source=source, require_selectable=True)
    return select_by_area(prepared.ds, polygon, prepared.spatial_ref, grid=prepared.grid)


def test_select_by_area_nodes(fvcom_dataset):
    """Node selection matches every node whose lon/lat falls inside the polygon."""
    lon = fvcom_dataset["lon"].values
    lat = fvcom_dataset["lat"].values
    minx, miny, maxx, maxy = AREA_POLYGON.bounds
    expected = np.flatnonzero((lon >= minx) & (lon <= maxx) & (lat >= miny) & (lat <= maxy))
    assert expected.size > 0

    ds = select_area(fvcom_dataset[["zeta"]], fvcom_dataset, AREA_POLYGON)

    assert ds["zeta"].dims == ("time", "pts")
    assert "node" not in ds.dims
    assert ds.sizes["pts"] == expected.size
    np.testing.assert_allclose(ds["lon"].values, lon[expected])
    np.testing.assert_allclose(ds["lat"].values, lat[expected])
    np.testing.assert_allclose(
        ds["zeta"].values,
        fvcom_dataset["zeta"].isel(node=expected).values,
    )


def test_select_by_area_faces(fvcom_dataset):
    """Face selection matches every face whose xugrid centroid falls inside the polygon.

    For this planar lattice fixture, xugrid's computed centroids coincide with
    the fixture's own ``lonc``/``latc`` UGRID face coordinates.
    """
    lonc = fvcom_dataset["lonc"].values
    latc = fvcom_dataset["latc"].values
    minx, miny, maxx, maxy = AREA_POLYGON.bounds
    expected = np.flatnonzero((lonc >= minx) & (lonc <= maxx) & (latc >= miny) & (latc <= maxy))
    assert expected.size > 0

    ds = select_area(fvcom_dataset[["u"]], fvcom_dataset, AREA_POLYGON)

    assert ds["u"].dims == ("time", "pts")
    assert "nele" not in ds.dims
    assert ds.sizes["pts"] == expected.size
    np.testing.assert_allclose(ds["lonc"].values, lonc[expected])
    np.testing.assert_allclose(ds["latc"].values, latc[expected])
    np.testing.assert_allclose(
        ds["u"].values,
        fvcom_dataset["u"].isel(nele=expected).values,
    )


def test_select_by_area_outside_mesh_is_empty(fvcom_dataset):
    """A polygon entirely outside the mesh yields a zero-length ``pts``."""
    outside = shapely.box(-80.0, 30.0, -79.0, 31.0)
    ds = select_area(fvcom_dataset[["zeta"]], fvcom_dataset, outside)
    assert ds.sizes["pts"] == 0


def test_select_by_area_rejects_mixed_locations(fvcom_dataset):
    """An area query over both node- and face-located parameters is an error."""
    with pytest.raises(ValueError, match="mix"):
        select_area(fvcom_dataset[["zeta", "u"]], fvcom_dataset, AREA_POLYGON)


def test_select_by_area_reprojected_polygon(fvcom_dataset):
    """A polygon supplied in EPSG:3857 selects the same nodes as the lon/lat polygon."""
    to_3857 = pyproj.Transformer.from_crs(4326, 3857, always_xy=True)
    x, y = to_3857.transform(*zip(*AREA_POLYGON.exterior.coords))
    polygon_3857 = shapely.Polygon(np.column_stack([x, y]))

    spatial_ref = dataset_spatial_ref(fvcom_dataset)
    polygon_native = project_geometry(fvcom_dataset, "EPSG:3857", polygon_3857, spatial_ref)

    lonlat_ds = select_area(fvcom_dataset[["zeta"]], fvcom_dataset, AREA_POLYGON)
    reprojected_ds = select_area(fvcom_dataset[["zeta"]], fvcom_dataset, polygon_native)

    assert reprojected_ds.sizes["pts"] == lonlat_ds.sizes["pts"]
    np.testing.assert_allclose(
        reprojected_ds["lon"].values,
        lonlat_ds["lon"].values,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        reprojected_ds["lat"].values,
        lonlat_ds["lat"].values,
        atol=1e-6,
    )


def test_run_query_area_unstructured():
    """An end to end area query returns a ``(t, pts)`` CoverageJSON coverage."""
    ds = make_fvcom_dataset()
    ds.attrs[DATASET_ID_ATTR_KEY] = "fvcom"

    lon = ds["lon"].values
    lat = ds["lat"].values
    minx, miny, maxx, maxy = AREA_POLYGON.bounds
    expected = np.flatnonzero((lon >= minx) & (lon <= maxx) & (lat >= miny) & (lat <= maxy))

    query = EDRAreaQueryGet.model_validate(
        {"coords": AREA_POLYGON_WKT, "parameter-name": "zeta"},
    )
    covjson = query.run_query(ds, {}, query.geometry, cache=cachey.Cache(1e9))

    assert covjson["type"] == "Coverage"
    assert set(covjson["ranges"]) == {"zeta"}
    assert covjson["ranges"]["zeta"]["axisNames"] == ["t", "pts"]
    assert list(covjson["ranges"]["zeta"]["shape"]) == [4, expected.size]
