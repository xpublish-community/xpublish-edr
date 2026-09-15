"""Tests for UGRID (FVCOM-style unstructured mesh) detection and X/Y resolution."""

import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xpublish_edr.geometry.common import (
    GridKind,
    dataset_spatial_ref,
    grid_kind,
    prepare_spatial_grid,
)
from xpublish_edr.geometry.ugrid import (
    UgridSupportUnavailable,
    _require_xugrid,
    detect_mesh,
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


@pytest.fixture
def fvcom_dataset() -> xr.Dataset:
    """A default 5x5-node FVCOM-style UGRID dataset."""
    return make_fvcom_dataset()


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
