import numpy as np
import pandas as pd
import pytest
import xarray as xr
import xpublish
from fastapi.testclient import TestClient
from xpublish_ogc_core.plugin import OgcCorePlugin

from xpublish_edr.plugin import CfEdrPlugin

# UGRID reference attrs that ``make_fvcom_dataset(attrs_in_encoding=True)`` moves
# from ``.attrs`` to ``.encoding`` to model ``decode_coords="all"`` datasets.
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
    start_index: int | None = 1,
    face_dimension: bool = True,
    dask: bool = False,
    attrs_in_encoding: bool = False,
) -> xr.Dataset:
    """Build a small FVCOM-flavored UGRID-1.0 dataset.

    The mesh is a ``side x side`` lattice of nodes over lon -70..-69 / lat 43..44,
    split into triangles. ``nv`` is transposed (vertex dimension first) and
    optionally 1-based, exactly like real FVCOM output.

    ``start_index=None`` models the common real-world case of a 1-based ``nv``
    that never declares ``start_index``: the values stay 1-based but the
    attribute is omitted. ``face_dimension=False`` likewise omits the topology's
    ``face_dimension`` attribute.

    Shared by ``tests/test_ugrid.py`` (unit-level UGRID tests) and
    ``tests/test_cf_router.py`` (end-to-end ``fvcom_client`` tests), which
    cannot import each other since ``tests`` has no ``__init__.py``.
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
    nv = (tris + (1 if start_index is None else start_index)).T.astype("int32")
    nv_attrs = {"cf_role": "face_node_connectivity"}
    if start_index is not None:
        nv_attrs["start_index"] = start_index

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
            "nv": (("three", "nele"), nv, nv_attrs),
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
            "time": (("time",), time, {"standard_name": "time", "long_name": "time"}),
        },
        attrs={"Conventions": "CF-1.11, UGRID-1.0", "source": "FVCOM_4.4.1"},
    )

    if dask:
        ds = ds.chunk({"node": 7, "nele": 9, "time": 2})

    if attrs_in_encoding:
        topology = ds["mesh_topology"]
        moved = {k: topology.attrs.pop(k) for k in UGRID_KEYS if k in topology.attrs}
        topology.encoding.update(moved)
        if "start_index" in ds["nv"].attrs:
            ds["nv"].encoding["start_index"] = ds["nv"].attrs.pop("start_index")

    return ds


def build_ogc_app():
    """Compose the xpublish-ogc-core + xpublish-edr app with the CF air dataset.

    Shared by the ``ogc_app`` fixture and the schemathesis tests, which build
    their schemas at module-collection time and so cannot use the fixture.
    """
    from cf_xarray.datasets import airds

    rest = xpublish.Rest(
        {"air": airds},
        plugins={
            "ogc": OgcCorePlugin(),
            "edr": CfEdrPlugin(),
        },
    )

    return rest.app


@pytest.fixture(scope="module")
def ogc_app():
    return build_ogc_app()


@pytest.fixture(scope="module")
def client(ogc_app):
    return TestClient(ogc_app)


@pytest.fixture(scope="session")
def cf_air_dataset():
    from cf_xarray.datasets import airds

    # Create a float16 version of the air variable
    airds["air_float16"] = airds["air"].astype("float16")

    return airds


@pytest.fixture(scope="session")
def cf_temp_dataset():
    from cf_xarray.datasets import rotds

    return rotds


@pytest.fixture(scope="session")
def cf_xpublish(cf_air_dataset, cf_temp_dataset):
    rest = xpublish.Rest(
        {"air": cf_air_dataset, "temp": cf_temp_dataset},
        plugins={"edr": CfEdrPlugin()},
    )

    return rest


@pytest.fixture(scope="session")
def cf_client(cf_xpublish):
    app = cf_xpublish.app
    client = TestClient(app)

    return client
