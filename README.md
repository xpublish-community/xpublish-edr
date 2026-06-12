## xpublish-edr

[![PyPI](https://img.shields.io/pypi/v/xpublish-edr)](https://pypi.org/project/xpublish-edr/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/xpublish-edr.svg)](https://anaconda.org/conda-forge/xpublish-edr)

[![Tests](https://github.com/gulfofmaine/xpublish-edr/actions/workflows/tests.yml/badge.svg)](https://github.com/gulfofmaine/xpublish-edr/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/xpublish-community/xpublish-edr/branch/main/graph/badge.svg?token=19AE9JWWWD)](https://codecov.io/gh/xpublish-community/xpublish-edr)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/xpublish-community/xpublish-edr/main.svg)](https://results.pre-commit.ci/latest/github/xpublish-community/xpublish-edr/main)

[Xpublish](https://xpublish.readthedocs.io/en/latest/) routers for the [OGC EDR API](https://ogcapi.ogc.org/edr/).

### Documentation and code

URLs for the docs and code.

### Installation

For `conda` users you can

```shell
conda install --channel conda-forge xpublish_edr
```

or, if you are a `pip` users

```shell
pip install xpublish_edr
```

### Example

```python
import xarray as xr
import xpublish
from xpublish.routers import base_router, zarr_router
from xpublish_edr.cf_edr_router import cf_edr_router

ds = xr.open_dataset("dataset.nc")

rest = xpublish.Rest(
    datasets,
    routers=[
        (base_router, {"tags": ["info"]}),
        (cf_edr_router, {"tags": ["edr"], "prefix": "/edr"}),
        (zarr_router, {"tags": ["zarr"], "prefix": "/zarr"}),
    ],
)
```


## OGC EDR Spec Compliance

This package attempts to follow [the spec](https://docs.ogc.org/is/19-086r6/19-086r6.html) where reasonable, adding functionality where the value is demonstrable.

> **Note:** `POST` is supported on `/position` and `/area` as a non-spec extension so that requests with large geometries (many points, complex polygons) can submit them in the request body instead of being limited by URL length. All selection parameters (`datetime`, `z`, `parameter-name`, `crs`, `f`, `method`) are still passed as query string parameters. See the per-query tables below for supported body content types.

### [collections](https://docs.ogc.org/is/19-086r6/19-086r6.html#_e55ba0f5-8f24-4f1b-a7e3-45775e39ef2e) and Resource Paths Support

On its own, `xpublish-edr` serves queries at `/{dataset_id}/edr/{query}` rather than the `/collections/{collectionId}/{query}` path template described in the spec, because of the path structure of xpublish. [Collection metadata](https://docs.ogc.org/is/19-086r6/19-086r6.html#_5d07dde9-231a-4652-a1f3-dd036c337bdc) is available at the dataset level through the `/{dataset_id}/edr/` resource.

When composed with [xpublish-ogc-core](https://github.com/xpublish-community/xpublish-ogc-core), the spec compliant resource paths are also served — see below.

### OGC API integration via xpublish-ogc-core

If [xpublish-ogc-core](https://github.com/xpublish-community/xpublish-ogc-core) is installed alongside `xpublish-edr` (both load automatically through their `xpublish.plugin` entry points), this plugin implements its OGC hookspecs so that the composed app serves:

- `/collections/{collection_id}/position`, `/collections/{collection_id}/area`, and `/collections/{collection_id}/cube` — every supported geometry query at the spec compliant resource paths, sharing the query handling with the dataset level routes.
- `/collections/{collection_id}` carrying the full EDR collection metadata (`extent`, `parameter_names`, `crs`, `output_formats`) and a [`data_queries`](https://docs.ogc.org/is/19-086r6/19-086r6.html#_df2c080b-949c-40c3-ad14-d20228270c2d) member describing each query with collection scoped hrefs.
- `/conformance` declaring the [EDR 1.1 conformance classes](https://docs.ogc.org/is/19-086r6/19-086r6.html) (`core`, `collections`, `json`, `geojson`, `covjson`, and `queries`).

The integration is validated end to end in `tests/test_ogc_core_integration.py` against the official OGC schemas vendored by `xpublish-ogc-core`, plus a [Schemathesis](https://schemathesis.readthedocs.io/) fuzz of the composed app's OpenAPI description. Those tests are skipped when `xpublish_ogc_core` isn't importable, and run with the rest of the suite via `uv run pytest`.

### Supported Queries

[8.2.1 Position query](https://docs.ogc.org/is/19-086r6/19-086r6.html#_bbda46d4-04c5-426b-bea3-230d592fe1c2)

| Query  | Compliant | Comments
| ------------- | ------------- | ------------- |
| `coords`  | ✅ | Required for `GET`; for `POST` the points are read from the request body |
| `z`  | ✅ | |
| `datetime`  | ✅ | |
| `parameter-name`  | ✅   | |
| `crs`  | ✅  | Requires a CF compliant [grid mapping](https://cf-xarray.readthedocs.io/en/latest/grid_mappings.html) on the target dataset. Default is `EPSG:4326` |
| `f`  | ✅ | Supports `cf_covjson`, `csv`, `geojson` `netcdf`, `parquet` |
| `method`  | ➕ | Optional: controls data selection. Use "nearest" for nearest neighbor selection, or "linear" for interpolated selection. Uses `nearest` if not specified |
| `POST` body  | ➕ | Non-spec extension. Supported content types: `text/csv` (columns `x`/`y`, `lon`/`lat`, or `longitude`/`latitude`); `application/geo+json` (Point, MultiPoint, Feature, FeatureCollection, or GeometryCollection) |

> Any additional query parameters are assumed to be additional selections to make on the dimensions/coordinates. These queries will use the specified selections `method`.

[8.2.3 Area query](https://docs.ogc.org/is/19-086r6/19-086r6.html#_c92d1888-dc80-454f-8452-e2f070b90dcd)

| Query  | Compliant | Comments
| ------------- | ------------- | ------------- |
| `coords`  | ✅ | `POLYGON` and `MULTIPOLYGON` supported. Required for `GET`; for `POST` the polygon is read from the request body |
| `z`  | ✅   | |
| `datetime`  | ✅ | |
| `parameter-name`  | ✅   | |
| `crs`  | ✅  | Requires a CF compliant [grid mapping](https://cf-xarray.readthedocs.io/en/latest/grid_mappings.html) on the target dataset. Default is `EPSG:4326` |
| `f`  | ✅   | Supports `cf_covjson`, `csv`, `geojson` `netcdf`, `parquet` |
| `method`  | ➕ | Optional: controls data selection. Use "nearest" for nearest neighbor selection, or "linear" for interpolated selection. Uses `nearest` if not specified |
| `POST` body  | ➕ | Non-spec extension. Supported content types: `application/geo+json` (Polygon, MultiPolygon, Feature, FeatureCollection, or GeometryCollection); `application/wkt` / `text/plain` (raw WKT Polygon or MultiPolygon) |

> `method` is not applicable for the coordinates of area queries, only for selecting datetime, z, or additional dimensions.

For `POLYGON` coordinates, points that are located within **OR** on the polygons boundary are included in the response.

[8.2.4 Cube query](https://docs.ogc.org/is/19-086r6/19-086r6.html#_c92d1888-dc80-454f-8452-e2f070b90dcd)

| Query  | Compliant | Comments
| ------------- | ------------- | ------------- |
| `bbox`  | ✅ | Bounding box in `minx,miny,maxx,maxy` format |
| `z`  | ✅ | |
| `datetime`  | ✅ | |
| `parameter-name`  | ✅ | |
| `crs`  | ✅ | Requires a CF compliant [grid mapping](https://cf-xarray.readthedocs.io/en/latest/grid_mappings.html) on the target dataset. Default is `EPSG:4326` |
| `f`  | ✅ | Supports `cf_covjson`, `csv`, `geojson` `netcdf`, `parquet`, `geotiff`|
| `method`  | ➕ | Optional: controls data selection. Use "nearest" for nearest neighbor selection, or "linear" for interpolated selection. Uses `nearest` if not specified |

> `method` is not applicable for the coordinates of cube queries, only for selecting datetime, z, or additional dimensions.

Cube queries are not flattened like area queries, so the response is returned as sliced by xarray. This is particularly useful for subsetting regular grids.

## Get in touch

Report bugs, suggest features or view the source code on [GitHub](https://github.com/gulfofmaine/xpublish-edr/issues).


## License and copyright

xpublish-edr is licensed under BSD 3-Clause "New" or "Revised" License (BSD-3-Clause).

Development occurs on GitHub at <https://github.com/gulfofmaine/xpublish-edr/issues>.
