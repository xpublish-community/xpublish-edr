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

### [collections](https://docs.ogc.org/is/19-086r6/19-086r6.html#_e55ba0f5-8f24-4f1b-a7e3-45775e39ef2e) and Resource Paths Support

`xpublish-edr` does not currently support the `/collections/{collectionId}/query` path template described in the spec. Instead the path resource appears as `/{dataset_id}/edr/{query}`. This is because of the path structure of xpublish. In the future, if `xpublish` supports [`DataTree`](https://docs.xarray.dev/en/stable/generated/xarray.DataTree.html) it could provide a path to supporting the spec compliant `collections` resource path.

 However, despite the collections resource not existing, this implementation supports [collection metadata](https://docs.ogc.org/is/19-086r6/19-086r6.html#_5d07dde9-231a-4652-a1f3-dd036c337bdc) at the dataset level through the `/{dataset_id}/edr/` resource.

### Supported Queries

[8.2.1 Position query](https://docs.ogc.org/is/19-086r6/19-086r6.html#_bbda46d4-04c5-426b-bea3-230d592fe1c2)

| Query  | Compliant | Comments
| ------------- | ------------- | ------------- |
| `coords`  | ✅ | |
| `z`  | ✅ | |
| `datetime`  | ✅ | |
| `parameter-name`  | ✅   | |
| `crs`  | ✅  | Requires a CF compliant [grid mapping](https://cf-xarray.readthedocs.io/en/latest/grid_mappings.html) on the target dataset. Default is `EPSG:4326` |
| `f`  | ✅ | Supports `cf_covjson`, `csv`, `geojson` `netcdf`, `parquet` |
| `method`  | ➕ | Optional: controls data selection. Use "nearest" for nearest neighbor selection, or "linear" for interpolated selection. Uses `nearest` if not specified |

> Any additional query parameters are assumed to be additional selections to make on the dimensions/coordinates. These queries will use the specified selections `method`.

[8.2.3 Area query](https://docs.ogc.org/is/19-086r6/19-086r6.html#_c92d1888-dc80-454f-8452-e2f070b90dcd)

| Query  | Compliant | Comments
| ------------- | ------------- | ------------- |
| `coords`  | ✅ | Only `POLYGON` supported currently |
| `z`  | ✅   | |
| `datetime`  | ✅ | |
| `parameter-name`  | ✅   | |
| `crs`  | ✅  | Requires a CF compliant [grid mapping](https://cf-xarray.readthedocs.io/en/latest/grid_mappings.html) on the target dataset. Default is `EPSG:4326` |
| `f`  | ✅   | Supports `cf_covjson`, `csv`, `geojson` `netcdf`, `parquet` |
| `method`  | ➕ | Optional: controls data selection. Use "nearest" for nearest neighbor selection, or "linear" for interpolated selection. Uses `nearest` if not specified |

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
