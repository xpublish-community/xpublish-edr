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

### Dataset metadata requirements

For `/position`, `/area`, and `/cube`, `xpublish-edr` needs to know the
dataset's native CRS and which dimensions are X and Y. There are three supported
ways to provide that information.

#### CF metadata

Use a `grid_mapping` attribute on each data variable and CF attrs on the X/Y
coordinates:

```python
import pyproj
import xarray as xr

crs = pyproj.CRS.from_epsg(3857)

ds["spatial_ref"] = xr.DataArray(0, attrs=crs.to_cf())
ds["temperature"].attrs["grid_mapping"] = "spatial_ref"
ds["x"].attrs.update(
    axis="X",
    standard_name="projection_x_coordinate",
)
ds["y"].attrs.update(
    axis="Y",
    standard_name="projection_y_coordinate",
)
```

For native longitude/latitude grids, use CF longitude/latitude coordinate attrs
instead of projected X/Y attrs:

```python
ds["lon"].attrs["standard_name"] = "longitude"
ds["lat"].attrs["standard_name"] = "latitude"
```

Datasets that already have a scalar `spatial_ref` or `crs` variable with CF CRS
attrs are also accepted.

#### GeoZarr metadata

Use `proj:` attrs for the CRS and `spatial:dimensions` for the Y/X dimensions:

```python
ds.attrs["proj:code"] = "EPSG:3857"
ds.attrs["spatial:dimensions"] = ["y", "x"]  # [Y, X] order
```

Use `proj:wkt2` instead of `proj:code` if the CRS is stored as WKT.

#### Automatic raster metadata

For raster-style datasets with `x`/`y` dimensions but no explicit X/Y coordinate
arrays, provide CRS and affine transform metadata. `rasterix` detects the raster
dimensions and materializes regular 1D X/Y coordinates before selection.

CF/GDAL form:

```python
ds["spatial_ref"] = xr.DataArray(
    0,
    attrs={
        **crs.to_cf(),
        "GeoTransform": "0 1000 0 3000 0 -1000",
    },
)
ds["temperature"].attrs["grid_mapping"] = "spatial_ref"
```

GeoZarr form:

```python
ds.attrs["proj:code"] = "EPSG:3857"
ds.attrs["spatial:transform"] = [1000, 0, 0, 0, -1000, 3000]
ds.attrs["spatial:dimensions"] = ["y", "x"]
```

`datetime` and `z` queries also require indexed CF `T` and `Z` coordinates.

Spatial selection currently expects regular 1D X/Y coordinate grids, or an
affine transform that can be materialized into regular 1D X/Y coordinates. 2D
curvilinear spatial selection and `proj:projjson` CRS attrs are not currently
supported.


## OGC EDR Spec Compliance

This package attempts to follow [the spec](https://docs.ogc.org/is/19-086r6/19-086r6.html) where reasonable, adding functionality where the value is demonstrable.

> **Note:** `POST` is supported on `/position` and `/area` as a non-spec extension so that requests with large geometries (many points, complex polygons) can submit them in the request body instead of being limited by URL length. All selection parameters (`datetime`, `z`, `parameter-name`, `crs`, `f`, `method`) are still passed as query string parameters. See the per-query tables below for supported body content types.

### [collections](https://docs.ogc.org/is/19-086r6/19-086r6.html#_e55ba0f5-8f24-4f1b-a7e3-45775e39ef2e) and Resource Paths Support

`xpublish-edr` does not currently support the `/collections/{collectionId}/query` path template described in the spec. Instead the path resource appears as `/{dataset_id}/edr/{query}`. This is because of the path structure of xpublish. In the future, if `xpublish` supports [`DataTree`](https://docs.xarray.dev/en/stable/generated/xarray.DataTree.html) it could provide a path to supporting the spec compliant `collections` resource path.

 However, despite the collections resource not existing, this implementation supports [collection metadata](https://docs.ogc.org/is/19-086r6/19-086r6.html#_5d07dde9-231a-4652-a1f3-dd036c337bdc) at the dataset level through the `/{dataset_id}/edr/` resource.

### Supported Queries

[8.2.1 Position query](https://docs.ogc.org/is/19-086r6/19-086r6.html#_bbda46d4-04c5-426b-bea3-230d592fe1c2)

| Query  | Compliant | Comments
| ------------- | ------------- | ------------- |
| `coords`  | ✅ | Required for `GET`; for `POST` the points are read from the request body |
| `z`  | ✅ | |
| `datetime`  | ✅ | |
| `parameter-name`  | ✅   | |
| `crs`  | ✅  | Uses the dataset metadata described above. Default request CRS is `EPSG:4326` |
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
| `crs`  | ✅  | Uses the dataset metadata described above. Default request CRS is `EPSG:4326` |
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
| `crs`  | ✅ | Uses the dataset metadata described above. Default request CRS is `EPSG:4326` |
| `f`  | ✅ | Supports `cf_covjson`, `csv`, `geojson` `netcdf`, `parquet`, `geotiff`|
| `method`  | ➕ | Optional: controls data selection. Use "nearest" for nearest neighbor selection, or "linear" for interpolated selection. Uses `nearest` if not specified |

> `method` is not applicable for the coordinates of cube queries, only for selecting datetime, z, or additional dimensions.

Cube queries are not flattened like area queries, so the response is returned as sliced by xarray. This is particularly useful for subsetting regular grids.

## Get in touch

Report bugs, suggest features or view the source code on [GitHub](https://github.com/gulfofmaine/xpublish-edr/issues).


## License and copyright

xpublish-edr is licensed under BSD 3-Clause "New" or "Revised" License (BSD-3-Clause).

Development occurs on GitHub at <https://github.com/gulfofmaine/xpublish-edr/issues>.
