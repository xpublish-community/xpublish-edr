"""
Common geometry handling functions
"""

import itertools
from functools import lru_cache
from typing import Union

import pyproj
import rioxarray  # noqa
import xarray as xr
from shapely import Geometry
from shapely.ops import transform

VECTORIZED_DIM = "pts"

# https://pyproj4.github.io/pyproj/stable/advanced_examples.html#caching-pyproj-objectshttps://pyproj4.github.io/pyproj/stable/advanced_examples.html#caching-pyproj-objects
transformer_from_crs = lru_cache(pyproj.Transformer.from_crs)


DEFAULT_CRS = pyproj.CRS.from_epsg(4326)


def coord_is_regular(da: xr.DataArray) -> bool:
    """
    Check if the DataArray has a regular grid
    """
    return len(da.shape) == 1 and da.name in da.dims


def is_regular_xy_coords(ds: xr.Dataset) -> bool:
    """
    Check if the dataset has 2D coordinates
    """
    return coord_is_regular(ds.cf["X"]) and coord_is_regular(ds.cf["Y"])


def spatial_bounds(ds: xr.Dataset) -> tuple[float, float, float, float]:
    """
    Get the spatial bounds of the dataset, naively, in whatever CRS it is in
    """
    x = ds.cf["X"]
    min_x = float(x.min().values)
    max_x = float(x.max().values)

    y = ds.cf["Y"]
    min_y = float(y.min().values)
    max_y = float(y.max().values)
    return min_x, min_y, max_x, max_y


def dataset_crs(ds: xr.Dataset) -> pyproj.CRS:
    grid_mapping_names = ds.cf.grid_mapping_names
    if len(grid_mapping_names) == 0:
        # Default to WGS84
        keys = ds.cf.keys()
        if "latitude" in keys and "longitude" in keys:
            crs = pyproj.CRS.from_epsg(4326)

            # Write the crs to the dataset so it is there on export
            ds.rio.write_crs(crs, inplace=True)
            return crs
        else:
            raise ValueError("Unknown coordinate system")
    if len(grid_mapping_names) > 1:
        raise ValueError(f"Multiple grid mappings found: {grid_mapping_names!r}!")
    (grid_mapping_var,) = tuple(itertools.chain(*ds.cf.grid_mapping_names.values()))

    grid_mapping = ds[grid_mapping_var]
    return pyproj.CRS.from_cf(grid_mapping.attrs)


def project_geometry(ds: xr.Dataset, geometry_crs: str, geometry: Geometry) -> Geometry:
    """
    Get the projection from the dataset
    """
    data_crs = dataset_crs(ds)

    transformer = transformer_from_crs(
        crs_from=geometry_crs,
        crs_to=data_crs,
        always_xy=True,
    )
    return transform(transformer.transform, geometry)


def project_bbox(
    ds: xr.Dataset,
    bbox_crs: Union[str, pyproj.CRS],
    bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """
    Project the bbox to the dataset's CRS
    """
    data_crs = dataset_crs(ds)
    if isinstance(bbox_crs, pyproj.CRS):
        target_crs = bbox_crs
    else:
        target_crs = pyproj.CRS.from_string(bbox_crs)
    if data_crs == target_crs:
        return bbox

    transformer: pyproj.Transformer = transformer_from_crs(
        crs_from=target_crs,
        crs_to=data_crs,
        always_xy=True,
    )
    projected_x, projected_y = transformer.transform(
        xx=[bbox[0], bbox[2]],
        yy=[bbox[1], bbox[3]],
    )

    min_x = min(projected_x)
    max_x = max(projected_x)
    min_y = min(projected_y)
    max_y = max(projected_y)
    return min_x, min_y, max_x, max_y


def project_dataset(ds: xr.Dataset, query_crs: Union[str, pyproj.CRS]) -> xr.Dataset:
    """
    Project the dataset to the given CRS
    """
    data_crs = dataset_crs(ds)
    if isinstance(query_crs, pyproj.CRS):
        target_crs = query_crs
    else:
        target_crs = pyproj.CRS.from_string(query_crs)
    if data_crs == target_crs:
        return ds

    transformer = transformer_from_crs(
        crs_from=data_crs,
        crs_to=target_crs,
        always_xy=True,
    )

    # Unpack the coordinates
    try:
        X = ds.cf["X"]
        Y = ds.cf["Y"]
    except KeyError:
        # If the dataset has multiple X axes, we can try to find the right one
        source_cf_coords = data_crs.coordinate_system.to_cf()

        source_x_coord = next(
            coord["standard_name"] for coord in source_cf_coords if coord["axis"] == "X"
        )
        source_y_coord = next(
            coord["standard_name"] for coord in source_cf_coords if coord["axis"] == "Y"
        )

        X = ds.cf[source_x_coord]
        Y = ds.cf[source_y_coord]

    # Transform the coordinates
    # If the data is vectorized, we just transform the points in full
    # TODO: Handle 2D coordinates
    if len(X.dims) > 1 or len(Y.dims) > 1:
        raise NotImplementedError("Only 1D coordinates are supported")

    x, y = xr.broadcast(X, Y)
    target_dims = x.dims

    x, y = transformer.transform(x, y)

    x_dim = X.dims[0]
    y_dim = Y.dims[0]

    coords_to_drop = [
        c for c in ds.coords if x_dim in ds[c].dims or y_dim in ds[c].dims
    ]

    # TODO: Handle rotated pole
    target_cf_coords = target_crs.coordinate_system.to_cf()

    # Get the new X and Y coordinates
    target_x_coord = next(coord for coord in target_cf_coords if coord["axis"] == "X")
    target_y_coord = next(coord for coord in target_cf_coords if coord["axis"] == "Y")

    target_x_coord_name = target_x_coord["standard_name"]
    target_y_coord_name = target_y_coord["standard_name"]

    stdnames = ds.cf.standard_names
    coords_to_drop += list(
        itertools.chain(
            stdnames.get(target_x_coord_name, []),
            stdnames.get(target_y_coord_name, []),
        ),
    )
    ds = ds.drop_vars(coords_to_drop)

    # Create the new dataset with vectorized coordinates
    x_var = xr.Variable(dims=target_dims, data=x, attrs=target_x_coord)
    y_var = xr.Variable(dims=target_dims, data=y, attrs=target_y_coord)
    ds = ds.assign_coords(
        {
            target_x_coord_name: x_var,
            target_y_coord_name: y_var,
        },
    )

    if x_dim != y_dim:
        ds = ds.transpose(..., y_dim, x_dim)

    # Write the crs to the dataset so it is there on export
    ds = ds.rio.write_crs(target_crs, inplace=True)
    return ds
