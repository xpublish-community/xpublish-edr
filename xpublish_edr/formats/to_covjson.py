"""
Generate CoverageJSON responses for xarray Dataset for EDR queries
"""
import sys
from typing import Dict, List, Optional, Union

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

import numpy as np
import xarray as xr


class Domain(TypedDict):
    """CovJSON Domain type"""

    type: Literal["Domain"]
    domainType: Literal["Grid"]
    axes: Dict
    referencing: List


class Parameter(TypedDict):
    """CovJSON Parameter type"""

    type: Literal["Parameter"]
    description: Dict[str, str]
    unit: Optional[Dict[str, Dict[str, str]]]
    observedProperty: Optional[Dict[str, Union[str, Dict[str, str]]]]


class CovJSON(TypedDict):
    """CovJSON type"""

    type: str
    domain: Domain
    parameters: Dict[str, Parameter]
    ranges: Dict


def invert_cf_dims(ds):
    """
    Return a mapping of dataset dimension name to CF axes
    """
    inverted = {}
    for key, values in ds.cf.axes.items():
        for value in values:
            inverted[value] = key.lower()
    return inverted


def to_cf_covjson(ds: xr.Dataset):
    """Transform an xarray dataset to CoverageJSON using CF conventions"""

    covjson: CovJSON = {
        "type": "Coverage",
        "domain": {
            "type": "Domain",
            "domainType": "Grid",
            "axes": {},
            "referencing": [],
        },
        "parameters": {},
        "ranges": {},
    }

    inverted_dims = invert_cf_dims(ds)

    for name, da in ds.coords.items():
        if "datetime" in str(da.dtype):
            values = da.dt.strftime("%Y-%m-%dT%H:%M:%S%Z").values.tolist()
        else:
            values = da.values
            values = np.where(np.isnan(values), None, values).tolist()
        try:
            if not isinstance(values, list):
                try:
                    values = [values.item()]
                except AttributeError:
                    values = [values]
            covjson["domain"]["axes"][inverted_dims.get(name, name)] = {
                "values": values,
            }
        except (ValueError, TypeError):
            pass

    for var in ds.data_vars:
        da = ds[var]

        parameter: Parameter = {
            "type": "Parameter",
            "observedProperty": {},
            "description": {},
            "unit": {},
        }

        try:
            parameter["description"]["en"] = da.attrs["long_name"]
            parameter["observedProperty"]["label"] = {"en": da.attrs["long_name"]}
        except KeyError:
            pass

        try:
            parameter["unit"]["label"] = {"en": da.attrs["units"]}
        except KeyError:
            pass

        covjson["parameters"][var] = parameter

        values = da.values.ravel()
        if "datetime" in str(da.dtype):
            values = da.dt.strftime("%Y-%m-%dT%H:%M:%S%Z").values.tolist()
            dataType = "string"
        else:
            values = np.where(np.isnan(values), None, values).tolist()

            if da.dtype.kind in ("i", "u"):
                values = [int(v) for v in values]
                dataType = "integer"
            elif da.dtype.kind in ("f", "c"):
                dataType = "float"
            else:
                dataType = "string"

        cov_range = {
            "type": "NdArray",
            "dataType": dataType,
            "axisNames": [inverted_dims.get(dim, dim) for dim in da.dims],
            "shape": da.shape,
            "values": values,
        }

        covjson["ranges"][var] = cov_range

    return covjson
