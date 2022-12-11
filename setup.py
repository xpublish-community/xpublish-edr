from setuptools import setup

pkg_name = "xpublish_edr"

setup(
    use_scm_version={
        "write_to": f"{pkg_name}/_version.py",
        "write_to_template": '__version__ = "{version}"',
        "tag_regex": r"^(?P<prefix>v)?(?P<version>[^\+]+)(?P<suffix>.*)?$",
    },
    entry_points={
        "xpublish_edr_position_formats": [
            "cf_covjson = xpublish_edr.formats.to_covjson:to_cf_covjson",
            "csv = xpublish_edr.formats.to_csv:to_csv",
            "nc = xpublish_edr.formats.to_netcdf:to_netcdf",
            "netcdf = xpublish_edr.formats.to_netcdf:to_netcdf",
            "nc4 = xpublish_edr.formats.to_netcdf:to_netcdf",
            "netcdf4 = xpublish_edr.formats.to_netcdf:to_netcdf",
        ],
        "xpublish.plugin": ["cf_edr = xpublish_edr.plugin:CfEdrPlugin"],
    },
)
