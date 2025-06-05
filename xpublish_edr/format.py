"""
Return response format functions from registered entry_points
"""

import importlib.metadata


def position_formats():
    """
    Return response format functions from registered
    `xpublish_edr_position_formats` entry_points
    """
    formats = {}

    entry_points = importlib.metadata.entry_points()
    for entry_point in entry_points.select(group="xpublish_edr_position_formats"):
        formats[entry_point.name] = entry_point.load()

    return formats


def area_formats():
    """
    Return response format functions from registered
    `xpublish_edr_area_formats` entry_points
    """
    formats = {}

    entry_points = importlib.metadata.entry_points()
    for entry_point in entry_points.select(group="xpublish_edr_area_formats"):
        formats[entry_point.name] = entry_point.load()

    return formats


def cube_formats():
    """
    Return response format functions from registered
    `xpublish_edr_cube_formats` entry_points
    """
    formats = {}

    entry_points = importlib.metadata.entry_points()
    for entry_point in entry_points.select(group="xpublish_edr_cube_formats"):
        formats[entry_point.name] = entry_point.load()

    return formats
