"""
OGC cube query functionality

params: Define the query parameters for an OGC cube query, including validation and conversion.
"""

from xpublish_edr.format import cube_formats as formats

from .query import EDRCubeQuery

__all__ = ["formats", "EDRCubeQuery"]
