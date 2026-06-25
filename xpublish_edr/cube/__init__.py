"""
OGC cube query functionality

handle_query: Handle an OGC cube query, including dataset selection, spatial filtering, and projection.
params: Define the query parameters for an OGC cube query, including validation and conversion.
"""

from xpublish_edr.format import cube_formats as formats

from .handle import handle_query
from .params import EDRCubeQuery

__all__ = ["formats", "EDRCubeQuery", "handle_query"]
