"""
OGC position query functionality

handle_query: Handle an OGC position query, including dataset selection, spatial filtering, and projection.
parse_body: Parse the request body for an OGC position query, including validation and conversion
params: Define the query parameters for an OGC position query, including validation and conversion.
"""

from xpublish_edr.format import position_formats as formats

from .handle import handle_query
from .params import EDRPositionQuery, EDRPositionQueryGet, EDRPositionQueryPost
from .parse import parse_body

__all__ = [
    "formats",
    "EDRPositionQuery",
    "EDRPositionQueryPost",
    "EDRPositionQueryGet",
    "handle_query",
    "parse_body",
]
