"""
OGC area query functionality

handle_query: Handle an OGC area query, including dataset selection, spatial filtering, and projection.
parse_body: Parse the request body for an OGC area query, including validation and conversion
params: Define the query parameters for an OGC area query, including validation and conversion.
"""

from xpublish_edr.format import area_formats as formats

from .handle import handle_query
from .params import EDRAreaQuery, EDRAreaQueryGet, EDRAreaQueryPost
from .parse import parse_body

__all__ = [
    "formats",
    "EDRAreaQuery",
    "EDRAreaQueryPost",
    "EDRAreaQueryGet",
    "handle_query",
    "parse_body",
]
