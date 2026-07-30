"""
OGC area query functionality

parse_body: Parse the request body for an OGC area query, including validation and conversion
params: Define the query parameters for an OGC area query, including validation and conversion.
"""

from xpublish_edr.format import area_formats as formats

from .parse import parse_body
from .query import EDRAreaQueryGet, EDRAreaQueryPost

__all__ = [
    "formats",
    "EDRAreaQueryPost",
    "EDRAreaQueryGet",
    "parse_body",
]
