"""
OGC position query functionality

parse_body: Parse the request body for an OGC position query, including validation and conversion
params: Define the query parameters for an OGC position query, including validation and conversion.
"""

from xpublish_edr.format import position_formats as formats

from .parse import parse_body
from .query import EDRPositionQueryGet, EDRPositionQueryPost

__all__ = [
    "formats",
    "EDRPositionQueryPost",
    "EDRPositionQueryGet",
    "parse_body",
]
