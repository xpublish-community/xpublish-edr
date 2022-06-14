"""
xpublish_edr is not a real package, just a set of best practices examples.
"""

from xpublish_edr.cf_edr_router import cf_edr_router

__all__ = ["cf_edr_router"]

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"
