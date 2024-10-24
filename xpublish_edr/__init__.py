"""
Xpublish routers for the OGC EDR API.
"""

from xpublish_edr.plugin import CfEdrPlugin

__all__ = ["CfEdrPlugin"]

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"
