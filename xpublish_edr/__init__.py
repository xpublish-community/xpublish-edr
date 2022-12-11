"""
xpublish_edr is not a real package, just a set of best practices examples.
"""

from xpublish_edr.plugin import CfEdrPlugin

__all__ = ["CfEdrPlugin"]

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"
