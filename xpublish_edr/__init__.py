"""
xpublish_edr is not a real package, just a set of best practices examples.
"""

from xpublish_edr.xpublish_edr import meaning_of_life, meaning_of_life_url

__all__ = ["meaning_of_life", "meaning_of_life_url"]

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"
