"""Shared pyproj helpers.

Kept in its own module so that both :mod:`xpublish_edr.geometry.common` and
:mod:`xpublish_edr.geometry.ugrid` can use a single cached transformer factory
without importing each other (``common`` already imports ``ugrid``).
"""

from functools import lru_cache, partial

import pyproj

# https://pyproj4.github.io/pyproj/stable/advanced_examples.html#caching-pyproj-objects
transformer_from_crs = lru_cache(partial(pyproj.Transformer.from_crs, always_xy=True))
