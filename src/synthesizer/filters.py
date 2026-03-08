"""A bridge module for the filters module.

The filters module is a submodule of the instruments module.
This module provides a convenient import path for backwards compatibility.
"""

from synthesizer.instruments.filters import UVJ, Filter, FilterCollection

__all__ = ["Filter", "FilterCollection", "UVJ"]
