"""A bridge module to handle the move of the filters module.

The filters module is now a submodule of the instruments module.
This module is a bridge to handle the move of the filters module
with a deprecation warning.
"""

from synthesizer.synth_warnings import deprecation

deprecation(
    "The filters module has been moved to the instruments module. "
    "Please update your imports "
    "synthesizer.filters -> synthesizer.instruments"
)

from synthesizer.instruments.filters import UVJ, Filter, FilterCollection

__all__ = ["Filter", "FilterCollection", "UVJ"]
