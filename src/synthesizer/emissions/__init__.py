"""A module defining the emission classes and related utilities."""

import sys

# Import line aliases
from synthesizer.emissions.utils import aliases as line_aliases

# Unpack the aliases
for alias, line in line_aliases.items():
    setattr(sys.modules[__name__], alias, line)

# Import the emissions classe
from synthesizer.emissions.line import LineCollection
from synthesizer.emissions.sed import (
    Sed,
    plot_observed_spectra,
    plot_spectra,
    plot_spectra_as_rainbow,
)

# Import some important utility functions
from synthesizer.emissions.utils import (
    combine_list_of_seds,
    get_attenuation,
    get_attenuation_at_1500,
    get_attenuation_at_5500,
    get_attenuation_at_lam,
    get_transmission,
)

__all__ = [
    "LineCollection",
    "Sed",
    "plot_observed_spectra",
    "plot_spectra",
    "plot_spectra_as_rainbow",
    "combine_list_of_seds",
    "get_attenuation",
    "get_attenuation_at_1500",
    "get_attenuation_at_5500",
    "get_attenuation_at_lam",
    "get_transmission",
]

# Add the line aliases to the __all__ list
__all__.extend(line_aliases.keys())
