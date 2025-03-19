"""A module defining the emission classes and related utilities."""

import sys

# Import line aliases
from synthesizer.emissions.utils import aliases as line_aliases

# Unpack the aliases
for alias, line in line_aliases.items():
    setattr(sys.modules[__name__], alias, line)

# Import the emissions classe
from synthesizer.emissions.line import LineCollection
from synthesizer.emissions.sed import Sed
