# Import the various extensions submodules to make them available
# at the top level
from synthesizer.extensions.openmp_check import check_openmp

# Import an alais the galaxy factory function
from synthesizer.galaxy import galaxy
from synthesizer.galaxy import galaxy as Galaxy

# Import the main classes to make them available at the top level
from synthesizer.grid import Grid

# Import the filters module to the top level to maintain old API
# before the filters module was moved to the instruments module
from synthesizer.instruments import filters

# Import the various utils submodules to make them available
# at the top level
from synthesizer.utils import art, integrate, plt, stats, util_funcs

__all__ = [
    art,
    integrate,
    plt,
    stats,
    util_funcs,
    Grid,
    galaxy,
    Galaxy,
    check_openmp,
    filters,
]
