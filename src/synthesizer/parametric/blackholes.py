"""A module for working with a parametric black holes.

Contains the BlackHole class for use with parametric systems. This houses
all the attributes and functionality related to parametric black holes.

Example usages:

    bhs = BlackHole(
        bolometric_luminosity,
        mass,
        accretion_rate,
        epsilon,
        inclination,
        spin,
        metallicity,
        offset,
)
"""
import numpy as np

from synthesizer import exceptions
from synthesizer.parametric.morphology import PointSource
from synthesizer.blackhole_emission_models import Template
from synthesizer.components import BlackholesComponent


class BlackHole(BlackholesComponent):
    """
    The base parametric BlackHole class.

    Attributes:
        morphology (PointSource)
            An instance of the PointSource morphology that describes the
            location of this blackhole
    """

    def __init__(
        self,
        bolometric_luminosity=None,
        mass=None,
        accretion_rate=None,
        epsilon=0.1,
        inclination=None,
        spin=None,
        metallicity=None,
        offset=None,
    ):
        """
        Intialise the Stars instance. The first two arguments are always
        required. All other arguments are optional attributes applicable
        in different situations.

        Args:
            bolometric_luminosity (float)
                The bolometric luminosity
            mass (float)
                The mass of each particle in Msun.
            accretion_rate (float)
                The accretion rate of the/each black hole in Msun/yr.
            metallicity (float)
                The metallicity of the region surrounding the/each black hole.
            epsilon (float)
                The radiative efficiency. By default set to 0.1.
            inclination (float)
                The inclination of the blackhole. Necessary for some disc
                models.
            spin (float)
                The spin of the blackhole. Necessary for some disc models.
            offset (unyt_array, float)
                The (x,y) offsets of the blackhole relative to the centre of
                the image. Units can be length or angle but should be
                consistent with the scene.

        """

        # Initialise base class
        BlackholesComponent.__init__(
            self,
            bolometric_luminosity=bolometric_luminosity,
            mass=mass,
            accretion_rate=accretion_rate,
            epsilon=epsilon,
            inclination=inclination,
            spin=spin,
            metallicity=metallicity,
        )

        # Initialise morphology using the in-built point-source class
        self.morphology = PointSource(offset=offset)

    def _prepare_sed_args(
        self,
        grid,
        fesc,
        spectra_type,
        grid_assignment_method,
    ):
        """
        A method to prepare the arguments for SED computation with the C
        functions.

        Args:
            grid (Grid)
                The SPS grid object to extract spectra from.
            fesc (float)
                The escape fraction.
            spectra_type (str)
                The type of spectra to extract from the Grid. This must match a
                type of spectra stored in the Grid.
            grid_assignment_method (string)
                The type of method used to assign particles to a SPS grid
                point. Allowed methods are cic (cloud in cell) or nearest
                grid point (ngp) or there uppercase equivalents (CIC, NGP).
                Defaults to cic.

        Returns:
            tuple
                A tuple of all the arguments required by the C extension.
        """

        # Set up the inputs to the C function.
        grid_props = [
            np.ascontiguousarray(getattr(grid, axis), dtype=np.float64)
            for axis in grid.axes
        ]
        props = [
            np.ascontiguousarray(getattr(self, axis), dtype=np.float64)
            for axis in grid.axes
        ]

        # For black holes mass is a grid parameter but we still need to
        # multiply by mass in the extensions so just multiply by 1
        mass = np.ones(1, dtype=np.float64)

        # Make sure we get the wavelength index of the grid array
        nlam = np.int32(grid.spectra[spectra_type].shape[-1])

        # Get the grid spctra
        grid_spectra = np.ascontiguousarray(
            grid.spectra[spectra_type],
            dtype=np.float64,
        )

        # Get the grid dimensions after slicing what we need
        grid_dims = np.zeros(len(grid_props) + 1, dtype=np.int32)
        for ind, g in enumerate(grid_props):
            grid_dims[ind] = len(g)
        grid_dims[ind + 1] = nlam

        # Convert inputs to tuples
        grid_props = tuple(grid_props)
        props = tuple(props)

        return (
            grid_spectra,
            grid_props,
            props,
            mass,
            fesc,
            grid_dims,
            len(grid_props),
            np.int32(1),
            nlam,
            grid_assignment_method,
        )

    def generate_lnu(
        self,
        emission_model,
        spectra_name,
        agn_region,
        fesc=0.0,
        verbose=False,
        grid_assignment_method="cic",
        bolometric_luminosity=None,
    ):
        """
        Generate the integrated rest frame spectra for a given grid key
        spectra.
        """
        # Handle the special Template case where a spectra is scaled
        # and returned
        if isinstance(emission_model, Template):
            return emission_model.get_spectra(bolometric_luminosity)

        # Ensure we have a key in the grid. If not error.
        if spectra_name not in list(emission_model.grid[agn_region].spectra.keys()):
            raise exceptions.MissingSpectraType(
                f"The Grid does not contain the key '{spectra_name}'"
            )

        from ..extensions.integrated_spectra import compute_integrated_sed

        # Prepare the arguments for the C function.
        args = self._prepare_sed_args(
            emission_model.grid[agn_region],
            fesc=fesc,
            spectra_type=spectra_name,
            grid_assignment_method=grid_assignment_method.lower(),
        )

        # Get the integrated spectra in grid units (erg / s / Hz)
        return compute_integrated_sed(*args)
