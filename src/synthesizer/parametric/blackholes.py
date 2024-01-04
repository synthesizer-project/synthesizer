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
from unyt import unyt_array, kpc

from synthesizer import exceptions
from synthesizer.parametric.morphology import PointSource
from synthesizer.components import BlackholesComponent
from synthesizer.utils import has_units


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
        offset=np.array([0.0, 0.0]) * kpc,
        **kwargs,
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
            offset (unyt_array)
                The (x,y) offsets of the blackhole relative to the centre of
                the image. Units can be length or angle but should be
                consistent with the scene.
            kwargs (dict)
                Any parameter for the emission models can be provided as kwargs
                here to override the defaults of the emission models.
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
            **kwargs,
        )

        # Ensure the offset has units
        if not has_units(offset):
            raise exceptions.MissingUnits(
                "The offset must be provided with units"
            )

        # Initialise morphology using the in-built point-source class
        self.morphology = PointSource(offset=offset)

    def _prepare_sed_args(
        self,
        emission_model,
        grid,
        fesc,
        spectra_type,
        line_region,
        grid_assignment_method,
        **kwargs,
    ):
        """
        A method to prepare the arguments for SED computation with the C
        functions.

        Args:
            emission_model (synthesizer.blackhole_emission_models.*)
                An instance of a blackhole emission model.
            grid (Grid)
                The SPS grid object to extract spectra from.
            fesc (float)
                The escape fraction.
            spectra_type (str)
                The type of spectra to extract from the Grid. This must match a
                type of spectra stored in the Grid.
            line_region (str)
                The specific line region, i.e. 'nlr' or 'blr'.
            grid_assignment_method (string)
                The type of method used to assign particles to a SPS grid
                point. Allowed methods are cic (cloud in cell) or nearest
                grid point (ngp) or there uppercase equivalents (CIC, NGP).
                Defaults to cic.
            kwargs (dict)
                Any other arguments. Mainly unused and here for consistency
                with particle version of this method which does have extra
                arguments.

        Returns:
            tuple
                A tuple of all the arguments required by the C extension.
        """

        # Set up the inputs to the C function.
        grid_props = [
            np.ascontiguousarray(getattr(grid, axis), dtype=np.float64)
            for axis in grid.axes
        ]
        props = []
        for axis in grid.axes:
            # Source parameters from the appropriate place
            if axis in emission_model.required_parameters:
                # Parameters that need to be provided from the black hole
                # (these already exclude any parameters fixed on the
                # emission model)
                props.append(getattr(self, axis))

            elif (
                axis in emission_model.variable_parameters
                and getattr(self, axis, None) is not None
            ):
                # Variable parameters defined on the black hole (not including
                # line region parameters)
                props.append(getattr(self, axis))

            elif (
                f"{axis} {line_region}" in emission_model.variable_parameters
                and getattr(self, axis + "_" + line_region, None) is not None
            ):
                # Variable line region parameters defined on the black hole
                props.append(getattr(self, axis + "_" + line_region))

            elif getattr(emission_model, axis, None) is not None:
                # Parameters required from the emission_model (not including
                # line region parameters)
                props.append(getattr(emission_model, axis))

            elif (
                getattr(emission_model, axis + "_" + line_region, None)
                is not None
            ):
                # Line region parameters required from the emission_model
                props.append(getattr(emission_model, axis + "_" + line_region))

            else:
                raise exceptions.MissingArgument(
                    f"{axis} can't be sourced from emission_model or self."
                )

        # Remove units from any unyt_arrays and make contiguous
        props = [
            prop.value if isinstance(prop, unyt_array) else prop
            for prop in props
        ]
        props = [
            np.ascontiguousarray(prop, dtype=np.float64) for prop in props
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
            np.array([fesc]),
            grid_dims,
            len(grid_props),
            np.int32(1),
            nlam,
            grid_assignment_method,
        )
