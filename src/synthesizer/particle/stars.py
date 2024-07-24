"""A module for working with arrays of stellar particles.

Contains the Stars class for use with particle based systems. This houses all
the data detailing collections of stellar particles. Each property is
stored in (N_star, ) shaped arrays for efficiency.

We also provide functions for creating "fake" stellar distributions by
sampling a SFZH.

In both cases a myriad of extra optional properties can be set by providing
them as keyword arguments.

Example usages:

    stars = Stars(initial_masses, ages, metallicities,
                  redshift=redshift, current_masses=current_masses, ...)
    stars = sample_sfhz(sfzh, n, total_initial_mass,
                        smoothing_lengths=smoothing_lengths,
                        tau_v=tau_vs, coordinates=coordinates, ...)
"""

import os

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from unyt import Hz, angstrom, erg, kpc, s

from synthesizer import exceptions
from synthesizer.components import StarsComponent
from synthesizer.extensions.timers import tic, toc
from synthesizer.line import Line
from synthesizer.parametric import SFH
from synthesizer.parametric import Stars as Para_Stars
from synthesizer.particle.particles import Particles
from synthesizer.units import Quantity
from synthesizer.utils import TableFormatter
from synthesizer.utils.plt import single_histxy
from synthesizer.warnings import warn


class Stars(Particles, StarsComponent):
    """
    The base Stars class. This contains all data a collection of stars could
    contain. It inherits from the base Particles class holding attributes and
    methods common to all particle types.

    The Stars class can be handed to methods elsewhere to pass information
    about the stars needed in other computations. For example a Galaxy object
    can be passed a stars object for use with any of the Galaxy helper methods.

    Note that due to the many possible operations, this class has a large
    number of optional attributes which are set to None if not provided.

    Attributes:
        initial_masses (array-like, float)
            The intial stellar mass of each particle in Msun.
        ages (array-like, float)
            The age of each stellar particle in yrs.
        metallicities (array-like, float)
            The metallicity of each stellar particle.
        tau_v (array-like, float)
            V-band dust optical depth of each stellar particle.
        alpha_enhancement (array-like, float)
            The alpha enhancement [alpha/Fe] of each stellar particle.
        resampled (bool)
            Flag for whether the young particles have been resampled.
        current_masses (array-like, float)
            The current mass of each stellar particle in Msun.
        smoothing_lengths (array-like, float)
            The smoothing lengths (describing the sph kernel) of each stellar
            particle in simulation length units.
        s_oxygen (array-like, float)
            fractional oxygen abundance.
        s_hydrogen (array-like, float)
            fractional hydrogen abundance.
        imf_hmass_slope (float)
            The slope of high mass end of the initial mass function (WIP).
        nstars (int)
            The number of stellar particles in the object.
    """

    # Define the allowed attributes
    attrs = [
        "nparticles",
        "tau_v",
        "alpha_enhancement",
        "imf_hmass_slope",
        "log10ages",
        "log10metallicities",
        "resampled",
        "velocities",
        "s_oxygen",
        "s_hydrogen",
        "nstars",
        "tau_v",
        "_coordinates",
        "_smoothing_lengths",
        "_softening_lengths",
        "_masses",
        "_initial_masses",
        "_current_masses",
    ]

    # Define class level Quantity attributes
    initial_masses = Quantity()
    current_masses = Quantity()
    smoothing_lengths = Quantity()

    def __init__(
        self,
        initial_masses,
        ages,
        metallicities,
        redshift=None,
        tau_v=None,
        alpha_enhancement=None,
        coordinates=None,
        velocities=None,
        current_masses=None,
        smoothing_lengths=None,
        s_oxygen=None,
        s_hydrogen=None,
        softening_length=None,
        centre=None,
        metallicity_floor=1e-5,
        **kwargs,
    ):
        """
        Intialise the Stars instance.

        The first 3 arguments are always required. All other arguments are
        optional attributes applicable in different situations.

        Args:
            initial_masses (array-like, float)
                The intial stellar mass of each particle in Msun.
            ages (array-like, float)
                The age of each stellar particle in yrs.
            metallicities (array-like, float)
                The metallicity of each stellar particle.
            redshift (float)
                The redshift/s of the stellar particles.
            tau_v (array-like, float)
                V-band dust optical depth of each stellar particle.
            alpha_enhancement (array-like, float)
                The alpha enhancement [alpha/Fe] of each stellar particle.
            coordinates (array-like, float)
                The 3D positions of the particles.
            velocities (array-like, float)
                The 3D velocities of the particles.
            current_masses (array-like, float)
                The current mass of each stellar particle in Msun.
            smoothing_lengths (array-like, float)
                The smoothing lengths (describing the sph kernel) of each
                stellar particle in simulation length units.
            s_oxygen (array-like, float)
                The fractional oxygen abundance.
            s_hydrogen (array-like, float)
                The fractional hydrogen abundance.
            softening_length (float)
                The gravitational softening lengths of each stellar
                particle in simulation units
            centre (array-like, float)
                The centre of the star particle. Can be defined in
                a number of way (e.g. centre of mass)
            metallicity_floor (float)
                The minimum metallicity allowed in the simulation.
            **kwargs
                Additional keyword arguments to be set as attributes.
        """
        # Instantiate parents
        Particles.__init__(
            self,
            coordinates=coordinates,
            velocities=velocities,
            masses=current_masses,
            redshift=redshift,
            softening_length=softening_length,
            nparticles=len(initial_masses),
            centre=centre,
            metallicity_floor=metallicity_floor,
        )
        StarsComponent.__init__(self, ages, metallicities)

        # Ensure initial masses is an accepted type to avoid
        # issues when masking
        if isinstance(initial_masses, list):
            raise exceptions.InconsistentArguments(
                "Initial mass should be numpy or unyt array."
            )

        # Set always required stellar particle properties
        self.initial_masses = initial_masses
        self.ages = ages
        self.metallicities = metallicities

        # Set the extra keyword arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Set the optional keyword arguments

        # Set the SPH kernel smoothing lengths
        self.smoothing_lengths = smoothing_lengths

        # Stellar particles also have a current mass, set it
        self.current_masses = self.masses

        # Set the V band optical depths
        self.tau_v = tau_v

        # Set the metallicity floor when using log properties
        self.metallicity_floor = metallicity_floor

        # Set the alpha enhancement [alpha/Fe] (only used for >2 dimensional
        # SPS grids)
        self.alpha_enhancement = alpha_enhancement

        # Set the fractional abundance of elements
        self.s_oxygen = s_oxygen
        self.s_hydrogen = s_hydrogen

        # Set up IMF properties (updated later)
        self.imf_hmass_slope = None  # slope of the imf

        # Intialise the flag for resampling
        self.resampled = False

        # Set a frontfacing clone of the number of particles
        # with clearer naming
        self.nstars = self.nparticles

        # Check the arguments we've been given
        self._check_star_args()

        # Particle stars can calculate and attach a SFZH analogous to a
        # parametric galaxy
        self.sfzh = None

    @property
    def log10ages(self):
        """
        Return stellar particle ages in log (base 10).

        Returns:
            log10ages (array)
                log10 stellar ages
        """
        return np.log10(self.ages)

    @property
    def log10metallicities(self):
        """
        Return stellar particle ages in log (base 10).

        Zero valued metallicities are set to `metallicity_floor`,
        which is set on initialisation of this stars object. To
        check it, run `stars.metallicity_floor`.

        Returns:
            log10metallicities (array)
                log10 stellar metallicities.
        """
        mets = self.metallicities
        mets[mets == 0.0] = self.metallicity_floor

        return np.log10(mets)

    def _check_star_args(self):
        """
        Sanitizes the inputs ensuring all arguments agree and are compatible.

        Raises:
            InconsistentArguments
                If any arguments are incompatible or not as expected an error
                is thrown.
        """

        # Ensure all arrays are the expected length
        for key in self.attrs:
            attr = getattr(self, key)
            if isinstance(attr, np.ndarray):
                if attr.shape[0] != self.nparticles:
                    raise exceptions.InconsistentArguments(
                        "Inconsistent stellar array sizes! (nparticles=%d, "
                        "%s=%d)" % (self.nparticles, key, attr.shape[0])
                    )

    def __str__(self):
        """
        Return a string representation of the stars object.

        Returns:
            table (str)
                A string representation of the particle object.
        """
        # Intialise the table formatter
        formatter = TableFormatter(self)

        return formatter.get_table("Stars")

    def _prepare_sed_args(
        self,
        grid,
        fesc,
        spectra_type,
        mask,
        grid_assignment_method,
        nthreads,
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
            mask (bool)
                A mask to be applied to the stars. Spectra will only be
                computed and returned for stars with True in the mask.
            grid_assignment_method (string)
                The type of method used to assign particles to a SPS grid
                point. Allowed methods are cic (cloud in cell) or nearest
                grid point (ngp) or there uppercase equivalents (CIC, NGP).
                Defaults to cic.
            nthreads (int)
                The number of threads to use in the C extension. If -1 then
                all available threads are used.

        Returns:
            tuple
                A tuple of all the arguments required by the C extension.
        """

        # Make a dummy mask if none has been passed
        if mask is None:
            mask = np.ones(self.nparticles, dtype=bool)

        # Set up the inputs to the C function.
        grid_props = [
            np.ascontiguousarray(grid.log10ages, dtype=np.float64),
            np.ascontiguousarray(grid.log10metallicities, dtype=np.float64),
        ]
        part_props = [
            np.ascontiguousarray(self.log10ages[mask], dtype=np.float64),
            np.ascontiguousarray(
                self.log10metallicities[mask], dtype=np.float64
            ),
        ]
        part_mass = np.ascontiguousarray(
            self._initial_masses[mask],
            dtype=np.float64,
        )

        # Make sure we set the number of particles to the size of the mask
        npart = np.int32(np.sum(mask))

        # Make sure we get the wavelength index of the grid array
        nlam = np.int32(grid.spectra[spectra_type].shape[-1])

        # Slice the spectral grids and pad them with copies of the edges.
        grid_spectra = np.ascontiguousarray(
            grid.spectra[spectra_type],
            np.float64,
        )

        # Get the grid dimensions after slicing what we need
        grid_dims = np.zeros(len(grid_props) + 1, dtype=np.int32)
        for ind, g in enumerate(grid_props):
            grid_dims[ind] = len(g)
        grid_dims[ind + 1] = nlam

        # If fesc isn't an array make it one
        if not isinstance(fesc, np.ndarray):
            fesc = np.ascontiguousarray(np.full(npart, fesc))

        # Convert inputs to tuples
        grid_props = tuple(grid_props)
        part_props = tuple(part_props)

        # If nthreads is -1 then use all available threads
        if nthreads == -1:
            nthreads = os.cpu_count()

        return (
            grid_spectra,
            grid_props,
            part_props,
            part_mass,
            fesc,
            grid_dims,
            len(grid_props),
            npart,
            nlam,
            grid_assignment_method,
            nthreads,
        )

    def generate_lnu(
        self,
        grid,
        spectra_name,
        fesc=0.0,
        young=None,
        old=None,
        mask=None,
        verbose=False,
        do_grid_check=False,
        grid_assignment_method="cic",
        parametric_young_stars=None,
        parametric_sfh="constant",
        aperture=None,
        nthreads=0,
    ):
        """
        Generate the integrated rest frame spectra for a given grid key
        spectra for all stars. Can optionally apply masks.

        Args:
            grid (Grid)
                The spectral grid object.
            spectra_name (string)
                The name of the target spectra inside the grid file.
            fesc (float/array-like, float)
                Fraction of stellar emission that escapes unattenuated from
                the birth cloud. Can either be a single value
                or an value per star (defaults to 0.0).
            young (bool/float)
                If not None, specifies age in Myr at which to filter
                for young star particles.
            old (bool/float)
                If not None, specifies age in Myr at which to filter
                for old star particles.
            mask (array)
                Boolean array of star particles to include
            verbose (bool)
                Flag for verbose output.
            do_grid_check (bool)
                Whether to check how many particles lie outside the grid. This
                is a sanity check that can be used to check the
                consistency of your particles with the grid. It is False by
                default because the check is extreme expensive.
            grid_assignment_method (string)
                The type of method used to assign particles to a SPS grid
                point. Allowed methods are cic (cloud in cell) or nearest
                grid point (ngp) or there uppercase equivalents (CIC, NGP).
                Defaults to cic.
            parametric_young_stars (bool/float)
                If not None, specifies age in Myr below which we replace
                individual star particles with a parametric SFH.
            parametric_sfh (string)
                Form of the parametric SFH to use for young stars.
                Currently two are supported, `Constant` and
                `TruncatedExponential`, selected using the keyword
                arguments `constant` and `exponential`.
            aperture (float)
                If not None, specifies the radius of a spherical aperture
                to apply to the particles.
            nthreads (int)
                The number of threads to use in the C extension. If -1 then
                all available threads are used.

        Returns:
            numpy.ndarray:
                Numpy array of integrated spectra in units of (erg / s / Hz).
        """

        # Ensure we have a key in the grid. If not error.
        if spectra_name not in list(grid.spectra.keys()):
            raise exceptions.MissingSpectraType(
                "The Grid does not contain the key '%s'" % spectra_name
            )

        # Are we checking the particles are consistent with the grid?
        if do_grid_check:
            # How many particles lie below the grid limits?
            n_below_age = self.log10ages[
                self.log10ages < grid.log10age[0]
            ].size
            n_below_metal = self.metallicities[
                self.metallicities < grid.metallicity[0]
            ].size

            # How many particles lie above the grid limits?
            n_above_age = self.log10ages[
                self.log10ages > grid.log10age[-1]
            ].size
            n_above_metal = self.metallicities[
                self.metallicities > grid.metallicity[-1]
            ].size

            # Check the fraction of particles outside of the grid (these
            # will be pinned to the edge of the grid) by finding
            # those inside
            age_inside_mask = np.logical_and(
                self.log10ages <= grid.log10age[-1],
                self.log10ages >= grid.log10age[0],
            )
            met_inside_mask = np.logical_and(
                self.metallicities < grid.metallicity[-1],
                grid.metallicity[0] < self.metallicities,
            )

            # Combine the boolean arrays for each axis
            inside_mask = np.logical_and(age_inside_mask, met_inside_mask)

            # Get the number outside
            n_out = self.metallicities[~inside_mask].size

            # Compute the ratio of those outside to total number
            ratio_out = n_out / self.nparticles

            # Tell the user if there are particles outside the grid
            if ratio_out > 0:
                print(
                    f"{ratio_out * 100:.2f}% of particles lie "
                    "outside the grid! "
                    "These will be pinned at the grid limits."
                )
                print("Of these:")
                print(
                    f"  {n_below_age / self.nparticles * 100:.2f}%"
                    f" have log10(ages/yr) > {grid.log10age[0]}"
                )
                print(
                    f"  {n_below_metal / self.nparticles * 100:.2f}%"
                    f" have metallicities < {grid.metallicity[0]}"
                )
                print(
                    f"  {n_above_age / self.nparticles * 100:.2f}%"
                    f" have log10(ages/yr) > {grid.log10age[-1]}"
                )
                print(
                    f"  {n_above_metal / self.nparticles * 100:.2f}%"
                    f" have metallicities > {grid.metallicity[-1]}"
                )

        # Get particle age masks
        if mask is None:
            mask = self._get_masks(young, old)

        # Ensure and warn that the masking hasn't removed everything
        if np.sum(mask) == 0:
            warn("Age mask has filtered out all particles")

            return np.zeros(len(grid.lam))

        if aperture is not None:
            # Get aperture mask
            aperture_mask = self._aperture_mask(aperture_radius=aperture)

            # Ensure and warn that the masking hasn't removed everything
            if np.sum(aperture_mask) == 0:
                warn("Aperture mask has filtered out all particles")

                return np.zeros(len(grid.lam))
        else:
            aperture_mask = np.ones(self.nparticles, dtype=bool)

        if parametric_young_stars:
            # Get mask for particles we're going to replace with parametric
            pmask = self._get_masks(parametric_young_stars, None)

            # Check we have particles to replace
            if np.sum(pmask) > 0:
                # Update the young/old mask to ignore those we're replacing
                mask[pmask] = False

                lnu_parametric = self._parametric_young_stars(
                    pmask=pmask & aperture_mask,
                    age=parametric_young_stars,
                    parametric_sfh=parametric_sfh,
                    grid=grid,
                    spectra_name=spectra_name,
                )
            else:
                # Create a dummy empty array
                lnu_parametric = np.zeros(len(grid.lam))

            if np.sum(mask) == 0:
                warn("All particles replaced with parametric forms")
                return lnu_parametric

        from ..extensions.integrated_spectra import compute_integrated_sed

        # Prepare the arguments for the C function.
        args = self._prepare_sed_args(
            grid,
            fesc=fesc,
            spectra_type=spectra_name,
            mask=mask & aperture_mask,
            grid_assignment_method=grid_assignment_method.lower(),
            nthreads=nthreads,
        )

        # Get the integrated spectra in grid units (erg / s / Hz)
        lnu_particle = compute_integrated_sed(*args)

        if parametric_young_stars:
            return lnu_particle + lnu_parametric
        else:
            return lnu_particle

    def _aperture_mask(self, aperture_radius):
        """
        Mask for particles within spherical aperture.

        Args:
            aperture_radius (float)
                Radius of spherical aperture in kpc
        """

        distance = np.sqrt(
            np.sum(
                (self.coordinates - self.centre).to(kpc) ** 2,
                axis=1,
            )
        )

        return distance < aperture_radius

    def _parametric_young_stars(
        self,
        pmask,
        age,
        parametric_sfh,
        grid,
        spectra_name,
    ):
        """
        Replace young stars with individual parametric SFH's. Can be either a
        constant or truncated exponential, selected with the `parametric_sfh`
        argument. Returns the emission from these replaced particles assuming
        this SFH. The metallicity is set to the metallicity of the parent
        star particle.

        Args:
            pmask (bool array)
                Star particles to replace
            age (float)
                Age in Myr below which we replace Star particles.
                Used to set the duration of parametric SFH
            parametric_sfh (string)
                Form of the parametric SFH to use for young stars.
                Currently two are supported, `Constant` and
                `TruncatedExponential`, selected using the keyword
                arguments `constant` and `exponential`.
            grid (Grid)
                The spectral grid object.
            spectra_name (string)
                The name of the target spectra inside the grid file.

        Returns:
            numpy.ndarray:
                Numpy array of integrated spectra in units of (erg / s / Hz).
        """

        # initialise SFH object
        if parametric_sfh == "constant":
            sfh = SFH.Constant(duration=age)
        elif parametric_sfh == "exponential":
            sfh = SFH.TruncatedExponential(tau=age / 2, max_age=age)
        else:
            raise ValueError(
                (
                    "Value of `parametric_sfh` provided, "
                    f"`{parametric_sfh}`, is not supported."
                    "Please use 'constant' or 'exponential'."
                )
            )

        stars = [None] * np.sum(pmask)

        # Loop through particles to be replaced
        for i, _pmask in enumerate(np.where(pmask)[0]):
            # Create a parametric Stars object
            stars[i] = Para_Stars(
                grid.log10age,
                grid.metallicity,
                sf_hist=sfh,
                metal_dist=self.metallicities[_pmask],
                initial_mass=self.initial_masses[_pmask],
            )

        if len(stars) > 1:
            # Combine the individual parametric forms for each particle
            stars = sum(stars[1:], stars[0])
        else:
            stars = stars[0]

        # Get the spectra for this parametric form
        return stars.generate_lnu(grid=grid, spectra_name=spectra_name)

    def _prepare_line_args(
        self,
        grid,
        line_id,
        fesc,
        mask,
        grid_assignment_method,
        nthreads,
    ):
        """
        Generate the arguments for the C extension to compute lines.

        Args:
            grid (Grid)
                The SPS grid object to extract spectra from.
            line_id (str)
                The id of the line to extract.
            fesc (float/array-like, float)
                Fraction of stellar emission that escapes unattenuated from
                the birth cloud. Can either be a single value
                or an value per star (defaults to 0.0).
            mask (bool)
                A mask to be applied to the stars. Spectra will only be
                computed and returned for stars with True in the mask.
            grid_assignment_method (string)
                The type of method used to assign particles to a SPS grid
                point. Allowed methods are cic (cloud in cell) or nearest
                grid point (ngp) or there uppercase equivalents (CIC, NGP).
                Defaults to cic.
            nthreads (int)
                The number of threads to use in the C extension. If -1 then
                all available threads are used.
        """
        # Make a dummy mask if none has been passed
        if mask is None:
            mask = np.ones(self.nparticles, dtype=bool)

        # Set up the inputs to the C function.
        grid_props = [
            np.ascontiguousarray(grid.log10ages, dtype=np.float64),
            np.ascontiguousarray(grid.log10metallicities, dtype=np.float64),
        ]
        part_props = [
            np.ascontiguousarray(self.log10ages[mask], dtype=np.float64),
            np.ascontiguousarray(
                self.log10metallicities[mask], dtype=np.float64
            ),
        ]
        part_mass = np.ascontiguousarray(
            self._initial_masses[mask],
            dtype=np.float64,
        )

        # Make sure we set the number of particles to the size of the mask
        npart = np.int32(np.sum(mask))

        # Get the line grid and continuum
        grid_line = np.ascontiguousarray(
            grid.lines[line_id]["luminosity"],
            np.float64,
        )
        grid_continuum = np.ascontiguousarray(
            grid.lines[line_id]["continuum"],
            np.float64,
        )

        # Get the grid dimensions after slicing what we need
        grid_dims = np.zeros(len(grid_props) + 1, dtype=np.int32)
        for ind, g in enumerate(grid_props):
            grid_dims[ind] = len(g)

        # If fesc isn't an array make it one
        if not isinstance(fesc, np.ndarray):
            fesc = np.ascontiguousarray(np.full(npart, fesc))

        # Convert inputs to tuples
        grid_props = tuple(grid_props)
        part_props = tuple(part_props)

        # If nthreads is -1 then use all available threads
        if nthreads == -1:
            nthreads = os.cpu_count()

        return (
            grid_line,
            grid_continuum,
            grid_props,
            part_props,
            part_mass,
            fesc,
            grid_dims,
            len(grid_props),
            npart,
            grid_assignment_method,
            nthreads,
        )

    def generate_line(
        self,
        grid,
        line_id,
        fesc,
        mask=None,
        method="cic",
        nthreads=0,
        verbose=False,
    ):
        """
        Calculate rest frame line luminosity and continuum from an SPS Grid.

        This is a flexible base method which extracts the rest frame line
        luminosity of this stellar population from the SPS grid based on the
        passed arguments.

        Args:
            grid (Grid):
                A Grid object.
            line_id (list/str):
                A list of line_ids or a str denoting a single line.
                Doublets can be specified as a nested list or using a
                comma (e.g. 'OIII4363,OIII4959').
            fesc (float/array-like, float)
                Fraction of stellar emission that escapes unattenuated from
                the birth cloud. Can either be a single value
                or an value per star (defaults to 0.0).
            mask (array)
                A mask to apply to the particles (only applicable to particle)
            method (str)
                The method to use for the interpolation. Options are:
                'cic' - Cloud in cell
                'ngp' - Nearest grid point
            nthreads (int)
                The number of threads to use in the C extension. If -1 then
                all available threads are used.

        Returns:
            Line
                An instance of Line contain this lines wavelenth, luminosity,
                and continuum.
        """
        from synthesizer.extensions.integrated_line import (
            compute_integrated_line,
        )

        # Ensure line_id is a string
        if not isinstance(line_id, str):
            raise exceptions.InconsistentArguments("line_id must be a string")

        # Ensure and warn that the masking hasn't removed everything
        if mask is not None and np.sum(mask) == 0:
            warn("Age mask has filtered out all particles")

            return Line(
                *[
                    Line(
                        line_id=line_id_,
                        wavelength=grid.lines[line_id_]["wavelength"]
                        * angstrom,
                        luminosity=np.zeros(self.nparticles) * erg / s,
                        continuum=np.zeros(self.nparticles) * erg / s / Hz,
                    )
                    for line_id_ in line_id.split(",")
                ]
            )

        # Set up a list to hold each individual Line
        lines = []

        # Loop over the ids in this container
        for line_id_ in line_id.split(","):
            # Strip off any whitespace (can be left by split)
            line_id_ = line_id_.strip()

            # Get this line's wavelength
            # TODO: The units here should be extracted from the grid but aren't
            # yet stored.
            lam = grid.lines[line_id_]["wavelength"] * angstrom

            # Get the luminosity and continuum
            lum, cont = compute_integrated_line(
                *self._prepare_line_args(
                    grid,
                    line_id_,
                    fesc,
                    mask=mask,
                    grid_assignment_method=method,
                    nthreads=nthreads,
                )
            )

            # Append this lines values to the containers
            lines.append(
                Line(
                    line_id=line_id_,
                    wavelength=lam,
                    luminosity=lum * erg / s,
                    continuum=cont * erg / s / Hz,
                )
            )

        # Don't init another line if there was only 1 in the first place
        if len(lines) == 1:
            return lines[0]
        else:
            return Line(*lines)

    def generate_particle_lnu(
        self,
        grid,
        spectra_name,
        fesc=0.0,
        verbose=False,
        do_grid_check=False,
        mask=None,
        grid_assignment_method="cic",
        nthreads=0,
    ):
        """
        Generate the particle rest frame spectra for a given grid key spectra
        for all stars. Can optionally apply masks.

        Args:
            grid (Grid)
                The spectral grid object.
            spectra_name (string)
                The name of the target spectra inside the grid file.
            fesc (float/array-like, float)
                Fraction of stellar emission that escapes unattenuated from
                the birth cloud. Can either be a single value
                or an value per star (defaults to 0.0).
            young (bool/float)
                If not None, specifies age in Myr at which to filter
                for young star particles.
            old (bool/float)
                If not None, specifies age in Myr at which to filter
                for old star particles.
            verbose (bool)
                Flag for verbose output. By default False.
            do_grid_check (bool)
                Whether to check how many particles lie outside the grid. This
                is a sanity check that can be used to check the
                consistency of your particles with the grid. It is False by
                default because the check is extreme expensive.
            grid_assignment_method (string)
                The type of method used to assign particles to a SPS grid
                point. Allowed methods are cic (cloud in cell) or nearest
                grid point (ngp) or there uppercase equivalents (CIC, NGP).
                Defaults to cic.
            nthreads (int)
                The number of threads to use in the C extension. If -1 then
                all available threads are used.

        Returns:
            numpy.ndarray:
                Numpy array of integrated spectra in units of (erg / s / Hz).
        """

        start = tic()

        # Ensure we have a total key in the grid. If not error.
        if spectra_name not in list(grid.spectra.keys()):
            raise exceptions.MissingSpectraType(
                f"The Grid does not contain the key '{spectra_name}'"
            )

        # Are we checking the particles are consistent with the grid?
        if do_grid_check:
            # How many particles lie below the grid limits?
            n_below_age = self.log10ages[
                self.log10ages < grid.log10age[0]
            ].size
            n_below_metal = self.metallicities[
                self.metallicities < grid.metallicity[0]
            ].size

            # How many particles lie above the grid limits?
            n_above_age = self.log10ages[
                self.log10ages > grid.log10age[-1]
            ].size
            n_above_metal = self.metallicities[
                self.metallicities > grid.metallicity[-1]
            ].size

            # Check the fraction of particles outside of the grid (these will
            # be pinned to the edge of the grid) by finding those inside
            age_inside_mask = np.logical_and(
                self.log10ages <= grid.log10age[-1],
                self.log10ages >= grid.log10age[0],
            )
            met_inside_mask = np.logical_and(
                self.metallicities < grid.metallicity[-1],
                grid.metallicity[0] < self.metallicities,
            )

            # Combine the boolean arrays for each axis
            inside_mask = np.logical_and(age_inside_mask, met_inside_mask)

            # Get the number outside
            n_out = self.metallicities[~inside_mask].size

            # Compute the ratio of those outside to total number
            ratio_out = n_out / self.nparticles

            # Tell the user if there are particles outside the grid
            if ratio_out > 0:
                print(
                    f"{ratio_out * 100:.2f}% of particles "
                    "lie outside the grid! "
                    "These will be pinned at the grid limits."
                )
                print("Of these:")
                print(
                    f"  {n_below_age / self.nparticles * 100:.2f}%"
                    f" have log10(ages/yr) < {grid.log10age[0]}"
                )
                print(
                    f"  {n_below_metal / self.nparticles * 100:.2f}% "
                    f"have metallicities < {grid.metallicity[0]}"
                )
                print(
                    f"  {n_above_age / self.nparticles * 100:.2f}% "
                    f"have log10(ages/yr) > {grid.log10age[-1]}"
                )
                print(
                    f"  {n_above_metal / self.nparticles * 100:.2f}% "
                    f"have metallicities > {grid.metallicity[-1]}"
                )

        # Ensure and warn that the masking hasn't removed everything
        if np.sum(mask) == 0:
            warn("Age mask has filtered out all particles")

            return np.zeros((self.nstars, len(grid.lam)))

        from ..extensions.particle_spectra import compute_particle_seds

        # Prepare the arguments for the C function.
        args = self._prepare_sed_args(
            grid,
            fesc=fesc,
            spectra_type=spectra_name,
            mask=mask,
            grid_assignment_method=grid_assignment_method.lower(),
            nthreads=nthreads,
        )
        toc("Preparing C args", start)

        # Get the integrated spectra in grid units (erg / s / Hz)
        masked_spec = compute_particle_seds(*args)

        start = tic()

        # If there's no mask we're done
        if mask is None:
            return masked_spec

        # If we have a mask we need to account for the zeroed spectra
        spec = np.zeros((self.nstars, masked_spec.shape[-1]))
        spec[mask] = masked_spec

        toc("Masking spectra and adding contribution", start)

        return spec

    def generate_particle_line(
        self,
        grid,
        line_id,
        fesc,
        mask=None,
        method="cic",
        nthreads=0,
        verbose=False,
    ):
        """
        Calculate rest frame line luminosity and continuum from an SPS Grid.

        This is a flexible base method which extracts the rest frame line
        luminosity of this stellar population from the SPS grid based on the
        passed arguments and calculate the luminosity and continuum for
        each individual particle.

        Args:
            grid (Grid):
                A Grid object.
            line_id (list/str):
                A list of line_ids or a str denoting a single line.
                Doublets can be specified as a nested list or using a
                comma (e.g. 'OIII4363,OIII4959').
            fesc (float/array-like, float)
                Fraction of stellar emission that escapes unattenuated from
                the birth cloud. Can either be a single value
                or an value per star (defaults to 0.0).
            mask (array)
                A mask to apply to the particles (only applicable to particle)
            method (str)
                The method to use for the interpolation. Options are:
                'cic' - Cloud in cell
                'ngp' - Nearest grid point
            nthreads (int)
                The number of threads to use in the C extension. If -1 then
                all available threads are used.

        Returns:
            Line
                An instance of Line contain this lines wavelenth, luminosity,
                and continuum.
        """
        from synthesizer.extensions.particle_line import (
            compute_particle_line,
        )

        # Ensure line_id is a string
        if not isinstance(line_id, str):
            raise exceptions.InconsistentArguments("line_id must be a string")

        # Ensure and warn that the masking hasn't removed everything
        if np.sum(mask) == 0:
            warn("Age mask has filtered out all particles")

            return Line(
                *[
                    Line(
                        line_id=line_id_,
                        wavelength=grid.lines[line_id_]["wavelength"]
                        * angstrom,
                        luminosity=0 * erg / s,
                        continuum=0 * erg / s / Hz,
                    )
                    for line_id_ in line_id.split(",")
                ]
            )

        # Set up a list to hold each individual Line
        lines = []

        # Loop over the ids in this container
        for line_id_ in line_id.split(","):
            # Strip off any whitespace (can be left by split)
            line_id_ = line_id_.strip()

            # Get this line's wavelength
            # TODO: The units here should be extracted from the grid but aren't
            # yet stored.
            lam = grid.lines[line_id_]["wavelength"] * angstrom

            # Get the luminosity and continuum
            _lum, _cont = compute_particle_line(
                *self._prepare_line_args(
                    grid,
                    line_id_,
                    fesc,
                    mask=mask,
                    grid_assignment_method=method,
                    nthreads=nthreads,
                )
            )

            # Account for the mask
            if mask is not None:
                lum = np.zeros(self.nparticles)
                cont = np.zeros(self.nparticles)
                lum[mask] = _lum
                cont[mask] = _cont
            else:
                lum = _lum
                cont = _cont

            # Append this lines values to the containers
            lines.append(
                Line(
                    line_id=line_id_,
                    wavelength=lam,
                    luminosity=lum * erg / s,
                    continuum=cont * erg / s / Hz,
                )
            )

        # Don't init another line if there was only 1 in the first place
        if len(lines) == 1:
            return lines[0]
        else:
            return Line(*lines)

    def _get_masks(self, young=None, old=None):
        """
        Get masks for which components we are handling, if a sub-component
        has not been requested it's necessarily all particles.

        Args:
            young (float):
                Age in Myr at which to filter for young star particles.
            old (float):
                Age in Myr at which to filter for old star particles.

        Raises:
            InconsistentParameter
                Can't select for both young and old components
                simultaneously

        """

        # We can't have both young and old set
        if young and old:
            raise exceptions.InconsistentParameter(
                "Galaxy sub-component can not be simultaneously young and old"
            )

        # Get the appropriate mask
        if young:
            # Mask out old stars
            s = self.log10ages <= np.log10(young.to("yr"))
        elif old:
            # Mask out young stars
            s = self.log10ages > np.log10(old.to("yr"))
        else:
            # Nothing to mask out
            s = np.ones(self.nparticles, dtype=bool)

        return s

    def renormalise_mass(self, stellar_mass):
        """
        Renormalises and overwrites the initial masses. Useful when rescaling
        the mass of the system of stellar particles.

        Args:
            stellar_mass (array-like, float)
                The stellar mass array to be renormalised.
        """

        self.initial_masses *= stellar_mass / np.sum(self.initial_masses)

    def _power_law_sample(self, low_lim, upp_lim, g, size=1):
        """
        Sample from a power law over an interval not containing zero.

        Power-law gen for pdf(x) propto x^{g-1} for a<=x<=b

        Args:
            low_lim (float)
                The lower bound of the interval over which to calulcate the
                power law.
            upp_lim (float)
                The upper bound of the interval over which to calulcate the
                power law.
            g (float)
                The power law index.
            size (int)
                The number of samples in the interval.

        Returns:
            array-like (float)
                The samples derived from the power law.
        """

        # Get a random sample
        rand = np.random.random(size=size)

        # Compute the value of the power law at the lower and upper bounds
        low_lim_g, upp_lim_g = low_lim**g, low_lim**g

        return (low_lim_g + (upp_lim_g - low_lim_g) * rand) ** (1 / g)

    def resample_young_stars(
        self,
        min_age=1e8,
        min_mass=700,
        max_mass=1e6,
        power_law_index=-1.3,
        n_samples=1e3,
        force_resample=False,
        verbose=False,
    ):
        """
        Resample young stellar particles into individual HII regions, with a
        power law distribution of masses. A young stellar particle is a
        stellar particle with an age < min_age (defined in Myr?).

        This function overwrites the properties stored in attributes with the
        resampled properties.

        Note: Resampling and imaging are not supported. If attempted an error
              is thrown.

        Args:
            min_age (float)
                The age below which stars will be resampled, in yrs.
            min_mass (float)
                The lower bound of the mass interval used in the power law
                sampling, in Msun.
            max_mass (float)
                The upper bound of the mass interval used in the power law
                sampling, in Msun.
            power_law_index (float)
                The index of the power law from which to sample stellar
            n_samples (int)
                The number of samples to generate for each stellar particles
                younger than min_age.
            force_resample (bool)
                A flag for whether resampling should be forced. Only applicable
                if trying to resample and already resampled Stars object.
            verbose (bool)
                Are we talking?
        """

        # Warn the user we are resampling a resampled population
        if self.resampled and not force_resample:
            warn(
                "Galaxy stars already resampled. "
                "To force resample, set force_resample=True. Returning..."
            )
            return None

        if verbose:
            print("Masking resample stars")

        # Get the indices of young stars for resampling
        resample_idxs = np.where(self.ages < min_age)[0]

        # No work to do here, stars are too old
        if len(resample_idxs) == 0:
            return None

        # Set up container for the resample stellar particles
        new_ages = {}
        new_masses = {}

        if verbose:
            print("Loop through resample stars")

        # Loop over the young stars we need to resample
        for _idx in resample_idxs:
            # Sample the power law
            rvs = self._power_law_sample(
                min_mass, max_mass, power_law_index, int(n_samples)
            )

            # If not enough mass has been sampled, repeat
            while np.sum(rvs) < self.masses[_idx]:
                n_samples *= 2
                rvs = self._power_law_sample(
                    min_mass, max_mass, power_law_index, int(n_samples)
                )

            # Sum masses up to the total mass limit
            _mask = np.cumsum(rvs) < self.masses[_idx]
            _masses = rvs[_mask]

            # Scale up to the original mass
            _masses *= self.masses[_idx] / np.sum(_masses)

            # Sample uniform distribution of ages
            _ages = np.random.rand(len(_masses)) * min_age

            # Store our resampled properties
            new_ages[_idx] = _ages
            new_masses[_idx] = _masses

        # Unpack the resample properties and make note of how many particles
        # were produced
        new_lens = [len(new_ages[_idx]) for _idx in resample_idxs]
        new_ages = np.hstack([new_ages[_idx] for _idx in resample_idxs])
        new_masses = np.hstack([new_masses[_idx] for _idx in resample_idxs])

        if verbose:
            print("Concatenate new arrays to existing")

        # Include the resampled particles in the attributes
        for attr, new_arr in zip(["masses", "ages"], [new_masses, new_ages]):
            attr_array = getattr(self, attr)
            setattr(self, attr, np.append(attr_array, new_arr))

        if verbose:
            print("Duplicate existing attributes")

        # Handle the other propertys that need duplicating
        for attr in Stars.attrs:
            # Skip unset attributes
            if getattr(self, attr) is None:
                continue

            # Include resampled stellar particles in this attribute
            attr_array = getattr(self, attr)[resample_idxs]
            setattr(
                self,
                attr,
                np.append(
                    getattr(self, attr),
                    np.repeat(attr_array, new_lens, axis=0),
                ),
            )

        if verbose:
            print("Delete old particles")

        # Loop over attributes
        for attr in Stars.attrs:
            # Skip unset attributes
            if getattr(self, attr) is None:
                continue

            # Delete the original stellar particles that have been resampled
            attr_array = getattr(self, attr)
            attr_array = np.delete(attr_array, resample_idxs)
            setattr(self, attr, attr_array)

        # Recalculate log attributes
        self.log10ages = np.log10(self.ages)
        self.log10metallicities = np.log10(self.metallicities)

        # Set resampled flag
        self.resampled = True

    def _prepare_sfzh_args(
        self,
        grid,
        grid_assignment_method,
        nthreads,
    ):
        """
        Prepare the arguments for SFZH computation with the C functions.

        Args:
            grid (Grid)
                The SPS grid object to extract spectra from.
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
            np.ascontiguousarray(grid.log10age, dtype=np.float64),
            np.ascontiguousarray(grid.metallicity, dtype=np.float64),
        ]
        part_props = [
            np.ascontiguousarray(self.log10ages, dtype=np.float64),
            np.ascontiguousarray(self.metallicities, dtype=np.float64),
        ]
        part_mass = np.ascontiguousarray(
            self._initial_masses, dtype=np.float64
        )

        # Make sure we set the number of particles to the size of the mask
        npart = np.int32(len(part_mass))

        # Get the grid dimensions after slicing what we need
        grid_dims = np.zeros(len(grid_props), dtype=np.int32)
        for ind, g in enumerate(grid_props):
            grid_dims[ind] = len(g)

        # Convert inputs to tuples
        grid_props = tuple(grid_props)
        part_props = tuple(part_props)

        # If nthreads = -1 we will use all available
        if nthreads == -1:
            nthreads = os.cpu_count()

        return (
            grid_props,
            part_props,
            part_mass,
            grid_dims,
            len(grid_props),
            npart,
            grid_assignment_method,
            nthreads,
        )

    def get_sfzh(
        self,
        grid,
        grid_assignment_method="cic",
        nthreads=0,
    ):
        """
        Generate the binned SFZH history of this collection of particles.

        The binned SFZH produced by this method is equivalent to the weights
        used to extract spectra from the grid.

        Args:
            grid (Grid)
                The spectral grid object.
            grid_assignment_method (string)
                The type of method used to assign particles to a SPS grid
                point. Allowed methods are cic (cloud in cell) or nearest
                grid point (ngp) or there uppercase equivalents (CIC, NGP).
                Defaults to cic.
            nthreads (int)
                The number of threads to use in the computation. If set to -1
                all available threads will be used.

        Returns:
            numpy.ndarray:
                Numpy array of containing the SFZH.
        """

        from synthesizer.extensions.sfzh import compute_sfzh

        # Prepare the arguments for the C function.
        args = self._prepare_sfzh_args(
            grid,
            grid_assignment_method=grid_assignment_method.lower(),
            nthreads=nthreads,
        )

        # Get the SFZH
        self.sfzh = compute_sfzh(*args)

        return self.sfzh

    def plot_sfzh(self, grid, grid_assignment_method="cic", show=True):
        """
        Plot the binned SZFH.

        Args:
            show (bool)
                Should we invoke plt.show()?

        Returns:
            fig
                The Figure object contain the plot axes.
            ax
                The Axes object containing the plotted data.
        """

        # Ensure we have the SFZH
        if self.sfzh is None:
            self.get_sfzh(grid, grid_assignment_method)

        # Get the grid axes
        log10ages = grid.log10age
        log10metallicities = np.log10(grid.metallicity)

        # Create the figure and extra axes for histograms
        fig, ax, haxx, haxy = single_histxy()

        # Visulise the SFZH grid
        ax.pcolormesh(
            log10ages,
            log10metallicities,
            self.sfzh.T,
            cmap=cmr.sunburst,
        )

        # Add binned Z to right of the plot
        metal_dist = np.sum(self.sfzh, axis=0)
        haxy.fill_betweenx(
            log10metallicities,
            metal_dist / np.max(metal_dist),
            step="mid",
            color="k",
            alpha=0.3,
        )

        # Add binned SF_HIST to top of the plot
        sf_hist = np.sum(self.sfzh, axis=1)
        haxx.fill_between(
            log10ages,
            sf_hist / np.max(sf_hist),
            step="mid",
            color="k",
            alpha=0.3,
        )

        # Set plot limits
        haxy.set_xlim([0.0, 1.2])
        haxy.set_ylim(log10metallicities[0], log10metallicities[-1])
        haxx.set_ylim([0.0, 1.2])
        haxx.set_xlim(log10ages[0], log10ages[-1])

        # Set labels
        ax.set_xlabel(r"$\log_{10}(\mathrm{age}/\mathrm{yr})$")
        ax.set_ylabel(r"$\log_{10}Z$")

        # Set the limits so all axes line up
        ax.set_ylim(log10metallicities[0], log10metallicities[-1])
        ax.set_xlim(log10ages[0], log10ages[-1])

        # Shall we show it?
        if show:
            plt.show()

        return fig, ax

    def get_particle_spectra(
        self,
        emission_model,
        dust_curves=None,
        tau_v=None,
        fesc=None,
        mask=None,
        verbose=True,
        **kwargs,
    ):
        """
        Generate stellar spectra as described by the emission model.

        Args:
            emission_model (EmissionModel):
                The emission model to use.
            dust_curves (dict):
                An override to the emission model dust curves. Either:
                    - None, indicating the dust_curves defined on the emission
                      models should be used.
                    - A single dust curve to apply to all emission models.
                    - A dictionary of the form {<label>: <dust_curve instance>}
                      to use a specific dust curve instance with particular
                      properties.
            tau_v (dict):
                An override to the dust model optical depth. Either:
                    - None, indicating the tau_v defined on the emission model
                      should be used.
                    - A float to use as the optical depth for all models.
                    - A dictionary of the form {<label>: float(<tau_v>)}
                      to use a specific optical depth with a particular
                      model or {<label>: str(<attribute>)} to use an attribute
                      of the component as the optical depth.
            fesc (dict):
                An override to the emission model escape fraction. Either:
                    - None, indicating the fesc defined on the emission model
                      should be used.
                    - A float to use as the escape fraction for all models.
                    - A dictionary of the form {<label>: float(<fesc>)}
                      to use a specific escape fraction with a particular
                      model or {<label>: str(<attribute>)} to use an
                      attribute of the component as the escape fraction.
            mask (dict):
                An override to the emission model mask. Either:
                    - None, indicating the mask defined on the emission model
                      should be used.
                    - A dictionary of the form {<label>: {"attr": attr,
                      "thresh": thresh, "op": op}} to add a specific mask to
                      a particular model.
            verbose (bool):
                Are we talking?
            kwargs (dict):
                Any additional keyword arguments to pass to the generator
                function.

        Returns:
            dict: A dictionary of spectra which can be attached to the
            appropriate spectra attribute of the component
            (spectra/particle_spectra).
        """
        # Get the spectra
        spectra = emission_model._get_spectra(
            emitters={"stellar": self},
            per_particle=True,
            dust_curves=dust_curves,
            tau_v=tau_v,
            fesc=fesc,
            mask=mask,
            verbose=verbose,
            **kwargs,
        )

        # Update the spectra dictionary
        self.particle_spectra.update(spectra)

        return self.particle_spectra[emission_model.label]

    def get_particle_lines(
        self,
        line_ids,
        emission_model,
        dust_curves=None,
        tau_v=None,
        fesc=None,
        mask=None,
        verbose=True,
        **kwargs,
    ):
        """
        Generate stellar lines as described by the emission model.

        Args:
            line_ids (list):
                A list of line_ids. Doublets can be specified as a nested
                list or using a comma (e.g., 'OIII4363,OIII4959').
            emission_model (EmissionModel):
                The emission model to use.
            dust_curves (dict):
                An override to the emission model dust curves. Either:
                    - None, indicating the dust_curves defined on the emission
                      models should be used.
                    - A single dust curve to apply to all emission models.
                    - A dictionary of the form {<label>: <dust_curve instance>}
                      to use a specific dust curve instance with particular
                      properties.
            tau_v (dict):
                An override to the dust model optical depth. Either:
                    - None, indicating the tau_v defined on the emission model
                      should be used.
                    - A float to use as the optical depth for all models.
                    - A dictionary of the form {<label>: float(<tau_v>)}
                      to use a specific optical depth with a particular
                      model or {<label>: str(<attribute>)} to use an attribute
                      of the component as the optical depth.
            fesc (dict):
                An override to the emission model escape fraction. Either:
                    - None, indicating the fesc defined on the emission model
                      should be used.
                    - A float to use as the escape fraction for all models.
                    - A dictionary of the form {<label>: float(<fesc>)}
                      to use a specific escape fraction with a particular
                      model or {<label>: str(<attribute>)} to use an
                      attribute of the component as the escape fraction.
            mask (dict):
                An override to the emission model mask. Either:
                    - None, indicating the mask defined on the emission model
                      should be used.
                    - A dictionary of the form {<label>: {"attr": attr,
                      "thresh": thresh, "op": op}} to add a specific mask to
                      a particular model.
            verbose (bool):
                Are we talking?
            kwargs (dict):
                Any additional keyword arguments to pass to the generator
                function.

        Returns:
            LineCollection:
                A LineCollection object containing the lines defined by the
                root model.
        """
        # Get the lines
        lines = emission_model._get_lines(
            line_ids=line_ids,
            emitters={"stellar": self},
            per_particle=True,
            dust_curves=dust_curves,
            tau_v=tau_v,
            fesc=fesc,
            mask=mask,
            verbose=verbose,
            **kwargs,
        )

        # Update the lines dictionary
        self.particle_lines.update(lines)

        return self.particle_lines[emission_model.label]


def sample_sfhz(
    sfzh,
    log10ages,
    log10metallicities,
    nstar,
    initial_mass=1,
    **kwargs,
):
    """
    Create "fake" stellar particles by sampling a SFZH.

    Args:
        sfhz (array-like, float)
            The Star Formation Metallicity History grid
            (from parametric.Stars).
        log10ages (array-like, float)
            The log of the SFZH age axis.
        log10metallicities (array-like, float)
            The log of the SFZH metallicities axis.
        nstar (int)
            The number of stellar particles to produce.
        intial_mass (int)
            The intial mass of the fake stellar particles.

    Returns:
        stars (Stars)
            An instance of Stars containing the fake stellar particles.
    """

    # Normalise the sfhz to produce a histogram (binned in time) between 0
    # and 1.
    hist = sfzh / np.sum(sfzh)

    # Compute the cumaltive distribution function
    cdf = np.cumsum(hist.flatten())
    cdf = cdf / cdf[-1]

    # Get a random sample from the cdf
    values = np.random.rand(nstar)
    value_bins = np.searchsorted(cdf, values)

    # Convert 1D random indices to 2D indices
    x_idx, y_idx = np.unravel_index(
        value_bins, (len(log10ages), len(log10metallicities))
    )

    # Extract the sampled ages and metallicites and create an array
    random_from_cdf = np.column_stack(
        (log10ages[x_idx], log10metallicities[y_idx])
    )

    # Extract the individual logged quantities
    log10ages, log10metallicities = random_from_cdf.T

    # Instantiate Stars object with extra keyword arguments
    stars = Stars(
        initial_mass * np.ones(nstar),
        10**log10ages,
        10**log10metallicities,
        **kwargs,
    )

    return stars
