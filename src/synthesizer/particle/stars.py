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
import warnings

import numpy as np
import matplotlib.pyplot as plt

from synthesizer.components import StarsComponent
from synthesizer.particle.particles import Particles
from synthesizer.sed import Sed
from synthesizer.units import Quantity
from synthesizer import exceptions


class Stars(Particles, StarsComponent):
    """
    The base Stars class. This contains all data a collection of stars could
    contain. It inherits from the base Particles class holding attributes and
    methods common to all particle types.

    The Stars class can be handed to methods elsewhere to pass information
    about the stars needed in other computations. For example a Galaxy object
    can be passed a stars object for use with any of the Galaxy helper methods.

    Note that due to the many possible operations, this class has a large number
    of optional attributes which are set to None if not provided.

    Attributes:
        initial_masses (array-like, float)
            The intial stellar mass of each particle in Msun.
        ages (array-like, float)
            The age of each stellar particle in Myrs.
        metallicities (array-like, float)
            The metallicity of each stellar particle.
        tau_v (array-like, float)
            V-band dust optical depth of each stellar particle.
        alpha_enhancement (array-like, float)
            The alpha enhancement [alpha/Fe] of each stellar particle.
        log10ages (array-like, float)
            Convnience attribute containing log10(age).
        log10metallicities (array-like, float)
            Convnience attribute containing log10(metallicity).
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
    __slots__ = [
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
    ):
        """
        Intialise the Stars instance. The first 3 arguments are always required.
        All other arguments are optional attributes applicable in different
        situations.

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
            imf_hmass_slope (float)
                The slope of high mass end of the initial mass function (WIP)
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
        )
        StarsComponent.__init__(self, ages, metallicities)

        # Set always required stellar particle properties
        self.initial_masses = initial_masses
        self.ages = ages
        self.metallicities = metallicities

        # Define the dictionary to hold particle spectra
        self.particle_spectra = {}

        # Set the optional keyword arguments

        # Set the SPH kernel smoothing lengths
        self.smoothing_lengths = smoothing_lengths

        # Stellar particles also have a current mass, set it
        self.current_masses = self.masses

        # Set the V band optical depths
        self.tau_v = tau_v

        # Set the alpha enhancement [alpha/Fe] (only used for >2 dimensional
        # SPS grids)
        self.alpha_enhancement = alpha_enhancement

        # Set the fractional abundance of elements
        self.s_oxygen = s_oxygen
        self.s_hydrogen = s_hydrogen

        # Compute useful logged quantities
        self.log10ages = np.log10(self.ages)
        self.log10metallicities = np.log10(self.metallicities)

        # Set up IMF properties (updated later)
        self.imf_hmass_slope = None  # slope of the imf

        # Intialise the flag for resampling
        self.resampled = False

        # Set a frontfacing clone of the number of particles with clearer naming
        self.nstars = self.nparticles

        # Check the arguments we've been given
        self._check_star_args()

    def _check_star_args(self):
        """
        Sanitizes the inputs ensuring all arguments agree and are compatible.

        Raises:
            InconsistentArguments
                If any arguments are incompatible or not as expected an error
                is thrown.
        """

        # Ensure all arrays are the expected length
        for key in self.__slots__:
            attr = getattr(self, key)
            if isinstance(attr, np.ndarray):
                if attr.shape[0] != self.nparticles:
                    raise exceptions.InconsistentArguments(
                        "Inconsistent stellar array sizes! (nparticles=%d, "
                        "%s=%d)" % (self.nparticles, key, attr.shape[0])
                    )

    def __str__(self):
        """
        Overloads the __str__ operator, enabling the printing of a summary of
        the Stars with print(stars) syntax, where stars is an instance of Stars.

        Returns:
            pstr (str)
                The summary string to be printed.
        """

        # Set up string for printing
        pstr = ""

        # Add the content of the summary to the string to be printed
        pstr += "-" * 10 + "\n"
        pstr += "SUMMARY OF STAR PARTICLES" + "\n"
        pstr += f"N_stars: {self.nparticles}" + "\n"
        pstr += "log10(total mass formed/Msol): "
        pstr += f"{np.log10(np.sum(self.initial_masses)): .2f}" + "\n"
        pstr += f"median(age/Myr): {np.median(self.ages)/1E6:.1f}" + "\n"
        pstr += "-" * 10

        return pstr

    def _prepare_sed_args(self, grid, fesc, spectra_type, mask=None):
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
                A mask to be applied to the stars. Spectra will only be computed
                and returned for stars with True in the mask.
        """

        # Make a dummy mask if none has been passed
        if mask is None:
            mask = np.ones(self.nparticles, dtype=bool)

        # Set up the inputs to the C function.
        grid_props = [
            np.ascontiguousarray(grid.log10age, dtype=np.float64),
            np.ascontiguousarray(np.log10(grid.metallicity), dtype=np.float64),
        ]
        part_props = [
            np.ascontiguousarray(self.log10ages[mask], dtype=np.float64),
            np.ascontiguousarray(self.log10metallicities[mask], dtype=np.float64),
        ]
        part_mass = np.ascontiguousarray(
            self._initial_masses[mask], dtype=np.float64
        )
        npart = np.int32(self.nparticles)
        nlam = np.int32(grid.spectra[spectra_type].shape[-1])

        # Slice the spectral grids and pad them with copies of the edges.
        grid_spectra = np.ascontiguousarray(grid.spectra[spectra_type], np.float64)

        # Get the grid dimensions after slicing what we need
        grid_dims = np.zeros(len(grid_props) + 1, dtype=np.int32)
        for ind, g in enumerate(grid_props):
            grid_dims[ind] = len(g)
        grid_dims[ind + 1] = nlam

        # Convert inputs to tuples
        grid_props = tuple(grid_props)
        part_props = tuple(part_props)

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
        )

    def generate_lnu(
            self,
            grid,
            spectra_name,
            fesc=0.0,
            young=False,
            old=False,
            verbose=False,
    ):
        """
        Generate the integrated rest frame spectra for a given grid key
        spectra for all stars. Can optionally apply masks.

        Args:
            grid (Grid)
                The spectral grid object.
            spectra_name (string)
                The name of the target spectra inside the grid file.
            fesc (float)
                Fraction of stellar emission that escapes unattenuated from
                the birth cloud (defaults to 0.0).
            young (bool/float)
                If not False, specifies age in Myr at which to filter
                for young star particles.
            old (bool/float)
                If not False, specifies age in Myr at which to filter
                for old star particles.
            verbose (bool)
                Flag for verbose output.

        Returns:
            Numpy array of integrated spectra in units of (erg / s / Hz).
        """

        # Ensure we have a total key in the grid. If not error.
        if spectra_name not in list(grid.spectra.keys()):
            raise exceptions.MissingSpectraType(
                "The Grid does not contain the key '%s'" % spectra_name
            )

        # Get particle age masks
        mask = self._get_masks(young, old)

        # Ensure and warn that the masking hasn't removed everything
        if np.sum(mask) == 0:
            if verbose:
                print("Age mask has filtered out all particles")

            return np.zeros(len(grid.lam))

        from ..extensions.integrated_spectra import compute_integrated_sed

        # Prepare the arguments for the C function.
        args = self._prepare_sed_args(
            grid, fesc=fesc, spectra_type=spectra_name, mask=mask
        )

        # Get the integrated spectra in grid units (erg / s / Hz)
        spec = compute_integrated_sed(*args)

        return spec

    def generate_particle_lnu(
        self, grid, spectra_name, fesc=0.0, young=False, old=False, verbose=False
    ):
        """
        Generate the particle rest frame spectra for a given grid key spectra
        for all stars. Can optionally apply masks.

        Args:
            grid (Grid)
                The spectral grid object.
            spectra_name (string)
                The name of the target spectra inside the grid file.
            fesc (float)
                Fraction of stellar emission that escapes unattenuated from
                the birth cloud (defaults to 0.0).
            young (bool/float)
                If not False, specifies age in Myr at which to filter
                for young star particles.
            old (bool/float)
                If not False, specifies age in Myr at which to filter
                for old star particles.
            verbose (bool)
                Flag for verbose output.

        Returns:
            Numpy array of integrated spectra in units of (erg / s / Hz).
        """

        # Ensure we have a total key in the grid. If not error.
        if spectra_name not in list(grid.spectra.keys()):
            raise exceptions.MissingSpectraType(
                "The Grid does not contain the key '%s'" % spectra_name
            )

        # Get particle age masks
        mask = self._get_masks(young, old)

        # Ensure and warn that the masking hasn't removed everything
        if np.sum(mask) == 0:
            if verbose:
                print("Age mask has filtered out all particles")

            return np.zeros(len(grid.lam))

        from ..extensions.particle_spectra import compute_particle_seds

        # Prepare the arguments for the C function.
        args = self._prepare_sed_args(
            grid, fesc=fesc, spectra_type=spectra_name, mask=mask
        )

        # Get the integrated spectra in grid units (erg / s / Hz)
        spec = compute_particle_seds(*args)

        return spec

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
            s = self.log10ages <= np.log10(young)
        elif old:
            # Mask out young stars
            s = self.log10ages > np.log10(old)
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
            warnings.warn(
                "Warning, galaxy stars already resampled. \
                    To force resample, set force_resample=True. Returning..."
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
        for attr in Stars.__slots__:
            # Skip unset attributes
            if getattr(self, attr) is None:
                continue

            # Include resampled stellar particles in this attribute
            attr_array = getattr(self, attr)[resample_idxs]
            setattr(
                self,
                attr,
                np.append(getattr(self, attr), np.repeat(attr_array, new_lens, axis=0)),
            )

        if verbose:
            print("Delete old particles")

        # Loop over attributes
        for attr in Stars.__slots__:
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

    def get_particle_spectra_linecont(
        self,
        grid,
        fesc=0.0,
        fesc_LyA=1.0,
        young=False,
        old=False,
    ):
        """
        Generate the line contribution spectra. This is only invoked if
        fesc_LyA < 1.

        Args:
            grid (obj):
                Spectral grid object.
            fesc (float):
                Fraction of stellar emission that escapeds unattenuated from
                the birth cloud (defaults to 0.0).
            fesc_LyA (float)
                Fraction of Lyman-alpha emission that can escape unimpeded
                by the ISM/IGM.
            young (bool, float):
                If not False, specifies age in Myr at which to filter
                for young star particles.
            old (bool, float):
                If not False, specifies age in Myr at which to filter
                for old star particles.

        Returns:
            numpy.ndarray
                The line contribution spectra.
        """

        # Generate contribution of line emission alone and reduce the
        # contribution of Lyman-alpha
        linecont = self.generate_particle_lnu(
            grid, spectra_name="linecont", old=old, young=young
        )

        # Multiply by the Lyamn-continuum escape fraction
        linecont *= 1 - fesc

        # Get index of Lyman-alpha
        idx = grid.get_nearest_index(1216.0, grid.lam)
        linecont[idx] *= fesc_LyA  # reduce the contribution of Lyman-alpha

        return linecont

    def get_particle_spectra_incident(self, grid, young=False, old=False, label=""):
        """
        Generate the incident (equivalent to pure stellar for stars) spectra
        using the provided Grid.

        Args:
            grid (obj):
                Spectral grid object.
            young (bool, float):
                If not False, specifies age in Myr at which to filter
                for young star particles.
            old (bool, float):
                If not False, specifies age in Myr at which to filter
                for old star particles.
            label (string)
                A modifier for the spectra dictionary key such that the
                key is label + "_incident".

        Returns:
            Sed
                An Sed object containing the stellar spectra.
        """

        # Get the incident spectra
        lnu = self.generate_particle_lnu(grid, "incident", young=young, old=old)

        # Create the Sed object
        sed = Sed(grid.lam, lnu)

        # Store the Sed object
        self.particle_spectra[label + "incident"] = sed

        return sed

    def get_particle_spectra_transmitted(
        self,
        grid,
        fesc=0.0,
        young=False,
        old=False,
        label="",
    ):
        """
        Generate the transmitted spectra using the provided Grid. This is the
        emission which is transmitted through the gas as calculated by the
        photoionisation code.

        Args:
            grid (obj):
                Spectral grid object.
            fesc (float):
                Fraction of stellar emission that escapeds unattenuated from
                the birth cloud (defaults to 0.0).
            young (bool, float):
                If not False, specifies age in Myr at which to filter
                for young star particles.
            old (bool, float):
                If not False, specifies age in Myr at which to filter
                for old star particles.
            label (string)
                A modifier for the spectra dictionary key such that the
                key is label + "_transmitted".

        Returns:
            Sed
                An Sed object containing the transmitted spectra.
        """

        # Get the transmitted spectra
        lnu = (1.0 - fesc) * self.generate_particle_lnu(
            grid, "transmitted", young=young, old=old
        )

        # Create the Sed object
        sed = Sed(grid.lam, lnu)

        # Store the Sed object
        self.particle_spectra[label + "transmitted"] = sed

        return sed

    def get_particle_spectra_nebular(
        self,
        grid,
        fesc=0.0,
        young=False,
        old=False,
        label="",
    ):
        """
        Generate nebular spectra from a grid object and star particles.
        The grid object must contain a nebular component.

        Args:
            grid (obj):
                Spectral grid object.
            fesc (float):
                Fraction of stellar emission that escapeds unattenuated from
                the birth cloud (defaults to 0.0).
            young (bool, float):
                If not False, specifies age in Myr at which to filter
                for young star particles.
            old (bool, float):
                If not False, specifies age in Myr at which to filter
                for old star particles.
            label (string)
                A modifier for the spectra dictionary key such that the
                key is label + "_nebular".

        Returns:
            Sed
                An Sed object containing the nebular spectra.
        """

        # Get the nebular emission spectra
        lnu = self.generate_particle_lnu(grid, "nebular", young=young, old=old)

        # Apply the escape fraction
        lnu *= 1 - fesc

        # Create the Sed object
        sed = Sed(grid.lam, lnu)

        # Store the Sed object
        self.particle_spectra[label + "nebular"] = sed

        return sed

    def get_particle_spectra_reprocessed(
        self,
        grid,
        fesc=0.0,
        fesc_LyA=1.0,
        young=False,
        old=False,
        label="",
    ):
        """
        Generates the intrinsic spectra, this is the sum of the escaping
        radiation (if fesc>0), the transmitted emission, and the nebular
        emission. The transmitted emission is the emission that is
        transmitted through the gas. In addition to returning the intrinsic
        spectra this saves the incident, nebular, and escaped spectra if
        update is set to True.

        Args:
            grid (obj):
                Spectral grid object.
            fesc (float):
                Fraction of stellar emission that escapeds unattenuated from
                the birth cloud (defaults to 0.0).
            fesc_LyA (float)
                Fraction of Lyman-alpha emission that can escape unimpeded
                by the ISM/IGM.
            young (bool, float):
                If not False, specifies age in Myr at which to filter
                for young star particles.
            old (bool, float):
                If not False, specifies age in Myr at which to filter
                for old star particles.
            label (string)
                A modifier for the spectra dictionary key such that the
                key is label + "_transmitted".

        Updates:
            incident:
            transmitted
            nebular
            reprocessed
            intrinsic

            if fesc>0:
                escaped

        Returns:
            Sed
                An Sed object containing the intrinsic spectra.
        """

        # The incident emission
        incident = self.get_particle_spectra_incident(
            grid,
            young=young,
            old=old,
            label=label,
        )

        # The emission which escapes the gas
        if fesc > 0:
            escaped = Sed(grid.lam, fesc * incident._lnu)

        # The stellar emission which **is** reprocessed by the gas
        transmitted = self.get_particle_spectra_transmitted(
            grid, fesc, young=young, old=old, label=label
        )

        # The nebular emission
        nebular = self.get_particle_spectra_nebular(
            grid, fesc, young=young, old=old, label=label
        )

        # If the Lyman-alpha escape fraction is <1.0 suppress it.
        if fesc_LyA < 1.0:
            # Get the new line contribution to the spectrum
            linecont = self.get_particle_spectra_linecont(
                grid,
                fesc=fesc,
                fesc_LyA=fesc_LyA,
            )

            # Get the nebular continuum emission
            nebular_continuum = self.generate_particle_lnu(
                grid, "nebular_continuum", young=young, old=old
            )
            nebular_continuum *= 1 - fesc

            # Redefine the nebular emission
            nebular._lnu = linecont + nebular_continuum

        # The reprocessed emission, the sum of transmitted, and nebular
        reprocessed = nebular + transmitted

        # The intrinsic emission, the sum of escaped, transmitted, and nebular
        # if escaped exists other its simply the reprocessed
        if fesc > 0:
            intrinsic = reprocessed + escaped
        else:
            intrinsic = reprocessed

        if fesc > 0:
            self.particle_spectra[label + "escaped"] = escaped
        self.particle_spectra[label + "reprocessed"] = reprocessed
        self.particle_spectra[label + "intrinsic"] = intrinsic

        return reprocessed

    def plot_spectra(
        self, show=False, spectra_to_plot=None, ylimits=("peak", 5), figsize=(3.5, 5)
    ):
        """
        plots all spectra associated with a galaxy object

        Args:
            show (bool):
                flag for whether to show the plot or just return the
                figure and axes
            spectra_to_plot (None, list):
                list of named spectra to plot that are present in
                `galaxy.spectra`
            figsize (tuple):
                tuple with size 2 defining the figure size

        Returns:
            fig (object)
            ax (object)
        """

        fig = plt.figure(figsize=figsize)

        left = 0.15
        height = 0.6
        bottom = 0.1
        width = 0.8

        ax = fig.add_axes((left, bottom, width, height))

        if not isinstance(spectra_to_plot, list):
            spectra_to_plot = list(self.spectra.keys())

        # only plot FIR if 'total' is plotted otherwise just plot UV-NIR
        if "total" in spectra_to_plot:
            xlim = [2.0, 7.0]
        else:
            xlim = [2.0, 4.5]

        ypeak = -100
        for sed_name in spectra_to_plot:
            sed = self.spectra[sed_name]
            ax.plot(
                np.log10(sed.lam), np.log10(sed.lnu), lw=1, alpha=0.8, label=sed_name
            )

            if np.max(np.log10(sed.lnu)) > ypeak:
                ypeak = np.max(np.log10(sed.lnu))

        # ax.set_xlim([2.5, 4.2])

        if ylimits[0] == "peak":
            if ypeak == ypeak:
                ylim = [ypeak - ylimits[1], ypeak + 0.5]
            ax.set_ylim(ylim)

        ax.set_xlim(xlim)

        ax.legend(fontsize=8, labelspacing=0.0)
        ax.set_xlabel(r"$\rm log_{10}(\lambda/\AA)$")
        ax.set_ylabel(r"$\rm log_{10}(L_{\nu}/erg\ s^{-1}\ Hz^{-1} M_{\odot}^{-1})$")

        if show:
            plt.show()

        return fig, ax

    def plot_observed_spectra(
        self,
        cosmo,
        z,
        fc=None,
        show=False,
        spectra_to_plot=None,
        figsize=(3.5, 5.0),
        verbose=True,
    ):
        """
        plots all spectra associated with a galaxy object

        Args:

        Returns:
        """

        fig = plt.figure(figsize=figsize)

        left = 0.15
        height = 0.7
        bottom = 0.1
        width = 0.8

        ax = fig.add_axes((left, bottom, width, height))
        filter_ax = ax.twinx()

        if not isinstance(spectra_to_plot, list):
            spectra_to_plot = list(self.spectra.keys())

        for sed_name in spectra_to_plot:
            sed = self.spectra[sed_name]
            sed.get_fnu(cosmo, z)
            ax.plot(sed.obslam, sed.fnu, lw=1, alpha=0.8, label=sed_name)
            print(sed_name)

            if fc:
                sed.get_broadband_fluxes(fc, verbose=verbose)
                for f in fc:
                    wv = f.pivwv()
                    filter_ax.plot(f.lam, f.t)
                    ax.scatter(wv, sed.broadband_fluxes[f.filter_code], zorder=4)

        # ax.set_xlim([5000., 100000.])
        # ax.set_ylim([0., 100])
        filter_ax.set_ylim([3, 0])
        ax.legend(fontsize=8, labelspacing=0.0)
        ax.set_xlabel(r"$\rm log_{10}(\lambda_{obs}/\AA)$")
        ax.set_ylabel(r"$\rm log_{10}(f_{\nu}/nJy)$")

        if show:
            plt.show()

        return fig, ax

def sample_sfhz(sfzh, nstar, initial_mass=1, **kwargs):
    """
    Create "fake" stellar particles by sampling a SFZH.

    Args:
        sfhz (Stars)
            The Star Formation Z (Metallicity) History object.
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
    hist = sfzh.sfzh / np.sum(sfzh.sfzh)

    # Compute the cumaltive distribution function
    cdf = np.cumsum(hist.flatten())
    cdf = cdf / cdf[-1]

    # Get a random sample from the cdf
    values = np.random.rand(nstar)
    value_bins = np.searchsorted(cdf, values)

    # Convert 1D random indices to 2D indices
    x_idx, y_idx = np.unravel_index(
        value_bins, (len(sfzh.log10ages), len(sfzh.log10metallicities))
    )

    # Extract the sampled ages and metallicites and create an array
    random_from_cdf = np.column_stack(
        (sfzh.log10ages[x_idx], sfzh.log10metallicities[y_idx])
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

