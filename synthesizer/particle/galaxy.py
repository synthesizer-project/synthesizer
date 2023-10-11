""" A module containing all the funtionality for Particle based galaxies.

Like it's parametric variant this module contains the Galaxy object definition
from which all galaxy focused functionality can be performed. This variant uses
Particle objects, which can either be derived from simulation data or generated
from parametric models. A Galaxy can contain Stars, Gas, and / or BlackHoles.

Despite its name a Particle based Galaxy can be used for any collection of
particles to enable certain functionality (e.g. imaging of a galaxy group, or
spectra for all particles in a simulation.)

Example usage:

    galaxy = Galaxy(stars, gas, black_holes, ...)
    galaxy.get_spectra_incident(...)

"""
import numpy as np
from unyt import kpc, unyt_quantity
from scipy.spatial import cKDTree

from synthesizer.particle import Stars
from synthesizer.particle import Gas
from synthesizer.sed import Sed
from synthesizer.dust.attenuation import PowerLaw
from synthesizer.base_galaxy import BaseGalaxy
from synthesizer import exceptions
from synthesizer.imaging.images import ParticleImage
from synthesizer.parametric import BinnedSFZH


class Galaxy(BaseGalaxy):
    """The Particle based Galaxy object.

    When working with particles this object provides interfaces for calculating
    spectra, galaxy properties and images. A galaxy can be composed of any
    combination of particle.Stars, particle.Gas, or particle.BlackHoles objects.

    Attributes:

    """

    __slots__ = [
        "spectra",
        "spectra_array",
        "lam",
        "stars",
        "gas",
        "sf_gas_metallicity",
        "sf_gas_mass",
        "gas_mass",
    ]

    def __init__(
        self,
        name="particle galaxy",
        stars=None,
        gas=None,
        black_holes=None,
        redshift=None,
    ):
        """Initialise a particle based Galaxy with objects derived from
           Particles.

        Args:
            name (str)
                A name to identify the galaxy. Only used for external labelling,
                has no internal use.
            stars (object, Stars/BinnedSFZH)
                An instance of Stars containing the stellar particle data
            gas (object, Gas)
                An instance of Gas containing the gas particle data.
            black_holes (object, BlackHoles)
                An instance of BlackHoles containing the black hole particle
                data.
            redshift (float)
                The redshift of the galaxy.

        Raises:
            InconsistentArguments
        """

        # Check we haven't been given a SFZH
        if isinstance(stars, BinnedSFZH):
            raise exceptions.InconsistentArguments(
                "BinnedSFZH passed instead of particle based Stars object."
                " Did you mean synthesizer.parametric.Galaxy instead?"
            )

        # Define a name for this galaxy
        self.name = name

        # What is the redshift of this galaxy?
        self.redshift = redshift

        self.spectra = {}  # integrated spectra dictionary
        self.spectra_array = {}  # spectra arrays dictionary

        # Particle components
        self.stars = stars  # a Stars object
        self.gas = gas  # a Gas object
        self.black_holes = black_holes  # A BlackHoles object

        # If we have them, record how many stellar / gas particles there are
        if self.stars:
            self.calculate_integrated_stellar_properties()

        if self.gas:
            self.calculate_integrated_gas_properties()

        # Ensure all attributes are initialised to None
        for attr in Galaxy.__slots__:
            try:
                getattr(self, attr)
            except AttributeError:
                setattr(self, attr, None)

    def calculate_integrated_stellar_properties(self):
        """
        Calculate integrated stellar properties
        """

        # Define integrated properties of this galaxy
        if self.stars.current_masses is not None:
            self.stellar_mass = np.sum(self.stars.current_masses)

    def calculate_integrated_gas_properties(self):
        """
        Calculate integrated gas properties
        """

        # Define integrated properties of this galaxy
        if self.gas.masses is not None:
            self.gas_mass = np.sum(self.gas.masses)

        if self.gas.star_forming is not None:
            mask = self.gas.star_forming
            if np.sum(mask) == 0:
                self.sf_gas_mass = 0.0
                self.sf_gas_metallicity = 0.0
            else:
                self.sf_gas_mass = np.sum(self.gas.masses[mask])

                # mass weighted gas phase metallicity
                self.sf_gas_metallicity = (
                    np.sum(self.gas.masses[mask] * self.gas.metallicities[mask])
                    / self.sf_gas_mass
                )

    def load_stars(self, initial_masses, ages, metals, **kwargs):
        """
        Load arrays for star properties into a `Stars`  object,
        and attach to this galaxy object

        Args:
        initial_masses : array_like (float)
            initial stellar particle masses (mass at birth), Msol
        ages : array_like (float)
            star particle age, Myr
        metals : array_like (float)
            star particle metallicity (total metal fraction)
        **kwargs

        Returns:
        None

        # TODO: this should be able to take a pre-existing stars object!
        """
        self.stars = Stars(initial_masses, ages, metals, **kwargs)
        self.calculate_integrated_stellar_properties()

    def load_gas(self, masses, metals, **kwargs):
        """
        Load arrays for gas particle properties into a `Gas` object,
        and attach to this galaxy object

        Args:
            masses : array_like (float)
                gas particle masses, Msol
            metals : array_like (float)
                gas particle metallicity (total metal fraction)
        **kwargs

        Returns:
        None

        # TODO: this should be able to take a pre-existing stars object!
        """
        self.gas = Gas(masses, metals, **kwargs)
        self.calculate_integrated_gas_properties()

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

        if mask is None:
            mask = np.ones(self.stars.nparticles, dtype=bool)

        # Set up the inputs to the C function.
        grid_props = [
            np.ascontiguousarray(grid.log10age, dtype=np.float64),
            np.ascontiguousarray(np.log10(grid.metallicity), dtype=np.float64),
        ]
        part_props = [
            np.ascontiguousarray(self.stars.log10ages[mask], dtype=np.float64),
            np.ascontiguousarray(self.stars.log10metallicities[mask], dtype=np.float64),
        ]
        part_mass = np.ascontiguousarray(
            self.stars._initial_masses[mask], dtype=np.float64
        )
        npart = np.int32(part_mass.size)
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

    def calculate_black_hole_metallicity(self, default_metallicity=0.012):
        """
        Calculates the metallicity of the region surrounding a black hole. This
        is defined as the mass weighted average metallicity of all gas particles
        whose SPH kernels intersect the black holes position.

        Args:
            default_metallicity (float)
                The metallicity value used when no gas particles are in range
                of the black hole. The default is solar metallcity.
        """

        # Ensure we actually have Gas and black holes
        if self.gas is None:
            raise exceptions.InconsistentArguments(
                "Calculating the metallicity of the region surrounding the black"
                " hole requires a Galaxy to be intialised with a Gas object!"
            )
        if self.black_holes is None:
            raise exceptions.InconsistentArguments(
                "This Galaxy does not have a black holes object!"
            )

        # Construct a KD-Tree to efficiently get all gas particles which
        # intersect the black hole
        tree = cKDTree(self.gas._coordinates)

        # Query the tree for gas particles in range of each black hole, here
        # we use the maximum smoothing length to get all possible intersections
        # without calculating the distance for every gas particle.
        inds = tree.query_ball_point(
            self.black_holes._coordinates, r=self.gas._smoothing_lengths.max()
        )

        # Loop over black holes
        metals = np.zeros(self.black_holes.nbh)
        for ind, gas_in_range in enumerate(inds):
            # Handle black holes with no neighbouring gas
            if len(gas_in_range) == 0:
                metals[ind] = default_metallicity

            # Calculate the separation between the black hole and gas particles
            sep = (
                self.gas._coordinates[gas_in_range, :]
                - self.black_holes._coordinates[ind, :]
            )

            dists = np.sqrt(sep[:, 0] ** 2 + sep[:, 1] ** 2 + sep[:, 2] ** 2)

            # Get only the gas particles with smoothing lengths that intersect
            okinds = dists < self.gas._smoothing_lengths[gas_in_range]
            gas_in_range = np.array(gas_in_range, dtype=int)[okinds]

            # The above operation can remove all gas neighbours...
            if len(gas_in_range) == 0:
                metals[ind] = default_metallicity
                continue

            # Calculate the mass weight metallicity of this black holes region
            metals[ind] = np.average(
                self.gas.metallicities[gas_in_range],
                weights=self.gas._masses[gas_in_range],
            )

        # Assign the metallicity we have found
        self.black_holes.metallicities = metals

    def _prepare_los_args(self, kernel, mask, threshold, force_loop):
        """
        A method to prepare the arguments for line of sight metal surface
        density computation with the C function.

        Args:
            kernel (array_like, float)
                A 1D description of the SPH kernel. Values must be in ascending
                order such that a k element array can be indexed for the value of
                impact parameter q via kernel[int(k*q)]. Note, this can be an
                arbitrary kernel.
            mask (bool)
                A mask to be applied to the stars. Surface densities will only be
                computed and returned for stars with True in the mask.
            threshold (float)
                The threshold above which the SPH kernel is 0. This is normally
                at a value of the impact parameter of q = r / h = 1.
        """

        # If we have no gas, throw an error
        if self.gas is None:
            raise exceptions.InconsistentArguments(
                "No Gas object has been provided! We can't calculate line of "
                "sight dust attenuation with at Gas object containing the "
                "dust!"
            )

        # Set up the kernel inputs to the C function.
        kernel = np.ascontiguousarray(kernel, dtype=np.float64)
        kdim = kernel.size

        # Set up the stellar inputs to the C function.
        star_pos = np.ascontiguousarray(
            self.stars._coordinates[mask, :], dtype=np.float64
        )
        nstar = self.stars._coordinates[mask, :].shape[0]

        # Set up the gas inputs to the C function.
        gas_pos = np.ascontiguousarray(self.gas._coordinates, dtype=np.float64)
        gas_sml = np.ascontiguousarray(self.gas._smoothing_lengths, dtype=np.float64)
        gas_met = np.ascontiguousarray(self.gas.metallicities, dtype=np.float64)
        gas_mass = np.ascontiguousarray(self.gas._masses, dtype=np.float64)
        if isinstance(self.gas.dust_to_metal_ratio, float):
            gas_dtm = np.ascontiguousarray(
                np.full_like(gas_mass, self.gas.dust_to_metal_ratio), dtype=np.float64
            )
        else:
            gas_dtm = np.ascontiguousarray(
                self.gas.dust_to_metal_ratio, dtype=np.float64
            )
        ngas = gas_mass.size

        return (
            kernel,
            star_pos,
            gas_pos,
            gas_sml,
            gas_met,
            gas_mass,
            gas_dtm,
            nstar,
            ngas,
            kdim,
            threshold,
            np.max(gas_sml),
            force_loop,
        )

    def generate_lnu(
        self, grid, spectra_name, fesc=0.0, young=False, old=False, verbose=False
    ):
        """
        Generate the luminosity for a given grid key spectra for all
        stars in this galaxy object. Can optionally apply masks.

        Base class for :func:`~particle.ParticleGalaxy.get_spectra_incident`
        and other related methods

        Args:
            grid (obj):
                spectral grid object
            spectra_name (string):
                name of the target spectra inside the grid file
            fesc (float):
                fraction of stellar emission that escapes unattenuated from
                the birth cloud (defaults to 0.0)
            young (bool, float):
                if not False, specifies age in Myr at which to filter
                for young star particles
            old (bool, float):
                if not False, specifies age in Myr at which to filter
                for old star particles
            verbose (bool):
                Flag for verbose output

        Returns:
            numpy array of integrated spectra in units of (erg / s / Hz)
        """

        # Ensure we have a total key in the grid. If not error.
        if spectra_name not in list(grid.spectra.keys()):
            raise exceptions.MissingSpectraType(
                "The Grid does not contain the key '%s'" % spectra_name
            )

        # get particle age masks
        mask = self._get_masks(young, old)

        if np.sum(mask) == 0:
            if verbose:
                print("Age mask has filtered out all particles")

            return np.zeros(len(grid.lam))

        from ..extensions.csed import compute_integrated_sed

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
        Generate the luminosity for a given grid key spectra for all
        stars in this galaxy object. Can optionally apply masks.

        Base class for :func:`~particle.ParticleGalaxy.get_spectra_incident`
        and other related methods

        Args:
            grid (obj):
                spectral grid object
            spectra_name (string):
                name of the target spectra inside the grid file
            fesc (float):
                fraction of stellar emission that escapes unattenuated from
                the birth cloud (defaults to 0.0)
            young (bool, float):
                if not False, specifies age in Myr at which to filter
                for young star particles
            old (bool, float):
                if not False, specifies age in Myr at which to filter
                for old star particles
            verbose (bool):
                Flag for verbose output

        Returns:
            numpy array of integrated spectra in units of (erg / s / Hz)
        """

        # Ensure we have a total key in the grid. If not error.
        if spectra_name not in list(grid.spectra.keys()):
            raise exceptions.MissingSpectraType(
                "The Grid does not contain the key '%s'" % spectra_name
            )

        # get particle age masks
        mask = self._get_masks(young, old)

        if np.sum(mask) == 0:
            if verbose:
                print("Age mask has filtered out all particles")

            return np.zeros(len(grid.lam))

        from ..extensions.csed import compute_particle_seds

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
                age in Myr at which to filter for young star particles
            old (float):
                age in Myr at which to filter for old star particles

        Raises:
            InconsistentParameter
                Can't select for both young and old components
                simultaneously

        """

        if young and old:
            raise exceptions.InconsistentParameter(
                "Galaxy sub-component can not be simultaneously young and old"
            )
        if young:
            s = self.stars.log10ages <= np.log10(young)
        elif old:
            s = self.stars.log10ages > np.log10(old)
        else:
            s = np.ones(self.stars.nparticles, dtype=bool)

        return s

    def get_line_los():
        """
        ParticleGalaxy specific method for obtaining the line luminosities
        subject to line of sight attenuation to each star particle.
        """

        pass

    def get_particle_spectra_linecont(
        self, grid, fesc=0.0, fesc_LyA=1.0, young=False, old=False
    ):
        """
        Generate the line contribution spectra. This is only invoked if
        fesc_LyA < 1.
        """

        # generate contribution of line emission alone and reduce the
        # contribution of Lyman-alpha
        linecont = self.generate_particle_lnu(
            grid, spectra_name="linecont", old=old, young=young
        )

        # multiply by the Lyamn-continuum escape fraction
        linecont *= 1 - fesc

        # get index of Lyman-alpha
        idx = grid.get_nearest_index(1216.0, grid.lam)
        linecont[idx] *= fesc_LyA  # reduce the contribution of Lyman-alpha

        return linecont

    def get_particle_spectra_incident(
            self, grid, young=False, old=False, label="", update=True
    ):
        """
        Generate the incident (equivalent to pure stellar for stars) spectra
        using the provided Grid.

        Args:
            grid (obj):
                spectral grid object
            update (bool):
                flag for whether to update the `stellar` spectra
                inside the galaxy object `spectra` dictionary.
                These are the combined values of young and old.
            young (bool, float):
                if not False, specifies age in Myr at which to filter
                for young star particles
            old (bool, float):
                if not False, specifies age in Myr at which to filter
                for old star particles

        Returns:
            An Sed object containing the stellar spectra
        """

        lnu = self.generate_particle_lnu(grid, "incident", young=young, old=old)

        sed = Sed(grid.lam, lnu)

        if update:
            self.spectra[label + "incident"] = sed

        return sed

    def get_particle_spectra_transmitted(
        self, grid, fesc=0.0, young=False, old=False, label="", update=True
    ):
        """
        Generate the transmitted spectra using the provided Grid. This is the
        emission which is transmitted through the gas as calculated by the
        photoionisation code.

        Args:
            grid (obj):
                spectral grid object
            fesc (float):
                fraction of stellar emission that escapeds unattenuated from
                the birth cloud (defaults to 0.0)
            update (bool):
                flag for whether to update the `stellar` spectra
                inside the galaxy object `spectra` dictionary.
                These are the combined values of young and old.
            young (bool, float):
                if not False, specifies age in Myr at which to filter
                for young star particles
            old (bool, float):
                if not False, specifies age in Myr at which to filter
                for old star particles

        Returns:
            An Sed object containing the transmitted spectra
        """

        lnu = (1.0 - fesc) * self.generate_particle_lnu(
            grid, "transmitted", young=young, old=old
        )

        sed = Sed(grid.lam, lnu)

        if update:
            self.spectra[label + "transmitted"] = sed

        return sed

    def get_particle_spectra_nebular(
        self, grid, fesc=0.0, update=True, young=False, label="", old=False
    ):
        """
        Generate nebular spectra from a grid object and star particles.
        The grid object must contain a nebular component.

        Args:
            grid (obj):
                spectral grid object
            fesc (float):
                fraction of stellar emission that escapeds unattenuated from
                the birth cloud (defaults to 0.0)
            update (bool):
                flag for whether to update the `intrinsic` and `attenuated`
                spectra inside the galaxy object `spectra` dictionary.
                These are the combined values of young and old.
            young (bool, float):
                if not False, specifies age in Myr at which to filter
                for young star particles
            old (bool, float):
                if not False, specifies age in Myr at which to filter
                for old star particles

        Returns:
            An Sed object containing the nebular spectra
        """

        lnu = self.generate_particle_lnu(grid, "nebular", young=young, old=old)

        lnu *= 1 - fesc

        sed = Sed(grid.lam, lnu)

        if update:
            self.spectra[label + "nebular"] = sed

        return sed

    def get_particle_spectra_reprocessed(
        self,
        grid,
        fesc=0.0,
        fesc_LyA=1.0,
        young=False,
        old=False,
        label="",
        update=True,
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
                spectral grid object
            fesc (float):
                fraction of stellar emission that escapeds unattenuated from
                the birth cloud (defaults to 0.0)
            fesc_LyA (float):
                fraction of Lyman-alpha emission that can escape unimpeded
                by the ISM/IGM.
            update (bool):
                flag for whether to update the `intrinsic`, `stellar` and
                `nebular` spectra inside the galaxy object `spectra`
                dictionary. These are the combined values of young and old.
            young (bool, float):
                if not False, specifies age in Myr at which to filter
                for young star particles
            old (bool, float):
                if not False, specifies age in Myr at which to filter
                for old star particles

        Updates:
            incident:
            transmitted
            nebular
            reprocessed
            intrinsic

            if fesc>0:
                escaped

        Returns:
            An Sed object containing the intrinsic spectra.
        """

        # the incident emission
        incident = self.get_particle_spectra_incident(
            grid, update=update, young=young, old=old, label=label
        )

        # the emission which escapes the gas
        if fesc > 0:
            escaped = Sed(grid.lam, fesc * incident._lnu)

        # the stellar emission which **is** reprocessed by the gas
        transmitted = self.get_particle_spectra_transmitted(
            grid, fesc, update=update, young=young, old=old, label=label
        )
        # the nebular emission
        nebular = self.get_particle_spectra_nebular(
            grid, fesc, update=update, young=young, old=old, label=label
        )

        # if the Lyman-alpha escape fraction is <1.0 suppress it.
        if fesc_LyA < 1.0:
            # get the new line contribution to the spectrum
            linecont = self.get_particle_spectra_linecont(grid, fesc=fesc, fesc_LyA=fesc_LyA)

            # get the nebular continuum emission
            nebular_continuum = self.generate_particle_lnu(
                grid, "nebular_continuum", young=young, old=old
            )
            nebular_continuum *= 1 - fesc

            # redefine the nebular emission
            nebular._lnu = linecont + nebular_continuum

        # the reprocessed emission, the sum of transmitted, and nebular
        reprocessed = nebular + transmitted

        # the intrinsic emission, the sum of escaped, transmitted, and nebular
        # if escaped exists other its simply the reprocessed
        if fesc > 0:
            intrinsic = reprocessed + escaped
        else:
            intrinsic = reprocessed

        if update:
            if fesc > 0:
                self.spectra[label + "escaped"] = escaped
            self.spectra[label + "reprocessed"] = reprocessed
            self.spectra[label + "intrinsic"] = intrinsic

        return reprocessed

    def get_particle_line_intrinsic(self, grid):
        # """
        # Calculate line luminosities from individual young stellar particles

        # Warning: slower than calculating integrated line luminosities,
        # particularly where young particles are resampled, as it does
        # not use vectorisation.

        # Args:
        #     grid (object):
        #         `Grid` object.
        # """
        # age_mask = self.stars.log10ages < grid.max_age
        # lum = np.zeros((np.sum(age_mask), len(grid.lines)))

        # if np.sum(age_mask) == 0:
        #     return lum
        # else:
        #     for i, (mass, age, metal) in enumerate(zip(
        #             self.stars.initial_masses[age_mask],
        #             self.stars.log10ages[age_mask],
        #             self.stars.log10metallicities[age_mask])):

        #         weights_temp = self._calculate_weights(grid, metal, age, mass,
        #                                                young_stars=True)
        #         lum[i] = np.sum(grid.line_luminosities * weights_temp,
        #                         axis=(1, 2))

        # return lum

        pass

    def get_particle_line_attenuated():
        pass

    def get_particle_line_screen():
        pass

    def get_particle_line_los():
        pass

    def calculate_los_tau_v(
        self,
        kappa,
        kernel,
        mask=None,
        threshold=1,
        force_loop=0,
    ):
        """
        Calculate tau_v for each star particle based on the distribution of
        stellar and gas particles.

        Note: the resulting tau_vs will be associated to the stars object at
        self.stars.tau_v.

        Args:
            kappa (float)
                ...
            kernel (array_like/float)
                A 1D description of the SPH kernel. Values must be in ascending
                order such that a k element array can be indexed for the value of
                impact parameter q via kernel[int(k*q)]. Note, this can be an
                arbitrary kernel.
            mask (bool)
                A mask to be applied to the stars. Surface densities will only be
                computed and returned for stars with True in the mask.
            threshold (float)
                The threshold above which the SPH kernel is 0. This is normally
                at a value of the impact parameter of q = r / h = 1.
        """

        from ..extensions.los import compute_dust_surface_dens

        # If we don't have a mask make a fake one for consistency
        if mask is None:
            mask = np.ones(self.stars.nparticles, dtype=bool)

        # Prepare the arguments
        args = self._prepare_los_args(kernel, mask, threshold, force_loop)

        # Compute the dust surface densities
        los_dustsds = compute_dust_surface_dens(*args)

        # Finalise the calculation
        tau_v = kappa * los_dustsds

        # Store the result in self.stars
        if self.stars.tau_v is None:
            self.stars.tau_v = np.zeros(self.stars.nparticles)
        self.stars.tau_v[mask] = tau_v

        return tau_v

    def apply_attenuation(
        self,
        tau_v,
        incident_spectra_type,
        dust_curve=PowerLaw(slope=-1.0),
        mask=None,
    ):
        """
        Apply attenuation to a spectra.

        Args:

            tau_v (array-like, float)
                The V-band optical depth for every star particle.

            incident_spectra_type (str)
                The key for the spectra to be attenuated.

            dust_curve (synthesizer.dust.attenuation.*)
                An instance of one of the dust attenuation models. (defined in
                synthesizer/dust/attenuation.py)

            mask (array-like, bool)
                A mask array with an entry for each star particle. Masked out
                particles (star particles with False in the mask) will be
                ignored when applying the attenuation.
        """

        # Get the wavelength grid associated to the spectra
        lam = self.spectra_array[incident_spectra_type]._lam

        # Compute the transmission
        if np.isscalar(tau_v):
            transmission = dust_curve.get_transmission(lam) * tau_v
        else:
            transmission = np.outer(
                tau_v,
                dust_curve.get_transmission(lam)
            )

        # These two should have the same shape so should work?
        if mask is None:
            sed = self.spectra_array[incident_spectra_type] * transmission
            self.spectra_array["attenuated"] = Sed(self.lam, sed)
            self.spectra["attenuated"] = Sed(self.lam, np.sum(sed, axis=0))
        else:
            sed = self.spectra_array[incident_spectra_type]
            sed[mask] *= transmission
            self.spectra_array["attenuated"] = Sed(self.lam, sed)
            self.spectra["attenuated"] = Sed(self.lam, np.sum(sed, axis=0))

        return self.spectra_array["attenuated"]

    def screen_dust_gamma_parameter(
        self,
        beta=0.1,
        Z_norm=0.035,
        sf_gas_metallicity=None,
        sf_gas_mass=None,
        stellar_mass=None,
        gamma_min=None,
        gamma_max=None,
    ):
        """
        Calculate the gamma parameter controlling the optical depth
        due to dust dependent on the mass and metallicity of star forming 
        gas.

        gamma = (Z_SF / Z_MW) * (M_SF / M_star) * (1 / beta)

        where Z_SF is the star forming gas metallicity, Z_MW is the Milky
        Way value (defaults to value from Zahid+14), M_SF is the star forming gas mass, M_star
        is the stellar mass, and beta is a normalisation value.


        Zahid+14:
        https://iopscience.iop.org/article/10.1088/0004-637X/791/2/130

        Args:
            beta (float):
                normalisation value, default 0.1
            Z_norm (float):
                metallicity normsalition value, defaults to Zahid+14
                value for the Milky Way (0.035)
            sf_gas_metallicity (array):
                custom star forming gas metallicity array. If None, 
                defaults to value attached to this galaxy object.
            sf_gas_mass (array):
                custom star forming gas mass array. If None, 
                defaults to value attached to this galaxy object.
            stellar_mass (array):
                custom stellar mass array. If None, defaults to value 
                attached to this galaxy object.
            gamma_min (float):
                lower limit of the gamma parameter. If None, no lower 
                limit implemented. Default = None                 
            gamma_max (float):
                upper limit of the gamma parameter. If None, no upper 
                limit implemented. Default = None                 
                
        Returns:
            gamma (array):
                gamma scaling parameter for this galaxy
        """

        if sf_gas_metallicity is None:
            if self.sf_gas_metallicity is None:
                raise ValueError("No sf_gas_metallicity provided")
            else:
                sf_gas_metallicity = self.sf_gas_metallicity

        if sf_gas_mass is None:
            if self.sf_gas_mass is None:
                raise ValueError("No sf_gas_mass provided")
            else:
                sf_gas_mass = self.sf_gas_mass

        if stellar_mass is None:
            if self.stellar_mass is None:
                raise ValueError("No stellar_mass provided")
            else:
                stellar_mass = self.stellar_mass

        if sf_gas_mass == 0.:
            gamma = 0.
        elif stellar_mass == 0.:
            gamma = 1.0
        else:
            gamma = (sf_gas_metallicity / Z_norm) * (sf_gas_mass / stellar_mass) * (1.0 / beta)

        if gamma_min is not None:
            gamma[gamma < gamma_min] = gamma_min

        if gamma_max is not None:
            gamma[gamma > gamma_max] = gamma_max

        return gamma

    def create_stellarmass_hist(self, resolution, npix=None, fov=None):
        """
        Calculate a 2D histogram of the galaxy's mass distribution.

        NOTE: Either npix or fov must be defined.

        Parameters
        ----------
        resolution : float
           The size of a pixel.
        npix : int
            The number of pixels along an axis.
        fov : float
            The width of the image in image coordinates.

        Returns
        -------
        Image : array-like
            A 2D array containing the image.

        """

        # Instantiate the Image object.
        img = ParticleImage(
            resolution,
            npix,
            fov,
            stars=self.stars,
            pixel_values=self.stars.initial_masses,
        )

        return img.get_hist_imgs()

    def make_images(
        self,
        resolution,
        fov=None,
        img_type="hist",
        sed=None,
        filters=(),
        pixel_values=None,
        psfs=None,
        depths=None,
        snrs=None,
        aperture=None,
        noises=None,
        rest_frame=True,
        cosmo=None,
        psf_resample_factor=1,
    ):
        """
        Makes images, either one or one per filter. This is a generic method
        that will make every sort of image using every possible combination of
        arguments allowed by the ParticleImage class. These methods can be
        either a simple histogram or smoothing particles over a kernel. Either
        of these operations can be done with or without a PSF and noise.
        NOTE: Either npix or fov must be defined.
        Parameters
        ----------
        resolution : float
           The size of a pixel.
           (Ignoring any supersampling defined by psf_resample_factor)
        npix : int
            The number of pixels along an axis.
        fov : float
            The width of the image in image coordinates.
        img_type : str
            The type of image to be made, either "hist" -> a histogram, or
            "smoothed" -> particles smoothed over a kernel.
        sed : obj (SED)
            An sed object containing the spectra for this image.
        filters : obj (FilterCollection)
            An imutable collection of Filter objects. If provided images are
            made for each filter.
        pixel_values : array-like (float)
            The values to be sorted/smoothed into pixels. Only needed if an sed
            and filters are not used.
        psfs : dict
            A dictionary containing the psf in each filter where the key is
            each filter code and the value is the psf in that filter.
        depths : dict
            A dictionary containing the depth of an observation in each filter
            where the key is each filter code and the value is the depth in
            that filter.
        aperture : float/dict
            Either a float describing the size of the aperture in which the
            depth is defined or a dictionary containing the size of the depth
            aperture in each filter.
        rest_frame : bool
            Are we making an observation in the rest frame?
        cosmo : obj (astropy.cosmology)
            A cosmology object from astropy, used for cosmological calculations
            when converting rest frame luminosity to flux.
        psf_resample_factor : float
            The factor by which the image should be resampled for robust PSF
            convolution. Note the images after PSF application will be
            downsampled to the native pixel scale.
        Returns
        -------
        Image : array-like
            A 2D array containing the image.
        """

        # Handle a super resolution image
        if psf_resample_factor is not None:
            if psf_resample_factor != 1:
                resolution /= psf_resample_factor

        # Instantiate the Image object.
        img = ParticleImage(
            resolution=resolution,
            fov=fov,
            sed=sed,
            stars=self.stars,
            filters=filters,
            pixel_values=pixel_values,
            rest_frame=rest_frame,
            redshift=self.redshift,
            cosmo=cosmo,
            psfs=psfs,
            depths=depths,
            apertures=aperture,
            snrs=snrs,
        )

        # Make the image, handling incorrect image types
        if img_type == "hist":
            # Compute the image
            img.get_hist_imgs()

            if psfs is not None:
                # Convolve the image/images
                img.get_psfed_imgs()

                # Downsample to the native resolution if we need to.
                if psf_resample_factor is not None:
                    if psf_resample_factor != 1:
                        img.downsample(1 / psf_resample_factor)

            if depths is not None or noises is not None:
                img.get_noisy_imgs(noises)

            return img

        elif img_type == "smoothed":
            # Compute image
            img.get_imgs()

            if psfs is not None:
                # Convolve the image/images
                img.get_psfed_imgs()

                # Downsample to the native resolution if we need to.
                if psf_resample_factor is not None:
                    if psf_resample_factor != 1:
                        img.downsample(1 / psf_resample_factor)

            if depths is not None or noises is not None:
                img.get_noisy_imgs(noises)

            return img

        else:
            raise exceptions.UnknownImageType(
                "Unknown img_type %s. (Options are 'hist' or " "'smoothed')" % img_type
            )
