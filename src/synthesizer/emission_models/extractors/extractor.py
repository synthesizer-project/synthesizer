"""A submodule containing the Extractor class."""

import os
from abc import ABC, abstractmethod

import numpy as np
from unyt import Hz, c, erg, s, unyt_array, unyt_quantity

from synthesizer import exceptions
from synthesizer.emission_models.utils import get_param
from synthesizer.extensions.integrated_spectra import compute_integrated_sed
from synthesizer.extensions.particle_spectra import (
    compute_part_seds_with_vel_shift,
    compute_particle_seds,
)
from synthesizer.extensions.timers import tic, toc
from synthesizer.sed import Sed
from synthesizer.synth_warnings import warn


class Extractor(ABC):
    """
    An abstract base class defining the framework for extraction.

    This class provides common methods and attributes for the extraction of
    emissions (sed and lines) from a grid for a given emitter. It also
    templates out what methods are needed by any child classes.

    In short, this class provides the machinery to extract emitter
    attributes corresponding to grid axes and run the extraction code to
    compute the emission from the grid. This functionality enables extraction
    from complete arbitrary grids and emitters, as long as the emitter contains
    the necessary attributes.

    Attributes:
        _emitter_attributes (tuple)
            The attributes to extract from the emitter.
        _grid_axes (tuple)
            The grid axes corresponding to the emitter attributes.
        _axes_units (tuple)
            The units for each grid axis.
        _weight_var (str)
            The weight variable to extract from the emitter and use to weight
            the emission grid. Note, this is set during grid generation and
            is stored on each grid object explictly from the grid file itself.
        _grid_dims (np.array)
            The grid dimensions, the shape of the spectra grid.
        _grid_naxes (int)
            The number of grid axes, i.e. the shape of the grid excluding the
            wavelength axis.
        _grid_nlam (int)
            The number of spectra grid wavelength elements.
        _log_emitter_attr (tuple)
            Whether to log the emitter data.
        _grid (Grid)
            The grid from which to extract the emission.
        _spectra_grid (unyt_array)
            The grid of spectra.
        _line_lum_grid (unyt_array)
            The grid of line luminosities.
        _line_cont_grid (unyt_array)
            The grid of line continua.
        _line_lams (unyt_array)
            The line wavelengths.
    """

    def __init__(self, grid, extract):
        """
        Initialize an Extractor instance with the given grid and emission type.
        
        This constructor sets up internal references required for emission extraction by obtaining
        emitter attributes, grid axes values, and corresponding unit information from the provided grid.
        It also assigns the spectra and line emission grids for the specified emission type, captures the
        weight variable, grid dimensions, and establishes logging flags for emitter attributes.
            
        Args:
            grid (Grid): The grid object containing emission data and extraction configuration.
            extract (str): The emission type identifier used to select the appropriate spectra and line grids.
        """
        start = tic()

        # Get the attribute names we will have to extract from the emitter
        self._emitter_attributes = grid._extract_axes

        # Attach the grid axes to the Extractor object (note that these
        # are already logged where this is required)
        self._grid_axes = tuple(
            grid._extract_axes_values[axis]
            for axis in self._emitter_attributes
        )

        # Attach the units for each axis so we can convert the emitter data
        # if needs be
        self._axes_units = tuple(grid._axes_units[axis] for axis in grid.axes)

        # Attach the weight variable we'll extract from the emitter
        self._weight_var = grid._weight_var

        # Attach the spectra and line grids to the Extractor object
        self._spectra_grid = grid.spectra[extract]
        self._line_lum_grid = grid.line_lums[extract]
        self._line_cont_grid = grid.line_conts[extract]
        self._line_lams = grid.line_lams

        # Attach the grid dimensions that we will need
        self._grid_dims = np.array(grid.shape, dtype=np.int32)
        self._grid_naxes = grid.naxes
        self._grid_nlam = grid.nlam

        # Record whether we need to log the emitter data
        self._log_emitter_attr = tuple(
            axis[:5] == "log10" for axis in grid.axes
        )

        # Finally, attach a pointer to the grid object
        self._grid = grid

        toc("Setting up the Extractor (including grid axis extraction)", start)

    def get_emitter_attrs(self, emitter, model, do_grid_check):
        """
        Extracts emitter attributes and an associated weight variable.
        
        This method retrieves the specified parameters from the emitter using the provided emission
        model. It converts any values with physical units to the expected grid units and ensures that
        all parameters are formatted as arrays for downstream processing. If enabled, it also checks
        that the extracted attributes lie within the valid grid limits.
        
        Args:
            emitter: The emitter instance (e.g., star, black hole, or gas).
            model: The emission model providing configuration details for the extraction.
            do_grid_check (bool): If True, validates that the extracted attributes fall within the grid
                boundaries, a process that may be computationally expensive.
                
        Returns:
            tuple: A pair where the first element is a tuple containing the emitter's attributes as arrays,
            and the second element is the weight variable.
        """
        start = tic()

        # Set up a list to store the extracted attributes
        extracted = []

        # Loop over the attributes we need to extract
        for axis, units, log in zip(
            self._emitter_attributes,
            self._axes_units,
            self._log_emitter_attr,
        ):
            # Get the attribute from the emitter
            value = get_param(axis, model, None, emitter)

            # Convert the units if necessary
            if units != "dimensionless" and isinstance(
                value, (unyt_array, unyt_quantity)
            ):
                value = value.to(units).value

            # We need these values to be arrays for the C code
            if not isinstance(value, np.ndarray):
                value = np.array(value)

            # Append the extracted value to the list
            extracted.append(value)

        # Check if the attributes are outside the grid axes if necessary
        if do_grid_check:
            self.check_emitter_attrs(extracted)

        # Also extract the weight variable
        weight = get_param(self._weight_var, model, None, emitter)

        toc("Preparing particle data for extraction", start)

        return tuple(extracted), weight

    def check_emitter_attrs(self, emitter, extracted_attrs):
        """
        Check whether emitter attributes lie outside the grid boundaries and issue a warning if so.
        
        This method computes the fraction of the emitter's attributes that fall outside the
        predefined grid axes. It is invoked only when grid boundary checking is enabled
        (via the do_grid_check argument in the generate_lnu and generate_line methods). If
        any attributes are found out-of-bound, a warning is logged.
            
        Args:
            emitter (Stars/BlackHoles/Gas): The emitter from which attributes are extracted.
            extracted_attrs (tuple): Tuple of attribute arrays extracted from the emitter.
        """
        start = tic()

        # Loop over the extracted attributes and check if they are outside the
        # grid axes, we'll do this by updating a mask for each attribute
        inside = np.zeros_like(extracted_attrs[0], dtype=bool)
        for i, (attr, axis) in enumerate(
            zip(extracted_attrs, self._grid_axes)
        ):
            inside |= (attr >= axis.min()) & (attr <= axis.max())

        # Compute the fraction of attributes outside the grid axes
        frac_outside = 1 - np.sum(inside) / len(inside)

        # Warn the user if the fraction is greater than 0.0
        if frac_outside > 0.0:
            warn(
                f"Found a {emitter.__class__.__name__} with "
                f"{frac_outside * 100:.2f}% of the attributes outside"
                " the grid axes."
            )

        toc("Checking the particle data against the grid axes", start)

    @abstractmethod
    def generate_lnu(self, *args, **kwargs):
        """
        Extracts the spectral luminosity from the grid for a given emitter.
        
        This method computes the emitter’s spectra by mapping its attributes onto a spectral
        grid. It accepts additional positional and keyword arguments to configure the extraction
        process (e.g., specifying the emission model, masks, grid assignment method, threading,
        and grid validation options). Derived classes must override this method to implement the
        specific logic for generating the spectral luminosity data.
        
        Returns:
            The computed spectral luminosity, structured as appropriate for the extractor’s design.
        """
        pass

    @abstractmethod
    def generate_line(self, *args, **kwargs):
        """
        Extract the line luminosities from the grid for an emitter.
        
        This method is a stub and should be overridden by subclasses if line emission extraction
        is supported. It accepts additional positional and keyword arguments to accommodate
        model-specific parameters.
        """
        pass


class IntegratedParticleExtractor(Extractor):
    """
    A class to extract the integrated emission from a particle.

    This Extractor will produce integrated emission from particle based
    components.

    If no mask is being used it will attempt to reuse any stored grid weights
    to reduce the computation time.
    """

    def generate_lnu(
        self,
        emitter,
        model,
        mask,
        lam_mask,
        grid_assignment_method,
        nthreads,
        do_grid_check,
    ):
        """
        Extract integrated spectral emission from an emitter using a spectral grid.
        
        This function computes the integrated spectral energy distribution (SED)
        by mapping emitter attributes onto a pre-defined spectral grid with a specified
        grid assignment method ("cic" or "ngp"). If the emitter has no particles or if a
        provided mask filters out all particles, a warning is issued and an empty SED is
        returned. Multithreading is supported via the nthreads parameter, and grid weights
        may be updated for later use when no mask is applied.
        
        Args:
            emitter (Stars/BlackHoles/Gas): The emitter object from which to extract emission.
            model (EmissionModel): The emission model defining the spectral properties.
            mask (np.array): Array mask to filter particles.
            lam_mask (np.array): Array mask to filter the wavelength axis of the spectrum.
            grid_assignment_method (str): Method for assigning particles to the grid, either "cic" (Cloud-in-Cell) or "ngp" (Nearest Grid Point).
            nthreads (int): Number of threads for computation; if -1, all available threads are used.
            do_grid_check (bool): Flag to perform an expensive check of emitter attributes against grid boundaries.
        
        Returns:
            Sed: The integrated spectral energy distribution with flux densities in erg/s/Hz.
        """
        start = tic()

        # Check we actually have to do the calculation
        if emitter.nparticles == 0:
            warn("Found emitter with no particles, returning empty Sed")
            return Sed(model.lam, np.zeros(self._grid_nlam) * erg / s / Hz)
        elif mask is not None and np.sum(mask) == 0:
            warn("A mask has filtered out all particles, returning empty Sed")
            return Sed(model.lam, np.zeros(self._grid_nlam) * erg / s / Hz)

        # Get the attributes from the emitter
        extracted, weight = self.get_emitter_attrs(
            emitter,
            model,
            do_grid_check,
        )

        # If nthreads is -1 then use all available threads
        if nthreads == -1:
            nthreads = os.cpu_count()

        # Get the grid_weights if they exist and we don't have a mask
        grid_weights = emitter._grid_weights.get(
            grid_assignment_method.lower(), {}
        ).get(self._grid.grid_name, None)

        # Compute the integrated lnu array (this is attached to an Sed
        # object elsewhere)
        spec, grid_weights = compute_integrated_sed(
            self._spectra_grid,
            self._grid_axes,
            extracted,
            weight,
            self._grid_dims,
            self._grid_naxes,
            emitter.nparticles,
            self._grid_nlam,
            grid_assignment_method.lower(),
            nthreads,
            grid_weights,
            mask,
            lam_mask,
        )

        # If we have no mask then lets store the grid weights in case
        # we can make use of them later
        if (
            mask is None
            and self._grid.grid_name
            not in emitter._grid_weights[grid_assignment_method.lower()]
        ):
            emitter._grid_weights[grid_assignment_method.lower()][
                self._grid.grid_name
            ] = grid_weights

        toc("Generating integrated lnu", start)

        return Sed(model.lam, spec * erg / s / Hz)

    def generate_line(self, *args, **kwargs):
        """
        Extracts line luminosities from the grid for the emitter.
        
        This method serves as a placeholder to be overridden by subclasses that
        compute emission line properties. Implementations should use the spectral
        grid and emitter attributes to derive the line luminosities, with additional
        arguments available to customize the extraction process.
        """
        pass


class DopplerShiftedParticleExtractor(Extractor):
    """
    A class to extract the Doppler shifted emission from a particle.

    This Extractor will produce a Doppler shifted spectra for each particle
    in a particle based component.

    This is not applicable for line emission which is treated without Doppler
    shifting. Doppler broadened lines can be extracted from spectra where the
    width of the lines is accounted for in the spectra grid.
    """

    def generate_lnu(
        self,
        emitter,
        model,
        mask,
        lam_mask,
        grid_assignment_method,
        nthreads,
        do_grid_check,
    ):
        """
        Computes Doppler-shifted spectra for each emitter particle using grid data.
        
        This function extracts the Doppler-shifted spectral energy distributions for individual
        particles by mapping their properties, including velocities, onto a precomputed grid.
        Optional particle and wavelength masks filter the inputs, and particles are assigned to
        grid cells using either "cic" (Cloud-in-Cell) or "ngp" (Nearest Grid Point) methods.
        If no particles are present or all particles are masked out, an empty Sed is returned.
        An exception is raised if the emitter lacks velocity data needed for Doppler shifting.
        
        Parameters:
            emitter:
                An emitter object (e.g., a star, black hole, or gas instance) containing particle
                properties and velocities.
            model:
                An emission model providing the wavelength grid and spectral properties.
            mask (np.array):
                A Boolean array to filter particles; only unmasked particles are processed.
            lam_mask (np.array):
                A Boolean array applied to the wavelength axis of the spectra.
            grid_assignment_method (str):
                The scheme used for assigning particles to grid cells ("cic" or "ngp"), case-insensitive.
            nthreads (int):
                The number of threads for parallel processing. If set to -1, all available threads are used.
            do_grid_check (bool):
                A flag indicating whether to verify that particle positions lie within the grid bounds,
                a check that may be computationally expensive.
        
        Returns:
            Sed:
                An object encapsulating the wavelength grid and the computed Doppler-shifted spectra in erg/s/Hz.
        
        Raises:
            InconsistentArguments:
                If the emitter does not provide velocity data necessary for computing Doppler shifts.
        """
        start = tic()

        # Check we actually have to do the calculation
        if emitter.nparticles == 0:
            warn("Found emitter with no particles, returning empty Sed")
            return Sed(
                model.lam,
                np.zeros((emitter.nparticles, self._grid_nlam)) * erg / s / Hz,
            )
        elif mask is not None and np.sum(mask) == 0:
            warn("A mask has filtered out all particles, returning empty Sed")
            return Sed(
                model.lam,
                np.zeros((emitter.nparticles, self._grid_nlam)) * erg / s / Hz,
            )

        # Get the attributes from the emitter
        extracted, weight = self.get_emitter_attrs(
            emitter,
            model,
            do_grid_check,
        )

        # Get the emitter velocities
        if emitter._velocities is None:
            raise exceptions.InconsistentArguments(
                "velocity shifted spectra requested but no "
                "star velocities provided."
            )
        vel_units = emitter.velocities.units

        # If nthreads is -1 then use all available threads
        if nthreads == -1:
            nthreads = os.cpu_count()

        # Compute the lnu array
        spec = compute_part_seds_with_vel_shift(
            self._spectra_grid,
            self._grid._lam,
            self._grid_axes,
            extracted,
            weight,
            emitter._velocities,
            self._grid_dims,
            self._grid_naxes,
            emitter.nparticles,
            self._grid_nlam,
            grid_assignment_method.lower(),
            nthreads,
            c.to(vel_units).value,
            mask,
            lam_mask,
        )

        toc("Generating doppler shifted particle lnu", start)

        return Sed(model.lam, spec * erg / s / Hz)

    def generate_line(self, *args, **kwargs):
        """
        Extracts the line luminosities for the emitter from the grid.
        
        This placeholder method should be overridden in subclasses to implement the
        appropriate line emission extraction strategy. Additional positional and keyword
        arguments may be provided to support various emitter configurations.
        """
        pass


class IntegratedDopplerShiftedParticleExtractor(Extractor):
    """
    A class to extract the Doppler shifted emission from a particle.

    This Extractor will produce the integrated Doppler shifted spectra for
    a particle based component.

    This is not applicable for line emission which is treated without Doppler
    shifting. Doppler broadened lines can be extracted from spectra where the
    width of the lines is accounted for in the spectra grid.
    """

    def generate_lnu(
        self,
        emitter,
        model,
        mask,
        lam_mask,
        grid_assignment_method,
        nthreads,
        do_grid_check,
    ):
        """
        Extract integrated Doppler-shifted spectra from the grid.
        
        This function computes the integrated emission spectrum by applying a velocity
        shift to the emitter's spectra. It extracts necessary attributes, verifies that
        velocity data is present, and computes the Doppler-shifted spectra using the grid.
        If the emitter has no particles or the mask filters out all particles, a Sed with
        zero flux is returned.
        
        Args:
            emitter (Stars/BlackHoles/Gas): The emitter object from which to extract emission.
            model (EmissionModel): The emission model defining the emission.
            mask (np.array): Mask to filter particles.
            lam_mask (np.array): Mask to apply along the wavelength axis of the spectra.
            grid_assignment_method (str): Method for assigning particles to the grid ("cic" or "ngp").
            nthreads (int): Number of threads for extraction; -1 uses all available threads.
            do_grid_check (bool): Whether to check for particles outside the grid.
        
        Returns:
            Sed: The integrated spectrum obtained by summing Doppler-shifted spectra.
        
        Raises:
            exceptions.InconsistentArguments: If velocity data is missing for Doppler shifting.
        """
        start = tic()

        # Check we actually have to do the calculation
        if emitter.nparticles == 0:
            warn("Found emitter with no particles, returning empty Sed")
            return Sed(model.lam, np.zeros(self._grid_nlam) * erg / s / Hz)
        elif mask is not None and np.sum(mask) == 0:
            warn("A mask has filtered out all particles, returning empty Sed")
            return Sed(model.lam, np.zeros(self._grid_nlam) * erg / s / Hz)

        # Get the attributes from the emitter
        extracted, weight = self.get_emitter_attrs(
            emitter,
            model,
            do_grid_check,
        )

        # Get the emitter velocities
        if emitter._velocities is None:
            raise exceptions.InconsistentArguments(
                "velocity shifted spectra requested but no "
                "star velocities provided."
            )
        vel_units = emitter.velocities.units

        # If nthreads is -1 then use all available threads
        if nthreads == -1:
            nthreads = os.cpu_count()

        # Compute the lnu array
        spec = compute_part_seds_with_vel_shift(
            self._spectra_grid,
            self._grid._lam,
            self._grid_axes,
            extracted,
            weight,
            emitter._velocities,
            self._grid_dims,
            self._grid_naxes,
            emitter.nparticles,
            self._grid_nlam,
            grid_assignment_method.lower(),
            nthreads,
            c.to(vel_units).value,
            mask,
            lam_mask,
        )

        # Sum the spectra over the particles
        spec = np.sum(spec, axis=0)

        toc("Generating doppler shifted integrated lnu", start)

        return Sed(model.lam, spec * erg / s / Hz)

    def generate_line(self, *args, **kwargs):
        """
        Extract line luminosities from the grid for the emitter.
        
        This method serves as a placeholder to be overridden by subclasses. Implementations
        should extract and return line luminosities based on the emitter's data and any additional
        parameters provided via positional or keyword arguments.
        """
        pass


class ParticleExtractor(Extractor):
    """
    A class to extract the emission from a particle.

    This Extractor will produce a spectra for each particle in a particle
    based component.
    """

    def generate_lnu(
        self,
        emitter,
        model,
        mask,
        lam_mask,
        grid_assignment_method,
        nthreads,
        do_grid_check,
    ):
        """
        Generate per-particle spectral energy distributions from the grid.
        
        This function computes the luminosity per unit frequency for each particle in the
        emitter by mapping particle attributes onto a spectral grid using the specified
        grid assignment method. If the emitter has no particles or if the particle mask
        filters out all particles, an empty spectral energy distribution (SED) is returned.
        
        Args:
            emitter:
                The emission source containing particle data.
            model:
                The emission model providing the wavelength grid.
            mask (np.array):
                A boolean mask applied to filter particles.
            lam_mask (np.array):
                A boolean mask applied along the wavelength axis.
            grid_assignment_method (str):
                The method to assign particles to the grid; either "cic" (Cloud-in-Cell) or
                "ngp" (Nearest Grid Point).
            nthreads (int):
                The number of threads to use, where -1 indicates that all available threads
                will be used.
            do_grid_check (bool):
                If True, verifies that particle attributes fall within grid bounds.
        
        Returns:
            Sed:
                A spectral energy distribution containing the per-particle spectra with
                units of erg/s/Hz.
        """
        start = tic()

        # Check we actually have to do the calculation
        if emitter.nparticles == 0:
            warn("Found emitter with no particles, returning empty Sed")
            return Sed(
                model.lam,
                np.zeros((emitter.nparticles, self._grid_nlam)) * erg / s / Hz,
            )
        elif mask is not None and np.sum(mask) == 0:
            warn("A mask has filtered out all particles, returning empty Sed")
            return Sed(
                model.lam,
                np.zeros((emitter.nparticles, self._grid_nlam)) * erg / s / Hz,
            )

        # Get the attributes from the emitter
        extracted, weight = self.get_emitter_attrs(
            emitter,
            model,
            do_grid_check,
        )

        # If nthreads is -1 then use all available threads
        if nthreads == -1:
            nthreads = os.cpu_count()

        # Compute the lnu array
        spec = compute_particle_seds(
            self._spectra_grid,
            self._grid_axes,
            extracted,
            weight,
            self._grid_dims,
            self._grid_naxes,
            emitter.nparticles,
            self._grid_nlam,
            grid_assignment_method.lower(),
            nthreads,
            mask,
            lam_mask,
        )

        toc("Generating particle lnu", start)

        return Sed(model.lam, spec * erg / s / Hz)

    def generate_line(self, *args, **kwargs):
        """
        Extracts line luminosities from the spectral grid.
        
        This method is intended to be overridden by subclasses to compute emission line
        luminosities based on the grid configuration and emitter properties. Additional
        positional and keyword arguments may be used for implementation-specific customization.
        """
        pass


class IntegratedParametricExtractor(Extractor):
    """
    A class to extract the integrated parametric emission from a particle.

    This Extractor will produce integrated emission from parametric based
    components. This differs from particle based components only in that we
    can straight multiply the SFZH by the grid spectra to get the integrated
    emission.
    """

    def generate_lnu(
        self,
        emitter,
        model,
        mask,
        lam_mask,
        grid_assignment_method,
        nthreads,
        do_grid_check,
    ):
        """
        Compute the integrated spectral energy distribution (SED) for a parametric emitter.
        
        This method calculates the integrated SED by multiplying the emitter's star formation
        history (sfzh) with the corresponding grid spectra and summing over the masked grid cells.
        The output SED uses the wavelength array from the emission model. Note that the parameters
        grid_assignment_method, nthreads, and do_grid_check are retained for interface consistency
        but are not used in this computation.
        
        Args:
            emitter: Emitter object providing a star formation history (sfzh) and a method to
                generate a mask.
            model: Emission model from which the wavelength array (lam) is taken for the SED.
            mask (np.ndarray): Filter applied to the emitter's sfzh to select valid bins (sfzh > 0).
            lam_mask (np.ndarray or None): Optional mask to restrict the grid spectra to specific wavelengths.
            grid_assignment_method (str): Unused; maintained for compatibility with particle-based extractors.
            nthreads (int): Unused; included for interface uniformity with multi-threaded extractors.
            do_grid_check (bool): Unused; reserved for potential grid consistency checks.
        
        Returns:
            Sed: Integrated spectral energy distribution computed from the parametric emitter.
        """
        start = tic()

        # Get a mask for non-zero bins in the SFZH
        mask = emitter.get_mask("sfzh", 0, ">", mask=mask)

        # Add an extra dimension to enable later summation
        sfzh = np.expand_dims(emitter.sfzh, axis=self._grid_naxes)

        # Get the grid spectra including any lam mask
        if lam_mask is None:
            grid_spectra = self._spectra_grid
        else:
            grid_spectra = self._spectra_grid[..., lam_mask]

        # Compute the integrated lnu array by multiplying the sfzh by the
        # grid spectra
        spec = np.sum(grid_spectra[mask] * sfzh[mask], axis=0)

        toc("Generating integrated lnu", start)

        return Sed(model.lam, spec * erg / s / Hz)

    def generate_line(self, *args, **kwargs):
        """
        Extract line luminosities from the grid for the emitter.
        
        Subclasses should override this method to implement specific strategies for
        extracting line emission data. Extra positional and keyword arguments may be used
        to customize the extraction process.
        """
        pass
