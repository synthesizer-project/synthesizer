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
from synthesizer.line import LineCollection
from synthesizer.sed import Sed
from synthesizer.synth_warnings import warn


class Extractor(ABC):
    """An abstract base class defining the framework for extraction."""

    def __init__(self, grid, extract):
        """
        Initialize the Extractor object.

        Args:
            grid (Grid): The grid object.
            extract (str): The emission type to extract from the grid.

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
        Get the attributes from the emitter.

        Args:
            emitter (Stars/BlackHoles/Gas):
                The emitter object.
            model (EmissionModel):
                The emission model object.
            do_grid_check (bool)
                Whether to check how many particles lie outside the grid. This
                is a sanity check that can be used to check the consistency
                of your particles with the grid. It is False by default
                because the check is extreme expensive.

        Returns:
            tuple: The extracted attributes and the weight variable.
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
        Compute the fraction of emitter attributes outside the grid axes.

        This is by default not run but can be invoked via the do_grid_check
        argument in the generate_lnu and generate_line methods.

        Args:
            extracted_attrs (tuple): The extracted attributes from the emitter.
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
        """Extract the spectra from the grid for the emitter."""
        pass

    @abstractmethod
    def generate_line(self, *args, **kwargs):
        """Extract the line luminosities from the grid for the emitter."""
        pass


class IntegratedParticleExtractor(Extractor):
    """A class to extract the integrated emission from a particle."""

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
        Extract the spectra from the grid for the emitter.

        Args:
            emitter (Stars/BlackHoles/Gas): The emitter object.
            model (EmissionModel): The emission model object.

        Returns:
            Sed: The integrated spectra.
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

    def generate_line(
        self,
        emitter,
        model,
        mask,
        lam_mask,
        grid_assignment_method,
        nthreads,
        do_grid_check,
    ):
        """Extract the line luminosities from the grid for the emitter."""
        start = tic()

        # Check we actually have to do the calculation
        if emitter.nparticles == 0:
            warn("Found emitter with no particles, returning empty Line")
            return LineCollection(
                line_ids=self._grid.line_ids,
                lam=self._line_lams,
                lum=np.zeros(self._grid.nlines) * erg / s,
                cont=np.zeros(self._grid.nlines) * erg / s / Hz,
            )
        elif mask is not None and np.sum(mask) == 0:
            warn("A mask has filtered out all particles, returning empty Line")
            return LineCollection(
                line_ids=self._grid.line_ids,
                lam=self._line_lams,
                lum=np.zeros(self._grid.nlines) * erg / s,
                cont=np.zeros(self._grid.nlines) * erg / s / Hz,
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

        # Get the grid_weights if they exist and we don't have a mask
        grid_weights = emitter._grid_weights.get(
            grid_assignment_method.lower(), {}
        ).get(self._grid.grid_name, None)

        # We need to modify the grid dims to account for the difference
        # in the final axis between the spectra and line grids
        grid_dims = np.array(self._grid_dims)
        grid_dims[-1] = self._grid.nlines

        # Compute the integrated line lum array
        lum, grid_weights = compute_integrated_sed(
            self._line_lum_grid,
            self._grid_axes,
            extracted,
            weight,
            grid_dims,
            self._grid_naxes,
            emitter.nparticles,
            self._grid.nlines,
            grid_assignment_method.lower(),
            nthreads,
            grid_weights,
            mask,
            lam_mask,
        )

        # Compute the integrated continuum array
        cont, _ = compute_integrated_sed(
            self._line_cont_grid,
            self._grid_axes,
            extracted,
            weight,
            grid_dims,
            self._grid_naxes,
            emitter.nparticles,
            self._grid.nlines,
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

        toc("Generating integrated line", start)

        return LineCollection(
            line_ids=self._grid.line_ids,
            lam=self._line_lams,
            lum=lum * erg / s,
            cont=cont * erg / s / Hz,
        )


class DopplerShiftedParticleExtractor(Extractor):
    """A class to extract the Doppler shifted emission from a particle."""

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
        Extract the spectra from the grid for the emitter.

        Args:
            emitter (Stars/BlackHoles/Gas): The emitter object.
            model (EmissionModel): The emission model object.

        Returns:
            Sed: The integrated spectra.
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
        """Doppler shifted line luminosities make no sense."""
        raise exceptions.UnimplementedFunctionality(
            "Doppler shifted emission lines aren't implemented since they "
            "have no width. To include the effects of velocity broadening"
            " on line emissions, extract them from the spectra."
        )


class IntegratedDopplerShiftedParticleExtractor(Extractor):
    """A class to extract the Doppler shifted emission from a particle."""

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
        Extract the spectra from the grid for the emitter.

        Args:
            emitter (Stars/BlackHoles/Gas): The emitter object.
            model (EmissionModel): The emission model object.

        Returns:
            Sed: The integrated spectra.
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
        """Doppler shifted line luminosities make no sense."""
        raise exceptions.UnimplementedFunctionality(
            "Doppler shifted emission lines aren't implemented since they "
            "have no width. To include the effects of velocity broadening"
            " on line emissions, extract them from the spectra."
        )


class ParticleExtractor(Extractor):
    """A class to extract the emission from a particle."""

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
        Extract the spectra from the grid for the emitter.

        Args:
            emitter (Stars/BlackHoles/Gas): The emitter object.
            model (EmissionModel): The emission model object.

        Returns:
            Sed: The integrated spectra.
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

    def generate_line(
        self,
        emitter,
        model,
        mask,
        lam_mask,
        grid_assignment_method,
        nthreads,
        do_grid_check,
    ):
        """Extract the line luminosities from the grid for the emitter."""
        start = tic()

        # Check we actually have to do the calculation
        if emitter.nparticles == 0:
            warn("Found emitter with no particles, returning empty Line")
            return LineCollection(
                line_ids=self._grid.line_ids,
                lam=self._line_lams,
                lum=np.zeros((emitter.nparticles, self._grid.nlines))
                * erg
                / s,
                cont=np.zeros((emitter.nparticles, self._grid.nlines))
                * erg
                / s
                / Hz,
            )
        elif mask is not None and np.sum(mask) == 0:
            warn("A mask has filtered out all particles, returning empty Line")
            return LineCollection(
                line_ids=self._grid.line_ids,
                lam=self._line_lams,
                lum=np.zeros((emitter.nparticles, self._grid.nlines))
                * erg
                / s,
                cont=np.zeros((emitter.nparticles, self._grid.nlines))
                * erg
                / s
                / Hz,
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

        # We need to modify the grid dims to account for the difference
        # in the final axis between the spectra and line grids
        grid_dims = np.array(self._grid_dims)
        grid_dims[-1] = self._grid.nlines

        # Compute the integrated line lum array
        lum = compute_particle_seds(
            self._line_lum_grid,
            self._grid_axes,
            extracted,
            weight,
            grid_dims,
            self._grid_naxes,
            emitter.nparticles,
            self._grid.nlines,
            grid_assignment_method.lower(),
            nthreads,
            mask,
            lam_mask,
        )

        # Compute the integrated continuum array
        cont = compute_particle_seds(
            self._line_cont_grid,
            self._grid_axes,
            extracted,
            weight,
            grid_dims,
            self._grid_naxes,
            emitter.nparticles,
            self._grid.nlines,
            grid_assignment_method.lower(),
            nthreads,
            mask,
            lam_mask,
        )

        toc("Generating particle line", start)

        return LineCollection(
            line_ids=self._grid.line_ids,
            lam=self._line_lams,
            lum=lum * erg / s,
            cont=cont * erg / s / Hz,
        )


class IntegratedParametricExtractor(Extractor):
    """
    A class to extract the integrated parametric emission from a particle.

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
        Extract the spectra from the grid for the emitter.

        Args:
            emitter (Stars/BlackHoles/Gas): The emitter object.
            model (EmissionModel): The emission model object.

        Returns:
            Sed: The integrated spectra.
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

    def generate_line(
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
        Extract the line luminosities from the grid for the emitter.

        Args:
            emitter (Stars/BlackHoles/Gas): The emitter object.
            model (EmissionModel): The emission model object.

        """
        start = tic()

        # Get a mask for non-zero bins in the SFZH
        mask = emitter.get_mask("sfzh", 0, ">", mask=mask)

        # Add an extra dimension to enable later summation
        sfzh = np.expand_dims(emitter.sfzh, axis=self._grid_naxes)

        # Get the grid line lunminosities and continua including any lam mask
        if lam_mask is None:
            grid_line_lums = self._line_lum_grid
            grid_line_conts = self._line_cont_grid
        else:
            grid_line_lums = self._line_lum_grid[..., lam_mask]
            grid_line_conts = self._line_cont_grid[..., lam_mask]

        # Compute the integrated line array by multiplying the sfzh by
        # the grids
        lum = np.sum(grid_line_lums[mask] * sfzh[mask], axis=0)
        cont = np.sum(grid_line_conts[mask] * sfzh[mask], axis=0)

        toc("Generating integrated lnu", start)

        return LineCollection(
            line_ids=self._grid.line_ids,
            lam=self._line_lams,
            lum=lum * erg / s,
            cont=cont * erg / s / Hz,
        )
