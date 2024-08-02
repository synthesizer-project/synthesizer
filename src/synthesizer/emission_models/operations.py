"""A submodule containing the operations performed by an EmissionModel.

An emission models inherits each of there opertaion classes but will only
ever instantiate one. This is because operations are isolated to one per
model. The correct operation is instantiated in EmissionMode._init_operations.

These classes should not be used directly.
"""

import numpy as np

from synthesizer import exceptions
from synthesizer.line import Line
from synthesizer.sed import Sed


class Extraction:
    """
    A class to define the extraction of spectra from a grid.

    Attributes:
        grid (Grid):
            The grid to extract from.
        extract (str):
            The key for the spectra to extract.
        fesc (float):
            The escape fraction.
    """

    def __init__(self, grid, extract, fesc):
        """
        Initialise the extraction model.

        Args:
            grid (Grid):
                The grid to extract from.
            extract (str):
                The key for the spectra to extract.
            fesc (float):
                The escape fraction.
        """
        # Attach the grid
        self._grid = grid

        # What base key will we be extracting?
        self._extract = extract

        # Attach the escape fraction
        self._fesc = fesc

    def _extract_spectra(
        self,
        emission_model,
        emitters,
        per_particle,
        spectra,
        verbose,
        **kwargs,
    ):
        """
        Extract spectra from the grid.

        Args:
            emission_model (EmissionModel):
                The emission model to extract from.
            emitters (dict):
                The emitters to extract the spectra for.
            per_particle (bool):
                Are we extracting per particle?
            spectra (dict):
                The dictionary to store the extracted spectra in.
            verbose (bool):
                Are we talking?
            kwargs (dict):
                Any additional keyword arguments to pass to the generator
                function.

        Returns:
            dict:
                The dictionary of extracted spectra.
        """
        # First step we need to extract each base spectra
        for label, spectra_key in emission_model._extract_keys.items():
            # Skip if we don't need to extract this spectra
            if label in spectra:
                continue

            # Get this model
            this_model = emission_model._models[label]

            # Skip models for a different emitter
            if this_model.emitter not in emitters:
                continue

            # Get the emitter
            emitter = emitters[this_model.emitter]

            # Do we have to define a mask?
            this_mask = None
            for mask_dict in this_model.masks:
                this_mask = emitter.get_mask(**mask_dict, mask=this_mask)

            # Fix any parameters we need to fix
            prev_properties = {}
            for prop in this_model.fixed_parameters:
                setattr(emitter, prop, this_model.fixed_parameters[prop])

            # Get the generator function
            if per_particle:
                generator_func = emitter.generate_particle_lnu
            else:
                generator_func = emitter.generate_lnu

            # Get this base spectra
            spectra[label] = Sed(
                emission_model.lam,
                generator_func(
                    this_model.grid,
                    spectra_key,
                    fesc=getattr(emitter, this_model.fesc)
                    if isinstance(this_model.fesc, str)
                    else this_model.fesc,
                    mask=this_mask,
                    verbose=verbose,
                    **kwargs,
                ),
            )

            # Replace any fixed parameters
            for prop in prev_properties:
                setattr(emitter, prop, prev_properties[prop])

        return spectra

    def _extract_lines(
        self,
        line_ids,
        emission_model,
        emitters,
        per_particle,
        lines,
        verbose,
        **kwargs,
    ):
        """
        Extract lines from the grid.

        Args:
            line_ids (list):
                The line ids to extract.
            emission_model (EmissionModel):
                The emission model to extract from.
            emitters (dict):
                The emitters to extract the lines for.
            per_particle (bool):
                Are we generating lines per particle?
            lines (dict):
                The dictionary to store the extracted lines in.
            verbose (bool):
                Are we talking?
            kwargs (dict):
                Any additional keyword arguments to pass to the generator
                function.

        Returns:
            dict:
                The dictionary of extracted lines.
        """
        # First step we need to extract each base lines
        for label in emission_model._extract_keys.keys():
            # Skip it if we happen to already have the lines
            if label in lines:
                continue

            # Get this model
            this_model = emission_model._models[label]

            # Skip models for a different emitter
            if this_model.emitter not in emitters:
                continue

            # Get the emitter
            emitter = emitters[this_model.emitter]

            # Do we have to define a mask?
            this_mask = None
            for mask_dict in this_model.masks:
                this_mask = emitter.get_mask(**mask_dict, mask=this_mask)

            # Fix any parameters we need to fix
            prev_properties = {}
            for prop in this_model.fixed_parameters:
                prev_properties[prop] = getattr(emitter, prop, None)
                setattr(emitter, prop, this_model.fixed_parameters[prop])

            # Get the generator function
            if per_particle:
                generator_func = emitter.generate_particle_line
            else:
                generator_func = emitter.generate_line

            # Initialise the lines dictionary for this label
            lines[label] = {}

            # Loop over the line ids
            for line_id in line_ids:
                # Get this base lines
                lines[label][line_id] = generator_func(
                    grid=this_model.grid,
                    line_id=line_id,
                    line_type=this_model.extract,
                    fesc=getattr(emitter, this_model.fesc)
                    if isinstance(this_model.fesc, str)
                    else this_model.fesc,
                    mask=this_mask,
                    verbose=verbose,
                    **kwargs,
                )

            # Replace any fixed parameters
            for prop in prev_properties:
                setattr(emitter, prop, prev_properties[prop])

        return lines

    def _extract_summary(self):
        """Return a summary of an extraction model."""
        # Create a list to hold the summary
        summary = []

        # Populate the list with the summary information
        summary.append("Extraction model:")
        summary.append(f"  Grid: {self._grid.grid_name}")
        summary.append(f"  Extract key: {self._extract}")
        summary.append(f"  Escape fraction: {self._fesc}")

        return summary


class Generation:
    """
    A class to define the generation of spectra.

    This can be used either to generate spectra for dust emission with the
    intrinsic and attenuated spectra used to scale the emission or to simply
    get a spectra from a generator.

    Attributes:
        generator (EmissionModel):
            The emission generation model. This must define a get_spectra
            method.
        lum_intrinsic_model (EmissionModel):
            The intrinsic model to use deriving the dust
            luminosity when computing dust emission.
        lum_attenuated_model (EmissionModel):
            The attenuated model to use deriving the dust
            luminosity when computing dust emission.
    """

    def __init__(self, generator, lum_intrinsic_model, lum_attenuated_model):
        """
        Initialise the generation model.

        Args:
            generator (EmissionModel):
                The emission generation model. This must define a get_spectra
                method.
            lum_intrinsic_model (EmissionModel):
                The intrinsic model to use deriving the dust
                luminosity when computing dust emission.
            lum_attenuated_model (EmissionModel):
                The attenuated model to use deriving the dust
                luminosity when computing dust emission.
        """
        # Attach the emission generation model
        self._generator = generator

        # Attach the keys for the intrinsic and attenuated spectra to use when
        # computing the dust luminosity
        self._lum_intrinsic_model = lum_intrinsic_model
        self._lum_attenuated_model = lum_attenuated_model

    def _generate_spectra(
        self, this_model, emission_model, spectra, lam, per_particle
    ):
        """
        Generate the spectra for a given model.

        Args:
            this_model (EmissionModel):
                The model to generate the spectra for.
            emission_model (EmissionModel):
                The root emission model.
            spectra (dict):
                The dictionary of spectra.
            lam (ndarray):
                The wavelength grid to generate the spectra on.
            per_particle (bool):
                Are we generating per particle?

        Returns:
            dict:
                The dictionary of spectra.
        """
        # Unpack what we need for dust emission
        generator = this_model.generator

        # Handle the dust emission case
        if this_model._is_dust_emitting:
            intrinsic = spectra[this_model.lum_intrinsic_model.label]
            attenuated = spectra[this_model.lum_attenuated_model.label]

            # Apply the dust emission model
            spectra[this_model.label] = generator.get_spectra(
                lam,
                intrinsic,
                attenuated,
            )
        elif this_model.lum_intrinsic_model is not None:
            # otherwise we are scaling by a single spectra
            spectra[this_model.label] = generator.get_spectra(
                lam,
                spectra[this_model.lum_intrinsic_model.label],
            )

        else:
            # Otherwise we have a bog standard generation
            spectra[this_model.label] = generator.get_spectra(
                lam,
            )

        return spectra

    def _generate_lines(
        self,
        line_ids,
        this_model,
        emission_model,
        lines,
        per_particle,
        emitter,
    ):
        """
        Generate the lines for a given model.

        This involves first generating the spectra and then extracting the
        emission at the line wavelengths.

        Args:
            line_ids (list):
                The line ids to extract.
            this_model (EmissionModel):
                The model to generate the lines for.
            emission_model (EmissionModel):
                The root emission model.
            lines (dict):
                The dictionary of lines.
            per_particle (bool):
                Are we generating lines per particle?
            emitter (Stars/BlackHoles/Galaxy):
                The emitter to generate the lines for.

        Returns:
            dict:
                The dictionary of lines.
        """
        # Do we already have the spectra?
        if per_particle and this_model.label in emitter.particle_spectra:
            spectra = emitter.particle_spectra[this_model.label]
        elif this_model.label in emitter.spectra:
            spectra = emitter.spectra[this_model.label]
        else:
            raise exceptions.MissingSpectraType(
                "To generate a line using a generator the corresponding "
                "spectra must be generated first."
            )

        # Now we have the spectra we can get the emission at each line
        # and include it
        lines[this_model.label] = {}
        for line_id in line_ids:
            # Get the emission at this lines wavelength
            lam = lines[this_model.lum_intrinsic_model.label][
                line_id
            ].wavelength

            # Get the luminosity at this wavelength
            luminosity = spectra.get_lnu_at_lam(lam)

            # Create the line (luminoisty = continuum)
            lines[this_model.label][line_id] = Line(
                line_id=line_id,
                wavelength=lam,
                luminosity=0,
                continuum=luminosity,
            )

        return lines

    def _generate_summary(self):
        """Return a summary of a generation model."""
        # Create a list to hold the summary
        summary = []

        # Populate the list with the summary information
        summary.append("Generation model:")
        summary.append(f"  Emission generation model: {self._generator}")
        if (
            self.lum_intrinsic_model is not None
            and self.lum_attenuated_model is not None
        ):
            summary.append(
                f"  Dust luminosity: "
                f"{self._lum_intrinsic_model.label} - "
                f"{self._lum_attenuated_model.label}"
            )
        elif self.lum_intrinsic_model is not None:
            summary.append(f"  Scale by: {self._lum_intrinsic_model.label}")

        return summary


class DustAttenuation:
    """
    A class to define the dust attenuation of spectra.

    Attributes:
        dust_curve (emission_models.attenuation.*):
            The dust curve to apply.
        apply_dust_to (EmissionModel):
            The model to apply the dust curve to.
        tau_v (float/ndarray/str/tuple):
            The optical depth to apply. Can be a float, ndarray, or a string
            to a component attribute. Can also be a tuple combining any of
            these.
    """

    def __init__(self, dust_curve, apply_dust_to, tau_v):
        """
        Initialise the dust attenuation model.

        Args:
            dust_curve (emission_models.attenuation.*):
                The dust curve to apply.
            apply_dust_to (EmissionModel):
                The model to apply the dust curve to.
            tau_v (float/ndarray/str/tuple):
                The optical depth to apply. Can be a float, ndarray, or a
                string to a component attribute. Can also be a tuple combining
                any of these.
        """
        # Attach the dust curve
        self._dust_curve = dust_curve

        # Attach the model to apply the dust curve to
        self._apply_dust_to = apply_dust_to

        # Attach the optical depth/s
        self._tau_v = (
            tau_v
            if isinstance(tau_v, (tuple, list)) or tau_v is None
            else [tau_v]
        )

    def _dust_attenuate_spectra(
        self,
        this_model,
        spectra,
        emitter,
        this_mask,
    ):
        """
        Dust attenuate the extracted spectra.

        Args:
            this_model (EmissionModel):
                The model defining the dust attenuation.
            spectra (dict):
                The dictionary of spectra.
            emitter (Stars/BlackHoles):
                The emitter to dust attenuate the spectra for.
            this_mask (dict):
                The mask to apply to the spectra.

        Returns:
            dict:
                The dictionary of spectra.
        """
        # Unpack the tau_v value unpacking any attributes we need
        # to extract from the emitter
        tau_v = 1
        for tv in this_model.tau_v:
            tau_v *= getattr(emitter, tv) if isinstance(tv, str) else tv
            tau_v *= getattr(emitter, tv) if isinstance(tv, str) else tv

        # Get the spectra to apply dust to
        apply_dust_to = spectra[this_model.apply_dust_to.label]

        # Otherwise, we are applying a dust curve (there's no
        # alternative)
        spectra[this_model.label] = apply_dust_to.apply_attenuation(
            tau_v,
            dust_curve=this_model.dust_curve,
            mask=this_mask,
        )

        return spectra

    def _dust_attenuate_lines(
        self,
        line_ids,
        this_model,
        lines,
        emitter,
        this_mask,
    ):
        """
        Dust attenuate the extracted lines.

        Args:
            line_ids (list):
                The line ids to extract.
            this_model (EmissionModel):
                The model defining the dust attenuation.
            lines (dict):
                The dictionary of lines.
            emitter (Stars/BlackHoles):
                The emitter to dust attenuate the lines for.
            this_mask (dict):
                The mask to apply to the lines.

        Returns:
            dict:
                The dictionary of lines.
        """
        # Unpack the tau_v value unpacking any attributes we need
        # to extract from the emitter
        tau_v = 1
        for tv in this_model.tau_v:
            tau_v *= getattr(emitter, tv) if isinstance(tv, str) else tv
            tau_v *= getattr(emitter, tv) if isinstance(tv, str) else tv

        # Get the lines to apply dust to
        apply_dust_to = lines[this_model.apply_dust_to.label]

        # Create dictionary to hold the dust attenuated lines
        lines[this_model.label] = {}

        # Loop over the line ids
        for line_id in line_ids:
            # Otherwise, we are applying a dust curve (there's no
            # alternative)
            lines[this_model.label][line_id] = apply_dust_to[
                line_id
            ].apply_attenuation(
                tau_v,
                dust_curve=this_model.dust_curve,
                mask=this_mask,
            )

        return lines

    def _attenuate_summary(self):
        """Return a summary of a dust attenuation model."""
        # Create a list to hold the summary
        summary = []

        # Populate the list with the summary information
        summary.append("Dust attenuation model:")
        summary.append(f"  Dust curve: {self._dust_curve}")
        summary.append(f"  Apply dust to: {self._apply_dust_to.label}")
        summary.append(f"  Optical depth (tau_v): {self._tau_v}")

        return summary


class Combination:
    """
    A class to define the combination of spectra.

    Attributes:
        combine (list):
            A list of models to combine.
    """

    def __init__(self, combine):
        """
        Initialise the combination model.

        Args:
            combine (list):
                A list of models to combine.
        """
        # Attach the models to combine
        self._combine = list(combine) if combine is not None else combine

    def _combine_spectra(self, emission_model, spectra, this_model):
        """
        Combine the extracted spectra.

        Args:
            emission_model (EmissionModel):
                The root emission model. This is used to get a consistent
                wavelength grid.
            spectra (dict):
                The dictionary of spectra.
            this_model (EmissionModel):
                The model defining the combination.

        Returns:
            dict:
                The dictionary of spectra.
        """
        # Create an empty spectra to add to
        spectra[this_model.label] = Sed(
            emission_model.lam,
            lnu=np.zeros_like(spectra[this_model.combine[0].label]._lnu),
        )

        # Combine the spectra
        for combine_model in this_model.combine:
            spectra[this_model.label]._lnu += spectra[combine_model.label]._lnu

        return spectra

    def _combine_lines(self, line_ids, emission_model, lines, this_model):
        """
        Combine the extracted lines.

        Args:
            line_ids (list):
                The line ids to extract.
            emission_model (EmissionModel):
                The root emission model. This is used to get a consistent
                wavelength grid.
            lines (dict):
                The dictionary of lines.
            this_model (EmissionModel):
                The model defining the combination.

        Returns:
            dict:
                The dictionary of lines.
        """
        # Create dictionary to hold the combined lines
        lines[this_model.label] = {}

        # Loop over lines copying over the first set of lines
        for line_id in line_ids:
            # Initialise the combined luminosity and continuum for the line
            lum = 0
            cont = 0

            # Get the wavelength of the line
            lam = lines[this_model.combine[0].label][line_id].wavelength

            # Combine the lines
            for combine_model in this_model.combine[1:]:
                lum += lines[combine_model.label][line_id]._luminosity
                cont += lines[combine_model.label][line_id]._continuum

            # Add the line to the dictionary
            lines[this_model.label][line_id] = Line(
                line_id=line_id,
                wavelength=lam,
                luminosity=lum,
                continuum=cont,
            )

        return lines

    def _combine_summary(self):
        """Return a summary of a combination model."""
        # Create a list to hold the summary
        summary = []

        # Populate the list with the summary information
        summary.append("Combination model:")
        summary.append(
            "  Combine models: "
            f"{', '.join([model.label for model in self._combine])}"
        )

        return summary
