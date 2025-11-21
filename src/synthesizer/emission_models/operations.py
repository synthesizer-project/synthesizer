"""A submodule containing the operations performed by an EmissionModel.

An emission models inherits each of there opertaion classes but will only
ever instantiate one. This is because operations are isolated to one per
model. The correct operation is instantiated in EmissionMode._init_operations.

These classes should not be used directly.
"""

import numpy as np
from unyt import Hz, erg, s

from synthesizer import exceptions
from synthesizer.emission_models.extractors.extractor import (
    DopplerShiftedParticleExtractor,
    IntegratedDopplerShiftedParticleExtractor,
    IntegratedParametricExtractor,
    IntegratedParticleExtractor,
    ParticleExtractor,
)
from synthesizer.emission_models.utils import cache_model_params
from synthesizer.emissions import LineCollection, Sed
from synthesizer.extensions.timers import tic, toc
from synthesizer.grid import Template
from synthesizer.imaging import Image, ImageCollection
from synthesizer.parametric import BlackHole


class Extraction:
    """A class to define the extraction of spectra from a grid.

    Attributes:
        grid (Grid):
            The grid to extract from.
        extract (str):
            The key for the spectra to extract.
    """

    def __init__(self, grid, extract, vel_shift):
        """Initialise the extraction model.

        Args:
            grid (Grid):
                The grid to extract from.
            extract (str):
                The key for the spectra to extract.
            vel_shift (bool):
                Should the emission take into account the velocity shift due
                to peculiar velocities? (Particle Only!)
        """
        # Attach the grid
        self._grid = grid

        # What base key will we be extracting?
        self._extract = extract

        # Ensure the grid has the right key
        if extract not in grid.spectra and extract not in grid.line_lums:
            raise exceptions.MissingSpectraType(
                f"The Grid does not contain the key '{extract}'."
                f"Available types for spectra:"
                f"{grid.available_spectra_emissions}"
                f"Available types for lines: "
                f"{grid.available_line_emissions}"
            )

        # Should the emission take into account the velocity shift due to
        # peculiar velocities? (Particle Only!)
        self._use_vel_shift = vel_shift

    def _extract_spectra(
        self,
        this_model,
        emitters,
        spectra,
        particle_spectra,
        verbose,
        nthreads,
        grid_assignment_method,
    ):
        """Extract spectra from the grid.

        Args:
            this_model (EmissionModel):
                The emission model defining the extraction.
            emitters (dict):
                The emitters to extract the spectra for.
            spectra (dict):
                The dictionary to store the extracted spectra in.
            particle_spectra (dict):
                The dictionary to store the extracted particle spectra in.
            verbose (bool):
                Are we talking?
            nthreads (int):
                The number of threads to use when generating spectra.
            grid_assignment_method (str):
                The method to use when assigning particles to the grid.
                Options are 'cic' (cloud-in-cell) and 'ngp' (nearest
                grid point).

        Returns:
            dict:
                The dictionary of extracted spectra.
        """
        # Extract the label for this model
        label = this_model.label

        # Skip models for a different emitter
        if this_model.emitter not in emitters:
            return spectra, particle_spectra

        # Get the emitter
        emitter = emitters[this_model.emitter]

        # Do we have to define a property mask?
        mask_start = tic()
        this_mask = None
        for mask_dict in this_model.masks:
            this_mask = emitter.get_mask(
                **mask_dict,
                mask=this_mask,
                attr_override_obj=this_model,
            )
        toc("Getting the mask", mask_start)

        # Get the appropriate extractor
        extractor_start = tic()
        if this_model.per_particle and this_model.vel_shift:
            extractor = DopplerShiftedParticleExtractor(
                this_model.grid,
                this_model.extract,
            )
        elif this_model.per_particle:
            extractor = ParticleExtractor(
                this_model.grid,
                this_model.extract,
            )
        elif this_model.vel_shift:
            extractor = IntegratedDopplerShiftedParticleExtractor(
                this_model.grid,
                this_model.extract,
            )
        elif emitter.is_parametric and not isinstance(emitter, BlackHole):
            extractor = IntegratedParametricExtractor(
                this_model.grid,
                this_model.extract,
            )
        else:
            extractor = IntegratedParticleExtractor(
                this_model.grid,
                this_model.extract,
            )
        toc("Getting the extractor", extractor_start)

        # Get the spectra (note that result is a tuple containing the
        # particle spectra and the integrated spectra if per_particle
        # is True, otherwise it is just the integrated spectra)
        result = extractor.generate_lnu(
            emitter,
            this_model,
            mask=this_mask,
            lam_mask=this_model._lam_mask,
            grid_assignment_method=grid_assignment_method,
            nthreads=nthreads,
            do_grid_check=False,
        )

        # Cache the model on the emitter
        cache_model_params(this_model, emitter)

        # Store the spectra in the right place
        if this_model.per_particle:
            particle_spectra[label] = result[0]
            spectra[label] = result[1]
        else:
            spectra[label] = result

        return spectra, particle_spectra

    def _extract_lines(
        self,
        line_ids,
        this_model,
        emitters,
        lines,
        particle_lines,
        verbose,
        nthreads,
        grid_assignment_method,
    ):
        """Extract lines from the grid.

        Args:
            line_ids (list):
                The line ids to extract.
            this_model (EmissionModel):
                The emission model defining the extraction.
            emitters (dict):
                The emitters to extract the lines for.
            lines (dict):
                The dictionary to store the extracted lines in.
            particle_lines (dict):
                The dictionary to store the extracted particle lines in.
            verbose (bool):
                Are we talking?
            nthreads (int):
                The number of threads to use when generating lines.
            grid_assignment_method (str):
                The method to use when assigning particles to the grid.
                Options are 'cic' (cloud-in-cell) and 'ngp' (nearest
                grid point).

        Returns:
            dict:
                The dictionary of extracted lines.
        """
        # We need to be certain that any composite lines we have been
        # passed are split into their constituent parts
        passed_line_ids = line_ids
        line_ids = []

        # If line_ids is None, use all available lines from the grid
        if passed_line_ids is None:
            line_ids = list(this_model.grid.available_lines)
        else:
            for lid in passed_line_ids:
                for ljd in lid.split(","):
                    line_ids.append(ljd.strip())

        # Remove any duplicated lines, will give the user exactly what they
        # asked for, but there's not point doing extra work!
        line_ids = list(set(line_ids))

        # Extract the label for this model
        label = this_model.label

        # Skip models for a different emitter
        if this_model.emitter not in emitters:
            return lines, particle_lines

        # Check the attached grid has all the right lines in it
        missing_lines = set(line_ids) - set(this_model.grid.available_lines)
        if missing_lines:
            raise exceptions.MissingLines(
                f"The Grid does not contain the following lines: "
                f"{missing_lines}."
            )

        # Get the emitter
        emitter = emitters[this_model.emitter]

        # Do we have to define a property mask?
        this_mask = None
        for mask_dict in this_model.masks:
            this_mask = emitter.get_mask(
                **mask_dict,
                mask=this_mask,
                attr_override_obj=this_model,
            )

        # Get the appropriate extractor
        if this_model.per_particle and this_model.vel_shift:
            extractor = DopplerShiftedParticleExtractor(
                this_model.grid,
                this_model.extract,
            )
        elif this_model.per_particle:
            extractor = ParticleExtractor(
                this_model.grid,
                this_model.extract,
            )
        elif this_model.vel_shift:
            extractor = IntegratedDopplerShiftedParticleExtractor(
                this_model.grid,
                this_model.extract,
            )
        elif emitter.is_parametric and not isinstance(emitter, BlackHole):
            extractor = IntegratedParametricExtractor(
                this_model.grid,
                this_model.extract,
            )
        else:
            extractor = IntegratedParticleExtractor(
                this_model.grid,
                this_model.extract,
            )

        # Get the lam_mask based on the line_ids that have been requested
        lam_mask = np.isin(extractor._grid.available_lines, line_ids)

        # Now combine the model's lam_mask with the requested lines
        # if it exists
        if this_model._lam_mask is not None:
            # Get the indices that would bin the lines into the
            # spectra grid wavelength array
            lam_grid = extractor._grid.lam
            raw_indices = np.digitize(
                extractor._line_lams,
                lam_grid,
                right=False,
            )

            # Remove any lines which are masked out in the lam_mask
            for i, raw in enumerate(raw_indices):
                # Outside the lam grid -> leave this line as-is (no
                # lam-based filtering)
                if raw == 0 or raw == lam_grid.size:
                    continue
                grid_ix = raw - 1  # map to 0-based left-bin index
                if not this_model._lam_mask[grid_ix]:
                    lam_mask[i] = False

        # Get the spectra (note that result is a tuple containing the
        # particle line and the integrated line if per_particle is True,
        # otherwise it is just the integrated line)
        result = extractor.generate_line(
            emitter,
            this_model,
            mask=this_mask,
            lam_mask=lam_mask,
            grid_assignment_method=grid_assignment_method,
            nthreads=nthreads,
            do_grid_check=False,
        )

        # Cache the model on the emitter
        cache_model_params(this_model, emitter)

        # Store the lines in the right place
        if this_model.per_particle:
            particle_lines[label] = result[0]
            lines[label] = result[1]
        else:
            lines[label] = result

        # Ok, we have our lines but this contains all lines currently
        # with any not asked for set to zero. We need to filter this
        # down to just the lines we want. Note that this will also
        # handle composite lines
        if len(passed_line_ids) < lines[label].nlines:
            lines[label] = lines[label][passed_line_ids]
            if this_model.per_particle:
                particle_lines[label] = particle_lines[label][passed_line_ids]

        return lines, particle_lines

    def _extract_summary(self):
        """Return a summary of an extraction model."""
        # Create a list to hold the summary
        summary = []

        # Populate the list with the summary information
        summary.append("Extraction model:")
        summary.append(f"  Grid: {self._grid.grid_name}")
        summary.append(f"  Extract key: {self._extract}")
        summary.append(f"  Use velocity shift: {self._use_vel_shift}")

        return summary

    def extract_to_hdf5(self, group):
        """Save the extraction model to an HDF5 group."""
        # Flag it's extraction
        group.attrs["type"] = "extraction"

        # Save the grid
        group.attrs["grid"] = self._grid.grid_name

        # Save the extract key
        group.attrs["extract"] = self._extract


class Generation:
    """A class to define the generation of spectra.

    This can be used either to generate spectra for dust emission with the
    intrinsic and attenuated spectra used to scale the emission or to simply
    get a spectra from a generator.

    Attributes:
        generator (EmissionModel):
            The emission generation model. This must define a get_spectra
            method.
    """

    def __init__(self, generator):
        """Initialise the generation model.

        Args:
            generator (EmissionModel):
                The emission generation model. This must define a get_spectra
                method.
        """
        # Attach the emission generation model
        self._generator = generator

    def _generate_spectra(
        self,
        this_model,
        emission_model,
        spectra,
        particle_spectra,
        lam,
        emitter,
    ):
        """Generate the spectra for a given model.

        Args:
            this_model (EmissionModel):
                The model to generate the spectra for.
            emission_model (EmissionModel):
                The root emission model.
            spectra (dict):
                The dictionary of spectra.
            particle_spectra (dict):
                The dictionary of particle
            lam (ndarray):
                The wavelength grid to generate the spectra on.
            emitter (dict):
                The emitter to generate the spectra for.

        Returns:
            dict:
                The dictionary of spectra.
        """
        # Unpack what we need for dust emission
        generator = this_model.generator
        per_particle = this_model.per_particle

        # If we have an empty emitter we can just return zeros (only applicable
        # when nparticles exists in the emitter)
        if getattr(emitter, "nparticles", 1) == 0:
            spectra[this_model.label] = Sed(
                lam,
                np.zeros(lam.size) * erg / s / Hz,
            )
            if per_particle:
                particle_spectra[this_model.label] = Sed(
                    lam,
                    np.zeros((emitter.nparticles, lam.size)) * erg / s / Hz,
                )
            return spectra, particle_spectra

        # Special handling for templates
        if isinstance(generator, Template):
            # If we have a template we need to generate the spectra
            # for each model
            sed = generator.get_spectra(emitter.bolometric_luminosity)
        else:
            # Generate the spectra
            sed = generator._generate_spectra(
                lam,
                emitter,
                this_model,
                particle_spectra if per_particle else spectra,
            )

        # Cache the model on the emitter
        cache_model_params(this_model, emitter)

        # Store the spectra in the right place (integrating if we need to)
        if per_particle:
            particle_spectra[this_model.label] = sed
            spectra[this_model.label] = sed.sum()
        else:
            spectra[this_model.label] = sed

        return spectra, particle_spectra

    def _generate_lines(
        self,
        this_model,
        emission_model,
        lines,
        particle_lines,
        emitter,
        lams,
        line_ids,
        spectra,
        particle_spectra,
    ):
        """Generate the lines for a given model.

        This involves first generating the spectra and then extracting the
        emission at the line wavelengths.

        Args:
            this_model (EmissionModel):
                The model to generate the lines for.
            emission_model (EmissionModel):
                The root emission model.
            lines (dict):
                The dictionary of lines.
            particle_lines (dict):
                The dictionary of particle lines.
            emitter (Stars/BlackHoles/Galaxy):
                The emitter to generate the lines for.
            lams (unyt_array):
                The line wavelengths to generate the lines at.
            line_ids (list):
                The line ids to generate.
            spectra (dict):
                Dictionary of existing spectra from all emitters for scaling.
            particle_spectra (dict):
                Dictionary of existing particle spectra from all emitters for
                scaling.

        Returns:
            dict:
                The dictionary of lines.
        """
        generator = this_model.generator
        per_particle = this_model.per_particle

        # If the emitter is empty we can just return zeros. This is only
        # applicable when nparticles exists in the emitter
        if getattr(emitter, "nparticles", 1) == 0:
            # Create the zeroed luminosity and continuum arrays
            lums = np.zeros((0, len(lams))) * erg / s
            conts = np.zeros((0, len(lams))) * erg / s / Hz

            zeroed_lines = LineCollection(
                line_ids=line_ids,
                lam=lams,
                lum=lums,
                cont=conts,
            )

            if per_particle:
                particle_lines[this_model.label] = zeroed_lines
                lines[this_model.label] = zeroed_lines
            else:
                lines[this_model.label] = zeroed_lines

            return lines, particle_lines

        # Special handling for templates
        if isinstance(generator, Template):
            # If we have a template we need to generate the spectra
            # for each model
            spectra = generator.get_spectra(emitter.bolometric_luminosity)
            out_lines = LineCollection(
                line_ids=line_ids,
                lam=lams,
                lum=np.zeros((emitter.nparticles, len(lams))) * erg / s
                if per_particle
                else np.zeros(len(lams)) * erg / s,
                cont=spectra.get_lnu_at_lam(lams),
            )
        else:
            # Generate the spectra
            out_lines = generator._generate_lines(
                line_ids,
                lams,
                emitter,
                this_model,
                particle_lines if per_particle else lines,
                particle_spectra if per_particle else spectra,
            )

        # Cache the model on the emitter
        cache_model_params(this_model, emitter)

        # Store the lines in the right place (integrating if we need to)
        if per_particle:
            particle_lines[this_model.label] = out_lines
            lines[this_model.label] = out_lines.sum()
        else:
            lines[this_model.label] = out_lines

        return lines, particle_lines

    def _generate_summary(self):
        """Return a summary of a generation model."""
        # Create a list to hold the summary
        summary = []

        # Populate the list with the summary information
        summary.append("Generation model:")
        summary.append(f"  Emission generation model: {self._generator}")

        # Do we have intrinsic/attenuated/scaler models?
        if (
            hasattr(self._generator, "_intrinsic")
            and self._generator._intrinsic is not None
        ):
            summary.append(
                f"  Intrinsic energy balance model: "
                f"{self.generator._intrinsic.label}"
            )
        if (
            hasattr(self._generator, "_attenuated")
            and self._generator._attenuated is not None
        ):
            summary.append(
                f"  Attenuated energy balance model: "
                f"{self.generator._attenuated.label}"
            )
        if (
            hasattr(self._generator, "_scaler")
            and self._generator._scaler is not None
        ):
            summary.append(f"  Scaler model: {self.generator._scaler.label}")

        return summary

    def generate_to_hdf5(self, group):
        """Save the generation model to an HDF5 group."""
        # Flag it's generation
        group.attrs["type"] = "generation"

        # Save the generator
        group.attrs["generator"] = str(type(self._generator))

        # Save the energy balance models if they exist
        if (
            hasattr(self._generator, "_intrinsic")
            and self._generator._intrinsic is not None
        ):
            group.attrs["intrinsic_model"] = self.generator._intrinsic.label
        if (
            hasattr(self._generator, "_attenuated")
            and self._generator._attenuated is not None
        ):
            group.attrs["attenuated_model"] = self.generator._attenuated.label

        # And the same for the scaler
        if (
            hasattr(self._generator, "_scaler")
            and self._generator._scaler is not None
        ):
            group.attrs["scaler_model"] = self.generator._scaler.label


class Transformation:
    """A class to define the transformation of an emission.

    A transformation can include attenuation of the spectra by an extinction
    curve, or it can involve any scaling of the emission.

    A Transformer is needed to apply the transformation to the emission. This
    class must inherit from the Transformer base class (defined in
    emission_models/transformers/transformer.py) and must define the
    get_transformation method. This method should return an array the size
    of the input that can multiply the emission.

    Attributes:
        transformer (Transformer):
            The transformer to apply to the emission.
        apply_to (EmissionModel):
            The model to apply the transformer to.
    """

    def __init__(self, transformer, apply_to):
        """Initialise the dust attenuation model.

        Args:
            transformer (Transformer):
                The model defining the transformation.
            apply_to (EmissionModel):
                The model to apply the transformation to.
        """
        # Attach the transformer
        self._transformer = transformer

        # Attach the model to apply the transformer to
        self._apply_to = apply_to

    @property
    def _apply_to_label(self):
        """Return the label of the model to apply the transformation to."""
        return (
            self._apply_to
            if isinstance(self._apply_to, str)
            else self._apply_to.label
        )

    def _transform_emission(
        self,
        this_model,
        emissions,
        particle_emissions,
        emitter,
        this_mask,
        lam,
    ):
        """Transform an emission.

        This can act on either an Sed or a LineCollection using dependency
        injection, i.e. the appropriate transform will be applied based
        on what is passed into it.

        Args:
            this_model (EmissionModel):
                The model defining the transformation.
            emissions (dict):
                The dictionary of emissions (Sed/LineCollection).
            particle_emissions (dict):
                The dictionary of particle emissions (Sed/LineCollection).
            emitter (Stars/BlackHoles):
                The emitter to transform the spectra for.
            this_mask (dict):
                The mask to apply to the spectra.
            lam (unyt_array):
                The wavelength grid corresponding to the lam_mask elements.

        Returns:
            dict:
                The dictionary of spectra.
        """
        # Get the spectra to apply dust to
        if this_model.per_particle:
            apply_to = particle_emissions[this_model._apply_to_label]
        else:
            apply_to = emissions[this_model._apply_to_label]

        # If we have a LineColleciton and a lam_mask, we need to translate
        # the nlam lam_mask into an nlines lam_mask
        if (
            isinstance(apply_to, LineCollection)
            and this_model._lam_mask is not None
        ):
            # Get the line wavelengths
            line_lams = apply_to.lam

            # Get the indices that would bin the lines into the
            # spectra grid wavelength array
            raw_indices = np.digitize(
                line_lams,
                lam,
                right=False,
            )

            # Translate these indices into a mask
            lam_mask = np.zeros(apply_to.nlines, dtype=bool)
            for i, raw in enumerate(raw_indices):
                # Skip lines outside the grid: raw == 0 (below first edge)
                # or raw == nlam (at/above last edge)
                if raw == 0 or raw == lam.size:
                    continue
                grid_ix = raw - 1
                if this_model._lam_mask[grid_ix]:
                    lam_mask[i] = True

        else:
            # Otherwise we can just use the lam_mask as is
            lam_mask = this_model.lam_mask

        # Apply the transform to the emission
        emission = self.transformer._transform(
            apply_to,
            emitter,
            this_model,
            this_mask if this_model.per_particle else None,
            lam_mask,
        )

        # Cache the model on the emitter
        cache_model_params(this_model, emitter)

        # Store the spectra in the right place (integrating if we need to)
        if this_model.per_particle:
            particle_emissions[this_model.label] = emission
            emissions[this_model.label] = emission.sum()
        else:
            emissions[this_model.label] = emission

        return emissions, particle_emissions

    def _transform_summary(self):
        """Return a summary of a transformation model."""
        # Create a list to hold the summary
        summary = []

        # Populate the list with the summary information
        summary.append("Transformer model:")
        summary.append(f"  Transformer: {type(self.transformer)}")
        summary.append(f"  Apply to: {self._apply_to_label}")

        return summary

    def transformation_to_hdf5(self, group):
        """Save the transformation model to an HDF5 group."""
        # Flag it's dust attenuation
        group.attrs["type"] = "transformation"

        # Save the dust curve
        group.attrs["transformer"] = str(type(self._transformer))

        # Save the model to apply the dust curve to
        group.attrs["apply_to"] = self._apply_to_label


class Combination:
    """A class to define the combination of spectra.

    Attributes:
        combine (list):
            A list of models to combine.
    """

    def __init__(self, combine):
        """Initialise the combination model.

        Args:
            combine (list):
                A list of models to combine.
        """
        # Attach the models to combine
        self._combine = list(combine) if combine is not None else combine

    @property
    def _combine_labels(self):
        """Return the labels of the models to combine."""
        return [
            model if isinstance(model, str) else model.label
            for model in self._combine
        ]

    def _combine_spectra(
        self,
        emission_model,
        spectra,
        particle_spectra,
        this_model,
        emitter,
    ):
        """Combine the extracted spectra.

        Args:
            emission_model (EmissionModel):
                The root emission model. This is used to get a consistent
                wavelength grid.
            spectra (dict):
                The dictionary of spectra.
            particle_spectra (dict):
                The dictionary of particle spectra.
            this_model (EmissionModel):
                The model defining the combination.
            emitter (Stars/BlackHoles/Galaxy):
                The emitter to generate the spectra for.

        Returns:
            dict:
                The dictionary of spectra.
        """
        # Create an empty spectra to add to
        if this_model.per_particle:
            out_spec = Sed(
                emission_model.lam,
                lnu=np.zeros_like(
                    particle_spectra[this_model._combine_labels[0]]._lnu
                )
                * erg
                / s
                / Hz,
            )
        else:
            out_spec = Sed(
                emission_model.lam,
                lnu=np.zeros_like(spectra[this_model._combine_labels[0]]._lnu)
                * erg
                / s
                / Hz,
            )

        # Combine the spectra
        for combine_label in this_model._combine_labels:
            if this_model.per_particle:
                nan_mask = np.isnan(particle_spectra[combine_label]._lnu)
                out_spec._lnu[~nan_mask] += particle_spectra[
                    combine_label
                ]._lnu[~nan_mask]
            else:
                nan_mask = np.isnan(spectra[combine_label]._lnu)
                out_spec._lnu[~nan_mask] += spectra[combine_label]._lnu[
                    ~nan_mask
                ]

        # Cache the model on the emitter
        cache_model_params(this_model, emitter)

        # Store the spectra in the right place (integrating if we need to)
        if this_model.per_particle:
            particle_spectra[this_model.label] = out_spec
            spectra[this_model.label] = out_spec.sum()
        else:
            spectra[this_model.label] = out_spec

        return spectra, particle_spectra

    def _combine_lines(
        self,
        emission_model,
        lines,
        particle_lines,
        this_model,
        emitter,
    ):
        """Combine the extracted lines.

        Args:
            emission_model (EmissionModel):
                The root emission model. This is used to get a consistent
                wavelength grid.
            lines (dict):
                The dictionary of lines.
            particle_lines (dict):
                The dictionary of particle lines.
            this_model (EmissionModel):
                The model defining the combination.
            emitter (Stars/BlackHoles/Galaxy):
                The emitter to generate the spectra for.

        Returns:
            dict:
                The dictionary of lines.
        """
        # Get the right out lines dict and create the right entry
        out_lines = {}

        # Get the right exist lines dict and create the output lines
        if this_model.per_particle:
            in_lines = particle_lines
            out_lines = LineCollection(
                line_ids=particle_lines[
                    this_model._combine_labels[0]
                ].line_ids,
                lam=particle_lines[this_model._combine_labels[0]].lam,
                lum=np.zeros_like(
                    particle_lines[this_model._combine_labels[0]].luminosity
                ),
                cont=np.zeros_like(
                    particle_lines[this_model._combine_labels[0]].continuum
                ),
            )
        else:
            in_lines = lines
            out_lines = LineCollection(
                line_ids=lines[this_model._combine_labels[0]].line_ids,
                lam=lines[this_model._combine_labels[0]].lam,
                lum=np.zeros_like(
                    lines[this_model._combine_labels[0]].luminosity
                ),
                cont=np.zeros_like(
                    lines[this_model._combine_labels[0]].continuum
                ),
            )

        # Loop over combination models adding the lines
        for combine_label in this_model._combine_labels:
            out_lines += in_lines[combine_label]

        # Cache the model on the emitter
        cache_model_params(this_model, emitter)

        # Store the lines in the right place (integrating if we need to)
        if this_model.per_particle:
            particle_lines[this_model.label] = out_lines
            lines[this_model.label] = out_lines.sum()
        else:
            lines[this_model.label] = out_lines

        return lines, particle_lines

    def _combine_images(
        self,
        images,
        this_model,
        instrument,
        fov,
        img_type,
        do_flux,
        emitters,
        kernel,
        kernel_threshold,
        nthreads,
    ):
        """Combine the images by addition.

        Args:
            images (dict):
                The dictionary of image collections.
            this_model (EmissionModel):
                The model defining the combination.
            instrument (Instrument):
                The instrument to use when generating images.
            fov (float):
                The field of view of the images.
            img_type (str):
                The type of image to generate.
            do_flux (bool):
                Are we generating flux images?
            emitters (dict):
                The emitters to generate the images for.
            kernel (str):
                The kernel to use when generating images.
            kernel_threshold (float):
                The threshold to use when generating images.
            nthreads (int):
                The number of threads to use when generating images.
        """
        # Check we saved the models we are combining
        missing = [
            model.label
            for model in this_model.combine
            if model.label not in images
        ]

        # Ok, we're missing some images. All other images that can be made
        # have been made
        if len(missing) > 0:
            raise exceptions.MissingImage(
                "Can't generate galaxy level images without saving the "
                f"spectra from the component models ({', '.join(missing)})."
            )

        # Get the image for each model we are combining
        combine_images = []
        for combine_label in this_model._combine_labels:
            combine_images.append(images[combine_label])

        # Get the first image to add to
        out_image = ImageCollection(
            resolution=combine_images[0].resolution,
            fov=combine_images[0].fov,
            imgs={
                f: Image(
                    resolution=combine_images[0].resolution,
                    fov=combine_images[0].fov,
                    img=np.zeros(combine_images[0].npix)
                    * combine_images[0].imgs[f].units,
                )
                for f in combine_images[0].imgs
            },
        )

        # Combine the images
        # Again, we have a problem if any don't exist
        for img in combine_images:
            out_image += img

        # Store the combined image
        images[this_model.label] = out_image

        return images

    def _combine_summary(self):
        """Return a summary of a combination model."""
        # Create a list to hold the summary
        summary = []

        # Populate the list with the summary information
        summary.append("Combination model:")
        summary.append(
            f"  Combine models: {', '.join(list(self._combine_labels))}"
        )

        return summary

    def combine_to_hdf5(self, group):
        """Save the combination model to an HDF5 group."""
        # Flag it's combination
        group.attrs["type"] = "combination"

        # Save the models to combine
        group.attrs["combine"] = list(self._combine_labels)
