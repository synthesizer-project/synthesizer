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
from synthesizer.grid import Template
from synthesizer.imaging.image_collection import (
    _generate_image_collection_generic,
)
from synthesizer.line import Line, LineCollection
from synthesizer.parametric import Stars as ParametricStars
from synthesizer.sed import Sed


class Extraction:
    """
    A class to define the extraction of spectra from a grid.

    Attributes:
        grid (Grid):
            The grid to extract from.
        extract (str):
            The key for the spectra to extract.
    """

    def __init__(self, grid, extract, vel_shift):
        """
        Initialise the extraction model.

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
        if extract not in grid.spectra and extract not in grid.lines:
            raise exceptions.MissingSpectraType(
                f"The Grid does not contain the key '{extract}' "
                f"(available types are {grid.available_spectra}."
            )

        # Should the emission take into account the velocity shift due to
        # peculiar velocities? (Particle Only!)
        self._use_vel_shift = vel_shift

    def _extract_spectra(
        self,
        emission_model,
        emitters,
        spectra,
        particle_spectra,
        verbose,
        nthreads,
        grid_assignment_method,
    ):
        """
        Extract spectra from the grid.

        Args:
            emission_model (EmissionModel):
                The emission model to extract from.
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

            # Are we doing a parametric Stars object? If so we have a special
            # case (TODO: In the future we should make this work without
            # needing to do this)
            if isinstance(emitter, ParametricStars):
                parametric_stars = True
            else:
                parametric_stars = False

            # Do we have to define a property mask?
            this_mask = None
            for mask_dict in this_model.masks:
                this_mask = emitter.get_mask(**mask_dict, mask=this_mask)

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
            elif parametric_stars:
                extractor = IntegratedParametricExtractor(
                    this_model.grid,
                    this_model.extract,
                )
            else:
                extractor = IntegratedParticleExtractor(
                    this_model.grid,
                    this_model.extract,
                )

            sed = extractor.generate_lnu(
                emitter,
                this_model,
                mask=this_mask,
                lam_mask=this_model._lam_mask,
                grid_assignment_method=grid_assignment_method,
                nthreads=nthreads,
                do_grid_check=False,
            )

            # Store the spectra in the right place (integrating if we
            # need to)
            if this_model.per_particle:
                particle_spectra[label] = sed
                spectra[label] = sed.sum()
            else:
                spectra[label] = sed

        return spectra, particle_spectra

    def _extract_lines(
        self,
        line_ids,
        emission_model,
        emitters,
        lines,
        particle_lines,
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
            particle_lines (dict):
                The dictionary to store the extracted particle lines in.
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
            if this_model.per_particle:
                generator_func = emitter.generate_particle_line
            else:
                generator_func = emitter.generate_line

            # Initialise the lines dictionary for this label
            out_lines = {}

            # Loop over the line ids
            for line_id in line_ids:
                # Get this base lines
                out_lines[line_id] = generator_func(
                    grid=this_model.grid,
                    line_id=line_id,
                    line_type=this_model.extract,
                    mask=this_mask,
                    verbose=verbose,
                    **kwargs,
                )

            # Store the lines in the right place (integrating if we need to)
            if this_model.per_particle:
                particle_lines[label] = LineCollection(out_lines)
                lines[label] = LineCollection(
                    {
                        line_id: line.sum()
                        for line_id, line in out_lines.items()
                    }
                )
            else:
                lines[label] = LineCollection(out_lines)

            # Replace any fixed parameters
            for prop in prev_properties:
                setattr(emitter, prop, prev_properties[prop])

        return lines, particle_lines

    def _extract_images(
        self,
        instrument,
        fov,
        img_type,
        do_flux,
        emitters,
        images,
        kernel,
        kernel_threshold,
        nthreads,
        limit_to,
    ):
        """
        Create images for all the extraction keys.

        Args:
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
            images (dict):
                The dictionary to store the images in.
            kernel (str):
                The kernel to use when generating images.
            kernel_threshold (float):
                The threshold to use when generating images.
            nthreads (int):
                The number of threads to use when generating images.
            limit_to (str):
                Limit the images to a specific model.

        Returns:
            dict:
                The dictionary of image collections.
        """
        # Loop over the extraction keys
        for label in self._extract_keys.keys():
            # If we are limiting to a specific model, skip all others
            if limit_to is not None and label != limit_to:
                continue

            # Also skip any models we didn't save
            if not self._models[label].save:
                continue

            # Get the model
            this_model = self._models[label]

            # Get the emitter
            emitter = emitters[this_model.emitter]

            # Store the resulting image collection
            try:
                images[label] = _generate_image_collection_generic(
                    instrument,
                    fov,
                    img_type,
                    do_flux,
                    this_model.per_particle,
                    kernel,
                    kernel_threshold,
                    nthreads,
                    label,
                    emitter,
                )
            except Exception as e:
                print(f"Failed to generate image for {label}: {e}")
                raise e

        return images

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
        self,
        this_model,
        emission_model,
        spectra,
        particle_spectra,
        lam,
        emitter,
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

        # Handle the dust emission case
        if this_model._is_dust_emitting:
            # Get the right spectra
            if per_particle:
                intrinsic = particle_spectra[
                    this_model.lum_intrinsic_model.label
                ]
                attenuated = particle_spectra[
                    this_model.lum_attenuated_model.label
                ]
            else:
                intrinsic = spectra[this_model.lum_intrinsic_model.label]
                attenuated = spectra[this_model.lum_attenuated_model.label]

            # Apply the dust emission model
            sed = generator.get_spectra(
                lam,
                intrinsic,
                attenuated,
            )

        elif this_model.lum_intrinsic_model is not None:
            # otherwise we are scaling by a single spectra
            sed = generator.get_spectra(
                lam,
                particle_spectra[this_model.lum_intrinsic_model.label]
                if per_particle
                else spectra[this_model.lum_intrinsic_model.label],
            )
        elif isinstance(generator, Template):
            # If we have a template we need to generate the spectra
            # for each model
            sed = generator.get_spectra(
                emitter.bolometric_luminosity,
            )

        else:
            # Otherwise we have a bog standard generation
            sed = generator.get_spectra(
                lam,
            )

        # Store the spectra in the right place (integrating if we need to)
        if per_particle:
            particle_spectra[this_model.label] = sed
            spectra[this_model.label] = sed.sum()
        else:
            spectra[this_model.label] = sed

        return spectra, particle_spectra

    def _generate_lines(
        self,
        line_ids,
        this_model,
        emission_model,
        lines,
        particle_lines,
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
            particle_lines (dict):
                The dictionary of particle lines.
            emitter (Stars/BlackHoles/Galaxy):
                The emitter to generate the lines for.

        Returns:
            dict:
                The dictionary of lines.
        """
        # Unpack what we need for dust emission
        per_particle = this_model.per_particle

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

        # If the emitter is empty we can just return zeros. This is only
        # applicable when nparticles exists in the emitter
        if getattr(emitter, "nparticles", 1) == 0:
            lines[this_model.label] = {}
            for line_id in line_ids:
                # Get the emission at this lines wavelength
                lam = lines[this_model.lum_intrinsic_model.label][
                    line_id
                ].wavelength

                lines[this_model.label][line_id] = Line(
                    line_id=line_id,
                    wavelength=lam * Hz,
                    luminosity=np.zeros(emitter.nparticles),
                    continuum=np.zeros(emitter.nparticles),
                )

            # We need to make sure we do this for each particle too if needs
            # be
            if per_particle:
                particle_lines[this_model.label] = lines[this_model.label]

            return lines, particle_lines

        # Now we have the spectra we can get the emission at each line
        # and include it
        out_lines = {}
        for line_id in line_ids:
            # Get the emission at this lines wavelength
            lam = lines[this_model.lum_intrinsic_model.label][
                line_id
            ].wavelength

            # Get the continuum at this wavelength
            cont = spectra.get_lnu_at_lam(lam)

            # Create the line (luminoisty = continuum)
            out_lines[line_id] = Line(
                line_id=line_id,
                wavelength=lam,
                luminosity=0.0 * erg / s,
                continuum=cont,
            )

        # Store the lines in the right place (integrating if we need to)
        if per_particle:
            particle_lines[this_model.label] = LineCollection(out_lines)
            lines[this_model.label] = LineCollection(
                {line_id: line.sum() for line_id, line in out_lines.items()}
            )
        else:
            lines[this_model.label] = LineCollection(out_lines)

        return lines, particle_lines

    def _generate_images(
        self,
        instrument,
        fov,
        this_model,
        img_type,
        do_flux,
        emitter,
        images,
        kernel,
        kernel_threshold,
        nthreads,
    ):
        """
        Create an image for a generation key.

        Args:
            instrument (Instrument):
                The instrument to use when generating images.
            fov (float):
                The field of view of the images.
            this_model (EmissionModel):
                The model to generate the images for.
            img_type (str):
                The type of image to generate.
            do_flux (bool):
                Are we generating flux images?
            emitter (dict):
                The emitter to generate the images for.
            images (dict):
                The dictionary to store the images in.
            kernel (str):
                The kernel to use when generating images.
            kernel_threshold (float):
                The threshold to use when generating images.
            nthreads (int):
                The number of threads to use when generating images.

        Returns:
            dict:
                The dictionary of image collections now containing the
                generated images.
        """
        # Store the resulting image collection
        images[this_model.label] = _generate_image_collection_generic(
            instrument,
            fov,
            img_type,
            do_flux,
            this_model.per_particle,
            kernel,
            kernel_threshold,
            nthreads,
            this_model.label,
            emitter,
        )

        return images

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

    def generate_to_hdf5(self, group):
        """Save the generation model to an HDF5 group."""
        # Flag it's generation
        group.attrs["type"] = "generation"

        # Save the generator
        group.attrs["generator"] = str(type(self._generator))

        # Save the dust luminosity models
        if self._lum_intrinsic_model is not None:
            group.attrs["lum_intrinsic_model"] = (
                self._lum_intrinsic_model.label
            )
        if self._lum_attenuated_model is not None:
            group.attrs["lum_attenuated_model"] = (
                self._lum_attenuated_model.label
            )


class Transformation:
    """
    A class to define the transformation of an emission.

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
        """
        Initialise the dust attenuation model.

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

    def _transform_emission(
        self,
        this_model,
        emissions,
        particle_emissions,
        emitter,
        this_mask,
    ):
        """
        Transform an emission.

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

        Returns:
            dict:
                The dictionary of spectra.
        """
        # Get the spectra to apply dust to
        if this_model.per_particle:
            apply_to = particle_emissions[this_model.apply_to.label]
        else:
            apply_to = emissions[this_model.apply_to.label]

        # Apply the transform to the spectra
        emission = self.transformer._transform(
            apply_to,
            emitter,
            this_model,
            this_mask if this_model.per_particle else None,
            this_model.lam_mask,
        )

        # Store the spectra in the right place (integrating if we need to)
        if this_model.per_particle:
            particle_emissions[this_model.label] = emission
            emissions[this_model.label] = emission.sum()
        else:
            emissions[this_model.label] = emission

        return emissions, particle_emissions

    def _transform_images(
        self,
        instrument,
        fov,
        this_model,
        img_type,
        do_flux,
        emitter,
        images,
        kernel,
        kernel_threshold,
        nthreads,
    ):
        """
        Create an image for a transformation model.

        Args:
            instrument (Instrument):
                The instrument to use when generating images.
            fov (float):
                The field of view of the images.
            this_model (EmissionModel):
                The model to generate the images for.
            img_type (str):
                The type of image to generate.
            do_flux (bool):
                Are we generating flux images?
            emitter (dict):
                The emitter to generate the images for.
            images (dict):
                The dictionary to store the images in.
            kernel (str):
                The kernel to use when generating images.
            kernel_threshold (float):
                The threshold to use when generating images.
            nthreads (int):
                The number of threads to use when generating images.
            instrument (Instrument):
                The instrument to use when generating images.

        Returns:
            dict:
                The dictionary of image collections now containing the
                generated images.
        """
        # Store the resulting image collection
        images[this_model.label] = _generate_image_collection_generic(
            instrument,
            fov,
            img_type,
            do_flux,
            this_model.per_particle,
            kernel,
            kernel_threshold,
            nthreads,
            this_model.label,
            emitter,
        )

        return images

    def _transform_summary(self):
        """Return a summary of a transformation model."""
        # Create a list to hold the summary
        summary = []

        # Populate the list with the summary information
        summary.append("Transformer model:")
        summary.append(f"  Transformer: {type(self.transformer)}")
        summary.append(f"  Apply to: {self._apply_to.label}")

        return summary

    def transformation_to_hdf5(self, group):
        """Save the transformation model to an HDF5 group."""
        # Flag it's dust attenuation
        group.attrs["type"] = "transformation"

        # Save the dust curve
        group.attrs["transformer"] = str(type(self._transformer))

        # Save the model to apply the dust curve to
        group.attrs["apply_to"] = self._apply_to.label


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

    def _combine_spectra(
        self,
        emission_model,
        spectra,
        particle_spectra,
        this_model,
    ):
        """
        Combine the extracted spectra.

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

        Returns:
            dict:
                The dictionary of spectra.
        """
        # Create an empty spectra to add to
        if this_model.per_particle:
            out_spec = Sed(
                emission_model.lam,
                lnu=np.zeros_like(
                    particle_spectra[this_model.combine[0].label]._lnu
                )
                * erg
                / s
                / Hz,
            )
        else:
            out_spec = Sed(
                emission_model.lam,
                lnu=np.zeros_like(spectra[this_model.combine[0].label]._lnu)
                * erg
                / s
                / Hz,
            )

        # Combine the spectra
        for combine_model in this_model.combine:
            if this_model.per_particle:
                nan_mask = np.isnan(particle_spectra[combine_model.label]._lnu)
                out_spec._lnu[~nan_mask] += particle_spectra[
                    combine_model.label
                ]._lnu[~nan_mask]
            else:
                nan_mask = np.isnan(spectra[combine_model.label]._lnu)
                out_spec._lnu[~nan_mask] += spectra[combine_model.label]._lnu[
                    ~nan_mask
                ]

        # Store the spectra in the right place (integrating if we need to)
        if this_model.per_particle:
            particle_spectra[this_model.label] = out_spec
            spectra[this_model.label] = out_spec.sum()
        else:
            spectra[this_model.label] = out_spec

        return spectra, particle_spectra

    def _combine_lines(
        self,
        line_ids,
        emission_model,
        lines,
        particle_lines,
        this_model,
    ):
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
            particle_lines (dict):
                The dictionary of particle lines.
            this_model (EmissionModel):
                The model defining the combination.

        Returns:
            dict:
                The dictionary of lines.
        """
        # Get the right out lines dict and create the right entry
        out_lines = {}

        # Get the right exist lines dict
        if this_model.per_particle:
            in_lines = particle_lines
        else:
            in_lines = lines

        # Loop over lines copying over the first set of lines
        for line_id in line_ids:
            # Initialise the combined luminosity and continuum for the line
            lum = 0
            cont = 0

            # Get the wavelength of the line
            lam = in_lines[this_model.combine[0].label][line_id].wavelength

            # Combine the lines
            for combine_model in this_model.combine:
                lum += in_lines[combine_model.label][line_id]._luminosity
                cont += in_lines[combine_model.label][line_id]._continuum

            # Add the line to the dictionary
            out_lines[line_id] = Line(
                line_id=line_id,
                wavelength=lam,
                luminosity=lum * erg / s,
                continuum=cont * erg / s / Hz,
            )

        # Store the lines in the right place (integrating if we need to)
        if this_model.per_particle:
            particle_lines[this_model.label] = LineCollection(out_lines)
            lines[this_model.label] = LineCollection(
                {line_id: line.sum() for line_id, line in out_lines.items()}
            )
        else:
            lines[this_model.label] = LineCollection(out_lines)

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
        """
        Combine the images by addition.

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

        # Ok, we don't have the models so we have no choice but to generate
        # the image directly from the spectra
        if len(missing) > 0 and this_model.emitter != "galaxy":
            images[this_model.label] = _generate_image_collection_generic(
                instrument,
                fov,
                img_type,
                do_flux,
                this_model.per_particle,
                kernel,
                kernel_threshold,
                nthreads,
                this_model.label,
                emitters[this_model.emitter],
            )
            return images

        elif len(missing) > 0 and this_model.emitter == "galaxy":
            raise exceptions.MissingImage(
                "Can't generate galaxy level images without saving the "
                f"spectra from the component models ({', '.join(missing)})."
            )

        # Get the image for each model we are combining
        combine_labels = []
        combine_images = []
        for model in this_model.combine:
            combine_labels.append(model.label)
            combine_images.append(images[model.label])

        # Get the first image to add to
        out_image = combine_images[0]

        # Combine the images
        # Again, we have a problem if any don't exist
        for img in combine_images[1:]:
            out_image += img

        # Store the image
        images[this_model.label] = out_image

        return images

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

    def combine_to_hdf5(self, group):
        """Save the combination model to an HDF5 group."""
        # Flag it's combination
        group.attrs["type"] = "combination"

        # Save the models to combine
        group.attrs["combine"] = [model.label for model in self._combine]
