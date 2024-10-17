"""A module containing a pipeline helper class."""

import time
from functools import partial

import h5py
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool

from synthesizer import check_openmp, exceptions
from synthesizer._version import __version__
from synthesizer.instruments.filters import FilterCollection
from synthesizer.survey.survey_utils import (
    sort_data_recursive,
    write_datasets_recursive,
)
from synthesizer.utils.art import Art
from synthesizer.warnings import warn


class Survey:
    """"""

    def __init__(
        self,
        gal_loader_func,
        emission_model,
        instruments,
        n_galaxies,
        nthreads=1,
        comm=None,
        verbose=1,
    ):
        """
        Initialise the Survey object.

        This will not perform any part of the calculation, it only sets it up.

        This will attach all the passed attributes of the survey and set up
        anything we'll need later like MPI variables (if applicable), flags
        to indicate what stages we've completed and containers for any
        ouputs and additional analysis functions.

        This will also check arguments are sensible, e.g.
            - The galaxy loader function is callable and takes at least one
              argument (the galaxy index).
            - Synthesizer has been installed with OpenMP support if multiple
              threads are requested.

        Args:
            gal_loader_func (callable): The function to load galaxies. This
                function must return a Galaxy object or None. It should take
                the galaxy index as the first argument with that arugment
                being called 'gal_index'. Beyond this it can take any number
                if additional arguments and keyword arguments which must be
                passed to the load_galaxies function.
            emission_model (EmissionModel): The emission model to use for the
                survey.
            instruments (list): A list of Instrument objects to use for the
                survey.
            n_galaxies (int): How many galaxies will we load in total (i.e.
                not per rank if using MPI)?
            nthreads (int): The number of threads to use for shared memory
                parallelism. Default is 1.
            comm (MPI.Comm): The MPI communicator to use for MPI parallelism.
                Default is None.
            verbose (int): How talkative are we? 0: No output beyond hello and
                goodbye. 1: Outputs with timings but only on rank 0 (when using
                MPI). 2: Outputs with timings on all ranks (when using MPI).
        """
        # Attributes to track timing
        self._start_time = time.perf_counter()

        # Attach all the attributes we need to know what to do
        self.gal_loader_func = self._validate_loader(gal_loader_func)
        self.emission_model = emission_model
        self.instruments = instruments

        # Set verbosity
        self.verbose = verbose

        # How many galaxies are we going to be looking at?
        self.n_galaxies = n_galaxies

        # Initialise an attribute we'll store our galaxy indices into (this
        # will either be 0-n_galaxies or a subset of these indices if we are
        # running with MPI). We'll constructed this in load_galaxies.
        self.galaxy_indices = None

        # Define the container to hold the galaxies
        self.galaxies = []

        # How many threads are we using for shared memory parallelism?
        self.nthreads = nthreads

        # Check if we can use OpenMP
        if self.nthreads > 1 and not check_openmp():
            raise exceptions.MissingPartition(
                "Can't use multiple threads without OpenMP support. "
                " Install with: `WITH_OPENMP=1 pip install .`"
            )

        # It's quicker for us to collect together all filters and apply them
        # in one go, so we collect them together here. Note that getting
        # photometry is the only process that can be done collectively like
        # this without complex logic to check we don't have to do things
        # on an instrument-by-instrument basis (e.g. check resolution are
        # the same for imaging, wavelength arrays for spectroscopy etc.).
        self.filters = FilterCollection()
        for inst in instruments:
            if inst.can_do_photometry:
                self.filters += inst.filters

        # Define flags to indicate when we completed the various stages
        self._loaded_galaxies = False
        self._got_lnu_spectra = False
        self._got_fnu_spectra = False
        self._got_luminosities = False
        self._got_fluxes = False
        self._got_lum_lines = False
        self._got_flux_lines = False
        self._got_images_lum = False
        self._got_images_flux = False
        self._got_lnu_data_cubes = False
        self._got_fnu_data_cubes = False
        self._got_spectroscopy = False
        self._got_sfzh = False

        # Define containers for any additional analysis functions
        self._analysis_funcs = {}
        self._analysis_args = {}
        self._analysis_kwargs = {}

        # It'll be helpful later if we know what line IDs have been requested
        # so we'll store these should get_lines be called.
        self._line_ids = []

        # Everything that follows is only needed for hybrid parallelism
        # (running with MPI in addition to shared memory parallelism)

        # If we are running with hybrid parallelism, we need to know about
        # the communicator for MPI
        self.comm = comm
        self.using_mpi = comm is not None

        # Get some MPI informaiton if we are using MPI
        if self.using_mpi:
            self.rank = comm.Get_rank()
            self.size = comm.Get_size()
        else:
            self.rank = 0
            self.size = 1

        if self.rank == 0:
            self._say_hello()
            self._report()

    def _validate_loader(self, func):
        """
        Validate the galaxy loader function.

        This function checks that the galaxy loader function is a callable
        and that it takes at least one argument (the galaxy index). If the
        function is not valid, an exception is raised.

        Args:
            func (callable): The galaxy loader function to validate.
        """
        # Ensure we have a callable function
        if not callable(func):
            raise exceptions.InconsistentArguments(
                "gal_loader_func is not a callable function"
                f" (found {type(func)})."
            )

        # Ensure we have at least 1 argument
        if len(func.__code__.co_varnames) < 1:
            raise exceptions.InconsistentArguments(
                "gal_loader_func must take at least one "
                f"argument (found {len(func.__code__.co_varnames)})."
            )

        # Ensure the first argument is called "gal_index"
        if func.__code__.co_varnames[0] != "gal_index":
            raise exceptions.InconsistentArguments(
                "The first argument of gal_loader_func must be the index "
                "of the galaxy to load and must be called 'gal_index'. "
                f"(found '{func.__code__.co_varnames[0]}')."
            )

        return func

    def _say_hello(self):
        """Print a nice welcome."""
        print()
        print("\n".join([" " * 25 + s for s in Art.galaxy.split("\n")]))
        print()

    def _report(self):
        """Print a report contain the Survey setup."""
        # Print the MPI setup if we are using MPI
        if self.using_mpi:
            self._print(f"Running with MPI on {self.size} ranks.")

        # Print the shared memory parallelism setup
        if self.nthreads > 1 and self.using_mpi:
            self._print(f"Running with {self.nthreads} threads per rank.")
        elif self.nthreads > 1:
            self._print(f"Running with {self.nthreads} threads.")

        # Print the number of galaxies we are going to load
        self._print(f"Will process {self.n_galaxies} galaxies.")

        # Print some information about the emission model
        self._print(f"Root emission model: {self.emission_model.label}")
        self._print(
            f"EmissionModel contains {len(self.emission_model._models)} "
            "individual models."
        )
        self._print("EmissionModels split by emitter:")
        label_width = max(
            len("   - galaxy"), len("   - stellar"), len("   - blackhole")
        )
        ngal_models = len(
            [
                m
                for m in self.emission_model._models.values()
                if m.emitter == "galaxy"
            ]
        )
        nstar_models = len(
            [
                m
                for m in self.emission_model._models.values()
                if m.emitter == "stellar"
            ]
        )
        nbh_models = len(
            [
                m
                for m in self.emission_model._models.values()
                if m.emitter == "blackhole"
            ]
        )
        self._print(f"{'   - galaxy:'.ljust(label_width + 2)} {ngal_models}")
        self._print(f"{'   - stellar:'.ljust(label_width + 2)} {nstar_models}")
        self._print(f"{'   - blackhole:'.ljust(label_width + 2)} {nbh_models}")

        self._print("EmissionModels split by operation type:")
        label_width = max(
            len("   - extraction"),
            len("   - combination"),
            len("   - attenuating"),
            len("   - generation"),
        )
        nextract_models = len(
            [
                m
                for m in self.emission_model._models.values()
                if m._is_extracting
            ]
        )
        ncombine_models = len(
            [
                m
                for m in self.emission_model._models.values()
                if m._is_combining
            ]
        )
        nattenuate_models = len(
            [
                m
                for m in self.emission_model._models.values()
                if m._is_dust_attenuating
            ]
        )
        ngen_models = len(
            [
                m
                for m in self.emission_model._models.values()
                if m._is_generating or m._is_dust_emitting
            ]
        )
        self._print(
            f"{'   - extraction:'.ljust(label_width + 2)} {nextract_models}"
        )
        self._print(
            f"{'   - combination:'.ljust(label_width + 2)} {ncombine_models}"
        )
        self._print(
            f"{'   - attenuating:'.ljust(label_width + 2)} {nattenuate_models}"
        )
        self._print(
            f"{'   - generation:'.ljust(label_width + 2)} {ngen_models}"
        )

        # Print the number of instruments we have
        self._print(f"Using {len(self.instruments)} instruments.")

        # Print the number of filters we have
        self._print(f"Instruments have {len(self.filters)} filters in total.")

        # Make a breakdown of the instruments
        self._print(
            "Included instruments:",
            ", ".join(list(self.instruments.instruments.keys())),
        )
        self._print("Instruments split by capability:")
        label_width = max(
            len("   - photometry"),
            len("   - spectroscopy"),
            len("   - imaging"),
            len("   - resolved spectroscopy"),
        )
        nphot_inst = len(
            [inst for inst in self.instruments if inst.can_do_photometry]
        )
        nspec_inst = len(
            [inst for inst in self.instruments if inst.can_do_spectroscopy]
        )
        nimg_inst = len(
            [inst for inst in self.instruments if inst.can_do_imaging]
        )
        nresspec_inst = len(
            [
                inst
                for inst in self.instruments
                if inst.can_do_resolved_spectroscopy
            ]
        )
        self._print(
            f"{'   - photometry:'.ljust(label_width + 2)} {nphot_inst}"
        )
        self._print(
            "   - spectroscopy:".ljust(label_width + 2),
            nspec_inst,
        )
        self._print(f"{'   - imaging:'.ljust(label_width + 2)} {nimg_inst}")
        self._print(
            f"{'   - resolved spectroscopy:'.ljust(label_width + 2)}"
            f" {nresspec_inst}"
        )

    def _say_goodbye(self):
        """Print a nice goodbye including timing."""
        elapsed = time.perf_counter() - self._start_time

        # Report in sensible units
        if elapsed < 1:
            elapsed *= 1000
            units = "ms"
        elif elapsed < 60:
            units = "s"
        elif elapsed < 3600:
            elapsed /= 60
            units = "mins"
        else:
            elapsed /= 3600
            units = "hours"

        # Report how blazingly fast we are
        self._print(f"Total synthesis took {elapsed:.3f} {units}.")
        self._print("Goodbye!")

    def _print(self, *args, **kwargs):
        """
        Print a message to the screen with extra information.

        The prints behave differently depending on whether we are using MPI or
        not. We can also set the verbosity level at the Survey level which will
        control the verbosity of the print statements.

        Verbosity:
            0: No output beyond hello and goodbye.
            1: Outputs with timings but only on rank 0 (when using MPI).
            2: Outputs with timings on all ranks (when using MPI).

        Args:
            message (str): The message to print.
        """
        # At verbosity 0 we are silent
        if self.verbose == 0:
            return

        # Get the current time code in seconds with 0 padding and 2
        # decimal places
        now = time.perf_counter() - self._start_time
        int_now = str(int(now)).zfill(
            len(str(int(now))) + 1 if now > 9999 else 5
        )
        decimal = str(now).split(".")[-1][:2]
        now_str = f"{int_now}.{decimal}"

        # Create the prefix for the print, theres extra info to output if
        # we are using MPI
        if self.using_mpi:
            # Only print on rank 0 if we are using MPI and have verbosity 1
            if self.verbose == 1 and self.rank != 0:
                return

            prefix = (
                f"[{str(self.rank).zfill(len(str(self.size)) + 1)}]"
                f"[{now_str}]:"
            )

        else:
            prefix = f"[{now_str}]:"

        print(prefix, *args, **kwargs)

    def _took(self, start, message):
        """
        Print a message with the time taken since the start time.

        Args:
            start (float): The start time of the process.
            message (str): The message to print.
        """
        elapsed = time.perf_counter() - start

        # Report in sensible units
        if elapsed < 1:
            elapsed *= 1000
            units = "ms"
        elif elapsed < 60:
            units = "s"
        else:
            elapsed /= 60
            units = "mins"

        # Report how blazingly fast we are
        self._print(f"{message} took {elapsed:.3f} {units}.")

    def add_analysis_func(self, func, *args, **kwargs):
        """"""
        # Ensure we have a callable function
        if not callable(func):
            raise exceptions.InconsistentArguments(
                "Analysis function is not a callable function"
                f" (found {type(func)})."
            )

        # Warn the user if theres a name clash, we'll take the new one
        if func.__name__ in self._analysis_funcs:
            warn(
                f"{func.__name__} already exists in the analysis functions. "
                "Overwriting with the passed function."
            )

        # Add the function to the dictionary
        self._analysis_funcs[func.__name__] = func
        self._analysis_args[func.__name__] = args
        self._analysis_kwargs[func.__name__] = kwargs

        self._print(f"Added analysis function: {func.__name__}")

    def load_galaxies(self, *args, **kwargs):
        """
        Load the galaxies using the provided loader function.

        This function will load each individual galaxy in parallel using
        the number of threads specified in the constructor. The loader
        function should take the galaxy index as the first argument and can
        then take any number of additional arguments and keyword arguments
        passed to this function. The loader function should return a single
        galaxy.

        Args:
            *args: Any additional arguments to pass to the galaxy loader
                function. A copy of these will be sent to each thread.
            **kwargs: Any additional keyword arguments to pass to the galaxy
                loader function. A copy of these will be sent to each thread.
        """
        start = time.perf_counter()

        if self.nthreads > 1:
            self._print(
                f"Loading {self.n_galaxies} galaxies"
                f" with {self.gal_loader_func.__name__}"
                f" distributed over {self.nthreads} threads..."
            )
        else:
            self._print(
                f"Loading {self.n_galaxies} galaxies"
                f" with {self.gal_loader_func.__name__}..."
            )

        # Create the galaxy indices array, if needed. If we are running
        # with MPI this is created when partitioning.
        if self.using_mpi:
            if self.galaxy_indices is None:
                raise exceptions.MissingPartition(
                    "Before loading the galaxies need to be partitioned. "
                    "Call partition_galaxies with optional weights first."
                )
        else:
            self.galaxy_indices = range(self.n_galaxies)

        # OK, we've got everything we need to load the galaxies, we'll do
        # this in parallel if we have more than one thread and pass any extra
        # args and kwargs to the galaxy loader function.
        if self.nthreads > 1:
            with Pool(self.nthreads) as pool:
                self.galaxies = pool.map(
                    partial(self.gal_loader_func, *args, **kwargs),
                    self.galaxy_indices,
                )
        else:
            self.galaxies = [
                self.gal_loader_func(i, *args, **kwargs)
                for i in self.galaxy_indices
            ]

        # Sanitise the galaxies list to remove any None values
        for i, g in enumerate(self.galaxies):
            if g is None:
                self.galaxies.remove(g)
                self.galaxy_indices.pop(i)

        # Done!
        self._loaded_galaxies = True
        self._took(start, "Loading galaxies")

    def get_sfzh(self, grid):
        """"""
        start = time.perf_counter()

        # Loop over galaxies and get thier SFZH, skip any without stars. This
        # Can use internal shared memory parallelism so we just loop over the
        # galaxies.
        for g in self.galaxies:
            # Parametric galaxies have this ready to go so we can skip them
            if getattr(g, "sfzh", None) is not None:
                continue
            elif g.stars is not None and g.stars.nstars > 0:
                g.get_sfzh(grid, nthreads=self.nthreads)

        # Done!
        self._got_sfzh = True
        self._took(start, "Getting SFZH")

    def get_spectra(self, cosmo=None):
        """"""
        start = time.perf_counter()

        # Ensure we are ready
        if not self._loaded_galaxies:
            raise exceptions.SurveyNotReady(
                "Cannot generate spectra before galaxies are loaded! "
                "Call load_galaxies first."
            )

        # Loop over the galaxies and get the spectra
        for g in self.galaxies:
            g.get_spectra(self.emission_model, nthreads=self.nthreads)

        # If we have a cosmology, get the observed spectra. We can do this
        # with a threadpool if we have multiple threads, but there's no
        # internal shared memory parallelism in this process.
        if cosmo is not None and self.nthreads > 1:

            def _get_observed_spectra(g):
                g.get_observed_spectra(cosmo=cosmo)
                return g

            with Pool(self.nthreads) as pool:
                self.galaxies = pool.map(
                    _get_observed_spectra,
                    self.galaxies,
                )
        elif cosmo is not None:
            for g in self.galaxies:
                g.get_observed_spectra(cosmo=cosmo)

        # Done!
        self._got_lnu_spectra = True
        self._got_fnu_spectra = True if cosmo is not None else False
        self._took(start, "Generating spectra")

    def get_photometry_luminosities(self):
        """"""
        start = time.perf_counter()

        # Ensure we are ready
        if not self._got_lnu_spectra:
            raise exceptions.SurveyNotReady(
                "Cannot generate photometry before lnu spectra are generated! "
                "Call get_spectra first."
            )

        # Loop over the galaxies and get the photometry, there is internal
        # shared memory parallelism in this process so we can just loop over
        # the galaxies at this level
        for g in self.galaxies:
            g.get_photo_lnu(filters=self.filters, nthreads=self.nthreads)

        # Done!
        self._got_luminosities = True
        self._took(start, "Getting photometric luminosities")

    def get_photometry_fluxes(self):
        """"""
        start = time.perf_counter()

        # Ensure we are ready
        if not self._got_fnu_spectra:
            raise exceptions.SurveyNotReady(
                "Cannot generate photometry before fnu spectra are generated! "
                "Call get_spectra with a cosmology object first."
            )

        # Loop over the galaxies and get the photometry, there is internal
        # shared memory parallelism in this process so we can just loop over
        # the galaxies at this level
        for g in self.galaxies:
            g.get_photo_fnu(filters=self.filters, nthreads=self.nthreads)

        # Done!
        self._got_fluxes = True
        self._took(start, "Getting photometric fluxes")

    def get_lines(self, line_ids):
        """"""
        start = time.perf_counter()

        self._print(f"Generating {len(line_ids)} emission lines...")

        # Ensure we are ready
        if not self._loaded_galaxies:
            raise exceptions.SurveyNotReady(
                "Cannot generate emission lines before galaxies are loaded! "
                "Call load_galaxies first."
            )

        # Loop over the galaxies and get the spectra
        for g in self.galaxies:
            g.get_lines(line_ids, self.emission_model, nthreads=self.nthreads)

        # Store the line IDs for later
        self._line_ids = line_ids

        # Done!
        self._got_lum_lines = True
        self._took(start, "Getting emission lines")

    def get_images_luminosity(
        self,
        fov,
        img_type="smoothed",
        kernel=None,
        kernel_threshold=1.0,
    ):
        """"""
        start = time.perf_counter()

        # Ensure we are ready
        if not self._got_luminosities:
            raise exceptions.SurveyNotReady(
                "Cannot generate images before luminosities are generated! "
                "Call get_photometry_luminosities first."
            )

        def _apply_psfs(g):
            """"""
            for inst in self.instruments:
                if inst.can_do_psf_imaging:
                    for img in g.images_lnu:
                        img.apply_psfs(inst.psfs)

        def _apply_noise(g):
            """"""
            for inst in self.instruments:
                if inst.can_do_noisy_imaging:
                    for img in g.images_lnu:
                        img.apply_noise(inst.noise_maps)

        # Loop over instruments and perform any imaging they define
        for inst in self.instruments:
            # Skip if the instrument can't do imaging
            if not inst.can_do_imaging:
                continue

            # Loop over galaxies getting the initial images. We do this on
            # an individual galaxy basis since we can use internal shared
            # memory parallelism to do this
            for g in self.galaxies:
                g.get_images_luminosity(
                    resolution=inst.resolution,
                    fov=fov,
                    emission_model=self.emission_model,
                    img_type=img_type,
                    kernel=kernel,
                    kernel_threshold=kernel_threshold,
                    nthreads=self.nthreads,
                )

            # If the instrument has a PSF we can apply that here. If we can
            # we'll use a pool of threads to do this in parallel since there
            # is no internal shared memory parallelism in this process.
            if inst.can_do_psf_imaging:
                # Do we have multiple threads?
                if self.nthreads > 1:
                    with Pool(self.nthreads) as pool:
                        pool.map(
                            partial(self._apply_psfs, psfs=inst.psfs),
                            self.galaxies,
                        )
                else:
                    for g in self.galaxies:
                        self._apply_psfs(inst.psfs)

            # If the instrument has noise we can apply that here. Again, if
            # we can we'll use a pool of threads to do this in parallel since
            # there is no internal shared memory parallelism in this process.
            if inst.can_do_noisy_imaging:
                # Do we have multiple threads?
                if self.nthreads > 1:
                    with Pool(self.nthreads) as pool:
                        pool.map(_apply_noise, self.galaxies)
                else:
                    for g in self.galaxies:
                        self._apply_noise(instrument=inst)

        # Done!
        self._got_images_lum = True
        self._took(start, "Getting luminosity images")

    def get_images_flux(
        self,
        fov,
        img_type="smoothed",
        kernel=None,
        kernel_threshold=1.0,
    ):
        """"""
        start = time.perf_counter()

        # Ensure we are ready
        if not self._got_fluxes:
            raise exceptions.SurveyNotReady(
                "Cannot generate images before fluxes are generated! "
                "Call get_photometry_fluxes first."
            )

        def _apply_psfs(g):
            """"""
            for inst in self.instruments:
                if inst.can_do_psf_imaging:
                    for img in g.images_fnu:
                        img.apply_psfs(inst.psfs)

        def _apply_noise(g):
            """"""
            for inst in self.instruments:
                if inst.can_do_noisy_imaging:
                    for img in g.images_fnu:
                        img.apply_noise(inst.noise_maps)

        # Loop over instruments and perform any imaging they define
        for inst in self.instruments:
            # Skip if the instrument can't do imaging
            if not inst.can_do_imaging:
                continue

            # Loop over galaxies getting the initial images. We do this on
            # an individual galaxy basis since we can use internal shared
            # memory parallelism to do this
            for g in self.galaxies:
                g.get_images_flux(
                    resolution=inst.resolution,
                    fov=fov,
                    emission_model=self.emission_model,
                    img_type=img_type,
                    kernel=kernel,
                    kernel_threshold=kernel_threshold,
                    nthreads=self.nthreads,
                )

            # If the instrument has a PSF we can apply that here. If we can
            # we'll use a pool of threads to do this in parallel since there
            # is no internal shared memory parallelism in this process.
            if inst.can_do_psf_imaging:
                # Do we have multiple threads?
                if self.nthreads > 1:
                    with Pool(self.nthreads) as pool:
                        pool.map(_apply_psfs, self.galaxies)
                else:
                    for g in self.galaxies:
                        _apply_psfs(g)

            # If the instrument has noise we can apply that here. Again, if
            # we can we'll use a pool of threads to do this in parallel since
            # there is no internal shared memory parallelism in this process.
            if inst.can_do_noisy_imaging:
                # Do we have multiple threads?
                if self.nthreads > 1:
                    with Pool(self.nthreads) as pool:
                        pool.map(_apply_noise, self.galaxies)
                else:
                    for g in self.galaxies:
                        _apply_noise(g)

        # Done!
        self._got_images_flux = True
        self._took(start, "Getting flux images")

    def get_lnu_data_cubes(self):
        """"""
        start = time.perf_counter()
        raise exceptions.NotImplemented(
            "Data cubes are not yet implemented in Surveys."
        )

        # Done!
        self._got_lnu_data_cubes = True
        self._took(start, "Getting lnu data cubes")

    def get_fnu_data_cubes(self):
        """"""
        start = time.perf_counter()
        raise exceptions.NotImplemented(
            "Data cubes are not yet implemented in Surveys."
        )

        # Done!
        self._got_fnu_data_cubes = True
        self._took(start, "Getting fnu data cubes")

    def _run_extra_analysis(self):
        """"""
        start = time.perf_counter()

        # Nothing to do if we have no analysis functions
        if len(self._analysis_funcs) == 0:
            return

        # Loop over the analysis functions and run them on each individual
        # galaxy. We can do this with a threadpool if we have multiple threads.
        for name, func in self._analysis_funcs.items():
            if self.nthreads > 1:
                with Pool(self.nthreads) as pool:
                    pool.map(
                        partial(
                            func,
                            *self._analysis_args[name],
                            **self._analysis_kwargs[name],
                        ),
                        self.galaxies,
                    )
            else:
                for g in self.galaxies:
                    func(
                        g,
                        *self._analysis_args[name],
                        **self._analysis_kwargs[name],
                    )

        # Done!
        self._took(start, "Extra analysis")

    def _setup_output(self):
        """"""
        # Define a dictionary in which we'll collect EVERYTHING
        output = {"Galaxies": {}}

        # Set up the structure we know we'll need
        if self._got_lnu_spectra:
            output["Galaxies"]["SpectralLuminosityDensities"] = {}
        if self._got_fnu_spectra:
            output["Galaxies"]["SpectralFluxDensities"] = {}
        if self._got_luminosities:
            output["Galaxies"]["PhotometricLuminosities"] = {}
        if self._got_fluxes:
            output["Galaxies"]["PhotometricFluxes"] = {}
        if self._got_lum_lines:
            output["Galaxies"]["RestFrameEmissionLines"] = {}
        if self._got_flux_lines:
            output["Galaxies"]["ObservedEmissionLines"] = {}
        if self._got_images_lum:
            output["Galaxies"]["ImagesLuminosity"] = {}
        if self._got_images_flux:
            output["Galaxies"]["ImagesFlux"] = {}
        if self._got_spectroscopy:
            output["Galaxies"]["Spectroscopy"] = {}
        for key in self.emission_model._models.keys():
            # Skip unsaved models
            if not self.emission_model._models[key].save:
                continue

            # Make the appropriate entries
            if self._got_lnu_spectra:
                output["Galaxies"]["SpectralLuminosityDensities"][key] = (
                    np.zeros((self.n_galaxies, self.emission_model.lam.size))
                )
            if self._got_fnu_spectra:
                output["Galaxies"]["SpectralFluxDensities"][key] = np.zeros(
                    (self.n_galaxies, self.emission_model.lam.size)
                )
            if self._got_luminosities:
                output["Galaxies"]["PhotometricLuminosities"][key] = {}
            if self._got_fluxes:
                output["Galaxies"]["PhotometricFluxes"][key] = {}
            if self._got_lum_lines:
                output["Galaxies"]["RestFrameEmissionLines"][key] = {
                    l_id: {"Luminosities": [], "Continua": []}
                    for l_id in self._line_ids
                }
            if self._got_flux_lines:
                output["Galaxies"]["ObservedEmissionLines"][key] = {
                    l_id: {"Fluxes": [], "Continua": []}
                    for l_id in self._line_ids
                }
            if self._got_images_lum:
                output["Galaxies"]["ImagesLuminosity"][key] = {}
            if self._got_images_flux:
                output["Galaxies"]["ImagesFlux"][key] = {}
            if self._got_spectroscopy:
                output["Galaxies"]["Spectroscopy"][key] = {}

            # Loop over instruments and make entries for any instruments with
            # photometry capabilities
            for inst in self.instruments:
                if inst.can_do_photometry:
                    # Make the appropriate entries
                    if self._got_luminosities:
                        output["Galaxies"]["PhotometricLuminosities"][key][
                            inst.label
                        ] = {}
                    if self._got_fluxes:
                        output["Galaxies"]["PhotometricFluxes"][key][
                            inst.label
                        ] = {}

                # Make entries for any instruments with imaging capabilities
                if inst.can_do_imaging:
                    # Make the appropriate entries
                    if self._got_images_lum:
                        output["Galaxies"]["ImagesLuminosity"][key][
                            inst.label
                        ] = {}
                    if self._got_images_flux:
                        output["Galaxies"]["ImagesFlux"][key][inst.label] = {}

        return output

    def _collect_threaded(self, output):
        """"""
        # Loop over the galaxies and collect everything we need
        with Pool(self.nthreads) as pool:
            for key, model in self.emission_model.items():
                # Skip unsaved models
                if not self.emission_model._models[key].save:
                    continue

                # Get the spectra if they were made
                if self._got_lnu_spectra:
                    if model.emitter == "galaxy":
                        output["Galaxies"]["SpectralLuminosityDensities"][
                            key
                        ] = pool.map(
                            partial(
                                lambda g, key: g.spectra[key].lnu,
                                key=key,
                            ),
                            self.galaxies,
                        )
                    elif model.emitter == "stellar":
                        output["Galaxies"]["SpectralLuminosityDensities"][
                            key
                        ] = pool.map(
                            partial(
                                lambda g, key: g.stars.spectra[key].lnu,
                                key=key,
                            ),
                            self.galaxies,
                        )
                    elif model.emitter == "blackhole":
                        output["Galaxies"]["SpectralLuminosityDensities"][
                            key
                        ] = pool.map(
                            partial(
                                lambda g, key: g.blackholes.spectra[key].lnu,
                                key=key,
                            ),
                            self.galaxies,
                        )
                    else:
                        raise exceptions.SurveyError(
                            f"Unknown emitter type '{model.emitter}'"
                        )

                if self._got_fnu_spectra:
                    output["Galaxies"]["SpectralFluxDensities"][key] = (
                        pool.map(
                            partial(
                                lambda g, key: g.spectra[key].fnu, key=key
                            ),
                            self.galaxies,
                        )
                    )

                # Get the lines if they were made
                if self._got_lum_lines:
                    for l_id in self._line_ids:
                        output["Galaxies"]["EmissionLines"][key][l_id][
                            "Luminosities"
                        ] = pool.map(
                            partial(
                                lambda g, key, l_id: g.lines[key][
                                    l_id
                                ].luminosity,
                                key=key,
                            ),
                            self.galaxies,
                        )
                        output["Galaxies"]["EmissionLines"][key][l_id][
                            "Continua"
                        ] = pool.map(
                            partial(
                                lambda g, key, l_id: g.lines[key][
                                    l_id
                                ].continuum,
                                key=key,
                            ),
                            self.galaxies,
                        )

                # Loop over instruments and collect any data created for them
                for inst in self.instruments:
                    # Get the photometry if it was made
                    if self._got_luminosities and inst.can_do_photometry:
                        for fcode in inst.filters.filter_codes:
                            output["Galaxies"]["PhotometricLuminosities"][key][
                                inst.label
                            ][fcode] = pool.map(
                                partial(
                                    lambda g, key, fcode: g.photo_lnu[key][
                                        fcode
                                    ].photo_lnu,
                                    key=key,
                                    fcode=fcode,
                                ),
                                self.galaxies,
                            )
                    if self._got_fluxes and inst.can_do_photometry:
                        for fcode in inst.filters.filter_codes:
                            output["Galaxies"]["PhotometricFluxes"][key][
                                inst.label
                            ][fcode] = pool.map(
                                partial(
                                    lambda g, key, fcode: g.photo_fnu[key][
                                        fcode
                                    ].photo_fnu,
                                    key=key,
                                    fcode=fcode,
                                ),
                                self.galaxies,
                            )

                    # Get the images if they were made
                    if self._got_images_lum and inst.can_do_imaging:
                        for fcode in inst.filters.filter_codes:
                            output["Galaxies"]["Images"][key][inst.label][
                                fcode
                            ] = pool.map(
                                partial(
                                    lambda g, key, fcode: g.images_lnu[key][
                                        inst.label
                                    ][fcode].arr
                                    * g.images_lnu[key][inst.label][
                                        fcode
                                    ].units,
                                    key=key,
                                    fcode=fcode,
                                ),
                                self.galaxies,
                            )
                    if self._got_images_flux and inst.can_do_imaging:
                        for fcode in inst.filters.filter_codes:
                            output["Galaxies"]["Images"][key][inst.label][
                                fcode
                            ] = pool.map(
                                partial(
                                    lambda g, key, fcode: g.images_fnu[key][
                                        inst.label
                                    ][fcode].arr
                                    * g.images_fnu[key][inst.label][
                                        fcode
                                    ].units,
                                    key=key,
                                    fcode=fcode,
                                ),
                                self.galaxies,
                            )

                    # Get the spectroscopy if it was made
                    if self._got_spectroscopy and inst.can_do_spectroscopy:
                        output["Galaxies"]["Spectroscopy"][key][inst.label] = (
                            pool.map(
                                partial(
                                    lambda g, key: g.spectra[key].fnu,
                                    key=key,
                                ),
                                self.galaxies,
                            )
                        )

        # With everything collected we can sort it to be in the original order
        # and return it
        return sort_data_recursive(output, self.galaxy_indices)

    def _parallel_write(self, outpath, rank_output):
        """"""
        # If we have parallel h5py we can write in parallel using MPI,
        # redirect to the function that does this.
        if hasattr(h5py.get_config(), "mpi") and h5py.get_config().mpi:
            self._true_parallel_write(outpath, rank_output)
            return

    def _true_parallel_write(self, outpath, rank_output):
        """"""
        pass

    def _serial_write(
        self,
        outpath,
        output,
    ):
        """"""
        # Collected everything from the individual galaxies
        output = self._collect_threaded(output)

        # Write out the data to the HDF5 file rooted at the galaxy group. We
        # do this recursively to handle arbitrarily nested dictionaries.
        with h5py.File(outpath, "a") as hdf:
            write_datasets_recursive(hdf, output["Galaxies"], "Galaxies")

    def write(self, outpath, particle_datasets=False):
        """"""
        # Particle datasets are not yet implemented
        if particle_datasets:
            raise exceptions.NotImplemented(
                "Particle datasets are not yet implemented."
            )

        # We're done with everything so we know we'll have what is needed for
        # any extra analysis asked for by the user. We'll run these now.
        self._run_extra_analysis()

        # Set up the outputs dictionary, this is where we'll collect everything
        # together from the individual galaxies before writing it out.
        output = self._setup_output()

        # Regardless of parallel HDF5, we first need to create our file, some
        # basic structure and store some top level metadata (we will
        # overwrite any existing file with the same name)
        if self.rank == 0:
            with h5py.File(outpath, "w") as hdf:
                # Write out some top level metadata
                hdf.attrs["synthesizer_version"] = __version__

                # Create groups for the instruments, emission model, and
                # galaxies
                inst_group = hdf.create_group("Instruments")
                model_group = hdf.create_group("EmissionModel")
                hdf.create_group("Galaxies")  # we'll use this in a mo

                # Write out the instruments
                inst_group.attrs["ninstruments"] = (
                    self.instruments.ninstruments
                )
                for label, instrument in self.instruments.items():
                    instrument.to_hdf5(inst_group.create_group(label))

                # Write out the emission model
                for label, model in self.emission_model.items():
                    model.to_hdf5(model_group.create_group(label))

        # Call the appropriate galaxy property writer based on whether we
        # have parallel h5py
        write_start = time.perf_counter()
        if self.using_mpi:
            self._parallel_write(outpath)
        else:
            self._serial_write(outpath, output)
        self._took(write_start, f"Writing output to {outpath}")

        # Totally done!
        self._say_goodbye()

    def partition_galaxies(self, galaxy_weights=None, random_seed=42):
        """"""
        start = time.perf_counter()

        # Early exit if we have no galaxies
        if self.n_galaxies == 0:
            self.galaxy_indices = []
            return

        # Set the random seed
        np.random.seed(random_seed)

        # If we have no weights, we'll just partition the galaxies evenly
        # after randomising the order
        if galaxy_weights is None:
            inds = np.random.permutation(self.n_galaxies)
            inds_per_rank = {i: [] for i in range(self.size)}
            split_inds = np.array_split(inds, self.size)
            for i in range(self.size):
                inds_per_rank[i] = split_inds[i]
            self.galaxy_indices = list(inds_per_rank[self.rank])
        else:
            # Randomise the order of the loaded galaxies
            galaxy_order = np.random.permutation(self.n_galaxies)

            # Partition the galaxies
            inds_per_rank = {i: [] for i in range(self.size)}
            weights_per_rank = np.zeros(self.size)
            for i in galaxy_order:
                # Find the rank with the lowest weight
                rank = np.argmin(weights_per_rank)
                inds_per_rank[rank].append(i)
                weights_per_rank[rank] += galaxy_weights[i]

            # Assign the indices to the current rank
            self.galaxy_indices = inds_per_rank[self.rank]

        # Finally we produce a nice horizontal bar graph to show the
        # distribution of galaxies across the ranks. This only needs printing
        # on rank 0 regardless of verbosity.
        if self.rank == 0:
            self._print("Partitioned galaxies across ranks:")
            # Find the maximum list length for scaling
            max_length = max(len(lst) for lst in inds_per_rank.values())

            for key, lst in inds_per_rank.items():
                # Calculate the length of the bar based on the relative size
                bar_length = int((len(lst) / max_length) * 50)

                # Create the bar and append the list length in brackets
                bar = "#" * bar_length
                self._print(
                    f"Rank {str(key).zfill(len(str(self.size)) + 1)} - "
                    f"{bar} ({len(lst)})"
                )

        self._took(start, "Partitioning galaxies")
