/******************************************************************************
 * C extension to calculate SEDs for star particles.
 * Calculates weights on an arbitrary dimensional grid given the mass.
 *****************************************************************************/
/* C includes */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Python includes */
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>

/* Local includes */
#include "macros.h"
#include "property_funcs.h"
#include "timers.h"
#include "weights.h"

/**
 * @brief Find nearest wavelength bin for a given lambda, in a given wavelength
 * array. Used by the spectra loop functions when considering doppler shift
 *
 * Note: binary search returns the index of the upper bin of those that straddle
 * the given lambda.
 */
int get_upper_lam_bin(double lambda, double *grid_wavelengths, int nlam) {
  return binary_search(0, nlam - 1, grid_wavelengths, lambda);
}

/**
 * @brief This calculates particle spectra using a cloud in cell approach.
 *
 * This is the serial version of the function.
 *
 * @param grid: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 */
static void spectra_loop_cic_serial(struct grid *grid, struct particles *parts,
                                    double *spectra, double *part_spectra) {

  /* Unpack the grid properties. */
  int *dims = grid->dims;
  int ndim = grid->ndim;
  int nlam = grid->nlam;
  double **grid_props = grid->props;
  double *grid_spectra = grid->spectra;

  /* Unpack the particles properties. */
  double *part_masses = parts->mass;
  double **part_props = parts->props;
  int npart = parts->npart;
  npy_bool *mask = parts->mask;

  /* Calculate the number of cell in a patch of the grid. */
  int ncells = 1 << ndim;

  /* Loop over particles. */
  for (int p = 0; p < npart; p++) {

    /* Skip masked particles. */
    if (mask != NULL && !mask[p]) {
      continue;
    }

    /* Get this particle's mass. */
    const double mass = part_masses[p];

    /* Setup the index and mass fraction arrays. */
    int part_indices[ndim];
    double axis_fracs[ndim];

    /* Get the grid indices and cell fractions for the particle. */
    get_part_ind_frac_cic(part_indices, axis_fracs, dims, ndim, grid_props,
                          part_props, p);

    /* To combine fractions we will need an array of dimensions for the
     * subset. These are always two in size, one for the low and one for high
     * grid point. */
    int sub_dims[ndim];
    for (int idim = 0; idim < ndim; idim++) {
      sub_dims[idim] = 2;
    }

    /* Now loop over this collection of cells collecting and setting their
     * weights. */
    for (int icell = 0; icell < ncells; icell++) {

      /* Set up some index arrays we'll need. */
      int subset_ind[ndim];
      int frac_ind[ndim];

      /* Get the multi-dimensional version of icell. */
      get_indices_from_flat(icell, ndim, sub_dims, subset_ind);

      /* Multiply all contributing fractions and get the fractions index
       * in the grid. */
      double frac = 1;
      for (int idim = 0; idim < ndim; idim++) {
        if (subset_ind[idim] == 0) {
          frac *= (1 - axis_fracs[idim]);
          frac_ind[idim] = part_indices[idim] - 1;
        } else {
          frac *= axis_fracs[idim];
          frac_ind[idim] = part_indices[idim];
        }
      }

      /* Nothing to do if fraction is 0. */
      if (frac == 0) {
        continue;
      }

      /* Define the weight. */
      double weight = frac * mass;

      /* Get the weight's index. */
      const int grid_ind = get_flat_index(frac_ind, dims, ndim);

      /* Get the spectra ind. */
      int unraveled_ind[ndim + 1];
      get_indices_from_flat(grid_ind, ndim, dims, unraveled_ind);
      unraveled_ind[ndim] = 0;
      int spectra_ind = get_flat_index(unraveled_ind, dims, ndim + 1);

      /* Add this grid cell's contribution to the spectra */
      for (int ilam = 0; ilam < nlam; ilam++) {

        /* Skip if this wavelength is masked. */
        if (grid->lam_mask != NULL && !grid->lam_mask[ilam]) {
          continue;
        }

        /* Add the contribution to this wavelength. */
        part_spectra[p * nlam + ilam] +=
            grid_spectra[spectra_ind + ilam] * weight;
        spectra[ilam] += grid_spectra[spectra_ind + ilam] * weight;
      }
    }
  }
}

/**
 * @brief This calculates the grid weights in each grid cell using a cloud
 *        in cell approach.
 *
 * This is the parallel version of the function.
 *
 * @param grid: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 * @param nthreads: The number of threads to use.
 */
#ifdef WITH_OPENMP
static void spectra_loop_cic_omp(struct grid *grid, struct particles *parts,
                                 double *spectra, double *part_spectra,
                                 int nthreads) {

  /* Unpack the grid properties. */
  int *dims = grid->dims;
  int ndim = grid->ndim;
  int nlam = grid->nlam;
  double **grid_props = grid->props;
  double *grid_spectra = grid->spectra;

  /* Unpack the particles properties. */
  double *part_masses = parts->mass;
  double **part_props = parts->props;
  int npart = parts->npart;
  npy_bool *mask = parts->mask;

  /* Calculate the number of cell in a patch of the grid. */
  int ncells = 1 << ndim;

#pragma omp parallel for schedule(static) num_threads(nthreads)
  /* Loop over particles. */
  for (int p = 0; p < npart; p++) {

    /* Skip masked particles. */
    if (mask != NULL && !mask[p]) {
      continue;
    }

    /* Get this particle's mass. */
    const double mass = part_masses[p];

    /* Setup the index and mass fraction arrays. */
    int part_indices[ndim];
    double axis_fracs[ndim];

    /* Get the grid indices and cell fractions for the particle. */
    get_part_ind_frac_cic(part_indices, axis_fracs, dims, ndim, grid_props,
                          part_props, p);

    /* To combine fractions we will need an array of dimensions for the
     * subset. These are always two in size, one for the low and one for high
     * grid point. */
    int sub_dims[ndim];
    for (int idim = 0; idim < ndim; idim++) {
      sub_dims[idim] = 2;
    }

    /* Now loop over this collection of cells collecting and setting their
     * weights. */
    for (int icell = 0; icell < ncells; icell++) {

      /* Set up some index arrays we'll need. */
      int subset_ind[ndim];
      int frac_ind[ndim];

      /* Get the multi-dimensional version of icell. */
      get_indices_from_flat(icell, ndim, sub_dims, subset_ind);

      /* Multiply all contributing fractions and get the fractions index
       * in the grid. */
      double frac = 1;
      for (int idim = 0; idim < ndim; idim++) {
        if (subset_ind[idim] == 0) {
          frac *= (1 - axis_fracs[idim]);
          frac_ind[idim] = part_indices[idim] - 1;
        } else {
          frac *= axis_fracs[idim];
          frac_ind[idim] = part_indices[idim];
        }
      }

      /* Nothing to do if fraction is 0. */
      if (frac == 0) {
        continue;
      }

      /* Define the weight. */
      double weight = frac * mass;

      /* Get the weight's index. */
      const int grid_ind = get_flat_index(frac_ind, dims, ndim);

      /* Get the spectra ind. */
      int unraveled_ind[ndim + 1];
      get_indices_from_flat(grid_ind, ndim, dims, unraveled_ind);
      unraveled_ind[ndim] = 0;
      int spectra_ind = get_flat_index(unraveled_ind, dims, ndim + 1);

      /* Add this grid cell's contribution to the spectra */
      for (int ilam = 0; ilam < nlam; ilam++) {

        /* Skip if this wavelength is masked. */
        if (grid->lam_mask != NULL && !grid->lam_mask[ilam]) {
          continue;
        }

        /* Add the contribution to this wavelength. */
        part_spectra[p * nlam + ilam] +=
            grid_spectra[spectra_ind + ilam] * weight;
        spectra[ilam] += grid_spectra[spectra_ind + ilam] * weight;
      }
    }
  }
}
#endif /* WITH_OPENMP */

/**
 * @brief This calculates particle spectra using a cloud in cell approach.
 *
 * This is a wrapper which calls the correct function based on the number of
 * threads requested and whether OpenMP is available.
 *
 * @param grid: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 * @param nthreads: The number of threads to use.
 */
void spectra_loop_cic(struct grid *grid, struct particles *parts,
                      double *spectra, double *part_spectra,
                      const int nthreads) {

  double start_time = tic();

  /* Call the correct function for the configuration/number of threads. */

#ifdef WITH_OPENMP

  /* If we have multiple threads and OpenMP we can parallelise. */
  if (nthreads > 1) {
    spectra_loop_cic_omp(grid, parts, spectra, part_spectra, nthreads);
  }
  /* Otherwise there's no point paying the OpenMP overhead. */
  else {
    spectra_loop_cic_serial(grid, parts, spectra, part_spectra);
  }

#else

  (void)nthreads;

  /* We don't have OpenMP, just call the serial version. */
  spectra_loop_cic_serial(grid, parts, spectra, part_spectra);

#endif
  toc("Cloud in Cell particle spectra loop", start_time);
}

/**
 * @brief This calculates particle spectra using a nearest grid point
 * approach.
 *
 * This is the serial version of the function.
 *
 * @param grid: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 */
static void spectra_loop_ngp_serial(struct grid *grid, struct particles *parts,
                                    double *spectra, double *part_spectra) {

  /* Unpack the grid properties. */
  int *dims = grid->dims;
  int ndim = grid->ndim;
  int nlam = grid->nlam;
  double **grid_props = grid->props;
  double *grid_spectra = grid->spectra;

  /* Unpack the particles properties. */
  double *part_masses = parts->mass;
  double **part_props = parts->props;
  int npart = parts->npart;
  npy_bool *mask = parts->mask;

  /* Loop over particles. */
  for (int p = 0; p < npart; p++) {

    /* Skip masked particles. */
    if (mask != NULL && !mask[p]) {
      continue;
    }

    /* Get this particle's mass. */
    const double weight = part_masses[p];

    /* Setup the index array. */
    int part_indices[ndim];

    /* Get the grid indices for the particle */
    get_part_inds_ngp(part_indices, dims, ndim, grid_props, part_props, p);

    /* Get the weight's index. */
    const int grid_ind = get_flat_index(part_indices, dims, ndim);

    /* Get the spectra ind. */
    int unraveled_ind[ndim + 1];
    get_indices_from_flat(grid_ind, ndim, dims, unraveled_ind);
    unraveled_ind[ndim] = 0;
    int spectra_ind = get_flat_index(unraveled_ind, dims, ndim + 1);

    /* Add this grid cell's contribution to the spectra */
    for (int ilam = 0; ilam < nlam; ilam++) {

      /* Skip if this wavelength is masked. */
      if (grid->lam_mask != NULL && !grid->lam_mask[ilam]) {
        continue;
      }

      /* Add the contribution to this wavelength. */
      part_spectra[p * nlam + ilam] +=
          grid_spectra[spectra_ind + ilam] * weight;
      spectra[ilam] += grid_spectra[spectra_ind + ilam] * weight;
    }
  }
}

/**
 * @brief This calculates particle spectra using a nearest grid point approach.
 *
 * This is the serial version of the function.
 *
 * @param grid: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 * @param nthreads: The number of threads to use.
 */
#ifdef WITH_OPENMP
static void spectra_loop_ngp_omp(struct grid *grid, struct particles *parts,
                                 double *spectra, double *part_spectra,
                                 int nthreads) {

  /* Unpack the grid properties. */
  int *dims = grid->dims;
  int ndim = grid->ndim;
  int nlam = grid->nlam;
  double **grid_props = grid->props;
  double *grid_spectra = grid->spectra;

  /* Unpack the particles properties. */
  double *part_masses = parts->mass;
  double **part_props = parts->props;
  int npart = parts->npart;
  npy_bool *mask = parts->mask;

/* Loop over particles. */
#pragma omp parallel for schedule(static) num_threads(nthreads)
  for (int p = 0; p < npart; p++) {

    /* Skip masked particles. */
    if (mask != NULL && !mask[p]) {
      continue;
    }

    /* Get this particle's mass. */
    const double weight = part_masses[p];

    /* Setup the index array. */
    int part_indices[ndim];

    /* Get the grid indices for the particle */
    get_part_inds_ngp(part_indices, dims, ndim, grid_props, part_props, p);

    /* Get the weight's index. */
    const int grid_ind = get_flat_index(part_indices, dims, ndim);

    /* Get the spectra ind. */
    int unraveled_ind[ndim + 1];
    get_indices_from_flat(grid_ind, ndim, dims, unraveled_ind);
    unraveled_ind[ndim] = 0;
    int spectra_ind = get_flat_index(unraveled_ind, dims, ndim + 1);

    /* Add this grid cell's contribution to the spectra */
    for (int ilam = 0; ilam < nlam; ilam++) {

      /* Skip if this wavelength is masked. */
      if (grid->lam_mask != NULL && !grid->lam_mask[ilam]) {
        continue;
      }

      /* Add the contribution to this wavelength. */
      part_spectra[p * nlam + ilam] +=
          grid_spectra[spectra_ind + ilam] * weight;
      spectra[ilam] += grid_spectra[spectra_ind + ilam] * weight;
    }
  }
}
#endif

/**
 * @brief This calculates particle spectra using a nearest grid point
 * approach.
 *
 * This is a wrapper which calls the correct function based on the number of
 * threads requested and whether OpenMP is available.
 *
 * @param grid: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 * @param nthreads: The number of threads to use.
 */
void spectra_loop_ngp(struct grid *grid, struct particles *parts,
                      double *spectra, double *part_spectra,
                      const int nthreads) {

  double start_time = tic();

  /* Call the correct function for the configuration/number of threads. */

#ifdef WITH_OPENMP

  /* If we have multiple threads and OpenMP we can parallelise. */
  if (nthreads > 1) {
    spectra_loop_ngp_omp(grid, parts, spectra, part_spectra, nthreads);
  }
  /* Otherwise there's no point paying the OpenMP overhead. */
  else {
    spectra_loop_ngp_serial(grid, parts, spectra, part_spectra);
  }

#else

  (void)nthreads;

  /* We don't have OpenMP, just call the serial version. */
  spectra_loop_ngp_serial(grid, parts, spectra, part_spectra);

#endif
  toc("Nearest Grid Point particle spectra loop", start_time);
}

/**
 * @brief Computes an integrated SED for a collection of particles.
 *
 * @param np_grid_spectra: The SPS spectra array.
 * @param grid_tuple: The tuple containing arrays of grid axis properties.
 * @param part_tuple: The tuple of particle property arrays (in the same
 * order as grid_tuple).
 * @param np_part_mass: The particle mass array.
 * @param np_velocities: The velocities array.
 * @param np_ndims: The size of each grid axis.
 * @param ndim: The number of grid axes.
 * @param npart: The number of particles.
 * @param nlam: The number of wavelength elements.
 * @param vel_shift: bool flag whether to consider doppler shift in spectra
 * computation. Defaults to False
 * @param c: speed of light
 */
PyObject *compute_particle_seds(PyObject *self, PyObject *args) {

  double start_time = tic();
  double setup_start = tic();

  /* We don't need the self argument but it has to be there. Tell the
   * compiler we don't care. */
  (void)self;

  int ndim, npart, nlam, nthreads;
  PyObject *grid_tuple, *part_tuple;
  PyArrayObject *np_grid_spectra;
  PyArrayObject *np_part_mass, *np_ndims;
  PyArrayObject *np_mask, *np_lam_mask;
  char *method;

  if (!PyArg_ParseTuple(args, "OOOOOiiisiOO", &np_grid_spectra, &grid_tuple,
                        &part_tuple, &np_part_mass, &np_ndims, &ndim, &npart,
                        &nlam, &method, &nthreads, &np_mask, &np_lam_mask)) {
    return NULL;
  }

  /* Extract the grid struct. */
  struct grid *grid_props =
      get_spectra_grid_struct(grid_tuple, np_ndims, np_grid_spectra,
                              /*np_lam*/ NULL, np_lam_mask, ndim, nlam);
  if (grid_props == NULL) {
    return NULL;
  }

  /* Extract the particle struct. */
  struct particles *part_props = get_part_struct(
      part_tuple, np_part_mass, /*np_velocities*/ NULL, np_mask, npart, ndim);
  if (part_props == NULL) {
    return NULL;
  }

  /* Allocate the spectra. */
  double *spectra = NULL;
  int err =
      posix_memalign((void **)&spectra, 64, grid_props->nlam * sizeof(double));
  if (err != 0 || spectra == NULL) {
    PyErr_SetString(PyExc_ValueError,
                    "Failed to allocate aligned memory for spectra.");
    return NULL;
  }
  bzero(spectra, grid_props->nlam * sizeof(double));
  double *part_spectra = calloc(npart * nlam, sizeof(double));
  if (part_spectra == NULL) {
    PyErr_SetString(PyExc_MemoryError,
                    "Could not allocate memory for spectra.");
    return NULL;
  }

  toc("Extracting Python data", setup_start);

  /* With everything set up we can compute the spectra for each particle
   * using the requested method. */
  if (strcmp(method, "cic") == 0) {
    spectra_loop_cic(grid_props, part_props, spectra, part_spectra, nthreads);
  } else if (strcmp(method, "ngp") == 0) {
    spectra_loop_ngp(grid_props, part_props, spectra, part_spectra, nthreads);
  } else {
    PyErr_Format(PyExc_ValueError, "Unknown grid assignment method (%s).",
                 method);
    return NULL;
  }

  /* Check we got the spectra sucessfully. (Any error messages will already
   * be set) */
  if (spectra == NULL) {
    return NULL;
  }
  if (part_spectra == NULL) {
    return NULL;
  }

  /* Clean up memory! */
  free(part_props);
  free(grid_props);

  /* Construct the particle spectra output numpy array. */
  npy_intp np_dims[2] = {npart, nlam};
  PyArrayObject *out_spectra =
      c_array_to_numpy(2, np_dims, NPY_FLOAT64, part_spectra);

  /* Construct the integrated spectra output numpy array. */
  npy_intp np_dims_int[1] = {nlam};
  PyArrayObject *out_integrated_spectra =
      c_array_to_numpy(1, np_dims_int, NPY_FLOAT64, spectra);

  /* Construct the output tuple. */
  PyObject *out_tuple =
      Py_BuildValue("NN", out_spectra, out_integrated_spectra);

  toc("Computing particle and integrated lnus", start_time);

  return out_tuple;
}

/**
 * @brief This calculates particle spectra using a cloud in cell approach.
 * This is the version of the function that accounts for doppler shift.
 * This is the serial version of the function.
 *
 * @param grid: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 * @param c: speed of light .
 */
static void shifted_spectra_loop_cic_serial(struct grid *grid,
                                            struct particles *parts,
                                            double *spectra,
                                            double *part_spectra,
                                            const double c) {

  /* Unpack the grid properties. */
  int *dims = grid->dims;
  int ndim = grid->ndim;
  int nlam = grid->nlam;
  double *wavelength = grid->lam;
  double **grid_props = grid->props;
  double *grid_spectra = grid->spectra;

  /* Unpack the particles properties. */
  double *part_masses = parts->mass;
  double **part_props = parts->props;
  double *velocity = parts->velocities;
  int npart = parts->npart;
  npy_bool *mask = parts->mask;

  /* Calculate the number of cell in a patch of the grid. */
  int ncells = 1 << ndim;

  /* Allocate the shifted wavelengths array and the mapped indices array. */
  double *shifted_wavelengths =
      synth_malloc(nlam * sizeof(double), "shifted wavelengths");
  int *mapped_indices = synth_malloc(nlam * sizeof(int), "mapped indices");

  /* Loop over particles. */
  for (int p = 0; p < npart; p++) {

    /* Skip masked particles. */
    if (mask != NULL && !mask[p]) {
      continue;
    }

    /* Get this particle's mass. */
    const double mass = part_masses[p];

    /* Get the particle velocity and red/blue shift factor. */
    double vel = velocity[p];
    double shift_factor = 1.0 + vel / c;

    /* Shift the wavelengths and get the mapping for each wavelength bin. We
     * do this for each element because there is no guarantee the input
     * wavelengths will be evenly spaced but we also don't want to repeat
     * the nearest bin search too many times. */
    for (int ilam = 0; ilam < nlam; ilam++) {
      shifted_wavelengths[ilam] = wavelength[ilam] * shift_factor;
      mapped_indices[ilam] =
          get_upper_lam_bin(shifted_wavelengths[ilam], wavelength, nlam);
    }

    /* Setup the index and mass fraction arrays. */
    int part_indices[ndim];
    double axis_fracs[ndim];

    /* Get the grid indices and cell fractions for the particle. */
    get_part_ind_frac_cic(part_indices, axis_fracs, dims, ndim, grid_props,
                          part_props, p);

    /* To combine fractions we will need an array of dimensions for the
     * subset. These are always two in size, one for the low and one for high
     * grid point. */
    int sub_dims[ndim];
    for (int idim = 0; idim < ndim; idim++) {
      sub_dims[idim] = 2;
    }

    /* Now loop over this collection of cells collecting and setting their
     * weights. */
    for (int icell = 0; icell < ncells; icell++) {

      /* Set up some index arrays we'll need. */
      int subset_ind[ndim];
      int frac_ind[ndim];

      /* Get the multi-dimensional version of icell. */
      get_indices_from_flat(icell, ndim, sub_dims, subset_ind);

      /* Multiply all contributing fractions and get the fractions index
       * in the grid. */
      double frac = 1;
      for (int idim = 0; idim < ndim; idim++) {
        if (subset_ind[idim] == 0) {
          frac *= (1 - axis_fracs[idim]);
          frac_ind[idim] = part_indices[idim] - 1;
        } else {
          frac *= axis_fracs[idim];
          frac_ind[idim] = part_indices[idim];
        }
      }

      /* Nothing to do if fraction is 0. */
      if (frac == 0) {
        continue;
      }

      /* Define the weight. */
      double weight = frac * mass;

      /* Get the weight's index. */
      const int grid_ind = get_flat_index(frac_ind, dims, ndim);

      /* Get the spectra ind. */
      int unraveled_ind[ndim + 1];
      get_indices_from_flat(grid_ind, ndim, dims, unraveled_ind);
      unraveled_ind[ndim] = 0;
      int spectra_ind = get_flat_index(unraveled_ind, dims, ndim + 1);

      /* Add this grid cell's contribution to the spectra */
      for (int ilam = 0; ilam < nlam; ilam++) {

        /* Get the shifted wavelength and index. */
        int ilam_shifted = mapped_indices[ilam];
        double shifted_lambda = shifted_wavelengths[ilam];

        /* Skip if this wavelength is masked. */
        /* Note: we skip the shifted wavelength here because the mask is
         * effectively saying "I want to ignore this wavelength in the
         * output spectra". */
        if (grid->lam_mask != NULL && !grid->lam_mask[ilam_shifted]) {
          continue;
        }

        /* Compute the fraction of the shifted wavelength between the two
         * closest wavelength elements. */
        double frac_shifted = 0.0;
        if (ilam_shifted > 0 && ilam_shifted <= nlam - 1) {
          frac_shifted =
              (shifted_lambda - wavelength[ilam_shifted - 1]) /
              (wavelength[ilam_shifted] - wavelength[ilam_shifted - 1]);
        } else {
          /* Out of bounds, skip this wavelength */
          continue;
        }

        /* Get the grid spectra value for this wavelength. */
        double grid_spectra_value = grid_spectra[spectra_ind + ilam] * weight;

        /* Add the contribution to the corresponding wavelength element. */
        part_spectra[p * nlam + ilam_shifted - 1] +=
            (1.0 - frac_shifted) * grid_spectra_value;
        part_spectra[p * nlam + ilam_shifted] +=
            frac_shifted * grid_spectra_value;
        spectra[ilam_shifted - 1] += (1.0 - frac_shifted) * grid_spectra_value;
        spectra[ilam_shifted] += frac_shifted * grid_spectra_value;
      }
    }
  }

  /* Free the allocated arrays. */
  free(shifted_wavelengths);
  free(mapped_indices);
}

/**
 * @brief This calculates the grid weights in each grid cell using a cloud
 *        in cell approach.
 * This is the version of the function that accounts for doppler shift.
 * This is the parallel version of the function.
 *
 * @param grid: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 * @param nthreads: The number of threads to use.
 * @param c: speed of light
 */
#ifdef WITH_OPENMP
static void shifted_spectra_loop_cic_omp(struct grid *grid,
                                         struct particles *parts,
                                         double *spectra, double *part_spectra,
                                         int nthreads, const double c) {

  /* Unpack the grid properties. */
  int *dims = grid->dims;
  int ndim = grid->ndim;
  int nlam = grid->nlam;
  double *wavelength = grid->lam;
  double **grid_props = grid->props;
  double *grid_spectra = grid->spectra;

  /* Unpack the particles properties. */
  double *part_masses = parts->mass;
  double **part_props = parts->props;
  double *velocity = parts->velocities;
  int npart = parts->npart;
  npy_bool *mask = parts->mask;

  /* Calculate the number of cell in a patch of the grid. */
  int ncells = 1 << ndim;
#pragma omp parallel num_threads(nthreads)
  {

    /* Allocate the shifted wavelengths array and the mapped indices array. */
    double *shifted_wavelengths =
        synth_malloc(nlam * sizeof(double), "shifted wavelengths");
    int *mapped_indices = synth_malloc(nlam * sizeof(int), "mapped indices");

    /* Loop over particles. */
#pragma omp for schedule(static)
    for (int p = 0; p < npart; p++) {

      /* Skip masked particles. */
      if (mask != NULL && !mask[p]) {
        continue;
      }

      /* Get this particle's mass. velocity and doppler shift. */
      const double mass = part_masses[p];

      /* Get the particle velocity and red/blue shift factor. */
      double vel = velocity[p];
      double shift_factor = 1.0 + vel / c;

      /* Shift the wavelengths and get the mapping for each wavelength bin. We
       * do this for each element because there is no guarantee the input
       * wavelengths will be evenly spaced but we also don't want to repeat
       * the nearest bin search too many times. */
      for (int ilam = 0; ilam < nlam; ilam++) {
        shifted_wavelengths[ilam] = wavelength[ilam] * shift_factor;
        mapped_indices[ilam] =
            get_upper_lam_bin(shifted_wavelengths[ilam], wavelength, nlam);
      }

      /* Setup the index and mass fraction arrays. */
      int part_indices[ndim];
      double axis_fracs[ndim];

      /* Get the grid indices and cell fractions for the particle. */
      get_part_ind_frac_cic(part_indices, axis_fracs, dims, ndim, grid_props,
                            part_props, p);

      /* To combine fractions we will need an array of dimensions for the
       * subset. These are always two in size, one for the low and one for
       * high grid point. */
      int sub_dims[ndim];
      for (int idim = 0; idim < ndim; idim++) {
        sub_dims[idim] = 2;
      }

      /* Now loop over this collection of cells collecting and setting their
       * weights. */
      for (int icell = 0; icell < ncells; icell++) {

        /* Set up some index arrays we'll need. */
        int subset_ind[ndim];
        int frac_ind[ndim];

        /* Get the multi-dimensional version of icell. */
        get_indices_from_flat(icell, ndim, sub_dims, subset_ind);

        /* Multiply all contributing fractions and get the fractions index
         * in the grid. */
        double frac = 1;
        for (int idim = 0; idim < ndim; idim++) {
          if (subset_ind[idim] == 0) {
            frac *= (1 - axis_fracs[idim]);
            frac_ind[idim] = part_indices[idim] - 1;
          } else {
            frac *= axis_fracs[idim];
            frac_ind[idim] = part_indices[idim];
          }
        }

        if (frac == 0) {
          continue;
        }

        /* Define the weight. */
        double weight = frac * mass;

        /* Get the weight's index. */
        const int grid_ind = get_flat_index(frac_ind, dims, ndim);

        /* Get the spectra ind. */
        int unraveled_ind[ndim + 1];
        get_indices_from_flat(grid_ind, ndim, dims, unraveled_ind);
        unraveled_ind[ndim] = 0;
        int spectra_ind = get_flat_index(unraveled_ind, dims, ndim + 1);

        /* Add this grid cell's contribution to the spectra */
        for (int ilam = 0; ilam < nlam; ilam++) {

          /* Get the shifted wavelength and index. */
          int ilam_shifted = mapped_indices[ilam];
          double shifted_lambda = shifted_wavelengths[ilam];

          /* Skip if this wavelength is masked. */
          /* Note: we skip the shifted wavelength here because the mask is
           * effectively saying "I want to ignore this wavelength in the
           * output spectra". */
          if (grid->lam_mask != NULL && !grid->lam_mask[ilam_shifted]) {
            continue;
          }

          /* Compute the fraction of the shifted wavelength between the two
           * closest wavelength elements. */
          double frac_shifted = 0.0;
          if (ilam_shifted > 0 && ilam_shifted <= nlam - 1) {
            frac_shifted =
                (shifted_lambda - wavelength[ilam_shifted - 1]) /
                (wavelength[ilam_shifted] - wavelength[ilam_shifted - 1]);
          } else {
            /* Out of bounds, skip this wavelength */
            continue;
          }

          /* Get the grid spectra value for this wavelength. */
          double grid_spectra_value = grid_spectra[spectra_ind + ilam] * weight;

          /* Add the contribution to the corresponding wavelength element. */
          part_spectra[p * nlam + ilam_shifted - 1] +=
              (1.0 - frac_shifted) * grid_spectra_value;
          part_spectra[p * nlam + ilam_shifted] +=
              frac_shifted * grid_spectra_value;
          spectra[ilam_shifted - 1] +=
              (1.0 - frac_shifted) * grid_spectra_value;
          spectra[ilam_shifted] += frac_shifted * grid_spectra_value;
        }
      }
    }

    /* Free the allocated arrays. */
    free(shifted_wavelengths);
    free(mapped_indices);
  }
}
#endif

/**
 * @brief This calculates doppler-shifted particle spectra using a cloud in
 * cell approach.
 *
 * This is a wrapper which calls the correct function based on the number of
 * threads requested and whether OpenMP is available.
 *
 * @param grid: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 * @param nthreads: The number of threads to use.
 */
void shifted_spectra_loop_cic(struct grid *grid, struct particles *parts,
                              double *spectra, double *part_spectra,
                              const int nthreads, const double c) {

  double start_time = tic();

  /* Call the correct function for the configuration/number of threads. */

#ifdef WITH_OPENMP

  /* If we have multiple threads and OpenMP we can parallelise. */
  if (nthreads > 1) {
    shifted_spectra_loop_cic_omp(grid, parts, spectra, part_spectra, nthreads,
                                 c);
  }
  /* Otherwise there's no point paying the OpenMP overhead. */
  else {
    shifted_spectra_loop_cic_serial(grid, parts, spectra, part_spectra, c);
  }

#else

  (void)nthreads;

  /* We don't have OpenMP, just call the serial version. */
  shifted_spectra_loop_cic_serial(grid, parts, spectra, part_spectra, c);

#endif
  toc("Cloud in Cell particle loop", start_time);
}

/**
 * @brief This calculates particle spectra using a nearest grid point
 * approach. This is the version of the function that accounts doppler shift
 * This is the serial version of the function.
 *
 * @param grid: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 * @param c: speed of light.
 */
static void shifted_spectra_loop_ngp_serial(struct grid *grid,
                                            struct particles *parts,
                                            double *spectra,
                                            double *part_spectra,
                                            const double c) {

  /* Unpack the grid properties. */
  int *dims = grid->dims;
  int ndim = grid->ndim;
  int nlam = grid->nlam;
  double *wavelength = grid->lam;
  double **grid_props = grid->props;
  double *grid_spectra = grid->spectra;

  /* Unpack the particles properties. */
  double *part_masses = parts->mass;
  double **part_props = parts->props;
  double *velocity = parts->velocities;
  int npart = parts->npart;
  npy_bool *mask = parts->mask;

  /* Allocate the shifted wavelengths array and the mapped indices array. */
  double *shifted_wavelengths =
      synth_malloc(nlam * sizeof(double), "shifted wavelengths");
  int *mapped_indices = synth_malloc(nlam * sizeof(int), "mapped indices");

  /* Loop over particles. */
  for (int p = 0; p < npart; p++) {

    /* Skip masked particles. */
    if (mask != NULL && !mask[p]) {
      continue;
    }

    /* Get this particle's mass, velocity and doppler shift. */
    const double weight = part_masses[p];

    /* Get the particle velocity and red/blue shift factor. */
    double vel = velocity[p];
    double shift_factor = 1.0 + vel / c;

    /* Shift the wavelengths and get the mapping for each wavelength bin. We
     * do this for each element because there is no guarantee the input
     * wavelengths will be evenly spaced but we also don't want to repeat
     * the nearest bin search too many times. */
    for (int ilam = 0; ilam < nlam; ilam++) {
      shifted_wavelengths[ilam] = wavelength[ilam] * shift_factor;
      mapped_indices[ilam] =
          get_upper_lam_bin(shifted_wavelengths[ilam], wavelength, nlam);
    }

    /* Setup the index array. */
    int part_indices[ndim];

    /* Get the grid indices for the particle */
    get_part_inds_ngp(part_indices, dims, ndim, grid_props, part_props, p);

    /* Get the weight's index. */
    const int grid_ind = get_flat_index(part_indices, dims, ndim);

    /* Get the spectra ind. */
    int unraveled_ind[ndim + 1];
    get_indices_from_flat(grid_ind, ndim, dims, unraveled_ind);
    unraveled_ind[ndim] = 0;
    int spectra_ind = get_flat_index(unraveled_ind, dims, ndim + 1);

    /* Add this grid cell's contribution to the spectra */
    for (int ilam = 0; ilam < nlam; ilam++) {

      /* Get the shifted wavelength and index. */
      int ilam_shifted = mapped_indices[ilam];
      double shifted_lambda = shifted_wavelengths[ilam];

      /* Skip if this wavelength is masked. */
      /* Note: we skip the shifted wavelength here because the mask is
       * effectively saying "I want to ignore this wavelength in the
       * output spectra". */
      if (grid->lam_mask != NULL && !grid->lam_mask[ilam_shifted]) {
        continue;
      }

      /* Compute the fraction of the shifted wavelength between the two
       * closest wavelength elements. */
      double frac_shifted = 0.0;
      if (ilam_shifted > 0 && ilam_shifted <= nlam - 1) {
        frac_shifted =
            (shifted_lambda - wavelength[ilam_shifted - 1]) /
            (wavelength[ilam_shifted] - wavelength[ilam_shifted - 1]);
      } else {
        /* Out of bounds, skip this wavelength */
        continue;
      }

      /* Get the grid spectra value for this wavelength. */
      double grid_spectra_value = grid_spectra[spectra_ind + ilam] * weight;

      /* Add the contribution to the corresponding wavelength element. */
      part_spectra[p * nlam + ilam_shifted - 1] +=
          (1.0 - frac_shifted) * grid_spectra_value;
      part_spectra[p * nlam + ilam_shifted] +=
          frac_shifted * grid_spectra_value;
      spectra[ilam_shifted - 1] += (1.0 - frac_shifted) * grid_spectra_value;
      spectra[ilam_shifted] += frac_shifted * grid_spectra_value;
    }
  }

  /* Free the allocated arrays. */
  free(shifted_wavelengths);
  free(mapped_indices);
}

/**
 * @brief This calculates particle spectra using a nearest grid point approach.
 * This is the version of the function that accounts for doppler shift.
 * This is the parallel version of the function.
 *
 * @param grid: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 * @param nthreads: The number of threads to use.
 */
#ifdef WITH_OPENMP
static void shifted_spectra_loop_ngp_omp(struct grid *grid,
                                         struct particles *parts,
                                         double *spectra, double *part_spectra,
                                         int nthreads, const double c) {

  /* Unpack the grid properties. */
  int *dims = grid->dims;
  int ndim = grid->ndim;
  int nlam = grid->nlam;
  double *wavelength = grid->lam;
  double **grid_props = grid->props;
  double *grid_spectra = grid->spectra;

  /* Unpack the particles properties. */
  double *part_masses = parts->mass;
  double **part_props = parts->props;
  double *velocity = parts->velocities;
  int npart = parts->npart;
  npy_bool *mask = parts->mask;

#pragma omp parallel num_threads(nthreads)
  {

    /* Allocate the shifted wavelengths array and the mapped indices array. */
    double *shifted_wavelengths =
        synth_malloc(nlam * sizeof(double), "shifted wavelengths");
    int *mapped_indices = synth_malloc(nlam * sizeof(int), "mapped indices");

#pragma omp for schedule(static)
    /* Loop over particles. */
    for (int p = 0; p < npart; p++) {

      /* Skip masked particles. */
      if (mask != NULL && !mask[p]) {
        continue;
      }

      /* Get this particle's mass, velocity and doppler shift contribution. */
      const double weight = part_masses[p];

      /* Get the particle velocity and red/blue shift factor. */
      double vel = velocity[p];
      double shift_factor = 1.0 + vel / c;

      /* Shift the wavelengths and get the mapping for each wavelength bin. We
       * do this for each element because there is no guarantee the input
       * wavelengths will be evenly spaced but we also don't want to repeat
       * the nearest bin search too many times. */
      for (int ilam = 0; ilam < nlam; ilam++) {
        shifted_wavelengths[ilam] = wavelength[ilam] * shift_factor;
        mapped_indices[ilam] =
            get_upper_lam_bin(shifted_wavelengths[ilam], wavelength, nlam);
      }

      /* Setup the index array. */
      int part_indices[ndim];

      /* Get the grid indices for the particle */
      get_part_inds_ngp(part_indices, dims, ndim, grid_props, part_props, p);

      /* Get the weight's index. */
      const int grid_ind = get_flat_index(part_indices, dims, ndim);

      /* Get the spectra ind. */
      int unraveled_ind[ndim + 1];
      get_indices_from_flat(grid_ind, ndim, dims, unraveled_ind);
      unraveled_ind[ndim] = 0;
      int spectra_ind = get_flat_index(unraveled_ind, dims, ndim + 1);

      /* Add this grid cell's contribution to the spectra */
      for (int ilam = 0; ilam < nlam; ilam++) {

        /* Get the shifted wavelength and index. */
        int ilam_shifted = mapped_indices[ilam];
        double shifted_lambda = shifted_wavelengths[ilam];

        /* Skip if this wavelength is masked. */
        /* Note: we skip the shifted wavelength here because the mask is
         * effectively saying "I want to ignore this wavelength in the
         * output spectra". */
        if (grid->lam_mask != NULL && !grid->lam_mask[ilam_shifted]) {
          continue;
        }

        /* Compute the fraction of the shifted wavelength between the two
         * closest wavelength elements. */
        double frac_shifted = 0.0;
        if (ilam_shifted > 0 && ilam_shifted <= nlam - 1) {
          frac_shifted =
              (shifted_lambda - wavelength[ilam_shifted - 1]) /
              (wavelength[ilam_shifted] - wavelength[ilam_shifted - 1]);
        } else {
          /* Out of bounds, skip this wavelength */
          continue;
        }

        /* Get the grid spectra value for this wavelength. */
        double grid_spectra_value = grid_spectra[spectra_ind + ilam] * weight;

        /* Add the contribution to the corresponding wavelength element. */
        part_spectra[p * nlam + ilam_shifted - 1] +=
            (1.0 - frac_shifted) * grid_spectra_value;
        part_spectra[p * nlam + ilam_shifted] +=
            frac_shifted * grid_spectra_value;
        spectra[ilam_shifted - 1] += (1.0 - frac_shifted) * grid_spectra_value;
        spectra[ilam_shifted] += frac_shifted * grid_spectra_value;
      }
    }
    /* Free the allocated arrays. */
    free(shifted_wavelengths);
    free(mapped_indices);
  }
}
#endif

/**
 * @brief This calculates doppler-shifted particle spectra using a nearest
 * grid point approach.
 *
 * This is a wrapper which calls the correct function based on the number of
 * threads requested and whether OpenMP is available.
 * This is the version of the wrapper that accounts for doppler shift.
 * @param grid: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 * @param nthreads: The number of threads to use.
 */
void shifted_spectra_loop_ngp(struct grid *grid, struct particles *parts,
                              double *spectra, double *part_spectra,
                              const int nthreads, const double c) {

  double start_time = tic();

  /* Call the correct function for the configuration/number of threads. */

#ifdef WITH_OPENMP

  /* If we have multiple threads and OpenMP we can parallelise. */
  if (nthreads > 1) {
    shifted_spectra_loop_ngp_omp(grid, parts, spectra, part_spectra, nthreads,
                                 c);
  }
  /* Otherwise there's no point paying the OpenMP overhead. */
  else {
    shifted_spectra_loop_ngp_serial(grid, parts, spectra, part_spectra, c);
  }

#else

  (void)nthreads;

  /* We don't have OpenMP, just call the serial version. */
  shifted_spectra_loop_ngp_serial(grid, parts, spectra, part_spectra, c);

#endif
  toc("Nearest Grid Point particle spectra loop", start_time);
}

/**
 * @brief Computes an integrated SED for a collection of particles.
 *
 * @param np_grid_spectra: The SPS spectra array.
 * @param grid_tuple: The tuple containing arrays of grid axis properties.
 * @param part_tuple: The tuple of particle property arrays (in the same
 * order as grid_tuple).
 * @param np_part_mass: The particle mass array.
 * @param np_velocities: The velocities array.
 * @param np_ndims: The size of each grid axis.
 * @param ndim: The number of grid axes.
 * @param npart: The number of particles.
 * @param nlam: The number of wavelength elements.
 * @param vel_shift: bool flag whether to consider doppler shift in spectra
 * computation. Defaults to False
 * @param c: speed of light
 */
PyObject *compute_part_seds_with_vel_shift(PyObject *self, PyObject *args) {

  double start_time = tic();
  double setup_start = tic();

  /* We don't need the self argument but it has to be there. Tell the
   * compiler we don't care. */
  (void)self;

  int ndim, npart, nlam, nthreads;
  PyObject *grid_tuple, *part_tuple;
  PyObject *py_vel_shift;
  PyObject *py_c;
  PyArrayObject *np_grid_spectra, *np_lam;
  PyArrayObject *np_velocities;
  PyArrayObject *np_part_mass, *np_ndims;
  PyArrayObject *np_mask, *np_lam_mask;
  char *method;

  if (!PyArg_ParseTuple(args, "OOOOOOOiiisiOOO", &np_grid_spectra, &np_lam,
                        &grid_tuple, &part_tuple, &np_part_mass, &np_velocities,
                        &np_ndims, &ndim, &npart, &nlam, &method, &nthreads,
                        &py_c, &np_mask, &np_lam_mask)) {
    return NULL;
  }

  /* Extract the grid struct. */
  struct grid *grid_props = get_spectra_grid_struct(
      grid_tuple, np_ndims, np_grid_spectra, np_lam, np_lam_mask, ndim, nlam);
  if (grid_props == NULL) {
    return NULL;
  }

  /* Extract the particle struct. */
  struct particles *part_props = get_part_struct(
      part_tuple, np_part_mass, np_velocities, np_mask, npart, ndim);
  if (part_props == NULL) {
    return NULL;
  }

  /* Allocate the spectra. */
  double *spectra = NULL;
  int err =
      posix_memalign((void **)&spectra, 64, grid_props->nlam * sizeof(double));
  if (err != 0 || spectra == NULL) {
    PyErr_SetString(PyExc_ValueError,
                    "Failed to allocate aligned memory for spectra.");
    return NULL;
  }
  bzero(spectra, grid_props->nlam * sizeof(double));
  double *part_spectra = calloc(npart * nlam, sizeof(double));
  if (part_spectra == NULL) {
    PyErr_SetString(PyExc_MemoryError,
                    "Could not allocate memory for spectra.");
    return NULL;
  }

  /* Convert c to double */
  double c = PyFloat_AsDouble(py_c);

  toc("Extracting Python data", setup_start);

  /* With everything set up we can compute the spectra for each particle
   * using the requested method. */
  if (strcmp(method, "cic") == 0) {
    shifted_spectra_loop_cic(grid_props, part_props, spectra, part_spectra,
                             nthreads, c);
  } else if (strcmp(method, "ngp") == 0) {
    shifted_spectra_loop_ngp(grid_props, part_props, spectra, part_spectra,
                             nthreads, c);
  } else {
    PyErr_SetString(PyExc_ValueError, "Unknown grid assignment method (%s).");
    return NULL;
  }

  /* Check we got the spectra sucessfully. (Any error messages will already
   * be set) */
  if (spectra == NULL) {
    return NULL;
  }
  if (part_spectra == NULL) {
    return NULL;
  }

  /* Clean up memory! */
  free(part_props);
  free(grid_props);

  /* Construct the particle spectra output numpy array. */
  npy_intp np_dims[2] = {npart, nlam};
  PyArrayObject *out_spectra =
      c_array_to_numpy(2, np_dims, NPY_FLOAT64, part_spectra);

  /* Construct the integrated spectra output numpy array. */
  npy_intp np_dims_int[1] = {nlam};
  PyArrayObject *out_integrated_spectra =
      c_array_to_numpy(1, np_dims_int, NPY_FLOAT64, spectra);

  /* Construct the output tuple. */
  PyObject *out_tuple =
      Py_BuildValue("NN", out_spectra, out_integrated_spectra);

  toc("Computing particle and integrated lnus", start_time);

  return out_tuple;
}

/* Below is all the gubbins needed to make the module importable in Python. */
static PyMethodDef SedMethods[] = {
    {"compute_particle_seds", (PyCFunction)compute_particle_seds, METH_VARARGS,
     "Method for calculating particle intrinsic spectra."},
    {"compute_part_seds_with_vel_shift",
     (PyCFunction)compute_part_seds_with_vel_shift, METH_VARARGS,
     "Method for calculating particle intrinsic spectra accounting for "
     "velocity shift."},
    {NULL, NULL, 0, NULL}};

/* Make this importable. */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "make_particle_sed",                   /* m_name */
    "A module to calculate particle seds", /* m_doc */
    -1,                                    /* m_size */
    SedMethods,                            /* m_methods */
    NULL,                                  /* m_reload */
    NULL,                                  /* m_traverse */
    NULL,                                  /* m_clear */
    NULL,                                  /* m_free */
};

PyMODINIT_FUNC PyInit_particle_spectra(void) {
  PyObject *m = PyModule_Create(&moduledef);
  import_array();
  return m;
};
