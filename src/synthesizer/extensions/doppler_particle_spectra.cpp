/******************************************************************************
 * C extension to calculate SEDs for star particles.
 * Calculates weights on an arbitrary dimensional grid given the mass.
 *****************************************************************************/
/* C includes */
#include <array>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Python includes */
#define PY_ARRAY_UNIQUE_SYMBOL SYNTHESIZER_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy_init.h"
#include <Python.h>

/* Local includes */
#include "cpp_to_python.h"
#include "grid_props.h"
#include "macros.h"
#include "part_props.h"
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
 * This is the version of the function that accounts for doppler shift.
 * This is the serial version of the function.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 * @param c: speed of light .
 */
static void shifted_spectra_loop_cic_serial(GridProps *grid_props,
                                            Particles *parts, double *spectra,
                                            double *part_spectra,
                                            const double c) {

  /* Unpack the grid properties. */
  const int ndim = grid_props->ndim;
  int nlam = grid_props->nlam;
  double *wavelength = grid_props->get_lam();

  /* Calculate the number of cell in a patch of the grid (2^ndim). */
  int ncells = 1 << ndim;

  /* Set up fixed sub-dimensions array (always {2, 2, ..., 2}) */
  std::array<int, MAX_GRID_NDIM> sub_dims;
  for (int idim = 0; idim < ndim; idim++) {
    sub_dims[idim] = 2;
  }

  /* Allocate the shifted wavelengths array and the mapped indices array. */
  double *shifted_wavelengths = new double[nlam];
  int *mapped_indices = new int[nlam];

  /* Loop over particles. */
  for (int p = 0; p < parts->npart; p++) {

    /* Skip masked particles. */
    if (parts->part_is_masked(p)) {
      continue;
    }

    /* Get the particle velocity and red/blue shift factor. */
    double vel = parts->get_vel_at(p);
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
    std::array<int, MAX_GRID_NDIM> part_indices;
    std::array<double, MAX_GRID_NDIM> axis_fracs;

    /* Get the grid indices and cell fractions for the particle. */
    get_part_ind_frac_cic(part_indices, axis_fracs, grid_props, parts, p);

    /* Now loop over this collection of cells collecting and setting their
     * weights. */
    for (int icell = 0; icell < ncells; icell++) {

      /* Set up some index arrays we'll need. */
      std::array<int, MAX_GRID_NDIM> subset_ind;
      std::array<int, MAX_GRID_NDIM> frac_ind;

      /* Get the multi-dimensional version of icell. */
      get_indices_from_flat(icell, ndim, sub_dims, subset_ind);

      /* Multiply all contributing fractions and get the fractions index
       * in the grid. */
      double frac = 1;
      for (int idim = 0; idim < ndim; idim++) {
        int offset = subset_ind[idim]; // 0 or 1
        frac *= offset ? axis_fracs[idim] : (1.0 - axis_fracs[idim]);
        frac_ind[idim] = part_indices[idim] + offset;
      }

      /* Nothing to do if fraction is 0. */
      if (frac == 0) {
        continue;
      }

      /* Define the weight. */
      double weight = frac * parts->get_weight_at(p);

      /* Get the weight's index. */
      const int grid_ind = grid_props->ravel_grid_index(frac_ind);

      /* Add this grid cell's contribution to the spectra */
      for (int ilam = 0; ilam < nlam; ilam++) {

        /* Get the shifted wavelength and index. */
        int ilam_shifted = mapped_indices[ilam];
        double shifted_lambda = shifted_wavelengths[ilam];

        /* Skip if this wavelength is masked. */
        if (grid_props->lam_is_masked(ilam_shifted)) {
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
        double grid_spectra_value =
            grid_props->get_spectra_at(grid_ind, ilam) * weight;

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
  delete[] shifted_wavelengths;
  delete[] mapped_indices;
}

/**
 * @brief This calculates the grid weights in each grid cell using a cloud
 *        in cell approach.
 * This is the version of the function that accounts for doppler shift.
 * This is the parallel version of the function.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 * @param nthreads: The number of threads to use.
 * @param c: speed of light
 */
#ifdef WITH_OPENMP
static void shifted_spectra_loop_cic_omp(GridProps *grid_props,
                                         Particles *parts, double *spectra,
                                         double *part_spectra, int nthreads,
                                         const double c) {

  /* Unpack the grid properties. */
  const int ndim = grid_props->ndim;
  const int nlam = grid_props->nlam;
  double *wavelength = grid_props->get_lam();

  /* Calculate the number of cells in a patch of the grid. */
  const int ncells = 1 << ndim;

  /* Subset dimensions are always 2 (low and high side). */
  std::array<int, MAX_GRID_NDIM> sub_dims;
  for (int i = 0; i < ndim; i++) {
    sub_dims[i] = 2;
  }

  /* Allocate the shifted wavelengths array and the mapped indices array. */
  double *shifted_wavelengths = new double[nlam];
  int *mapped_indices = new int[nlam];

#pragma omp parallel for schedule(static)                                      \
    num_threads(nthreads) private(shifted_wavelengths, mapped_indices)
  for (int p = 0; p < parts->npart; p++) {

    /* Skip masked particles. */
    if (parts->part_is_masked(p)) {
      continue;
    }

    /* Get the particle velocity and red/blue shift factor. */
    double vel = parts->get_vel_at(p);
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

    /* Get the grid indices and cell fractions for the particle. */
    std::array<int, MAX_GRID_NDIM> part_indices;
    std::array<double, MAX_GRID_NDIM> axis_fracs;
    get_part_ind_frac_cic(part_indices, axis_fracs, grid_props, parts, p);

    /* Loop over sub-cells collecting their weighted contributions. */
    for (int icell = 0; icell < ncells; icell++) {

      /* Get the multi-dimensional version of icell. */
      std::array<int, MAX_GRID_NDIM> subset_ind;
      get_indices_from_flat(icell, ndim, sub_dims, subset_ind);

      /* Compute the contribution from this sub-cell. */
      std::array<int, MAX_GRID_NDIM> frac_ind;
      double frac = 1.0;
      for (int idim = 0; idim < ndim; idim++) {
        const int offset = subset_ind[idim]; // 0 or 1
        frac *= offset ? axis_fracs[idim] : (1.0 - axis_fracs[idim]);
        frac_ind[idim] = part_indices[idim] + offset;
      }

      /* Nothing to do if fraction is 0. */
      if (frac == 0.0) {
        continue;
      }

      /* Define the weighted contribution from this cell. */
      const double weight = frac * parts->get_weight_at(p);

      /* Get the index of the grid cell. */
      const int grid_ind = grid_props->ravel_grid_index(frac_ind);

      /* Add this grid cell's contribution to the spectra */
      for (int ilam = 0; ilam < nlam; ilam++) {

        /* Get the shifted wavelength and index. */
        int ilam_shifted = mapped_indices[ilam];
        double shifted_lambda = shifted_wavelengths[ilam];

        /* Skip if this wavelength is masked. */
        if (grid_props->lam_is_masked(ilam_shifted)) {
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
        double grid_spectra_value =
            grid_props->get_spectra_at(grid_ind, ilam) * weight;

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
  delete[] shifted_wavelengths;
  delete[] mapped_indices;
}
#endif

/**
 * @brief This calculates doppler-shifted particle spectra using a cloud in
 * cell approach.
 *
 * This is a wrapper which calls the correct function based on the number of
 * threads requested and whether OpenMP is available.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 * @param nthreads: The number of threads to use.
 */
void shifted_spectra_loop_cic(GridProps *grid_props, Particles *parts,
                              double *spectra, double *part_spectra,
                              const int nthreads, const double c) {

  double start_time = tic();

  /* Call the correct function for the configuration/number of threads. */

#ifdef WITH_OPENMP

  /* If we have multiple threads and OpenMP we can parallelise. */
  if (nthreads > 1) {
    shifted_spectra_loop_cic_omp(grid_props, parts, spectra, part_spectra,
                                 nthreads, c);
  }
  /* Otherwise there's no point paying the OpenMP overhead. */
  else {
    shifted_spectra_loop_cic_serial(grid_props, parts, spectra, part_spectra,
                                    c);
  }

#else

  (void)nthreads;

  /* We don't have OpenMP, just call the serial version. */
  shifted_spectra_loop_cic_serial(grid_props, parts, spectra, part_spectra, c);

#endif
  toc("Cloud in Cell particle loop", start_time);
}

/**
 * @brief This calculates particle spectra using a nearest grid point
 * approach. This is the version of the function that accounts doppler shift
 * This is the serial version of the function.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 * @param c: speed of light.
 */
static void shifted_spectra_loop_ngp_serial(GridProps *grid_props,
                                            Particles *parts, double *spectra,
                                            double *part_spectra,
                                            const double c) {

  /* Unpack the grid properties. */
  const int ndim = grid_props->ndim;
  int nlam = grid_props->nlam;
  double *wavelength = grid_props->get_lam();

  /* Allocate the shifted wavelengths array and the mapped indices array. */
  double *shifted_wavelengths = new double[nlam];
  int *mapped_indices = new int[nlam];

  /* Loop over particles. */
  for (int p = 0; p < parts->npart; p++) {

    /* Skip masked particles. */
    if (parts->part_is_masked(p)) {
      continue;
    }

    /* Get the particle velocity and red/blue shift factor. */
    double vel = parts->get_vel_at(p);
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
    std::array<int, MAX_GRID_NDIM> part_indices;

    /* Get the grid indices and cell fractions for the particle. */
    get_part_inds_ngp(part_indices, grid_props, parts, p);

    /* Define the weight. */
    double weight = parts->get_weight_at(p);

    /* Get the weight's index. */
    const int grid_ind = grid_props->ravel_grid_index(part_indices);

    /* Add this grid cell's contribution to the spectra */
    for (int ilam = 0; ilam < nlam; ilam++) {

      /* Get the shifted wavelength and index. */
      int ilam_shifted = mapped_indices[ilam];
      double shifted_lambda = shifted_wavelengths[ilam];

      /* Skip if this wavelength is masked. */
      if (grid_props->lam_is_masked(ilam_shifted)) {
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
      double grid_spectra_value =
          grid_props->get_spectra_at(grid_ind, ilam) * weight;

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
  delete[] shifted_wavelengths;
  delete[] mapped_indices;
}

/**
 * @brief This calculates particle spectra using a nearest grid point approach.
 * This is the version of the function that accounts for doppler shift.
 * This is the parallel version of the function.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 * @param nthreads: The number of threads to use.
 */
#ifdef WITH_OPENMP
static void shifted_spectra_loop_ngp_omp(GridProps *grid_props,
                                         Particles *parts, double *spectra,
                                         double *part_spectra, int nthreads,
                                         const double c) {

  /* Unpack the grid properties. */
  const int ndim = grid_props->ndim;
  const int nlam = grid_props->nlam;
  double *wavelength = grid_props->get_lam();

  /* Allocate the shifted wavelengths array and the mapped indices array. */
  double *shifted_wavelengths = new double[nlam];
  int *mapped_indices = new int[nlam];

#pragma omp parallel for schedule(static)                                      \
    num_threads(nthreads) private(shifted_wavelengths, mapped_indices)
  for (int p = 0; p < parts->npart; p++) {

    /* Skip masked particles. */
    if (parts->part_is_masked(p)) {
      continue;
    }

    /* Get the particle velocity and red/blue shift factor. */
    double vel = parts->get_vel_at(p);
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

    /* Get the grid indices and cell fractions for the particle. */
    std::array<int, MAX_GRID_NDIM> part_indices;
    get_part_inds_ngp(part_indices, grid_props, parts, p);

    /* Define the weighted contribution from this cell. */
    const double weight = parts->get_weight_at(p);

    /* Get the index of the grid cell. */
    const int grid_ind = grid_props->ravel_grid_index(part_indices);

    /* Add this grid cell's contribution to the spectra */
    for (int ilam = 0; ilam < nlam; ilam++) {

      /* Get the shifted wavelength and index. */
      int ilam_shifted = mapped_indices[ilam];
      double shifted_lambda = shifted_wavelengths[ilam];

      /* Skip if this wavelength is masked. */
      if (grid_props->lam_is_masked(ilam_shifted)) {
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
      double grid_spectra_value =
          grid_props->get_spectra_at(grid_ind, ilam) * weight;

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
  delete[] shifted_wavelengths;
  delete[] mapped_indices;
}
#endif

/**
 * @brief This calculates doppler-shifted particle spectra using a nearest
 * grid point approach.
 *
 * This is a wrapper which calls the correct function based on the number of
 * threads requested and whether OpenMP is available.
 * This is the version of the wrapper that accounts for doppler shift.
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 * @param nthreads: The number of threads to use.
 */
void shifted_spectra_loop_ngp(GridProps *grid_props, Particles *parts,
                              double *spectra, double *part_spectra,
                              const int nthreads, const double c) {

  double start_time = tic();

  /* Call the correct function for the configuration/number of threads. */

#ifdef WITH_OPENMP

  /* If we have multiple threads and OpenMP we can parallelise. */
  if (nthreads > 1) {
    shifted_spectra_loop_ngp_omp(grid_props, parts, spectra, part_spectra,
                                 nthreads, c);
  }
  /* Otherwise there's no point paying the OpenMP overhead. */
  else {
    shifted_spectra_loop_ngp_serial(grid_props, parts, spectra, part_spectra,
                                    c);
  }

#else

  (void)nthreads;

  /* We don't have OpenMP, just call the serial version. */
  shifted_spectra_loop_ngp_serial(grid_props, parts, spectra, part_spectra, c);

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
  GridProps *grid_props =
      new GridProps(np_grid_spectra, grid_tuple, np_lam, np_lam_mask, nlam);
  RETURN_IF_PYERR();

  /* Create the object that holds the particle properties. */
  Particles *part_props =
      new Particles(np_part_mass, np_velocities, np_mask, part_tuple, npart);
  RETURN_IF_PYERR();

  /* Allocate the spectra. */
  double *spectra = new double[grid_props->nlam]();
  double *part_spectra = new double[npart * grid_props->nlam]();

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
  delete part_props;
  delete grid_props;

  /* Construct the particle spectra output numpy array. */
  npy_intp np_dims[2] = {npart, nlam};
  PyArrayObject *out_part_spectra =
      wrap_array_to_numpy<double>(2, np_dims, part_spectra);

  /* Construct the integrated spectra output numpy array. */
  npy_intp np_dims_int[1] = {nlam};
  PyArrayObject *out_integrated_spectra =
      wrap_array_to_numpy<double>(1, np_dims_int, spectra);

  /* Construct the output tuple. */
  PyObject *out_tuple =
      Py_BuildValue("NN", out_part_spectra, out_integrated_spectra);

  toc("Computing particle and integrated lnus", start_time);

  return out_tuple;
}

/* Below is all the gubbins needed to make the module importable in Python. */
static PyMethodDef SedMethods[] = {
    {"compute_part_seds_with_vel_shift",
     (PyCFunction)compute_part_seds_with_vel_shift, METH_VARARGS,
     "Method for calculating particle intrinsic spectra accounting for "
     "velocity shift."},
    {NULL, NULL, 0, NULL}};

/* Make this importable. */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "make_shifted_particle_sed",                            /* m_name */
    "A module to calculate doppler shifted  particle seds", /* m_doc */
    -1,                                                     /* m_size */
    SedMethods,                                             /* m_methods */
    NULL,                                                   /* m_reload */
    NULL,                                                   /* m_traverse */
    NULL,                                                   /* m_clear */
    NULL,                                                   /* m_free */
};

PyMODINIT_FUNC PyInit_doppler_particle_spectra(void) {
  PyObject *m = PyModule_Create(&moduledef);
  if (numpy_import() < 0) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to import numpy.");
    return NULL;
  }
  return m;
};
