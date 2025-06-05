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
 * @brief This calculates particle spectra using a cloud in cell approach.
 *
 * This is the serial version of the function.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 */
static void spectra_loop_cic_serial(GridProps *grid_props, Particles *parts,
                                    double *spectra, double *part_spectra) {

  /* Unpack the grid properties. */
  const int ndim = grid_props->ndim;
  int nlam = grid_props->nlam;

  /* Calculate the number of cell in a patch of the grid (2^ndim). */
  int ncells = 1 << ndim;

  /* Set up fixed sub-dimensions array (always {2, 2, ..., 2}) */
  std::array<int, MAX_GRID_NDIM> sub_dims;
  for (int idim = 0; idim < ndim; idim++) {
    sub_dims[idim] = 2;
  }

  /* Loop over particles. */
  for (int p = 0; p < parts->npart; p++) {

    /* Skip masked particles. */
    if (parts->part_is_masked(p)) {
      continue;
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

        /* Skip if this wavelength is masked. */
        if (grid_props->lam_is_masked(ilam)) {
          continue;
        }

        /* Get the spectra value at this index and wavelength. */
        double spec_val = grid_props->get_spectra_at(grid_ind, ilam);

        /* Add the contribution to this wavelength. */
        part_spectra[p * nlam + ilam] =
            std::fma(spec_val, weight, part_spectra[p * nlam + ilam]);
        spectra[ilam] = std::fma(spec_val, weight, spectra[ilam]);
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
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 * @param part_spectra: The per-particle output array.
 * @param nthreads: The number of threads to use.
 */
#ifdef WITH_OPENMP
static void spectra_loop_cic_omp(GridProps *grid_props, Particles *parts,
                                 double *spectra, double *part_spectra,
                                 int nthreads) {

  /* Unpack the grid properties. */
  const int ndim = grid_props->ndim;
  const int nlam = grid_props->nlam;

  /* Calculate the number of cells in a patch of the grid. */
  const int ncells = 1 << ndim;

  /* Subset dimensions are always 2 (low and high side). */
  std::array<int, MAX_GRID_NDIM> sub_dims;
  for (int i = 0; i < ndim; i++) {
    sub_dims[i] = 2;
  }

#pragma omp parallel for schedule(static) num_threads(nthreads)
  for (int p = 0; p < parts->npart; p++) {

    /* Skip masked particles. */
    if (parts->part_is_masked(p)) {
      continue;
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

      /* Add this grid cell's contribution to the spectra. */
      for (int ilam = 0; ilam < nlam; ilam++) {

        /* Skip if this wavelength is masked. */
        if (grid_props->lam_is_masked(ilam)) {
          continue;
        }

        /* Get the spectra value at this index and wavelength. */
        const double spec_val = grid_props->get_spectra_at(grid_ind, ilam);

        /* Use fused multiply-add to accumulate with better precision.
         * Equivalent to: += spec_val * weight, but with a single rounding. */
        part_spectra[p * nlam + ilam] =
            std::fma(spec_val, weight, part_spectra[p * nlam + ilam]);
        spectra[ilam] = std::fma(spec_val, weight, spectra[ilam]);
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
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 * @param nthreads: The number of threads to use.
 */
void spectra_loop_cic(GridProps *grid_props, Particles *parts, double *spectra,
                      double *part_spectra, const int nthreads) {

  double start_time = tic();

  /* Call the correct function for the configuration/number of threads. */

#ifdef WITH_OPENMP

  /* If we have multiple threads and OpenMP we can parallelise. */
  if (nthreads > 1) {
    spectra_loop_cic_omp(grid_props, parts, spectra, part_spectra, nthreads);
  }
  /* Otherwise there's no point paying the OpenMP overhead. */
  else {
    spectra_loop_cic_serial(grid_props, parts, spectra, part_spectra);
  }

#else

  (void)nthreads;

  /* We don't have OpenMP, just call the serial version. */
  spectra_loop_cic_serial(grid_props, parts, spectra, part_spectra);

#endif
  toc("Cloud in Cell particle spectra loop", start_time);
}

/**
 * @brief This calculates particle spectra using a nearest grid point
 *        approach.
 *
 * This is the serial version of the function.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 * @param part_spectra: The per-particle output array.
 */
static void spectra_loop_ngp_serial(GridProps *grid_props, Particles *parts,
                                    double *spectra, double *part_spectra) {

  /* Unpack the grid properties. */
  const int nlam = grid_props->nlam;

  /* Loop over particles. */
  for (int p = 0; p < parts->npart; p++) {

    /* Skip masked particles. */
    if (parts->part_is_masked(p)) {
      continue;
    }

    /* Setup the index array. */
    std::array<int, MAX_GRID_NDIM> part_indices;

    /* Get the grid indices for the particle */
    get_part_inds_ngp(part_indices, grid_props, parts, p);

    /* Get the weight's index. */
    const int grid_ind = grid_props->ravel_grid_index(part_indices);

    /* Get the weight of this particle. */
    const double weight = parts->get_weight_at(p);

    /* Add this grid cell's contribution to the spectra */
    for (int ilam = 0; ilam < nlam; ilam++) {

      /* Skip if this wavelength is masked. */
      if (grid_props->lam_is_masked(ilam)) {
        continue;
      }

      /* Get the spectra value at this index and wavelength. */
      const double spec_val = grid_props->get_spectra_at(grid_ind, ilam);

      /* Use fused multiply-add to accumulate with better precision.
       * Equivalent to: += spec_val * weight, but with a single rounding. */
      part_spectra[p * nlam + ilam] =
          std::fma(spec_val, weight, part_spectra[p * nlam + ilam]);
      spectra[ilam] = std::fma(spec_val, weight, spectra[ilam]);
    }
  }
}

/**
 * @brief This calculates particle spectra using a nearest grid point approach.
 *
 * This is the parallel version of the function.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 * @param part_spectra: The per-particle output array.
 * @param nthreads: The number of threads to use.
 */
#ifdef WITH_OPENMP
static void spectra_loop_ngp_omp(GridProps *grid_props, Particles *parts,
                                 double *spectra, double *part_spectra,
                                 int nthreads) {

  /* Unpack the grid properties. */
  const int nlam = grid_props->nlam;

  /* Loop over particles. */
#pragma omp parallel for schedule(static) num_threads(nthreads) reduction(+:spectra[:nlam])
  for (int p = 0; p < parts->npart; p++) {

    /* Skip masked particles. */
    if (parts->part_is_masked(p)) {
      continue;
    }

    /* Setup the index array. */
    std::array<int, MAX_GRID_NDIM> part_indices;

    /* Get the grid indices for the particle */
    get_part_inds_ngp(part_indices, grid_props, parts, p);

    /* Get the weight's index. */
    const int grid_ind = grid_props->ravel_grid_index(part_indices);

    /* Get the weight of this particle. */
    const double weight = parts->get_weight_at(p);

    /* Add this grid cell's contribution to the spectra */
    for (int ilam = 0; ilam < nlam; ilam++) {

      /* Skip if this wavelength is masked. */
      if (grid_props->lam_is_masked(ilam)) {
        continue;
      }

      /* Get the spectra value at this index and wavelength. */
      const double spec_val = grid_props->get_spectra_at(grid_ind, ilam);

      /* Use fused multiply-add to accumulate with better precision.
       * Equivalent to: += spec_val * weight, but with a single rounding. */
      part_spectra[p * nlam + ilam] =
          std::fma(spec_val, weight, part_spectra[p * nlam + ilam]);
      spectra[ilam] = std::fma(spec_val, weight, spectra[ilam]);
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
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 * @param nthreads: The number of threads to use.
 */
void spectra_loop_ngp(GridProps *grid_props, Particles *parts, double *spectra,
                      double *part_spectra, const int nthreads) {

  double start_time = tic();

  /* Call the correct function for the configuration/number of threads. */

#ifdef WITH_OPENMP

  /* If we have multiple threads and OpenMP we can parallelise. */
  if (nthreads > 1) {
    spectra_loop_ngp_omp(grid_props, parts, spectra, part_spectra, nthreads);
  }
  /* Otherwise there's no point paying the OpenMP overhead. */
  else {
    spectra_loop_ngp_serial(grid_props, parts, spectra, part_spectra);
  }

#else

  (void)nthreads;

  /* We don't have OpenMP, just call the serial version. */
  spectra_loop_ngp_serial(grid_props, parts, spectra, part_spectra);

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
  GridProps *grid_props = new GridProps(np_grid_spectra, grid_tuple,
                                        /*np_lam*/ nullptr, np_lam_mask, nlam);
  RETURN_IF_PYERR();

  /* Create the object that holds the particle properties. */
  Particles *part_props = new Particles(np_part_mass, /*np_velocities*/ NULL,
                                        np_mask, part_tuple, npart);
  RETURN_IF_PYERR();

  double out_tic = tic();

  /* Define the output dimensions. */
  npy_intp np_int_dims[1] = {nlam};
  npy_intp np_part_dims[2] = {npart, nlam};

  /* Allocate the spectra. */
  PyArrayObject *np_spectra =
      (PyArrayObject *)PyArray_ZEROS(1, np_int_dims, NPY_DOUBLE, 0);
  double *spectra = static_cast<double *>(PyArray_DATA(np_spectra));
  PyArrayObject *np_part_spectra =
      (PyArrayObject *)PyArray_ZEROS(2, np_part_dims, NPY_DOUBLE, 0);
  double *part_spectra = static_cast<double *>(PyArray_DATA(np_part_spectra));

  toc("Setting up output arrays", out_tic);

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
  RETURN_IF_PYERR();

  /* Clean up memory! */
  delete part_props;
  delete grid_props;

  /* Construct the output tuple. */
  PyObject *out_tuple = Py_BuildValue("NN", np_part_spectra, np_spectra);

  toc("Computing particle and integrated lnus", start_time);

  return out_tuple;
}

/* Below is all the gubbins needed to make the module importable in Python. */
static PyMethodDef SedMethods[] = {
    {"compute_particle_seds", (PyCFunction)compute_particle_seds, METH_VARARGS,
     "Method for calculating particle intrinsic spectra."},
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
  if (numpy_import() < 0) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to import numpy.");
    return NULL;
  }
  return m;
};
