/******************************************************************************
 * C extension to calculate integrated SEDs for a galaxy's star particles.
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
 * @brief Compute the integrated spectra from the grid weights.
 *
 * @param grid_props: The grid properties.
 * @param grid_weights: The grid weights computed from the particles.
 *
 * @return The integrated spectra.
 */
static double *get_spectra_serial(struct grid *grid_props,
                                  double *grid_weights) {

  /* Set up array to hold the SED itself. */
  double *spectra = (double *)malloc(grid_props->nlam * sizeof(double));
  if (spectra == NULL) {
    PyErr_SetString(PyExc_ValueError, "Failed to allocate memory for spectra.");
    return NULL;
  }
  bzero(spectra, grid_props->nlam * sizeof(double));

  /* Loop over wavelengths */
  for (int ilam = 0; ilam < grid_props->nlam; ilam++) {

    /* Loop over grid cells. */
    for (int grid_ind = 0; grid_ind < grid_props->size; grid_ind++) {

      /* Get the weight. */
      const double weight = grid_weights[grid_ind];

      /* Skip zero weight cells. */
      if (weight <= 0)
        continue;

      /* Get the spectra ind. */
      int unraveled_ind[grid_props->ndim + 1];
      get_indices_from_flat(grid_ind, grid_props->ndim, grid_props->dims,
                            unraveled_ind);
      unraveled_ind[grid_props->ndim] = 0;
      int spectra_ind =
          get_flat_index(unraveled_ind, grid_props->dims, grid_props->ndim + 1);

      /* Add the contribution to this wavelength. */
      /* fesc is already included in the weight */
      spectra[ilam] += grid_props->spectra[spectra_ind + ilam] * weight;
    }
  }

  return spectra;
}

/**
 * @brief Compute the integrated spectra from the grid weights.
 *
 * @param grid_props: The grid properties.
 * @param grid_weights: The grid weights computed from the particles.
 * @param nthreads: The number of threads to use.
 *
 * @return The integrated spectra.
 */
#ifdef WITH_OPENMP
static double *get_spectra_omp(struct grid *grid_props, double *grid_weights,
                               int nthreads) {

  /* Set up array to hold the SED itself. */
  double *spectra = calloc(grid_props->nlam, sizeof(double));
  if (spectra == NULL) {
    PyErr_SetString(PyExc_ValueError, "Failed to allocate memory for spectra.");
    return NULL;
  }

  /* Allocate a continuous chunk of data for each thread to use. */
  double *out_data = calloc(nthreads * grid_props->nlam, sizeof(double));
  if (out_data == NULL) {
    PyErr_SetString(PyExc_ValueError,
                    "Failed to allocate memory for thread spectra.");
    return NULL;
  }

  /* Allocate thread spectra. */
  double **thread_spectra = malloc(nthreads * sizeof(double *));
  if (thread_spectra == NULL) {
    PyErr_SetString(PyExc_ValueError,
                    "Failed to allocate memory for thread spectra.");
    return NULL;
  }
  for (int t = 0; t < nthreads; t++) {
    thread_spectra[t] = out_data + t * grid_props->nlam;
  }

#pragma omp parallel num_threads(nthreads)
  {
    /* Get the thread id. */
    int tid = omp_get_thread_num();

    /* Get a local pointer to this threads spectra. */
    double *local_spectra = thread_spectra[tid];

#pragma omp for
    /* Loop over wavelengths. */
    for (int ilam = 0; ilam < grid_props->nlam; ilam++) {

      /* Loop over grid cells. */
      for (int grid_ind = 0; grid_ind < grid_props->size; grid_ind++) {

        /* Get the weight. */
        const double weight = grid_weights[grid_ind];

        /* Skip zero weight cells. */
        if (weight <= 0)
          continue;

        /* Get the spectra ind. */
        int unraveled_ind[grid_props->ndim + 1];
        get_indices_from_flat(grid_ind, grid_props->ndim, grid_props->dims,
                              unraveled_ind);
        unraveled_ind[grid_props->ndim] = 0;
        int spectra_ind = get_flat_index(unraveled_ind, grid_props->dims,
                                         grid_props->ndim + 1);

        /* Add the contribution to this wavelength. */
        /* fesc is already included in the weight */
        local_spectra[ilam] += grid_props->spectra[spectra_ind + ilam] * weight;
      }
    }
  }

  /* Hierarchical reduction */
  for (int step = 1; step < nthreads; step *= 2) {
    for (int tid = 0; tid < nthreads; tid += 2 * step) {
      if (tid + step < nthreads) {
#pragma omp parallel for num_threads(nthreads)
        for (int i = 0; i < grid_props->nlam; i++) {
          thread_spectra[tid][i] += thread_spectra[tid + step][i];
        }
      }
    }
  }

  /* Copy the final reduced result to spectra */
  memcpy(spectra, thread_spectra[0], grid_props->nlam * sizeof(double));

  /* Clean up. */
  free(out_data);
  free(thread_spectra);

  return spectra;
}
#endif

/**
 * @brief Compute the integrated spectra from the grid weights.
 *
 * @param grid_props: The grid properties.
 * @param grid_weights: The grid weights computed from the particles.
 * @param nthreads: The number of threads to use.
 *
 * @return The integrated spectra.
 */
static double *get_spectra(struct grid *grid_props, double *grid_weights,
                           int nthreads) {

  double reduction_start = tic();
#ifdef WITH_OPENMP
  /* Do we have multiple threads to do the reduction on to the spectra? */
  double *spectra;
  if (nthreads > 1) {
    spectra = get_spectra_omp(grid_props, grid_weights, nthreads);
  } else {
    spectra = get_spectra_serial(grid_props, grid_weights);
  }
#else
  /* We can't do the reduction in parallel without OpenMP. */
  double *spectra = get_spectra_serial(grid_props, grid_weights);
#endif

  toc("Compute integrated spectra from weights", reduction_start);

  return spectra;
}

/**
 * @brief Computes an integrated SED for a collection of particles.
 *
 * @param np_grid_spectra: The SPS spectra array.
 *o
 * @param grid_tuple: The tuple containing arrays of grid axis properties.
 * @param part_tuple: The tuple of particle property arrays (in the same order
 *                    as grid_tuple).
 * @param np_part_mass: The particle mass array.
 * @param fesc: The escape fraction.
 * @param np_ndims: The size of each grid axis.
 * @param ndim: The number of grid axes.
 * @param npart: The number of particles.
 * @param nlam: The number of wavelength elements.
 */
PyObject *compute_integrated_sed(PyObject *self, PyObject *args) {

  double start_time = tic();
  double setup_start = tic();

  /* We don't need the self argument but it has to be there. Tell the compiler
   * we don't care. */
  (void)self;

  /* Note that many of the following arguements are specific to the particle
   * spectra, but since we have a single argument preparation function we need
   * to have all arguments present, regardless of whether they are used. */
  int ndim, npart, nlam, nthreads;
  PyObject *grid_tuple, *part_tuple;
  PyObject *py_vel_shift;
  PyObject *py_c;
  PyArrayObject *np_grid_spectra, *np_lam;
  PyArrayObject *np_fesc;
  PyArrayObject *np_velocities;
  PyArrayObject *np_part_mass, *np_ndims;
  char *method;

  if (!PyArg_ParseTuple(args, "OOOOOOOOiiisiOO", &np_grid_spectra, &np_lam,
                        &grid_tuple, &part_tuple, &np_part_mass, &np_fesc,
                        &np_velocities, &np_ndims, &ndim, &npart, &nlam,
                        &method, &nthreads, &py_vel_shift, &py_c)) {
    return NULL;
  }

  /* Extract the grid struct. */
  struct grid *grid_props = get_spectra_grid_struct(
      grid_tuple, np_ndims, np_grid_spectra, /*np_lam*/ NULL, ndim, nlam);
  if (grid_props == NULL) {
    return NULL;
  }

  /* Extract the particle struct. */
  struct particles *part_props = get_part_struct(
      part_tuple, np_part_mass, /*np_velocities*/ NULL, np_fesc, npart, ndim);
  if (part_props == NULL) {
    return NULL;
  }

  /* Allocate the grid weights. */
  double *grid_weights = calloc(grid_props->size, sizeof(double));
  if (grid_weights == NULL) {
    PyErr_SetString(PyExc_MemoryError,
                    "Could not allocate memory for grid weights.");
    return NULL;
  }

  toc("Extracting Python data", setup_start);

  /* With everything set up we can compute the weights for each particle using
   * the requested method. */
  if (strcmp(method, "cic") == 0) {
    weight_loop_cic(grid_props, part_props, grid_props->size, grid_weights,
                    nthreads);
  } else if (strcmp(method, "ngp") == 0) {
    weight_loop_ngp(grid_props, part_props, grid_props->size, grid_weights,
                    nthreads);
  } else {
    PyErr_SetString(PyExc_ValueError, "Unknown grid assignment method (%s).");
    return NULL;
  }

  /* Check we got the weights sucessfully. (Any error messages will already be
   * set) */
  if (grid_weights == NULL) {
    return NULL;
  }

  /* Compute the integrated SED. */
  double *spectra = get_spectra(grid_props, grid_weights, nthreads);
  if (spectra == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "Could not compute integrated SED.");
    return NULL;
  }

  /* Clean up memory! */
  free(grid_weights);
  free(part_props);
  free(grid_props);

  /* Reconstruct the python array to return. */
  npy_intp np_dims[1] = {
      nlam,
  };
  PyArrayObject *out_spectra = (PyArrayObject *)PyArray_SimpleNewFromData(
      1, np_dims, NPY_FLOAT64, spectra);

  toc("Compute integrated SED", start_time);

  return Py_BuildValue("N", out_spectra);
}

static PyMethodDef SedMethods[] = {
    {"compute_integrated_sed", (PyCFunction)compute_integrated_sed,
     METH_VARARGS, "Method for calculating integrated intrinsic spectra."},
    {NULL, NULL, 0, NULL}};

/* Make this importable. */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "make_sed",                              /* m_name */
    "A module to calculate integrated seds", /* m_doc */
    -1,                                      /* m_size */
    SedMethods,                              /* m_methods */
    NULL,                                    /* m_reload */
    NULL,                                    /* m_traverse */
    NULL,                                    /* m_clear */
    NULL,                                    /* m_free */
};

PyMODINIT_FUNC PyInit_integrated_spectra(void) {
  PyObject *m = PyModule_Create(&moduledef);
  import_array();
  return m;
}
