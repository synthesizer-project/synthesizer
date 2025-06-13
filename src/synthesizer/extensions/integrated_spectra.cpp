/******************************************************************************
 * C extension to calculate integrated SEDs for a galaxy's star particles.
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
 * @brief Compute the integrated spectra from the grid weights.
 *
 * @param grid_props: The grid properties.
 *
 * @return The integrated spectra.
 */
static PyArrayObject *get_spectra_serial(GridProps *grid_props) {

  /* Define the output dimensions. */
  npy_intp np_int_dims[1] = {grid_props->nlam};

  /* Allocate the output spectra. */
  PyArrayObject *np_spectra =
      (PyArrayObject *)PyArray_ZEROS(1, np_int_dims, NPY_DOUBLE, 0);
  double *spectra = static_cast<double *>(PyArray_DATA(np_spectra));

  /* Loop over wavelengths */
  for (int ilam = 0; ilam < grid_props->nlam; ilam++) {

    /* Skip if this wavelength is masked. */
    if (grid_props->lam_is_masked(ilam)) {
      continue;
    }

    /* Loop over grid cells. */
    for (int grid_ind = 0; grid_ind < grid_props->size; grid_ind++) {

      /* Get the weight. */
      const double weight = grid_props->get_grid_weight_at(grid_ind);

      /* Skip zero weight cells. */
      if (weight <= 0)
        continue;

      /* Get the grid spectra value at this index and wavelength. */
      double spec_val = grid_props->get_spectra_at(grid_ind, ilam);

      /* Add the contribution to this wavelength. */
      spectra[ilam] += spec_val * weight;
    }
  }

  return np_spectra;
}

/**
 * @brief Compute the integrated spectra from the grid weights.
 *
 * @param grid_props: The grid properties.
 * @param nthreads: The number of threads to use.
 *
 * @return The integrated spectra.
 */
#ifdef WITH_OPENMP
static PyArrayObject *get_spectra_omp(GridProps *grid_props, int nthreads) {

  /* Define the output dimensions. */
  npy_intp np_int_dims[1] = {grid_props->nlam};

  /* Allocate the output spectra. */
  PyArrayObject *np_spectra =
      (PyArrayObject *)PyArray_ZEROS(1, np_int_dims, NPY_DOUBLE, 0);
  double *spectra = static_cast<double *>(PyArray_DATA(np_spectra));

#pragma omp parallel num_threads(nthreads)
  {
    /* Get the thread id. */
    int tid = omp_get_thread_num();

    /* We will give each thread a chunk of the spectra to work on. */

    /* How many wavelength elements should each thread get? */
    int nlam_per_thread = (grid_props->nlam + nthreads - 1) / nthreads;

    /* Calculate the start and end indices for this thread. */
    int start = tid * nlam_per_thread;
    int end = start + nlam_per_thread;
    if (end >= grid_props->nlam) {
      end = grid_props->nlam;
    }

    /* Loop over wavelengths. */
    for (int ilam = 0; ilam < end - start; ilam++) {

      /* Skip if this wavelength is masked. */
      if (grid_props->lam_is_masked(start + ilam)) {
        continue;
      }

      /* Temporary value to hold the the spectra for this wavelength. */
      double this_element = 0.0;

      /* Loop over grid cells. */
      for (int grid_ind = 0; grid_ind < grid_props->size; grid_ind++) {

        /* Get the weight. */
        const double weight = grid_props->get_grid_weight_at(grid_ind);

        /* Skip zero weight cells. */
        if (weight <= 0)
          continue;

        /* Get the grid spectra value at this index and wavelength. */
        double spec_val = grid_props->get_spectra_at(grid_ind, start + ilam);

        /* Add the contribution to this wavelength. */
        this_element += spec_val * weight;
      }

      spectra[start + ilam] = this_element;
    }
  }

  return np_spectra;
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
static PyArrayObject *get_spectra(GridProps *grid_props, int nthreads) {

  double reduction_start = tic();
#ifdef WITH_OPENMP
  /* Do we have multiple threads to do the reduction on to the spectra? */
  PyArrayObject *np_spectra;
  if (nthreads > 1) {
    np_spectra = get_spectra_omp(grid_props, nthreads);
  } else {
    np_spectra = get_spectra_serial(grid_props);
  }
#else
  /* We can't do the reduction in parallel without OpenMP. */
  PyArrayObject *np_spectra = get_spectra_serial(grid_props);
#endif

  toc("Compute integrated spectra from weights", reduction_start);

  return np_spectra;
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
 * @param np_ndims: The size of each grid axis.
 * @param ndim: The number of grid axes.
 * @param npart: The number of particles.
 * @param nlam: The number of wavelength elements.
 */
PyObject *compute_integrated_sed(PyObject *self, PyObject *args) {

  double start_time = tic();

  /* We don't need the self argument but it has to be there. Tell the compiler
   * we don't care. */
  (void)self;

  int ndim, npart, nlam, nthreads;
  PyObject *grid_tuple, *part_tuple;
  PyArrayObject *np_grid_spectra, *np_grid_weights;
  PyArrayObject *np_part_mass, *np_ndims;
  PyArrayObject *np_mask, *np_lam_mask;
  char *method;

  if (!PyArg_ParseTuple(args, "OOOOOiiisiOOO", &np_grid_spectra, &grid_tuple,
                        &part_tuple, &np_part_mass, &np_ndims, &ndim, &npart,
                        &nlam, &method, &nthreads, &np_grid_weights, &np_mask,
                        &np_lam_mask))
    return NULL;

  /* Extract the grid struct. */
  GridProps *grid_props = new GridProps(np_grid_spectra, grid_tuple,
                                        /*np_lam*/ NULL, np_lam_mask, nlam);
  RETURN_IF_PYERR();

  /* Create the object that holds the particle properties. */
  Particles *part_props = new Particles(np_part_mass, /*np_velocities*/ NULL,
                                        np_mask, part_tuple, npart);
  RETURN_IF_PYERR();

  /* Get existing grid weights or allocate new ones. */
  double *grid_weights = grid_props->get_grid_weights();
  RETURN_IF_PYERR();

  /* With everything set up we can compute the weights for each particle using
   * the requested method if we need to. */
  if (grid_props->need_grid_weights()) {
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
  }
  RETURN_IF_PYERR();

  /* Compute the integrated SED. */
  PyArrayObject *np_spectra = get_spectra(grid_props, nthreads);
  if (np_spectra == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "Could not compute integrated SED.");
    return NULL;
  }
  RETURN_IF_PYERR();

  /* Extract the output grid weights before we free the grid object. */
  np_grid_weights = grid_props->get_np_grid_weights();

  /* Clean up memory! */
  delete grid_props;
  delete part_props;

  toc("Compute integrated SED", start_time);

  return Py_BuildValue("NN", np_spectra, np_grid_weights);
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
  if (numpy_import() < 0) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to import numpy.");
    return NULL;
  }
  return m;
}
