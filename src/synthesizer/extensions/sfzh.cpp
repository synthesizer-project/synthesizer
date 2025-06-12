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
 * @brief Computes an integrated SED for a collection of particles.
 *
 * @param np_grid_spectra: The SPS spectra array.
 * o
 * @param grid_tuple: The tuple containing arrays of grid axis properties.
 * @param part_tuple: The tuple of particle property arrays (in the same order
 *                    as grid_tuple).
 * @param np_part_mass: The particle mass array.
 * @param np_ndims: The size of each grid axis.
 * @param ndim: The number of grid axes.
 * @param npart: The number of particles.
 * @param nlam: The number of wavelength elements.
 */
PyObject *compute_sfzh(PyObject *self, PyObject *args) {

  double start_time = tic();

  /* We don't need the self argument but it has to be there. Tell the compiler
   * we don't care. */
  (void)self;

  int ndim, npart, nthreads;
  PyObject *grid_tuple, *part_tuple;
  PyArrayObject *np_part_mass, *np_ndims;
  PyArrayObject *np_mask;
  char *method;

  if (!PyArg_ParseTuple(args, "OOOOiisiO", &grid_tuple, &part_tuple,
                        &np_part_mass, &np_ndims, &ndim, &npart, &method,
                        &nthreads, &np_mask))
    return NULL;

  /* Extract the grid struct. */
  GridProps *grid_props =
      new GridProps(/*np_grid_spectra*/ nullptr, grid_tuple,
                    /*np_lam*/ nullptr, /*np_lam_mask*/ nullptr, 1);
  RETURN_IF_PYERR();

  Particles *parts = new Particles(np_part_mass, /*np_velocities*/ NULL,
                                   np_mask, part_tuple, npart);
  RETURN_IF_PYERR();

  /* Get the grid weights we'll work on. */
  double *sfzh = grid_props->get_grid_weights();
  RETURN_IF_PYERR();

  /* With everything set up we can compute the weights for each particle using
   * the requested method. */
  if (strcmp(method, "cic") == 0) {
    weight_loop_cic(grid_props, parts, grid_props->size, sfzh, nthreads);
  } else if (strcmp(method, "ngp") == 0) {
    weight_loop_ngp(grid_props, parts, grid_props->size, sfzh, nthreads);
  } else {
    PyErr_SetString(PyExc_ValueError, "Unknown grid assignment method (%s).");
    return NULL;
  }
  RETURN_IF_PYERR();

  /* Extract the grid weights we'll write out. */
  PyArrayObject *np_sfzh = grid_props->get_np_grid_weights();

  /* Clean up memory! */
  delete parts;
  delete grid_props;

  toc("Computing SFZH", start_time);

  return Py_BuildValue("N", np_sfzh);
}

/* Below is all the gubbins needed to make the module importable in Python. */
static PyMethodDef SFZHMethods[] = {{"compute_sfzh", (PyCFunction)compute_sfzh,
                                     METH_VARARGS,
                                     "Method for calculating the SFZH."},
                                    {NULL, NULL, 0, NULL}};

/* Make this importable. */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "make_sfzh",                             /* m_name */
    "A module to calculating particle SFZH", /* m_doc */
    -1,                                      /* m_size */
    SFZHMethods,                             /* m_methods */
    NULL,                                    /* m_reload */
    NULL,                                    /* m_traverse */
    NULL,                                    /* m_clear */
    NULL,                                    /* m_free */
};

PyMODINIT_FUNC PyInit_sfzh(void) {
  PyObject *m = PyModule_Create(&moduledef);
  if (numpy_import() < 0) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to import numpy.");
    return NULL;
  }
  return m;
}
