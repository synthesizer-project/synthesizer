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
  double setup_start = tic();

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
  struct grid *grid_props =
      get_spectra_grid_struct(grid_tuple, np_ndims, /*np_grid_spectra*/ NULL,
                              /*np_lam*/ NULL, NULL, ndim, /*nlam*/ 1);
  if (grid_props == NULL) {
    return NULL;
  }

  /* Extract the particle struct. */
  struct particles *part_props = get_part_struct(
      part_tuple, np_part_mass, /*np_velocities*/ NULL, np_mask, npart, ndim);
  if (part_props == NULL) {
    return NULL;
  }

  /* Allocate the sfzh array to output. */
  double *sfzh = calloc(grid_props->size, sizeof(double));
  if (sfzh == NULL) {
    PyErr_SetString(PyExc_MemoryError, "Could not allocate memory for sfzh.");
    return NULL;
  }

  toc("Extracting Python data", setup_start);

  /* With everything set up we can compute the weights for each particle using
   * the requested method. */
  if (strcmp(method, "cic") == 0) {
    weight_loop_cic(grid_props, part_props, grid_props->size, sfzh, nthreads);
  } else if (strcmp(method, "ngp") == 0) {
    weight_loop_ngp(grid_props, part_props, grid_props->size, sfzh, nthreads);
  } else {
    PyErr_SetString(PyExc_ValueError, "Unknown grid assignment method (%s).");
    return NULL;
  }

  /* Check we got the output. (Any error messages will already be set) */
  if (sfzh == NULL) {
    return NULL;
  }

  /* Reconstruct the python array to return. */
  npy_intp np_dims[ndim];
  for (int idim = 0; idim < ndim; idim++) {
    np_dims[idim] = grid_props->dims[idim];
  }

  PyArrayObject *out_sfzh = c_array_to_numpy(ndim, np_dims, NPY_FLOAT64, sfzh);

  /* Clean up memory! */
  free(part_props);
  free(grid_props);

  toc("Computing SFZH", start_time);

  return Py_BuildValue("N", out_sfzh);
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
  import_array();
  return m;
}
