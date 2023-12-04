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
 * @param fesc: The escape fraction.
 * @param np_ndims: The size of each grid axis.
 * @param ndim: The number of grid axes.
 * @param npart: The number of particles.
 * @param nlam: The number of wavelength elements.
 */
PyObject *compute_sfzh(PyObject *self, PyObject *args) {

  const int ndim, npart;
  const PyObject *grid_tuple, *part_tuple;
  const PyArrayObject *np_part_mass, *np_ndims;
  const char *method;

  if (!PyArg_ParseTuple(args, "OOOOiis", &grid_tuple, &part_tuple,
                        &np_part_mass, &np_ndims, &ndim, &npart, &method))
    return NULL;

  /* Quick check to make sure our inputs are valid. */
  if (ndim == 0)
    return NULL;
  if (npart == 0)
    return NULL;

  /* Extract a pointer to the grid dims */
  const int *dims = PyArray_DATA(np_ndims);

  /* Extract a pointer to the particle masses. */
  const double *part_mass = PyArray_DATA(np_part_mass);

  /* Allocate a single array for grid properties*/
  int nprops = 0;
  for (int dim = 0; dim < ndim; dim++)
    nprops += dims[dim];
  const double **grid_props = malloc(nprops * sizeof(double *));

  /* How many grid elements are there? (excluding wavelength axis)*/
  int grid_size = 1;
  for (int dim = 0; dim < ndim; dim++)
    grid_size *= dims[dim];

  /* Allocate an array to hold the grid weights. */
  double *sfzh = malloc(grid_size * sizeof(double));
  bzero(sfzh, grid_size * sizeof(double));

  /* Unpack the grid property arrays into a single contiguous array. */
  for (int idim = 0; idim < ndim; idim++) {

    /* Extract the data from the numpy array. */
    const PyArrayObject *np_grid_arr = PyTuple_GetItem(grid_tuple, idim);
    const double *grid_arr = PyArray_DATA(np_grid_arr);

    /* Assign this data to the property array. */
    grid_props[idim] = grid_arr;
  }

  /* Allocate a single array for particle properties. */
  const double **part_props = malloc(npart * ndim * sizeof(double *));

  /* Unpack the particle property arrays into a single contiguous array. */
  for (int idim = 0; idim < ndim; idim++) {

    /* Extract the data from the numpy array. */
    const PyArrayObject *np_part_arr = PyTuple_GetItem(part_tuple, idim);
    const double *part_arr = PyArray_DATA(np_part_arr);

    /* Assign this data to the property array. */
    part_props[idim] = part_arr;
  }

  /* Loop over particles. */
  for (int p = 0; p < npart; p++) {

    /* Get this particle's mass. */
    const double mass = part_mass[p];

    /* Finally, compute the weights for this particle using the
     * requested method. */
    if (strcmp(method, "cic") == 0) {
      weight_loop_cic(grid_props, part_props, mass, sfzh, dims, ndim, p, 0);
    } else if (strcmp(method, "ngp") == 0) {
      weight_loop_ngp(grid_props, part_props, mass, sfzh, dims, ndim, p, 0);
    } else {
      /* Only print this warning once! */
      if (p == 0)
        printf(
            "Unrecognised gird assignment method (%s)! Falling back on CIC\n",
            method);
      weight_loop_cic(grid_props, part_props, mass, sfzh, dims, ndim, p, 0);
    }

  } /* Loop over particles. */

  /* Clean up memory! */
  free(part_props);
  free(grid_props);

  /* Reconstruct the python array to return. */
  npy_intp np_dims[ndim];
  for (int idim = 0; idim < ndim; idim++) {
    np_dims[idim] = dims[idim];
  }

  PyArrayObject *out_sfzh = (PyArrayObject *)PyArray_SimpleNewFromData(
      ndim, np_dims, NPY_FLOAT64, sfzh);

  return Py_BuildValue("N", out_sfzh);
}

/* Below is all the gubbins needed to make the module importable in Python. */
static PyMethodDef SFZHMethods[] = {{"compute_sfzh", compute_sfzh, METH_VARARGS,
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
