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
 * @param np_ndims: The size of each grid axis.
 * @param ndim: The number of grid axes.
 * @param npart: The number of particles.
 * @param nlam: The number of wavelength elements.
 */
PyObject *compute_sfzh(PyObject *self, PyObject *args) {

  /* We don't need the self argument but it has to be there. Tell the compiler
   * we don't care. */
  (void)self;

  int ndim, npart;
  PyObject *grid_tuple, *part_tuple;
  PyArrayObject *np_part_mass, *np_ndims;
  char *method;

  if (!PyArg_ParseTuple(args, "OOOOiis", &grid_tuple, &part_tuple,
                        &np_part_mass, &np_ndims, &ndim, &npart, &method))
    return NULL;

  /* Quick check to make sure our inputs are valid. */
  if (ndim == 0) {
    PyErr_SetString(PyExc_ValueError, "ndim must be greater than 0.");
    return NULL;
  }
  if (npart == 0) {
    PyErr_SetString(PyExc_ValueError, "npart must be greater than 0.");
    return NULL;
  }

  /* Extract a pointer to the grid dims */
  const int *dims = PyArray_DATA(np_ndims);
  if (dims == NULL) {
    PyErr_SetString(PyExc_ValueError, "Failed to extract dims from np_ndims.");
    return NULL;
  }

  /* Extract a pointer to the particle masses. */
  const double *part_mass = PyArray_DATA(np_part_mass);
  if (part_mass == NULL) {
    PyErr_SetString(PyExc_ValueError,
                    "Failed to extract part_mass from np_part_mass.");
    return NULL;
  }

  /* Allocate a single array for grid properties*/
  int nprops = 0;
  for (int dim = 0; dim < ndim; dim++)
    nprops += dims[dim];
  const double **grid_props = malloc(nprops * sizeof(double *));
  if (grid_props == NULL) {
    PyErr_SetString(PyExc_MemoryError,
                    "Failed to allocate memory for grid_props.");
    return NULL;
  }

  /* How many grid elements are there? (excluding wavelength axis)*/
  int grid_size = 1;
  for (int dim = 0; dim < ndim; dim++)
    grid_size *= dims[dim];

  /* Allocate an array to hold the grid weights. */
  double *sfzh = malloc(grid_size * sizeof(double));
  if (sfzh == NULL) {
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for sfzh.");
    return NULL;
  }
  bzero(sfzh, grid_size * sizeof(double));

  /* Unpack the grid property arrays into a single contiguous array. */
  for (int idim = 0; idim < ndim; idim++) {

    /* Extract the data from the numpy array. */
    PyArrayObject *np_grid_arr =
        (PyArrayObject *)PyTuple_GetItem(grid_tuple, idim);
    const double *grid_arr = PyArray_DATA(np_grid_arr);

    /* Assign this data to the property array. */
    grid_props[idim] = grid_arr;
  }

  /* Allocate a single array for particle properties. */
  const double **part_props = malloc(npart * ndim * sizeof(double *));
  if (part_props == NULL) {
    PyErr_SetString(PyExc_MemoryError,
                    "Failed to allocate memory for part_props.");
    return NULL;
  }

  /* Unpack the particle property arrays into a single contiguous array. */
  for (int idim = 0; idim < ndim; idim++) {

    /* Extract the data from the numpy array. */
    PyArrayObject *np_part_arr =
        (PyArrayObject *)PyTuple_GetItem(part_tuple, idim);
    const double *part_arr = PyArray_DATA(np_part_arr);

    /* Assign this data to the property array. */
    part_props[idim] = part_arr;
  }

  /* With everything set up we can compute the weights for each particle using
   * the requested method. */
  Weights *weights;
  if (strcmp(method, "cic") == 0) {
    weights =
        weight_loop_cic(grid_props, part_props, part_mass, dims, ndim, npart);
  } else if (strcmp(method, "ngp") == 0) {
    weights =
        weight_loop_ngp(grid_props, part_props, part_mass, dims, ndim, npart);
  } else {
    PyErr_SetString(PyExc_ValueError, "Unknown grid assignment method (%s).");
    return NULL;
  }

  /* Populate the SFZH. */
  for (int weight_ind = 0; weight_ind < weights->size; weight_ind++) {

    /* Get the weight. */
    const double weight = weights->values[weight_ind];

    /* Get the flattened index. */
    int flat_ind = get_flat_index(weights->indices[weight_ind], dims, ndim);

    /* Add the weight to the SFZH. */
    sfzh[flat_ind] += weight;
  }

  /* Clean up memory! */
  for (int i = 0; i < ndim; i++) {
    free(weights->indices[i]);
  }
  free(weights->axis_size);
  free(weights->indices);
  free(weights->values);
  free(weights);
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
