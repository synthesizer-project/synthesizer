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
#include "hashmap.h"
#include "macros.h"
#include "property_funcs.h"
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
  const int *dims = extract_data_int(np_ndims, "dims");
  if (dims == NULL) {
    return NULL;
  }

  /* Extract a pointer to the particle masses. */
  const double *part_mass = extract_data_double(np_part_mass, "part_mass");
  if (part_mass == NULL) {
    return NULL;
  }

  /* Extract the grid properties from the tuple of numpy arrays. */
  const double **grid_props = extract_grid_props(grid_tuple, ndim, dims);
  if (grid_props == NULL) {
    return NULL;
  }

  /* Extract the particle properties from the tuple of numpy arrays. */
  const double **part_props = extract_part_props(part_tuple, ndim, npart);
  if (part_props == NULL) {
    return NULL;
  }

  /* How many grid elements are there? (excluding wavelength axis)*/
  int grid_size = 1;
  for (int dim = 0; dim < ndim; dim++)
    grid_size *= dims[dim];

  /* With everything set up we can compute the weights for each particle using
   * the requested method. */
  HashMap *weights;
  if (strcmp(method, "cic") == 0) {
    weights = weight_loop_cic(grid_props, part_props, part_mass, dims, ndim,
                              npart, /*per_part*/ 0);
  } else if (strcmp(method, "ngp") == 0) {
    weights = weight_loop_ngp(grid_props, part_props, part_mass, dims, ndim,
                              npart, /*per_part*/ 0);
  } else {
    PyErr_SetString(PyExc_ValueError, "Unknown grid assignment method (%s).");
    return NULL;
  }

  /* Allocate an array to hold the grid weights. */
  double *sfzh = malloc(grid_size * sizeof(double));
  if (sfzh == NULL) {
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for sfzh.");
    return NULL;
  }
  bzero(sfzh, grid_size * sizeof(double));

  /* Populate the SFZH. */
  for (int i = 0; i < weights->size; i++) {
    /* Get the hash map node. */
    Node *node = weights->buckets[i];

    /* Traverse the node linked list. */
    while (node) {

      /* Get the weight and indices. */
      const double weight = node->value;
      const IndexKey key = node->key;
      const int flat_ind = get_flat_index(key.grid_indices, dims, ndim);

      /* Add the weight to the SFZH. */
      sfzh[flat_ind] += weight;

      /* Next... */
      node = node->next;
    }
  }

  /* Clean up memory! */
  free_hash_map(weights);
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
