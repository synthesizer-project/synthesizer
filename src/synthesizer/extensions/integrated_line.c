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
#include "weights.h"

/**
 * @brief Computes an integrated line emission for a collection of particles.
 *
 * @param np_grid_line: The SPS line emission array.
 * @param np_grid_continuum: The SPS continuum emission array.
 * @param grid_tuple: The tuple containing arrays of grid axis properties.
 * @param part_tuple: The tuple of particle property arrays (in the same order
 *                    as grid_tuple).
 * @param np_part_mass: The particle mass array.
 * @param fesc: The escape fraction.
 * @param np_ndims: The size of each grid axis.
 * @param ndim: The number of grid axes.
 * @param npart: The number of particles.
 * @param method: The method to use for assigning weights.
 */
PyObject *compute_integrated_line(PyObject *self, PyObject *args) {

  /* We don't need the self argument but it has to be there. Tell the compiler
   * we don't care. */
  (void)self;

  int ndim, npart;
  PyObject *grid_tuple, *part_tuple;
  PyArrayObject *np_grid_lines, *np_grid_continuum;
  PyArrayObject *np_fesc;
  PyArrayObject *np_part_mass, *np_ndims;
  char *method;

  if (!PyArg_ParseTuple(args, "OOOOOOOiis", &np_grid_lines, &np_grid_continuum,
                        &grid_tuple, &part_tuple, &np_part_mass, &np_fesc,
                        &np_ndims, &ndim, &npart, &method))
    return NULL;

  /* Quick check to make sure our inputs are valid. */
  if (ndim == 0) {
    PyErr_SetString(PyExc_ValueError, "ndim must be greater than 0.");
    return NULL;
  }
  if (npart == 0) {
    PyErr_SetString(PyExc_ValueError, "No particles to process!");
    return NULL;
  }

  /* Extract a pointer to the lines grids */
  const double *grid_lines = PyArray_DATA(np_grid_lines);
  if (grid_lines == NULL) {
    PyErr_SetString(PyExc_ValueError, "Failed to extract grid_lines.");
    return NULL;
  }

  /* Extract a pointer to the continuum grid. */
  const double *grid_continuum = PyArray_DATA(np_grid_continuum);
  if (grid_continuum == NULL) {
    PyErr_SetString(PyExc_ValueError, "Failed to extract grid_continuum.");
    return NULL;
  }

  /* Declare and initialise the vairbales we'll store our result in. */
  double line_lum = 0.0;
  double line_cont = 0.0;

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

  /* Extract a pointer to the fesc array. */
  const double *fesc = PyArray_DATA(np_fesc);
  if (fesc == NULL) {
    PyErr_SetString(PyExc_ValueError, "Failed to extract fesc from np_fesc.");
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

  /* Unpack the grid property arrays into a single contiguous array. */
  for (int idim = 0; idim < ndim; idim++) {

    /* Extract the data from the numpy array. */
    PyArrayObject *np_grid_arr =
        (PyArrayObject *)PyTuple_GetItem(grid_tuple, idim);
    if (np_grid_arr == NULL) {
      PyErr_SetString(PyExc_ValueError, "Failed to extract grid_arr.");
      return NULL;
    }
    const double *grid_arr = PyArray_DATA(np_grid_arr);
    if (grid_arr == NULL) {
      PyErr_SetString(PyExc_ValueError, "Failed to extract grid_arr.");
      return NULL;
    }

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
    if (np_part_arr == NULL) {
      PyErr_SetString(PyExc_ValueError, "Failed to extract part_arr.");
      return NULL;
    }
    const double *part_arr = PyArray_DATA(np_part_arr);

    /* Assign this data to the property array. */
    part_props[idim] = part_arr;
  }

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

  /* Populate the integrated line. */
  for (int i = 0; i < weights->size; i++) {
    /* Get the hash map node. */
    Node *node = weights->buckets[i];

    /* Traverse the node linked list. */
    while (node) {

      /* Get the weight and indices. */
      const double weight = node->value;
      const IndexKey key = node->key;
      const int p = key.particle_index;

      /* Get the grid index. */
      int grid_ind = get_flat_index(key.grid_indices, dims, ndim);

      /* Add this grid cell's contribution to the lines */
      line_lum += grid_lines[grid_ind] * weight * (1.0 - fesc[p]);
      line_cont += grid_continuum[grid_ind] * weight * (1.0 - fesc[p]);

      /* Next... */
      node = node->next;
    }
  }

  /* Clean up memory! */
  free_hash_map(weights);
  free(part_props);
  free(grid_props);

  // Create a Python tuple containing the two doubles
  PyObject *result_tuple = Py_BuildValue("dd", line_lum, line_cont);

  // Return the tuple
  return result_tuple;
}

/* Below is all the gubbins needed to make the module importable in Python. */
static PyMethodDef LineMethods[] = {
    {"compute_integrated_line", (PyCFunction)compute_integrated_line,
     METH_VARARGS, "Method for calculating integrated intrinsic lines."},
    {NULL, NULL, 0, NULL}};

/* Make this importable. */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "get_line",                               /* m_name */
    "A module to calculate integrated lines", /* m_doc */
    -1,                                       /* m_size */
    LineMethods,                              /* m_methods */
    NULL,                                     /* m_reload */
    NULL,                                     /* m_traverse */
    NULL,                                     /* m_clear */
    NULL,                                     /* m_free */
};

PyMODINIT_FUNC PyInit_integrated_line(void) {
  PyObject *m = PyModule_Create(&moduledef);
  import_array();
  return m;
}
