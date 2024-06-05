/******************************************************************************
 * C extension to calculate SEDs for star particles.
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
 * @brief Computes per particle line emission for a collection of particles.
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
PyObject *compute_particle_line(PyObject *self, PyObject *args) {

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
    /* Error message is already set here. */
    return NULL;

  /* Quick check to make sure our inputs are valid. */
  if (ndim == 0) {
    PyErr_SetString(
        PyExc_ValueError,
        "Grid appears to be dimensionless! Something awful has happened!");
    return NULL;
  }
  if (npart == 0) {
    PyErr_SetString(PyExc_ValueError, "No particles to process!");
    return NULL;
  }

  /* Extract a pointer to the line grids */
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

  /* Set up arrays to hold the line emission and continuum. */
  double *line_lum = malloc(npart * sizeof(double));
  if (line_lum == NULL) {
    PyErr_SetString(PyExc_MemoryError,
                    "Failed to allocate memory for line_lum.");
    return NULL;
  }
  bzero(line_lum, npart * sizeof(double));
  double *line_cont = malloc(npart * sizeof(double));
  if (line_cont == NULL) {
    PyErr_SetString(PyExc_MemoryError,
                    "Failed to allocate memory for line_cont.");
    return NULL;
  }
  bzero(line_cont, npart * sizeof(double));

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
    if (part_arr == NULL) {
      PyErr_SetString(PyExc_ValueError, "Failed to extract part_arr.");
      return NULL;
    }

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

  /* Loop over the weights summing the line. */
  for (int weight_ind = 0; weight_ind < weights->size; weight_ind++) {

    /* Get the particle index. */
    int p = weights->part_indices[weight_ind];

    /* Get the weight. */
    double weight = weights->values[weight_ind];

    /* Get the grid cell index. */
    int grid_ind = get_flat_index(weights->indices[weight_ind], dims, ndim);

    /* Add the contribution to this particle. */
    line_lum[p] += grid_lines[grid_ind] * (1 - fesc[p]) * weight;
    line_cont[p] += grid_continuum[grid_ind] * (1 - fesc[p]) * weight;
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
  npy_intp np_dims[1] = {
      npart,
  };
  PyArrayObject *out_line = (PyArrayObject *)PyArray_SimpleNewFromData(
      1, np_dims, NPY_FLOAT64, line_lum);
  PyArrayObject *out_cont = (PyArrayObject *)PyArray_SimpleNewFromData(
      1, np_dims, NPY_FLOAT64, line_cont);

  return Py_BuildValue("(OO)", out_line, out_cont);
}

/* Below is all the gubbins needed to make the module importable in Python. */
static PyMethodDef LineMethods[] = {
    {"compute_particle_line", (PyCFunction)compute_particle_line, METH_VARARGS,
     "Method for calculating particle intrinsic line emission."},
    {NULL, NULL, 0, NULL}};

/* Make this importable. */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "make_particle_line",                           /* m_name */
    "A module to calculate particle line emission", /* m_doc */
    -1,                                             /* m_size */
    LineMethods,                                    /* m_methods */
    NULL,                                           /* m_reload */
    NULL,                                           /* m_traverse */
    NULL,                                           /* m_clear */
    NULL,                                           /* m_free */
};

PyMODINIT_FUNC PyInit_particle_line(void) {
  PyObject *m = PyModule_Create(&moduledef);
  import_array();
  return m;
}
