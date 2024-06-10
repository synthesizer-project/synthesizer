/******************************************************************************
 * A C module containing helper functions for extracting properties from the
 * numpy objects.
 *****************************************************************************/

/* Header */
#include "property_funcs.h"

/**
 * @brief Extract double data from a numpy array.
 *
 * @param np_arr: The numpy array to extract.
 * @param name: The name of the numpy array. (For error messages)
 */
const double *extract_data_double(PyArrayObject *np_arr, char *name) {

  /* Extract a pointer to the spectra grids */
  const double *data = PyArray_DATA(np_arr);
  if (data == NULL) {
    char error_msg[100];
    snprintf(error_msg, sizeof(error_msg), "Failed to extract %s.", name);
    PyErr_SetString(PyExc_ValueError, error_msg);
  }
  /* Success. */
  return data;
}

/**
 * @brief Extract int data from a numpy array.
 *
 * @param np_arr: The numpy array to extract.
 * @param name: The name of the numpy array. (For error messages)
 */
const int *extract_data_int(PyArrayObject *np_arr, char *name) {

  /* Extract a pointer to the spectra grids */
  const int *data = PyArray_DATA(np_arr);
  if (data == NULL) {
    char error_msg[100];
    snprintf(error_msg, sizeof(error_msg), "Failed to extract %s.", name);
    PyErr_SetString(PyExc_ValueError, error_msg);
  }
  /* Success. */
  return data;
}

/**
 * @brief Extract the grid properties from a tuple of numpy arrays.
 *
 * @param grid_tuple: A tuple of numpy arrays containing the grid properties.
 * @param ndim: The number of dimensions in the grid.
 * @param dims: The dimensions of the grid.
 */
const double **extract_grid_props(PyObject *grid_tuple, int ndim, int *dims) {

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

  /* Success. */
  return grid_props;
}

/**
 * @brief Extract the particle properties from a tuple of numpy arrays.
 *
 * @param part_tuple: A tuple of numpy arrays containing the particle
 * properties.
 * @param ndim: The number of dimensions in the grid.
 * @param npart: The number of particles.
 */
const double **extract_part_props(PyObject *part_tuple, int ndim, int npart) {

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
    for (int ipart = 0; ipart < npart; ipart++) {
      part_props[ipart * ndim + idim] = part_arr + ipart;
    }
  }

  /* Success. */
  return part_props;
}
