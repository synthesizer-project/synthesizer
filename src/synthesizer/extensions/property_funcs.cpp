/******************************************************************************
 * A C module containing helper functions for extracting properties from the
 * numpy objects.
 *****************************************************************************/

/* C headers. */
#include <Python.h>
#include <iostream>
#include <string.h>

/* Header */
#include "property_funcs.h"

/**
 * @brief Extract double data from a numpy array.
 *
 * @param np_arr: The numpy array to extract.
 * @param name: The name of the numpy array. (For error messages)
 */
double *extract_data_double(PyArrayObject *np_arr, const char *name) {

  if (np_arr == NULL) {
    char error_msg[100];
    snprintf(error_msg, sizeof(error_msg), "Missing array for %s.", name);
    PyErr_SetString(PyExc_ValueError, error_msg);
    return NULL;
  }

  if (PyArray_TYPE(np_arr) != NPY_DOUBLE) {
    char error_msg[120];
    snprintf(error_msg, sizeof(error_msg),
             "%s must be a float64 (NPY_DOUBLE) array.", name);
    PyErr_SetString(PyExc_TypeError, error_msg);
    return NULL;
  }

  if (!PyArray_IS_C_CONTIGUOUS(np_arr)) {
    char error_msg[120];
    snprintf(error_msg, sizeof(error_msg),
             "%s must be C-contiguous.", name);
    PyErr_SetString(PyExc_ValueError, error_msg);
    return NULL;
  }

  /* Extract a pointer to the spectra grids */
  double *data = reinterpret_cast<double *>(PyArray_DATA(np_arr));
  if (data == NULL) {
    char error_msg[100];
    snprintf(error_msg, sizeof(error_msg), "Failed to extract %s.", name);
    PyErr_SetString(PyExc_ValueError, error_msg);
    return NULL;
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
int *extract_data_int(PyArrayObject *np_arr, const char *name) {

  /* Extract a pointer to the spectra grids */
  int *data = reinterpret_cast<int *>(PyArray_DATA(np_arr));
  if (data == NULL) {
    char error_msg[100];
    snprintf(error_msg, sizeof(error_msg), "Failed to extract %s.", name);
    PyErr_SetString(PyExc_ValueError, error_msg);
    return NULL;
  }
  /* Success. */
  return data;
}

/**
 * @brief Extract an int64-compatible 1D index array.
 *
 * Accepts NPY_INT64 or NPY_INTP (when sizes match). The array must be
 * C-contiguous.
 *
 * @param np_arr: The numpy array to extract.
 * @param name: The name of the numpy array (for error messages).
 * @return Pointer to the npy_int64 data, or NULL on error.
 */
const npy_int64 *extract_index_array(PyArrayObject *np_arr, const char *name) {
  if (np_arr == NULL) {
    PyErr_Format(PyExc_ValueError, "%s array is NULL.", name);
    return NULL;
  }

  if (PyArray_NDIM(np_arr) != 1) {
    PyErr_Format(PyExc_ValueError, "%s must be a 1D array.", name);
    return NULL;
  }

  if (!PyArray_IS_C_CONTIGUOUS(np_arr)) {
    PyErr_Format(PyExc_ValueError, "%s must be C-contiguous.", name);
    return NULL;
  }

  const int dtype = PyArray_TYPE(np_arr);
  if (dtype == NPY_INT64) {
    return (npy_int64 *)PyArray_DATA(np_arr);
  }

  if (dtype == NPY_INTP) {
    if (sizeof(npy_intp) != sizeof(npy_int64)) {
      PyErr_Format(PyExc_TypeError,
                   "%s has incompatible intp size for int64 use.", name);
      return NULL;
    }
    return (npy_int64 *)PyArray_DATA(np_arr);
  }

  PyErr_Format(PyExc_TypeError, "%s must be int64 or intp.", name);
  return NULL;
}

/**
 * @brief Extract boolean data from a numpy array.
 *
 * This function returns a pointer to the underlying boolean data stored
 * as npy_bool values (typically unsigned char).
 *
 * @param np_arr: The numpy array to extract.
 * @param name: The name of the numpy array (for error messages).
 * @return Pointer to the npy_bool data, or NULL on error.
 */
npy_bool *extract_data_bool(PyArrayObject *np_arr, const char *name) {
  npy_bool *data = reinterpret_cast<npy_bool *>(PyArray_DATA(np_arr));
  if (data == NULL) {
    char error_msg[100];
    snprintf(error_msg, sizeof(error_msg), "Failed to extract %s.", name);
    PyErr_SetString(PyExc_ValueError, error_msg);
    return NULL;
  }
  return data;
}
