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
#include "data_types.h"

/**
 * @brief Extract double data from a numpy array.
 *
 * @param np_arr: The numpy array to extract.
 * @param name: The name of the numpy array. (For error messages)
 */
double *extract_data_double(PyArrayObject *np_arr, const char *name) {

  /* Check that the array is of the correct type. */
  if (PyArray_TYPE(np_arr) != NPY_FLOAT64) {
    char error_msg[256];
    snprintf(error_msg, sizeof(error_msg),
             "[extract_data_double]: Array '%s' must be of type float64.", name);
    PyErr_SetString(PyExc_TypeError, error_msg);
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
 * @brief Extract Float data from a numpy array.
 *
 * This extracts data matching the compiled precision (float32 or float64).
 *
 * @param np_arr: The numpy array to extract.
 * @param name: The name of the numpy array. (For error messages)
 */
Float *extract_data_float(PyArrayObject *np_arr, const char *name) {

  /* Check that the array is of the correct type. */
  if (PyArray_TYPE(np_arr) != NPY_FLOAT_T) {
    char error_msg[256];
    snprintf(error_msg, sizeof(error_msg),
             "[extract_data_float]: Array '%s' must be of type %s.", name,
             FLOAT_NAME);
    PyErr_SetString(PyExc_TypeError, error_msg);
    return NULL;
  }

  /* Extract a pointer to the data */
  Float *data = reinterpret_cast<Float *>(PyArray_DATA(np_arr));
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

  /* Check that the array is of the correct type. */
  if (PyArray_TYPE(np_arr) != NPY_INT32) {
    char error_msg[256];
    snprintf(error_msg, sizeof(error_msg),
             "[extract_data_int]: Array '%s' must be of type int32.", name);
    PyErr_SetString(PyExc_TypeError, error_msg);
    return NULL;
  }

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

  /* Check that the array is of the correct type. */
  if (PyArray_TYPE(np_arr) != NPY_BOOL) {
    char error_msg[256];
    snprintf(error_msg, sizeof(error_msg),
             "[extract_data_bool]: Array '%s' must be of type bool.", name);
    PyErr_SetString(PyExc_TypeError, error_msg);
    return NULL;
  }

  npy_bool *data = reinterpret_cast<npy_bool *>(PyArray_DATA(np_arr));
  if (data == NULL) {
    char error_msg[100];
    snprintf(error_msg, sizeof(error_msg), "Failed to extract %s.", name);
    PyErr_SetString(PyExc_ValueError, error_msg);
    return NULL;
  }
  return data;
}
