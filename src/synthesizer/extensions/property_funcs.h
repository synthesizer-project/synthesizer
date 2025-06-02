/******************************************************************************
 * A C module containing helper functions for extracting properties from the
 * numpy objects.
 *****************************************************************************/
#ifndef PROPERTY_FUNCS_H_
#define PROPERTY_FUNCS_H_

/* Standard includes */
#include <stdlib.h>

/* Python includes */
#define PY_ARRAY_UNIQUE_SYMBOL SYNTHESIZER_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy_init.h"
#include <Python.h>

/**
 * @brief Allocate an array.
 *
 * Just a wrapper around malloc with a check for NULL.
 *
 * @param n: The number of pointers to allocate.
 */
template <typename T> T *synth_malloc(size_t n, const char *msg) {
  T *ptr = reinterpret_cast<T *>(malloc(n));
  if (ptr == NULL) {
    char error_msg[100];
    snprintf(error_msg, sizeof(error_msg), "Failed to allocate memory for %s.",
             msg);
    PyErr_SetString(PyExc_MemoryError, error_msg);
  }
  bzero(ptr, n);
  return ptr;
}

/**
 * @brief Get a double value at a specific index in a numpy array.
 *
 * This function assumes the numpy array is of type float64 and contiguous.
 * If the array is not of type float64, it will raise a TypeError.
 * If the index is out of bounds, it will raise an IndexError.
 *
 * @param np_arr: The numpy array to access.
 * @param ind: The index to access.
 * @return The double value at the specified index.
 */
static inline double get_double_at(PyArrayObject *np_arr, npy_intp ind) {
  if (PyArray_TYPE(np_arr) != NPY_FLOAT64) {
    PyErr_SetString(PyExc_TypeError, "Array must be of type float64.");
    return 0.0;
  }

  if (ind < 0 || ind >= PyArray_SIZE(np_arr)) {
    PyErr_SetString(PyExc_IndexError, "Index out of bounds.");
    return 0.0;
  }

  if (PyArray_ISCONTIGUOUS(np_arr)) {
    const double *data_ptr = static_cast<const double *>(PyArray_DATA(np_arr));
    return data_ptr[ind];
  } else {
    PyErr_SetString(PyExc_ValueError,
                    "Array must be contiguous to use get_double_at.");
    return 0.0;
  }
}

/**
 * @brief Get an integer value at a specific index in a numpy array.
 *
 * This function assumes the numpy array is of type int32 and contiguous.
 * If the array is not of type int32, it will raise a TypeError.
 * If the index is out of bounds, it will raise an IndexError.
 *
 * @param np_arr: The numpy array to access.
 * @param ind: The index to access.
 * @return The integer value at the specified index.
 */
static inline int get_int_at(PyArrayObject *np_arr, npy_intp ind) {
  if (PyArray_TYPE(np_arr) != NPY_INT32) {
    PyErr_SetString(PyExc_TypeError, "Array must be of type int32.");
    return 0;
  }

  if (ind < 0 || ind >= PyArray_SIZE(np_arr)) {
    PyErr_SetString(PyExc_IndexError, "Index out of bounds.");
    return 0;
  }

  if (PyArray_ISCONTIGUOUS(np_arr)) {
    const int *data_ptr = static_cast<const int *>(PyArray_DATA(np_arr));
    return data_ptr[ind];
  } else {
    PyErr_SetString(PyExc_ValueError,
                    "Array must be contiguous to use get_int_at.");
    return 0;
  }
}

/**
 * @brief Get a boolean value at a specific index in a numpy array.
 *
 * This function assumes the numpy array is of type bool and contiguous.
 * If the array is not of type bool, it will raise a TypeError.
 * If the index is out of bounds, it will raise an IndexError.
 *
 * @param np_arr: The numpy array to access.
 * @param ind: The index to access.
 * @return The boolean value at the specified index.
 */
static inline npy_bool get_bool_at(PyArrayObject *np_arr, npy_intp ind) {
  if (PyArray_TYPE(np_arr) != NPY_BOOL) {
    PyErr_SetString(PyExc_TypeError, "Array must be of type bool.");
    return false;
  }

  if (ind < 0 || ind >= PyArray_SIZE(np_arr)) {
    PyErr_SetString(PyExc_IndexError, "Index out of bounds.");
    return false;
  }

  if (PyArray_ISCONTIGUOUS(np_arr)) {
    const npy_bool *data_ptr =
        static_cast<const npy_bool *>(PyArray_DATA(np_arr));
    return data_ptr[ind];
  } else {
    PyErr_SetString(PyExc_ValueError,
                    "Array must be contiguous to use get_bool_at.");
    return false;
  }
}

/* Prototypes */
double *extract_data_double(PyArrayObject *np_arr, const char *name);
int *extract_data_int(PyArrayObject *np_arr, const char *name);
npy_bool *extract_data_bool(PyArrayObject *np_arr, const char *name);
double **extract_grid_props(PyObject *grid_tuple, int ndim, int *dims);

#endif // PROPERTY_FUNCS_H_
