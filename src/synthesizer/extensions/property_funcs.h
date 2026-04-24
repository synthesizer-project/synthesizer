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
 * @brief Get a double value at a specific index in a numpy array.
 *
 * This function assumes the numpy array is of type float64 and contiguous.
 * If the array is not of type float64, it will raise a TypeError.
 * If the index is out of bounds, it will raise an IndexError.
 *
 * @param np_arr: The numpy array to access.
 * @param ind: The index to access.
 * @param array_name: A descriptive name for the array, used in errors.
 * @return The double value at the specified index.
 */
static inline double get_double_at(PyArrayObject *np_arr, npy_intp ind,
                                   const char *array_name) {
  const char *name = array_name == NULL ? "array" : array_name;

  if (PyArray_TYPE(np_arr) != NPY_FLOAT64) {
    PyErr_Format(PyExc_TypeError,
                 "[get_double_at]: Array '%s' must be of type float64.",
                 name);
    return 0.0;
  }

  if (ind < 0 || ind >= PyArray_SIZE(np_arr)) {
    PyErr_Format(PyExc_IndexError,
                 "[get_double_at]: Index (%ld) out of bounds for array '%s'. "
                 "Valid range is [0, %ld).",
                 ind, name, PyArray_SIZE(np_arr));
    return 0.0;
  }

  if (PyArray_ISCONTIGUOUS(np_arr)) {
    const double *data_ptr = static_cast<const double *>(PyArray_DATA(np_arr));
    return data_ptr[ind];
  } else {
    PyErr_Format(PyExc_ValueError,
                 "[get_double_at]: Array '%s' must be contiguous to use "
                 "get_double_at.",
                 name);
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
 * @param array_name: A descriptive name for the array, used in errors.
 * @return The integer value at the specified index.
 */
static inline int get_int_at(PyArrayObject *np_arr, npy_intp ind,
                             const char *array_name) {
  const char *name = array_name == NULL ? "array" : array_name;

  if (PyArray_TYPE(np_arr) != NPY_INT32) {
    PyErr_Format(PyExc_TypeError,
                 "[get_int_at]: Array '%s' must be of type int32.", name);
    return 0;
  }

  if (ind < 0 || ind >= PyArray_SIZE(np_arr)) {
    PyErr_Format(PyExc_IndexError,
                 "[get_int_at]: Index (%ld) out of bounds for array '%s'. "
                 "Valid range is [0, %ld).",
                 ind, name, PyArray_SIZE(np_arr));
    return 0;
  }

  if (PyArray_ISCONTIGUOUS(np_arr)) {
    const int *data_ptr = static_cast<const int *>(PyArray_DATA(np_arr));
    return data_ptr[ind];
  } else {
    PyErr_Format(PyExc_ValueError,
                 "[get_int_at]: Array '%s' must be contiguous to use "
                 "get_int_at.",
                 name);
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
 * @param array_name: A descriptive name for the array, used in errors.
 * @return The boolean value at the specified index.
 */
static inline npy_bool get_bool_at(PyArrayObject *np_arr, npy_intp ind,
                                   const char *array_name) {
  const char *name = array_name == NULL ? "array" : array_name;

  if (PyArray_TYPE(np_arr) != NPY_BOOL) {
    PyErr_Format(PyExc_TypeError,
                 "[get_bool_at]: Array '%s' must be of type bool.", name);
    return false;
  }

  if (ind < 0 || ind >= PyArray_SIZE(np_arr)) {
    PyErr_Format(PyExc_IndexError,
                 "[get_bool_at]: Index (%ld) out of bounds for array '%s'. "
                 "Valid range is [0, %ld).",
                 ind, name, PyArray_SIZE(np_arr));
    return false;
  }

  if (PyArray_ISCONTIGUOUS(np_arr)) {
    const npy_bool *data_ptr =
        static_cast<const npy_bool *>(PyArray_DATA(np_arr));
    return data_ptr[ind];
  } else {
    PyErr_Format(PyExc_ValueError,
                 "[get_bool_at]: Array '%s' must be contiguous to use "
                 "get_bool_at.",
                 name);
    return false;
  }
}

/* Prototypes */
double *extract_data_double(PyArrayObject *np_arr, const char *name);
int *extract_data_int(PyArrayObject *np_arr, const char *name);
npy_bool *extract_data_bool(PyArrayObject *np_arr, const char *name);
const npy_int64 *extract_index_array(PyArrayObject *np_arr, const char *name);
double **extract_grid_props(PyObject *grid_tuple, int ndim, int *dims);

#endif // PROPERTY_FUNCS_H_
