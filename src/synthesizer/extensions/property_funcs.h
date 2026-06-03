/******************************************************************************
 * A C module containing helper functions for extracting properties from the
 * numpy objects.
 *****************************************************************************/
#ifndef PROPERTY_FUNCS_H_
#define PROPERTY_FUNCS_H_

/* Standard includes */
#include <stdlib.h>
#include <type_traits>

/* Python includes */
#define PY_ARRAY_UNIQUE_SYMBOL SYNTHESIZER_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy_init.h"
#include <Python.h>

/**
 * @brief Extract typed floating-point data from a numpy array.
 *
 * Accepts both float32 and float64 arrays and returns a typed pointer
 * matching the requested Real type.
 *
 * @tparam Real The floating-point type (float or double).
 * @param np_arr The numpy array to extract.
 * @param name The name of the numpy array (for error messages).
 * @return Pointer to the data, or NULL on error (Python exception set).
 */
template <typename Real>
static inline Real *extract_data(PyArrayObject *np_arr, const char *name) {
  static_assert(std::is_same_v<Real, float> || std::is_same_v<Real, double>,
                "Real must be float or double");

  if (np_arr == NULL) {
    char error_msg[100];
    snprintf(error_msg, sizeof(error_msg), "Missing array for %s.", name);
    PyErr_SetString(PyExc_ValueError, error_msg);
    return NULL;
  }

  const int expected_type =
      std::is_same_v<Real, float> ? NPY_FLOAT32 : NPY_FLOAT64;
  const char *type_name =
      std::is_same_v<Real, float> ? "float32" : "float64";

  if (PyArray_TYPE(np_arr) != expected_type) {
    char error_msg[120];
    snprintf(error_msg, sizeof(error_msg),
             "%s must be a %s array.", name, type_name);
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

  Real *data = reinterpret_cast<Real *>(PyArray_DATA(np_arr));
  if (data == NULL) {
    char error_msg[100];
    snprintf(error_msg, sizeof(error_msg), "Failed to extract %s.", name);
    PyErr_SetString(PyExc_ValueError, error_msg);
    return NULL;
  }
  return data;
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

/**
 * @brief Get a typed value at a specific index in a numpy array.
 *
 * Accepts float32 or float64 arrays and returns the value at the given
 * index as the requested type.
 *
 * @tparam T The floating-point type (float or double).
 * @param np_arr The numpy array to access.
 * @param ind The index to access.
 * @param array_name A descriptive name for the array, used in errors.
 * @return The value at the specified index, or T(0) on error.
 */
template <typename T>
static inline T get_at(PyArrayObject *np_arr, npy_intp ind,
                       const char *array_name) {
  const char *name = array_name == NULL ? "array" : array_name;

  int expected_type;
  const char *type_name;
  if (std::is_same_v<T, float>) {
    expected_type = NPY_FLOAT32;
    type_name = "float32";
  } else if (std::is_same_v<T, double>) {
    expected_type = NPY_FLOAT64;
    type_name = "float64";
  } else {
    PyErr_Format(PyExc_TypeError,
                 "[get_at]: Unsupported type for array '%s'.", name);
    return T(0);
  }

  if (PyArray_TYPE(np_arr) != expected_type) {
    PyErr_Format(PyExc_TypeError,
                 "[get_at]: Array '%s' must be of type %s.", name, type_name);
    return T(0);
  }

  if (ind < 0 || ind >= PyArray_SIZE(np_arr)) {
    PyErr_Format(PyExc_IndexError,
                 "[get_at]: Index (%ld) out of bounds for array '%s'. "
                 "Valid range is [0, %ld).",
                 ind, name, PyArray_SIZE(np_arr));
    return T(0);
  }

  if (PyArray_ISCONTIGUOUS(np_arr)) {
    const T *data_ptr = static_cast<const T *>(PyArray_DATA(np_arr));
    return data_ptr[ind];
  } else {
    PyErr_Format(PyExc_ValueError,
                 "[get_at]: Array '%s' must be contiguous.", name);
    return T(0);
  }
}

/* Prototypes */
int *extract_data_int(PyArrayObject *np_arr, const char *name);
npy_bool *extract_data_bool(PyArrayObject *np_arr, const char *name);
const npy_int64 *extract_index_array(PyArrayObject *np_arr, const char *name);
double **extract_grid_props(PyObject *grid_tuple, int ndim, int *dims);

#endif // PROPERTY_FUNCS_H_
