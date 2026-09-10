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
  const char *type_name = std::is_same_v<Real, float> ? "float32" : "float64";

  if (PyArray_TYPE(np_arr) != expected_type) {
    char error_msg[120];
    snprintf(error_msg, sizeof(error_msg), "%s must be a %s array.", name,
             type_name);
    PyErr_SetString(PyExc_TypeError, error_msg);
    return NULL;
  }

  if (!PyArray_IS_C_CONTIGUOUS(np_arr)) {
    char error_msg[120];
    snprintf(error_msg, sizeof(error_msg), "%s must be C-contiguous.", name);
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
const npy_int64 *extract_index_array(PyArrayObject *np_arr, const char *name);

#endif  // PROPERTY_FUNCS_H_
