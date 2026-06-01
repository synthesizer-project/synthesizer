/******************************************************************************
 * C/C++ helpers for validating Python inputs before entering typed kernels.
 *
 * This module mirrors cpp_to_python.h by collecting shared boundary helpers
 * used when converting Python-facing NumPy arrays into validated C++ inputs.
 *****************************************************************************/
#ifndef PYTHON_TO_CPP_H
#define PYTHON_TO_CPP_H

/* Python includes */
#define PY_ARRAY_UNIQUE_SYMBOL SYNTHESIZER_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy_init.h"
#include <Python.h>

/* Prototypes for Python-to-C++ boundary helpers. */
const char *typenum_to_string(int typenum);
bool is_c_contiguous(PyArrayObject *np_arr, const char *name);
bool is_float32_or_float64(PyArrayObject *np_arr, const char *name);
bool is_matching_float_dtypes(PyArrayObject **arrays, const char **names,
                              int count, int *resolved_typenum);

/* Inline pointer extraction helpers. */
/**
 * @brief Extract a typed pointer from a validated NumPy array.
 *
 * Callers are expected to validate dtype and contiguity before using this
 * helper.
 *
 * @param np_arr: The NumPy array.
 * @return Typed pointer to the underlying buffer.
 */
template <typename T> inline T *data_ptr(PyArrayObject *np_arr) {
  return static_cast<T *>(PyArray_DATA(np_arr));
}

/**
 * @brief Extract a typed const pointer from a validated NumPy array.
 *
 * @param np_arr: The NumPy array.
 * @return Typed const pointer to the underlying buffer.
 */
template <typename T> inline const T *data_ptr(const PyArrayObject *np_arr) {
  return static_cast<const T *>(PyArray_DATA(np_arr));
}

#endif // PYTHON_TO_CPP_H
