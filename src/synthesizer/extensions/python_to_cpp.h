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
bool is_supported_float_typenum(int typenum);
int promoted_float_typenum(int lhs, int rhs);
bool is_matching_float_dtypes(PyArrayObject **arrays, const char **names,
                              int count, int *resolved_typenum);

/* Float-type dispatch helpers.
 *
 * These map a NumPy float typenum (NPY_FLOAT32/NPY_FLOAT64) onto a C++
 * scalar type and invoke a callable with a value of that type. Nesting
 * calls dispatches over several independent typenums without hand-written
 * switch tables, e.g.:
 *
 *   dispatch_float(part_typenum, [&](auto p) {
 *     dispatch_float(grid_typenum, [&](auto g) {
 *       dispatch_float(out_typenum, [&](auto o) {
 *         using PartReal = decltype(p);
 *         using GridReal = decltype(g);
 *         using OutT = decltype(o);
 *         kernel<PartReal, GridReal, OutT>(...);
 *       });
 *     });
 *   });
 *
 * Callers must validate that the typenum is float32 or float64 first
 * (anything else dispatches as float64's opposite branch never fires;
 * float32 is the fallback branch).
 */

/**
 * @brief Invoke a callable with a scalar of the C++ type matching typenum.
 *
 * @tparam F The callable type (typically a generic lambda).
 * @param typenum: The NumPy float typenum (NPY_FLOAT32 or NPY_FLOAT64).
 * @param f: The callable invoked with float{} or double{}.
 *
 * @return Whatever the callable returns.
 */
template <typename F>
inline decltype(auto) dispatch_float(int typenum, F &&f) {
  if (typenum == NPY_FLOAT64) {
    return f(double{});
  }
  return f(float{});
}

/* Inline pointer extraction helpers. */
/**
 * @brief Extract a typed pointer from a validated NumPy array.
 *
 * Callers are expected to validate dtype and contiguity before using this
 * helper.
 *
 * @tparam T The scalar type of the NumPy array buffer.
 * @param np_arr: The NumPy array.
 * @return Typed pointer to the underlying buffer.
 */
template <typename T>
inline T *data_ptr(PyArrayObject *np_arr) {
  return static_cast<T *>(PyArray_DATA(np_arr));
}

/**
 * @brief Extract a typed const pointer from a validated NumPy array.
 *
 * @tparam T The scalar type of the NumPy array buffer.
 * @param np_arr: The NumPy array.
 * @return Typed const pointer to the underlying buffer.
 */
template <typename T>
inline const T *data_ptr(const PyArrayObject *np_arr) {
  return static_cast<const T *>(PyArray_DATA(np_arr));
}

#endif  // PYTHON_TO_CPP_H
