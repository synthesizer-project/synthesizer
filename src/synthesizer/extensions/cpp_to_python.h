#ifndef CPP_TO_PYTHON_H
#define CPP_TO_PYTHON_H

#define PY_ARRAY_UNIQUE_SYMBOL SYNTHESIZER_ARRAY_API
#define NO_IMPORT_ARRAY
#include <Python.h>

#include <cstdint>
#include <memory>

#include "numpy_init.h"

/**
 * @brief Map a C++ scalar type onto its NumPy typenum.
 *
 * This template is used by the header-only array wrapping helpers to infer the
 * NumPy dtype that should back the returned array when the caller does not
 * pass an explicit typenum.
 *
 * @tparam T The C++ scalar type to map.
 */
template <typename T>
struct NumpyTypenum;

template <>
struct NumpyTypenum<float> {
  static constexpr int typenum = NPY_FLOAT32;
};

template <>
struct NumpyTypenum<double> {
  static constexpr int typenum = NPY_FLOAT64;
};

template <>
struct NumpyTypenum<int32_t> {
  static constexpr int typenum = NPY_INT32;
};

template <>
struct NumpyTypenum<int64_t> {
  static constexpr int typenum = NPY_INT64;
};

template <>
struct NumpyTypenum<uint8_t> {
  static constexpr int typenum = NPY_UINT8;
};

/**
 * @brief Wrap an owned C++ buffer in a NumPy array with an explicit dtype.
 *
 * Ownership of ``buffer`` is transferred to the returned NumPy array by
 * attaching a capsule-based deleter as the array base object.
 *
 * @tparam T The scalar type stored in the buffer.
 *
 * @param ndim The dimensionality of the output array.
 * @param dims The output array dimensions.
 * @param typenum The NumPy typenum to expose on the returned array.
 * @param buffer Pointer to a heap-allocated ``new[]`` buffer.
 *
 * @return A NumPy array owning ``buffer``, or ``nullptr`` on failure.
 */
template <typename T>
PyArrayObject *wrap_array_to_numpy(int ndim, npy_intp *dims, int typenum,
                                   T *buffer) {

  /* If allocation failed upstream there is nothing to wrap. */
  if (!buffer) {
    return nullptr;
  }

  /* Resolve the NumPy dtype descriptor for the requested output typenum. */
  PyArray_Descr *descr = PyArray_DescrFromType(typenum);
  if (!descr) {
    delete[] buffer;
    return nullptr;
  }

  /* Build a NumPy array view over the owned C++ buffer. */
  PyArrayObject *arr = reinterpret_cast<PyArrayObject *>(PyArray_NewFromDescr(
      &PyArray_Type, descr, ndim, dims, nullptr, static_cast<void *>(buffer),
      NPY_ARRAY_CARRAY, nullptr));
  if (!arr) {
    delete[] buffer;
    return nullptr;
  }

  /* Hand NumPy a capsule so the buffer is released when the array dies. */
  PyObject *capsule = PyCapsule_New(buffer, nullptr, [](PyObject *cap) {
    T *ptr = static_cast<T *>(PyCapsule_GetPointer(cap, nullptr));
    delete[] ptr;
  });

  if (!capsule) {
    Py_DECREF(arr);
    delete[] buffer;
    return nullptr;
  }

  /* Attach the capsule as the base object so NumPy owns the lifetime. */
  if (PyArray_SetBaseObject(arr, capsule) < 0) {
    Py_DECREF(arr);
    Py_DECREF(capsule);
    delete[] buffer;
    return nullptr;
  }

  return arr;
}

/**
 * @brief Wrap an owned C++ buffer in a NumPy array using the native type map.
 *
 * This overload resolves the output typenum from ``NumpyTypenum<T>`` before
 * forwarding to the explicit-typenum wrapper.
 *
 * @tparam T The scalar type stored in the buffer.
 *
 * @param ndim The dimensionality of the output array.
 * @param dims The output array dimensions.
 * @param buffer Pointer to a heap-allocated ``new[]`` buffer.
 *
 * @return A NumPy array owning ``buffer``, or ``nullptr`` on failure.
 */
template <typename T>
PyArrayObject *wrap_array_to_numpy(int ndim, npy_intp *dims, T *buffer) {
  /* Infer the NumPy typenum from the native scalar type and forward. */
  return wrap_array_to_numpy<T>(ndim, dims, NumpyTypenum<T>::typenum, buffer);
}

/**
 * @brief Wrap a ``std::unique_ptr`` buffer in a NumPy array.
 *
 * This overload releases the incoming unique pointer and forwards the raw
 * buffer to the raw-pointer wrapper so NumPy takes ownership.
 *
 * @tparam T The scalar type stored in the buffer.
 *
 * @param ndim The dimensionality of the output array.
 * @param dims The output array dimensions.
 * @param ptr Unique pointer owning a heap-allocated ``new[]`` buffer.
 *
 * @return A NumPy array owning the released buffer, or ``nullptr`` on failure.
 */
template <typename T>
PyArrayObject *wrap_array_to_numpy(int ndim, npy_intp *dims,
                                   std::unique_ptr<T[]> &&ptr) {
  /* Release ownership from the smart pointer before forwarding to NumPy. */
  T *raw = ptr.release();
  return wrap_array_to_numpy<T>(ndim, dims, raw);
}

int resolve_output_typenum(PyObject *dtype_obj,
                           const char *argument_name = "out_dtype");

PyArrayObject *array_or_none(PyObject *obj, const char *name = "argument");

#define RETURN_IF_PYERR()                 \
  do {                                    \
    if (PyErr_Occurred()) return nullptr; \
  } while (0)

#endif  // CPP_TO_PYTHON_H
