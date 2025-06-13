#include <type_traits>

#define PY_ARRAY_UNIQUE_SYMBOL SYNTHESIZER_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy_init.h"

#include "cpp_to_python.h"

// Map C++ type T to NumPy typenum at compile time
template <typename T> struct NumpyTypenum;

template <> struct NumpyTypenum<float> {
  static constexpr int typenum = NPY_FLOAT32;
};
template <> struct NumpyTypenum<double> {
  static constexpr int typenum = NPY_FLOAT64;
};
template <> struct NumpyTypenum<int32_t> {
  static constexpr int typenum = NPY_INT32;
};
template <> struct NumpyTypenum<int64_t> {
  static constexpr int typenum = NPY_INT64;
};
template <> struct NumpyTypenum<uint8_t> {
  static constexpr int typenum = NPY_UINT8;
};

// Core function for raw T* (allocated with new T[])
template <typename T>
PyArrayObject *wrap_array_to_numpy(int ndim, npy_intp *dims, int typenum,
                                   T *buffer) {

  if (!buffer) {
    return nullptr;
  }

  PyArray_Descr *descr = PyArray_DescrFromType(typenum);
  if (!descr) {
    delete[] buffer;
    return nullptr;
  }

  PyArrayObject *arr = reinterpret_cast<PyArrayObject *>(PyArray_NewFromDescr(
      &PyArray_Type, descr, ndim, dims, nullptr, static_cast<void *>(buffer),
      NPY_ARRAY_CARRAY, nullptr));
  if (!arr) {
    delete[] buffer;
    return nullptr;
  }

  // Capsule with correct deleter
  PyObject *capsule = PyCapsule_New(buffer, nullptr, [](PyObject *cap) {
    T *ptr = static_cast<T *>(PyCapsule_GetPointer(cap, nullptr));
    delete[] ptr;
  });

  if (!capsule) {
    Py_DECREF(arr);
    delete[] buffer;
    return nullptr;
  }

  if (PyArray_SetBaseObject(arr, capsule) < 0) {
    Py_DECREF(arr);
    Py_DECREF(capsule);
    delete[] buffer;
    return nullptr;
  }

  return arr;
}

// Overload: deduce typenum automatically from type T
template <typename T>
PyArrayObject *wrap_array_to_numpy(int ndim, npy_intp *dims, T *buffer) {
  return wrap_array_to_numpy<T>(ndim, dims, NumpyTypenum<T>::typenum, buffer);
}

// Overload: accept unique_ptr<T[]> and transfer ownership
template <typename T>
PyArrayObject *wrap_array_to_numpy(int ndim, npy_intp *dims,
                                   std::unique_ptr<T[]> &&ptr) {
  T *raw = ptr.release(); // transfer ownership
  return wrap_array_to_numpy<T>(ndim, dims, raw);
}

/* Declarations of specialized functions for common types */
template PyArrayObject *wrap_array_to_numpy<double>(int, npy_intp *, double *);
template PyArrayObject *wrap_array_to_numpy<float>(int, npy_intp *, float *);
template PyArrayObject *wrap_array_to_numpy<int32_t>(int, npy_intp *,
                                                     int32_t *);
template PyArrayObject *wrap_array_to_numpy<int64_t>(int, npy_intp *,
                                                     int64_t *);
template PyArrayObject *
wrap_array_to_numpy<double>(int, npy_intp *, std::unique_ptr<double[]> &&);
template PyArrayObject *wrap_array_to_numpy<float>(int, npy_intp *,
                                                   std::unique_ptr<float[]> &&);
template PyArrayObject *
wrap_array_to_numpy<int32_t>(int, npy_intp *, std::unique_ptr<int32_t[]> &&);
template PyArrayObject *
wrap_array_to_numpy<int64_t>(int, npy_intp *, std::unique_ptr<int64_t[]> &&);

/**
 * @brief Check that the given PyObject* is either None or a NumPy array.
 *
 * @param obj  The Python object to check (e.g. from argument parsing).
 * @param name Optional name to include in the error message.
 * @return     nullptr if obj is Py_None, or a PyArrayObject* if it's a valid
 * array. Returns nullptr and sets a Python error if the type is invalid.
 */
PyArrayObject *array_or_none(PyObject *obj, const char *name) {
  if (obj == Py_None) {
    return nullptr;
  }

  if (!PyArray_Check(obj)) {
    PyErr_Format(PyExc_TypeError, "%s must be a NumPy array or None", name);
    return nullptr;
  }

  return reinterpret_cast<PyArrayObject *>(obj);
}
