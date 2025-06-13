#ifndef CPP_TO_PYTHON_H
#define CPP_TO_PYTHON_H

#define PY_ARRAY_UNIQUE_SYMBOL SYNTHESIZER_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy_init.h"
#include <Python.h>
#include <cstdint>
#include <memory>

// Typenum mapping template (declare only)
template <typename T> struct NumpyTypenum;

// Main wrapper functions (declarations only)

// Raw pointer version with explicit typenum
template <typename T>
PyArrayObject *wrap_array_to_numpy(int ndim, npy_intp *dims, int typenum,
                                   T *buffer);

// Raw pointer version with inferred typenum
template <typename T>
PyArrayObject *wrap_array_to_numpy(int ndim, npy_intp *dims, T *buffer);

// unique_ptr version
template <typename T>
PyArrayObject *wrap_array_to_numpy(int ndim, npy_intp *dims,
                                   std::unique_ptr<T[]> &&ptr);

PyArrayObject *array_or_none(PyObject *obj, const char *name = "argument");

#define RETURN_IF_PYERR()                                                      \
  do {                                                                         \
    if (PyErr_Occurred())                                                      \
      return nullptr;                                                          \
  } while (0)

#endif // CPP_TO_PYTHON_H
