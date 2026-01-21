/******************************************************************************
 * C++ extension helper functions for working with NumPy arrays.
 *****************************************************************************/

#ifndef NUMPY_HELPERS_H
#define NUMPY_HELPERS_H

#include <Python.h>
#include <numpy/arrayobject.h>

#include "data_types.h"

/*----------------------------------------------------------------------------
 * require_float_array
 *
 * Ensures:
 *   - NumPy array
 *   - dtype == FLOAT
 *   - C-contiguous
 *   - aligned
 *
 * allow_copy = 0 → strict (error if mismatch)
 * allow_copy = 1 → will cast / copy if needed
 *
 * Returns new reference (caller must Py_DECREF)
 *----------------------------------------------------------------------------*/
static inline PyArrayObject *require_float_array(PyObject *obj, int allow_copy,
                                                 const char *name) {
  int flags = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED;

  if (allow_copy) {
    flags |= NPY_ARRAY_ENSUREARRAY;
    return (PyArrayObject *)PyArray_FROM_OTF(obj, NPY_FLOAT_T, flags);
  }

  /* Strict path */
  PyArrayObject *arr =
      (PyArrayObject *)PyArray_FROM_OTF(obj, NPY_FLOAT_T, flags);

  if (arr == NULL) {
    return NULL; /* NumPy already raised error */
  }

  if (PyArray_TYPE(arr) != NPY_FLOAT_T) {
    PyErr_Format(PyExc_TypeError, "%s must have dtype %s", name, FLOAT_NAME);
    Py_DECREF(arr);
    return NULL;
  }

  if (!PyArray_ISCARRAY(arr)) {
    PyErr_Format(PyExc_ValueError, "%s must be C-contiguous", name);
    Py_DECREF(arr);
    return NULL;
  }

  return arr;
}

/*----------------------------------------------------------------------------
 * Convenience accessor
 *----------------------------------------------------------------------------*/
static inline FLOAT *float_array_data(PyArrayObject *arr) {
  return (FLOAT *)PyArray_DATA(arr);
}

#endif /* NUMPY_HELPERS_H */
