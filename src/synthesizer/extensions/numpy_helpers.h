/******************************************************************************
 * C++ extension helper functions for working with NumPy arrays.
 *****************************************************************************/

#ifndef NUMPY_HELPERS_H
#define NUMPY_HELPERS_H

#include <Python.h>
#include <numpy/arrayobject.h>

#include "data_types.h"

/*----------------------------------------------------------------------------
 * Array validation helpers (strict, no copies)
 *----------------------------------------------------------------------------*/
static inline int ensure_c_contiguous(PyArrayObject *arr, const char *name) {
  if (!PyArray_ISCARRAY(arr)) {
    PyErr_Format(PyExc_ValueError,
                 "Array is not C contiguous. Use np.ascontiguousarray() "
                 "to convert the array before passing to C extensions.");
    return 0;
  }
  return 1;
}

static inline int ensure_float_array(PyArrayObject *arr, const char *name) {
  if (!PyArray_Check((PyObject *)arr)) {
    PyErr_Format(PyExc_TypeError,
                 "Expected a numpy array but got %s. Lists and other "
                 "iterables should be converted to numpy arrays before "
                 "being passed to C extensions.",
                 Py_TYPE((PyObject *)arr)->tp_name);
    return 0;
  }
  if (PyArray_TYPE(arr) != NPY_FLOAT_T) {
    PyErr_Format(PyExc_TypeError,
                 "Array has incorrect dtype. Expected %s but got %s.",
                 FLOAT_NAME, PyArray_DESCR(arr)->typeobj->tp_name);
    return 0;
  }
  return ensure_c_contiguous(arr, name);
}

static inline int ensure_double_array(PyArrayObject *arr, const char *name) {
  if (!PyArray_Check((PyObject *)arr)) {
    PyErr_Format(PyExc_TypeError,
                 "Expected a numpy array but got %s. Lists and other "
                 "iterables should be converted to numpy arrays before "
                 "being passed to C extensions.",
                 Py_TYPE((PyObject *)arr)->tp_name);
    return 0;
  }
  if (PyArray_TYPE(arr) != NPY_FLOAT64) {
    PyErr_Format(PyExc_TypeError,
                 "Array has incorrect dtype. Expected float64 but got %s.",
                 PyArray_DESCR(arr)->typeobj->tp_name);
    return 0;
  }
  return ensure_c_contiguous(arr, name);
}

static inline int ensure_int_array(PyArrayObject *arr, const char *name) {
  if (!PyArray_Check((PyObject *)arr)) {
    PyErr_Format(PyExc_TypeError,
                 "Expected a numpy array but got %s. Lists and other "
                 "iterables should be converted to numpy arrays before "
                 "being passed to C extensions.",
                 Py_TYPE((PyObject *)arr)->tp_name);
    return 0;
  }
  if (PyArray_TYPE(arr) != NPY_INT_T) {
    PyErr_Format(PyExc_TypeError,
                 "Array has incorrect dtype. Expected %s but got %s.", INT_NAME,
                 PyArray_DESCR(arr)->typeobj->tp_name);
    return 0;
  }
  return ensure_c_contiguous(arr, name);
}

static inline int ensure_bool_array(PyArrayObject *arr, const char *name) {
  if (!PyArray_Check((PyObject *)arr)) {
    PyErr_Format(PyExc_TypeError,
                 "Expected a numpy array but got %s. Lists and other "
                 "iterables should be converted to numpy arrays before "
                 "being passed to C extensions.",
                 Py_TYPE((PyObject *)arr)->tp_name);
    return 0;
  }
  if (PyArray_TYPE(arr) != NPY_BOOL) {
    PyErr_Format(PyExc_TypeError,
                 "Array has incorrect dtype. Expected bool but got %s.",
                 PyArray_DESCR(arr)->typeobj->tp_name);
    return 0;
  }
  return ensure_c_contiguous(arr, name);
}

#endif /* NUMPY_HELPERS_H */
