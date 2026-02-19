/******************************************************************************
 * A C++ extension to expose compile-time precision information to Python.
 *
 * This module provides functions to query the floating-point precision
 * that the synthesizer C extensions were compiled with.
 *****************************************************************************/

/* Python includes */
#define PY_ARRAY_UNIQUE_SYMBOL SYNTHESIZER_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy_init.h"
#include <Python.h>

/* Local includes */
#include "data_types.h"

/**
 * @brief Get the precision string ("float32" or "float64").
 *
 * @return A Python string indicating the compiled precision.
 */
static PyObject *get_precision(PyObject *self, PyObject *args) {
  (void)self;
  (void)args;
  return PyUnicode_FromString(FLOAT_NAME);
}

/**
 * @brief Get the number of bytes per floating-point value.
 *
 * @return A Python integer (4 or 8).
 */
static PyObject *get_float_bytes(PyObject *self, PyObject *args) {
  (void)self;
  (void)args;
  return PyLong_FromLong(FLOAT_BYTES);
}

/* Method definitions */
static PyMethodDef PrecisionMethods[] = {
    {"get_precision", get_precision, METH_NOARGS,
     "Return the compiled floating-point precision ('float32' or 'float64')."},
    {"get_float_bytes", get_float_bytes, METH_NOARGS,
     "Return the number of bytes per floating-point value (4 or 8)."},
    {NULL, NULL, 0, NULL}};

/* Module definition */
static struct PyModuleDef precisionmodule = {
    PyModuleDef_HEAD_INIT,
    "precision_info",                                   /* m_name */
    "Module to query compile-time precision settings.", /* m_doc */
    -1,                                                 /* m_size */
    PrecisionMethods,                                   /* m_methods */
    NULL,                                               /* m_reload */
    NULL,                                               /* m_traverse */
    NULL,                                               /* m_clear */
    NULL,                                               /* m_free */
};

/* Module initialization */
PyMODINIT_FUNC PyInit_precision_info(void) {
  PyObject *m = PyModule_Create(&precisionmodule);
  if (numpy_import() < 0) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to import numpy.");
    return NULL;
  }
  return m;
}
