/******************************************************************************
 * Kernel extension Python bindings.
 *
 * This module provides the Python entry points for the LOS kernel table
 * builders. It initializes the NumPy C API and exposes the analytic kernel
 * evaluation function to Python.
 *****************************************************************************/

#include "kernels.h"
#include "kernel_functions.h"

/**
 * @brief Evaluate a named kernel on a 1D radius array.
 *
 * This exposes the analytic kernel implementations to Python so the public
 * ``uniform`` / ``cubic`` / etc. helpers can delegate to the same code used by
 * the C++ table builders instead of duplicating the formulas.
 *
 * @param self The module instance (unused).
 * @param args Python arguments containing a 1D radius array and kernel name.
 *
 * @return A 1D float64 NumPy array of kernel values.
 */
PyObject *evaluate_kernel(PyObject *self, PyObject *args) {

  (void)self;

  PyArrayObject *np_radii;
  const char *kernel_name;

  if (!PyArg_ParseTuple(args, "O!s", &PyArray_Type, &np_radii, &kernel_name)) {
    return NULL;
  }

  if (PyArray_NDIM(np_radii) != 1) {
    PyErr_SetString(PyExc_ValueError, "radii must be a 1D array.");
    return NULL;
  }
  if (PyArray_TYPE(np_radii) != NPY_DOUBLE) {
    PyErr_SetString(PyExc_TypeError, "radii must be float64.");
    return NULL;
  }

  /* The helper expects a contiguous float64 buffer and returns a borrowed data
   * pointer owned by the NumPy array object. */
  const double *radii = extract_data_double(np_radii, "radii");
  if (radii == NULL) {
    return NULL;
  }

  kernel_func func = get_kernel_function(kernel_name);
  if (func == NULL) {
    PyErr_SetString(PyExc_ValueError, "Kernel name not defined");
    return NULL;
  }

  /* Allocate the result array on the Python side and fill it in place from
   * the shared analytic kernel implementation. */
  const int ndim = static_cast<int>(PyArray_DIM(np_radii, 0));
  npy_intp dims[1] = {ndim};
  PyArrayObject *np_values =
      (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
  if (np_values == NULL) {
    PyErr_NoMemory();
    return NULL;
  }
  double *values = static_cast<double *>(PyArray_DATA(np_values));

#ifdef WITH_OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < ndim; i++) {
    values[i] = func(radii[i]);
  }

  return Py_BuildValue("N", np_values);
}

/* Expose the Python-callable entry points for this extension module. */
static PyMethodDef KernelMethods[] = {
    {"evaluate_kernel", (PyCFunction)evaluate_kernel, METH_VARARGS,
     "Evaluate a named kernel on a 1D array of radii."},
    {"compute_projected_kernel", (PyCFunction)compute_projected_kernel,
     METH_VARARGS, "Build the projected LOS kernel table."},
    {"compute_truncated_los_kernel", (PyCFunction)compute_truncated_los_kernel,
     METH_VARARGS, "Build the truncated LOS kernel table."},
    {"compute_overlap_kernel", (PyCFunction)compute_overlap_kernel,
     METH_VARARGS,
     "Method for building the smoothed-input overlap kernel table."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "kernel",
    "A module to build LOS kernel tables",
    -1,
    KernelMethods,
    NULL,
    NULL,
    NULL,
    NULL,
};

/* Create the module and initialise the NumPy C API before any of the wrapped
 * entry points are called from Python. */
PyMODINIT_FUNC PyInit_kernel(void) {
  PyObject *m = PyModule_Create(&moduledef);
  if (m == NULL) {
    return NULL;
  }
  if (numpy_import() < 0) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to import numpy.");
    Py_DECREF(m);
    return NULL;
  }
#ifdef ATOMIC_TIMING
  if (import_toc_capsule() < 0) {
    Py_DECREF(m);
    return NULL;
  }
#endif
  return m;
}