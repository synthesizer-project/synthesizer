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
 * @brief Templated implementation of kernel evaluation.
 *
 * @tparam Real The floating-point type (float or double).
 */
template <typename Real>
static PyObject *evaluate_kernel_impl(PyObject *self,
                                      PyArrayObject *np_radii,
                                      const char *kernel_name) {
  (void)self;

  const Real *radii = extract_data<Real>(np_radii, "radii");
  if (radii == NULL) {
    return NULL;
  }

  kernel_func<Real> func = get_kernel_function<Real>(kernel_name);
  if (func == NULL) {
    PyErr_SetString(PyExc_ValueError, "Kernel name not defined");
    return NULL;
  }

  const int ndim = static_cast<int>(PyArray_DIM(np_radii, 0));
  const int typenum =
      std::is_same_v<Real, float> ? NPY_FLOAT32 : NPY_FLOAT64;
  npy_intp dims[1] = {ndim};
  PyArrayObject *np_values =
      (PyArrayObject *)PyArray_ZEROS(1, dims, typenum, 0);
  if (np_values == NULL) {
    PyErr_NoMemory();
    return NULL;
  }
  Real *values = static_cast<Real *>(PyArray_DATA(np_values));

#ifdef WITH_OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < ndim; i++) {
    values[i] = func(radii[i]);
  }

  return Py_BuildValue("N", np_values);
}

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
 * @return A 1D float64 or float32 NumPy array of kernel values.
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

  const int input_typenum = PyArray_TYPE(np_radii);
  /* Dispatch: encode input precision into a 1-bit key. */
  int dispatch_key = (input_typenum == NPY_FLOAT64);

  /* Dispatch: call the matching typed kernel based on the dispatch key. */
  switch (dispatch_key) {
  case 0:
    return evaluate_kernel_impl<float>(self, np_radii, kernel_name);
  case 1:
    return evaluate_kernel_impl<double>(self, np_radii, kernel_name);
  default:
    PyErr_SetString(PyExc_TypeError,
                    "radii must be float32 or float64.");
    return NULL;
  }
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