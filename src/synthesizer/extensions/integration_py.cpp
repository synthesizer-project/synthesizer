/******************************************************************************
 * Python bindings for the integration helpers.
 *****************************************************************************/
#define PY_ARRAY_UNIQUE_SYMBOL SYNTHESIZER_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy_init.h"
#include <Python.h>

#ifdef WITH_OPENMP
#include <omp.h>
#endif

/* Local includes. */
#include "cpp_to_python.h"
#include "integration.h"
#include "property_funcs.h"
#include "timers.h"
#ifdef ATOMIC_TIMING
#include "timers_init.h"
#endif

/**
 * @brief Check whether a 1D x grid is uniformly spaced.
 *
 * @param x 1D array of x values.
 * @param n Number of samples.
 *
 * @return True if the spacing is uniform within a small tolerance.
 */
static bool is_uniform_grid(const double *x, size_t n) {
  if (n < 3) {
    return true;
  }

  const double dx = x[1] - x[0];
  const double tol = 1.0e-12 * fmax(1.0, fabs(dx));

  for (size_t i = 1; i < n - 1; ++i) {
    if (fabs((x[i + 1] - x[i]) - dx) > tol) {
      return false;
    }
  }

  return true;
}

/**
 * @brief Serial trapezoidal integration over the final axis.
 */
static double *trapz_last_axis_serial(double *x, double *y, npy_intp n,
                                      npy_intp num_elements) {
  double *integral = (double *)calloc(num_elements, sizeof(double));
  if (integral == NULL) {
    return NULL;
  }

  for (npy_intp i = 0; i < num_elements; ++i) {
    integral[i] = trapz_1d(x, y + i * n, static_cast<size_t>(n));
  }

  return integral;
}

/**
 * @brief Parallel trapezoidal integration over the final axis.
 */
#ifdef WITH_OPENMP
static double *trapz_last_axis_parallel(double *x, double *y, npy_intp n,
                                        npy_intp num_elements, int nthreads) {
  double *integral = (double *)calloc(num_elements, sizeof(double));
  if (integral == NULL) {
    return NULL;
  }

#pragma omp parallel for num_threads(nthreads)
  for (npy_intp i = 0; i < num_elements; ++i) {
    integral[i] = trapz_1d(x, y + i * n, static_cast<size_t>(n));
  }
  return integral;
}
#endif

/**
 * @brief Trapezoidal integration over the final axis of an ND array.
 */
static PyObject *trapz_last_axis_integration(PyObject *self, PyObject *args) {

  (void)self;

  PyArrayObject *xs, *ys;
  int nthreads;

  if (!PyArg_ParseTuple(args, "O!O!i", &PyArray_Type, &xs, &PyArray_Type, &ys,
                        &nthreads)) {
    return NULL;
  }

  npy_intp ndim = PyArray_NDIM(ys);
  npy_intp *shape = PyArray_SHAPE(ys);
  npy_intp n = shape[ndim - 1];

  if (n == 0) {
    PyErr_SetString(PyExc_ValueError,
                    "ys final axis must contain at least one element.");
    return NULL;
  }

  double *x = extract_data_double(xs, "xs");
  double *y = (double *)PyArray_DATA(ys);
  npy_intp num_elements = PyArray_SIZE(ys) / n;

  if (x == NULL) {
    return NULL;
  }

  double *integral;
#ifdef WITH_OPENMP
  if (nthreads > 1) {
    integral = trapz_last_axis_parallel(x, y, n, num_elements, nthreads);
  } else {
    integral = trapz_last_axis_serial(x, y, n, num_elements);
  }
#else
  integral = trapz_last_axis_serial(x, y, n, num_elements);
#endif

  if (integral == NULL) {
    PyErr_NoMemory();
    return NULL;
  }

  npy_intp result_shape[NPY_MAXDIMS];
  for (npy_intp i = 0; i < ndim - 1; ++i) {
    result_shape[i] = shape[i];
  }
  PyArrayObject *result =
      wrap_array_to_numpy<double>(ndim - 1, result_shape, integral);
  if (result == NULL) {
    free(integral);
    return NULL;
  }

  return (PyObject *)result;
}

/**
 * @brief Serial Simpson integration over the final axis.
 */
static double *simps_last_axis_serial(double *x, double *y, npy_intp n,
                                      npy_intp num_elements) {
  double *integral = (double *)calloc(num_elements, sizeof(double));
  if (integral == NULL) {
    return NULL;
  }

  for (npy_intp i = 0; i < num_elements; ++i) {
    integral[i] = simps_1d(x, y + i * n, static_cast<size_t>(n));
  }

  return integral;
}

/**
 * @brief Parallel Simpson integration over the final axis.
 */
#ifdef WITH_OPENMP
static double *simps_last_axis_parallel(double *x, double *y, npy_intp n,
                                        npy_intp num_elements, int nthreads) {
  double *integral = (double *)calloc(num_elements, sizeof(double));
  if (integral == NULL) {
    return NULL;
  }

#pragma omp parallel for num_threads(nthreads)
  for (npy_intp i = 0; i < num_elements; ++i) {
    integral[i] = simps_1d(x, y + i * n, static_cast<size_t>(n));
  }

  return integral;
}
#endif

/**
 * @brief Simpson integration over the final axis of a ND array.
 */
static PyObject *simps_last_axis_integration(PyObject *self, PyObject *args) {
  (void)self;

  PyArrayObject *xs, *ys;
  int nthreads;

  if (!PyArg_ParseTuple(args, "O!O!i", &PyArray_Type, &xs, &PyArray_Type, &ys,
                        &nthreads)) {
    return NULL;
  }

  npy_intp ndim = PyArray_NDIM(ys);
  npy_intp *shape = PyArray_SHAPE(ys);
  npy_intp n = shape[ndim - 1];

  if (n == 0) {
    PyErr_SetString(PyExc_ValueError,
                    "ys final axis must contain at least one element.");
    return NULL;
  }

  double *x = extract_data_double(xs, "xs");
  double *y = (double *)PyArray_DATA(ys);
  npy_intp num_elements = PyArray_SIZE(ys) / n;

  if (x == NULL) {
    return NULL;
  }
  if (!is_uniform_grid(x, static_cast<size_t>(n))) {
    PyErr_SetString(PyExc_ValueError,
                    "Simpson integration requires uniformly spaced xs.");
    return NULL;
  }

  double *integral;
#ifdef WITH_OPENMP
  if (nthreads > 1) {
    integral = simps_last_axis_parallel(x, y, n, num_elements, nthreads);
  } else {
    integral = simps_last_axis_serial(x, y, n, num_elements);
  }
#else
  integral = simps_last_axis_serial(x, y, n, num_elements);
#endif

  if (integral == NULL) {
    PyErr_NoMemory();
    return NULL;
  }

  npy_intp result_shape[NPY_MAXDIMS];
  for (npy_intp i = 0; i < ndim - 1; ++i) {
    result_shape[i] = shape[i];
  }
  PyArrayObject *result =
      wrap_array_to_numpy<double>(ndim - 1, result_shape, integral);
  if (result == NULL) {
    free(integral);
    return NULL;
  }

  return (PyObject *)result;
}

/**
 * @brief Serial weighted trapezoidal integration over the final axis.
 */
static double *weighted_trapz_last_axis_serial(double *x, double *y, double *w,
                                               npy_intp n,
                                               npy_intp num_elements) {
  double *result = (double *)calloc(num_elements, sizeof(double));
  if (result == NULL) {
    return NULL;
  }

  double den = 0.0;
  for (npy_intp j = 0; j < n - 1; ++j) {
    den += 0.5 * (x[j + 1] - x[j]) * (w[j + 1] + w[j]);
  }

  if (den == 0.0) {
    return result;
  }

  for (npy_intp i = 0; i < num_elements; ++i) {
    double num = 0.0;
    for (npy_intp j = 0; j < n - 1; ++j) {
      num += 0.5 * (x[j + 1] - x[j]) *
             (y[i * n + j + 1] * w[j + 1] + y[i * n + j] * w[j]);
    }
    result[i] = num / den;
  }

  return result;
}

/**
 * @brief Parallel weighted trapezoidal integration over the final axis.
 */
#ifdef WITH_OPENMP
static double *weighted_trapz_last_axis_parallel(double *x, double *y, double *w,
                                                 npy_intp n,
                                                 npy_intp num_elements,
                                                 int nthreads) {
  double *result = (double *)calloc(num_elements, sizeof(double));
  if (result == NULL) {
    return NULL;
  }

  double den = 0.0;
  for (npy_intp j = 0; j < n - 1; ++j) {
    den += 0.5 * (x[j + 1] - x[j]) * (w[j + 1] + w[j]);
  }

  if (den == 0.0) {
    return result;
  }

#pragma omp parallel for num_threads(nthreads)
  for (npy_intp i = 0; i < num_elements; ++i) {
    double num = 0.0;
    for (npy_intp j = 0; j < n - 1; ++j) {
      num += 0.5 * (x[j + 1] - x[j]) *
             (y[i * n + j + 1] * w[j + 1] + y[i * n + j] * w[j]);
    }
    result[i] = num / den;
  }

  return result;
}
#endif

/**
 * @brief Weighted trapezoidal integration over the final axis of an ND array.
 */
static PyObject *weighted_trapz_last_axis_integration(PyObject *self,
                                                      PyObject *args) {
  (void)self;

  PyArrayObject *xs, *ys, *ws;
  int nthreads;

  if (!PyArg_ParseTuple(args, "O!O!O!i", &PyArray_Type, &xs, &PyArray_Type,
                        &ys, &PyArray_Type, &ws, &nthreads)) {
    return NULL;
  }

  if (PyArray_NDIM(xs) != 1) {
    PyErr_SetString(PyExc_ValueError, "xs must be a 1D array.");
    return NULL;
  }
  if (PyArray_NDIM(ws) != 1) {
    PyErr_SetString(PyExc_ValueError, "weights must be a 1D array.");
    return NULL;
  }

  npy_intp ndim = PyArray_NDIM(ys);
  if (ndim < 1) {
    PyErr_SetString(PyExc_ValueError, "ys must have at least 1 dimension.");
    return NULL;
  }
  npy_intp *shape = PyArray_SHAPE(ys);
  npy_intp n = shape[ndim - 1];

  if (n == 0) {
    PyErr_SetString(PyExc_ValueError,
                    "ys final axis must contain at least one element.");
    return NULL;
  }
  if (PyArray_DIM(xs, 0) != n || PyArray_DIM(ws, 0) != n) {
    PyErr_SetString(PyExc_ValueError,
                    "xs and weights must match ys along the final axis.");
    return NULL;
  }

  double *x = extract_data_double(xs, "xs");
  if (x == NULL) {
    return NULL;
  }
  double *y = extract_data_double(ys, "ys");
  if (y == NULL) {
    return NULL;
  }
  double *w = extract_data_double(ws, "weights");
  if (w == NULL) {
    return NULL;
  }

  npy_intp num_elements = PyArray_SIZE(ys) / n;
  double *result_arr;
#ifdef WITH_OPENMP
  if (nthreads > 1) {
    result_arr =
        weighted_trapz_last_axis_parallel(x, y, w, n, num_elements, nthreads);
  } else {
    result_arr = weighted_trapz_last_axis_serial(x, y, w, n, num_elements);
  }
#else
  result_arr = weighted_trapz_last_axis_serial(x, y, w, n, num_elements);
#endif

  if (result_arr == NULL) {
    PyErr_SetString(PyExc_MemoryError,
                    "Failed to allocate output for weighted trapz.");
    return NULL;
  }

  npy_intp result_shape[NPY_MAXDIMS];
  for (npy_intp i = 0; i < ndim - 1; ++i) {
    result_shape[i] = shape[i];
  }
  PyArrayObject *result =
      wrap_array_to_numpy<double>(ndim - 1, result_shape, result_arr);
  if (result == NULL) {
    free(result_arr);
    return NULL;
  }

  return (PyObject *)result;
}

/**
 * @brief Serial weighted Simpson integration over the final axis.
 */
static double *weighted_simps_last_axis_serial(double *x, double *y, double *w,
                                               npy_intp n,
                                               npy_intp num_elements) {
  double *result = (double *)calloc(num_elements, sizeof(double));
  if (result == NULL) {
    return NULL;
  }

  if (n < 2) {
    return result;
  }

  double den = 0.0;
  for (npy_intp j = 0; j < (n - 1) / 2; ++j) {
    npy_intp k = 2 * j;
    den += (x[k + 2] - x[k]) * (w[k] + 4 * w[k + 1] + w[k + 2]) / 6.0;
  }
  if ((n - 1) % 2 != 0) {
    den += 0.5 * (x[n - 1] - x[n - 2]) * (w[n - 1] + w[n - 2]);
  }

  if (den == 0.0) {
    return result;
  }

  for (npy_intp i = 0; i < num_elements; ++i) {
    double num = 0.0;
    for (npy_intp j = 0; j < (n - 1) / 2; ++j) {
      npy_intp k = 2 * j;
      num += (x[k + 2] - x[k]) *
             (y[i * n + k] * w[k] + 4 * y[i * n + k + 1] * w[k + 1] +
              y[i * n + k + 2] * w[k + 2]) /
             6.0;
    }
    if ((n - 1) % 2 != 0) {
      num += 0.5 * (x[n - 1] - x[n - 2]) *
             (y[i * n + n - 1] * w[n - 1] + y[i * n + n - 2] * w[n - 2]);
    }
    result[i] = num / den;
  }

  return result;
}

/**
 * @brief Parallel weighted Simpson integration over the final axis.
 */
#ifdef WITH_OPENMP
static double *weighted_simps_last_axis_parallel(double *x, double *y, double *w,
                                                 npy_intp n,
                                                 npy_intp num_elements,
                                                 int nthreads) {
  double *result = (double *)calloc(num_elements, sizeof(double));
  if (result == NULL) {
    return NULL;
  }

  if (n < 2) {
    return result;
  }

  double den = 0.0;
  for (npy_intp j = 0; j < (n - 1) / 2; ++j) {
    npy_intp k = 2 * j;
    den += (x[k + 2] - x[k]) * (w[k] + 4 * w[k + 1] + w[k + 2]) / 6.0;
  }
  if ((n - 1) % 2 != 0) {
    den += 0.5 * (x[n - 1] - x[n - 2]) * (w[n - 1] + w[n - 2]);
  }

  if (den == 0.0) {
    return result;
  }

#pragma omp parallel for num_threads(nthreads)
  for (npy_intp i = 0; i < num_elements; ++i) {
    double num = 0.0;
    for (npy_intp j = 0; j < (n - 1) / 2; ++j) {
      npy_intp k = 2 * j;
      num += (x[k + 2] - x[k]) *
             (y[i * n + k] * w[k] + 4 * y[i * n + k + 1] * w[k + 1] +
              y[i * n + k + 2] * w[k + 2]) /
             6.0;
    }
    if ((n - 1) % 2 != 0) {
      num += 0.5 * (x[n - 1] - x[n - 2]) *
             (y[i * n + n - 1] * w[n - 1] + y[i * n + n - 2] * w[n - 2]);
    }
    result[i] = num / den;
  }

  return result;
}
#endif

/**
 * @brief Weighted Simpson integration over the final axis of an ND array.
 */
static PyObject *weighted_simps_last_axis_integration(PyObject *self,
                                                      PyObject *args) {
  (void)self;

  PyArrayObject *xs, *ys, *ws;
  int nthreads;

  if (!PyArg_ParseTuple(args, "O!O!O!i", &PyArray_Type, &xs, &PyArray_Type,
                        &ys, &PyArray_Type, &ws, &nthreads)) {
    return NULL;
  }

  if (PyArray_NDIM(xs) != 1) {
    PyErr_SetString(PyExc_ValueError, "xs must be a 1D array.");
    return NULL;
  }
  if (PyArray_NDIM(ws) != 1) {
    PyErr_SetString(PyExc_ValueError, "weights must be a 1D array.");
    return NULL;
  }

  npy_intp ndim = PyArray_NDIM(ys);
  if (ndim < 1) {
    PyErr_SetString(PyExc_ValueError, "ys must have at least 1 dimension.");
    return NULL;
  }
  npy_intp *shape = PyArray_SHAPE(ys);
  npy_intp n = shape[ndim - 1];

  if (n == 0) {
    PyErr_SetString(PyExc_ValueError,
                    "ys final axis must contain at least one element.");
    return NULL;
  }
  if (PyArray_DIM(xs, 0) != n || PyArray_DIM(ws, 0) != n) {
    PyErr_SetString(PyExc_ValueError,
                    "xs and weights must match ys along the final axis.");
    return NULL;
  }

  double *x = extract_data_double(xs, "xs");
  if (x == NULL) {
    return NULL;
  }
  double *y = extract_data_double(ys, "ys");
  if (y == NULL) {
    return NULL;
  }
  double *w = extract_data_double(ws, "weights");
  if (w == NULL) {
    return NULL;
  }

  npy_intp num_elements = PyArray_SIZE(ys) / n;
  double *result_arr;
#ifdef WITH_OPENMP
  if (nthreads > 1) {
    result_arr =
        weighted_simps_last_axis_parallel(x, y, w, n, num_elements, nthreads);
  } else {
    result_arr = weighted_simps_last_axis_serial(x, y, w, n, num_elements);
  }
#else
  result_arr = weighted_simps_last_axis_serial(x, y, w, n, num_elements);
#endif

  if (result_arr == NULL) {
    PyErr_SetString(PyExc_MemoryError,
                    "Failed to allocate output for weighted simps.");
    return NULL;
  }

  npy_intp result_shape[NPY_MAXDIMS];
  for (npy_intp i = 0; i < ndim - 1; ++i) {
    result_shape[i] = shape[i];
  }
  PyArrayObject *result =
      wrap_array_to_numpy<double>(ndim - 1, result_shape, result_arr);
  if (result == NULL) {
    free(result_arr);
    return NULL;
  }

  return (PyObject *)result;
}

static PyMethodDef IntegrationMethods[] = {
    {"trapz_last_axis", trapz_last_axis_integration, METH_VARARGS,
     "Trapezoidal integration with OpenMP"},
    {"simps_last_axis", simps_last_axis_integration, METH_VARARGS,
     "Simpson's integration with OpenMP"},
    {"weighted_trapz_last_axis", weighted_trapz_last_axis_integration,
     METH_VARARGS, "Weighted trapezoidal integration with OpenMP"},
    {"weighted_simps_last_axis", weighted_simps_last_axis_integration,
     METH_VARARGS, "Weighted Simpson integration with OpenMP"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef integrationmodule = {
    PyModuleDef_HEAD_INIT,
    "integration",
    NULL,
    -1,
    IntegrationMethods,
    NULL,
    NULL,
    NULL,
    NULL};

PyMODINIT_FUNC PyInit_integration(void) {
  if (numpy_import() < 0) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to import numpy.");
    return NULL;
  }
  PyObject *m = PyModule_Create(&integrationmodule);
  if (m == NULL)
    return NULL;
#ifdef ATOMIC_TIMING
  if (import_toc_capsule() < 0) {
    Py_DECREF(m);
    return NULL;
  }
#endif
  return m;
}
