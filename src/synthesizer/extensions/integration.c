/******************************************************************************
 * A C module containing helper functions for integration.
 *****************************************************************************/
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#ifdef WITH_OPENMP
#include <omp.h>
#endif

/* Local includes. */
#include "property_funcs.h"

/**
 * @brief Serial trapezoidal integration.
 *
 * @param xs 1D array of x values.
 * @param ys 1D array of y values.
 */
static double *trapz_last_axis_serial(double *x, double *y, npy_intp n,
                                      npy_intp num_elements) {
  double *integral = (double *)calloc(num_elements, sizeof(double));

  for (npy_intp i = 0; i < num_elements; ++i) {
    for (npy_intp j = 0; j < n - 1; ++j) {
      integral[i] +=
          0.5 * (x[j + 1] - x[j]) * (y[i * n + j + 1] + y[i * n + j]);
    }
  }

  return integral;
}

/**
 * @brief Parallel trapezoidal integration.
 *
 * @param xs 1D array of x values.
 * @param ys 1D array of y values.
 * @param nthreads Number of threads to use.
 */
#ifdef WITH_OPENMP
static double *trapz_last_axis_parallel(double *x, double *y, npy_intp n,
                                        npy_intp num_elements, int nthreads) {
  double *integral = (double *)calloc(num_elements, sizeof(double));

#pragma omp parallel for num_threads(nthreads)                                 \
    reduction(+ : integral[ : num_elements])
  for (npy_intp i = 0; i < num_elements; ++i) {
    for (npy_intp j = 0; j < n - 1; ++j) {
      integral[i] +=
          0.5 * (x[j + 1] - x[j]) * (y[i * n + j + 1] + y[i * n + j]);
    }
  }
  return integral;
}
#endif

/**
 * @brief Trapezoidal integration over the final axis of an ND array.
 *
 * @param xs 1D array of x values.
 * @param ys ND array of y values.
 * @param num_threads Number of threads to use.
 */
static PyObject *trapz_last_axis_integration(PyObject *self, PyObject *args) {

  (void)self; /* Unused variable */

  PyArrayObject *xs, *ys;
  int nthreads;

  /* Parse the input tuple */
  if (!PyArg_ParseTuple(args, "O!O!i", &PyArray_Type, &xs, &PyArray_Type, &ys,
                        &nthreads)) {
    return NULL; /* Return NULL in case of parsing error */
  }

  /* Get the array dimensions. */
  npy_intp ndim = PyArray_NDIM(ys);
  npy_intp *shape = PyArray_SHAPE(ys);

  /* Number of elements along the last axis */
  npy_intp n = shape[ndim - 1];

  /* Get the data pointer of the xs array */
  double *x = extract_data_double(xs, "xs");

  /* Get the data pointer of the ys array */
  double *y = (double *)PyArray_DATA(ys);

  /* Number of elements excluding the last axis */
  npy_intp num_elements = PyArray_SIZE(ys) / n;

  /* Compute the integral with the appropriate function. */
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

  /* Construct the output. */
  npy_intp result_shape[NPY_MAXDIMS];
  for (npy_intp i = 0; i < ndim - 1; ++i) {
    result_shape[i] = shape[i];
  }
  PyObject *result =
      c_array_to_numpy(ndim - 1, result_shape, NPY_DOUBLE, integral);

  return result; /* Return the computed integral */
}

/**
 * @brief Serial Simpson's integration.
 *
 * @param xs 1D array of x values.
 * @param ys ND array of y values.
 */
static double *simps_last_axis_serial(double *x, double *y, npy_intp n,
                                      npy_intp num_elements) {
  double *integral = (double *)calloc(num_elements, sizeof(double));

  for (npy_intp i = 0; i < num_elements; ++i) {
    if (n < 2) {
      continue; /* If the array has less than 2 elements, skip */
    }
    for (npy_intp j = 0; j < (n - 1) / 2; ++j) {
      npy_intp k = 2 * j;
      integral[i] += (x[k + 2] - x[k]) *
                     (y[i * n + k] + 4 * y[i * n + k + 1] + y[i * n + k + 2]) /
                     6.0;
    }
    if ((n - 1) % 2 != 0) {
      integral[i] +=
          0.5 * (x[n - 1] - x[n - 2]) * (y[i * n + n - 1] + y[i * n + n - 2]);
    }
  }

  return integral;
}

/**
 * @brief Parallel Simpson's integration.
 *
 * @param xs 1D array of x values.
 * @param ys ND array of y values.
 * @param nthreads Number of threads to use.
 */
#ifdef WITH_OPENMP
static double *simps_last_axis_parallel(double *x, double *y, npy_intp n,
                                        npy_intp num_elements, int nthreads) {
  double *integral = (double *)calloc(num_elements, sizeof(double));

#pragma omp parallel for num_threads(nthreads)                                 \
    reduction(+ : integral[ : num_elements])
  for (npy_intp i = 0; i < num_elements; ++i) {
    if (n < 2) {
      continue; /* If the array has less than 2 elements, skip */
    }

    for (npy_intp j = 0; j < (n - 1) / 2; ++j) {
      npy_intp k = 2 * j;
      integral[i] += (x[k + 2] - x[k]) *
                     (y[i * n + k] + 4 * y[i * n + k + 1] + y[i * n + k + 2]) /
                     6.0;
    }
    if ((n - 1) % 2 != 0) {
      integral[i] +=
          0.5 * (x[n - 1] - x[n - 2]) * (y[i * n + n - 1] + y[i * n + n - 2]);
    }
  }

  return integral;
}
#endif

/**
 * @brief Simpson's integration over the final axis of a ND array.
 *
 * @param xs 1D array of x values.
 * @param ys ND array of y values.
 * @param nthreads Number of threads to use.
 */
static PyObject *simps_last_axis_integration(PyObject *self, PyObject *args) {
  (void)self; /* Unused variable */

  PyArrayObject *xs, *ys;
  int nthreads;

  /* Parse the input tuple */
  if (!PyArg_ParseTuple(args, "O!O!i", &PyArray_Type, &xs, &PyArray_Type, &ys,
                        &nthreads)) {
    return NULL; /* Return NULL in case of parsing error */
  }

  /* Get the array dimensions. */
  npy_intp ndim = PyArray_NDIM(ys);
  npy_intp *shape = PyArray_SHAPE(ys);

  /* Number of elements along the last axis */
  npy_intp n = shape[ndim - 1];

  /* Get the data pointer of the xs array */
  double *x = extract_data_double(xs, "xs");

  /* Get the data pointer of the ys array */
  double *y = (double *)PyArray_DATA(ys);

  /* Number of elements excluding the last axis */
  npy_intp num_elements = PyArray_SIZE(ys) / n;

  /* Compute the integral with the appropriate function. */
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

  /* Construct the output. */
  npy_intp result_shape[NPY_MAXDIMS];
  for (npy_intp i = 0; i < ndim - 1; ++i) {
    result_shape[i] = shape[i];
  }
  PyObject *result =
      c_array_to_numpy(ndim - 1, result_shape, NPY_DOUBLE, integral);

  return result; /* Return the computed integral */
}

static PyMethodDef IntegrationMethods[] = {
    {"trapz_last_axis", trapz_last_axis_integration, METH_VARARGS,
     "Trapezoidal integration with OpenMP"},
    {"simps_last_axis", simps_last_axis_integration, METH_VARARGS,
     "Simpson's integration with OpenMP"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef integrationmodule = {
    PyModuleDef_HEAD_INIT,
    "integration", /* name of module */
    NULL,
    -1,
    IntegrationMethods,
    NULL,
    NULL,
    NULL,
    NULL};

PyMODINIT_FUNC PyInit_integration(void) {
  import_array(); /* Initialize the NumPy C-API */
  return PyModule_Create(&integrationmodule);
}
