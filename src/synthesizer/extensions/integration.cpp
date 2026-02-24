/******************************************************************************
 * A C module containing helper functions for integration.
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
#include "property_funcs.h"
#include "timers.h"
#ifdef ATOMIC_TIMING
#include "timers_init.h"
#endif

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
  PyArrayObject *result =
      wrap_array_to_numpy<double>(ndim - 1, result_shape, integral);

  /* Create the output object. */
  if (result == NULL) {
    free(integral); /* Free the allocated memory on error */
    return NULL;    /* Return NULL in case of error */
  }
  PyObject *output = (PyObject *)result;

  return output;
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
  PyArrayObject *result =
      wrap_array_to_numpy<double>(ndim - 1, result_shape, integral);

  /* Create the output object. */
  if (result == NULL) {
    free(integral); /* Free the allocated memory on error */
    return NULL;    /* Return NULL in case of error */
  }
  PyObject *output = (PyObject *)result;

  return output;
}

/**
 * @brief Serial weighted trapezoidal integration over the final axis.
 *
 * Computes trapz(y * w, x) / trapz(w, x) for each spectrum in ys.
 *
 * @param x 1D array of x values.
 * @param y ND array flattened to (num_elements, n).
 * @param w 1D array of weights along the integrated axis.
 * @param n Number of elements along integrated axis.
 * @param num_elements Number of spectra (all axes except final).
 */
static double *weighted_trapz_last_axis_serial(double *x, double *y, double *w,
                                               npy_intp n,
                                               npy_intp num_elements) {
  double *result = (double *)calloc(num_elements, sizeof(double));

  /* Compute denominator once; it is shared by all spectra. */
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
 *
 * Computes trapz(y * w, x) / trapz(w, x) for each spectrum in ys.
 *
 * @param x 1D array of x values.
 * @param y ND array flattened to (num_elements, n).
 * @param w 1D array of weights along the integrated axis.
 * @param n Number of elements along integrated axis.
 * @param num_elements Number of spectra (all axes except final).
 * @param nthreads Number of threads to use.
 */
#ifdef WITH_OPENMP
static double *weighted_trapz_last_axis_parallel(double *x, double *y, double *w,
                                                 npy_intp n,
                                                 npy_intp num_elements,
                                                 int nthreads) {
  double *result = (double *)calloc(num_elements, sizeof(double));

  /* Compute denominator once; it is shared by all spectra. */
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
 *
 * Computes trapz(y * w, x) / trapz(w, x) for each spectrum in ys.
 */
static PyObject *weighted_trapz_last_axis_integration(PyObject *self,
                                                      PyObject *args) {
  (void)self; /* Unused variable */

  PyArrayObject *xs, *ys, *ws;
  int nthreads;

  /* Parse the input tuple */
  if (!PyArg_ParseTuple(args, "O!O!O!i", &PyArray_Type, &xs, &PyArray_Type,
                        &ys, &PyArray_Type, &ws, &nthreads)) {
    return NULL; /* Return NULL in case of parsing error */
  }

  if (PyArray_NDIM(xs) != 1) {
    PyErr_SetString(PyExc_ValueError, "xs must be a 1D array.");
    return NULL;
  }
  if (PyArray_NDIM(ws) != 1) {
    PyErr_SetString(PyExc_ValueError, "weights must be a 1D array.");
    return NULL;
  }

  /* Get the ys array dimensions. */
  npy_intp ndim = PyArray_NDIM(ys);
  npy_intp *shape = PyArray_SHAPE(ys);

  /* Number of elements along the last axis */
  npy_intp n = shape[ndim - 1];

  if (PyArray_DIM(xs, 0) != n || PyArray_DIM(ws, 0) != n) {
    PyErr_SetString(PyExc_ValueError,
                    "xs and weights must match ys along the final axis.");
    return NULL;
  }

  /* Get the data pointers. */
  double *x = extract_data_double(xs, "xs");
  if (x == NULL) {
    return NULL;
  }
  double *y = (double *)PyArray_DATA(ys);
  double *w = extract_data_double(ws, "weights");
  if (w == NULL) {
    return NULL;
  }

  /* Number of elements excluding the last axis */
  npy_intp num_elements = PyArray_SIZE(ys) / n;

  /* Compute the weighted result with the appropriate function. */
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

  /* Construct the output. */
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
 * @brief Serial batched weighted trapezoidal integration.
 *
 * Computes trapz(y * w_f, x) / den_f for each filter f and spectrum row.
 *
 * @param x 1D x-values of length n.
 * @param y Flattened ND spectra of shape (num_elements, n).
 * @param w Flattened filter weights of shape (nfilters, n).
 * @param den Precomputed denominators of shape (nfilters).
 * @param n Length of integrated axis.
 * @param num_elements Number of spectra rows.
 * @param nfilters Number of filters.
 */
static double *weighted_trapz_last_axis_many_serial(
    double *x, double *y, double *w, double *den, npy_int64 *starts,
    npy_int64 *ends, npy_intp n, npy_intp num_elements, npy_intp nfilters) {
  double *result =
      (double *)calloc((size_t)(num_elements * nfilters), sizeof(double));

  for (npy_intp i = 0; i < num_elements; ++i) {
    for (npy_intp f = 0; f < nfilters; ++f) {
      if (den[f] == 0.0) {
        continue;
      }

      double num = 0.0;
      double *wf = &w[f * n];
      npy_int64 j0 = starts[f];
      npy_int64 j1 = ends[f];
      if (j1 - j0 < 2) {
        continue;
      }
      for (npy_int64 j = j0; j < j1 - 1; ++j) {
        num += 0.5 * (x[j + 1] - x[j]) *
               (y[i * n + j + 1] * wf[j + 1] + y[i * n + j] * wf[j]);
      }
      result[i * nfilters + f] = num / den[f];
    }
  }

  return result;
}

#ifdef WITH_OPENMP
/**
 * @brief Parallel batched weighted trapezoidal integration.
 */
static double *weighted_trapz_last_axis_many_parallel(
    double *x, double *y, double *w, double *den, npy_int64 *starts,
    npy_int64 *ends, npy_intp n, npy_intp num_elements, npy_intp nfilters,
    int nthreads) {
  double *result =
      (double *)calloc((size_t)(num_elements * nfilters), sizeof(double));

#pragma omp parallel for num_threads(nthreads)
  for (npy_intp i = 0; i < num_elements; ++i) {
    for (npy_intp f = 0; f < nfilters; ++f) {
      if (den[f] == 0.0) {
        continue;
      }

      double num = 0.0;
      double *wf = &w[f * n];
      npy_int64 j0 = starts[f];
      npy_int64 j1 = ends[f];
      if (j1 - j0 < 2) {
        continue;
      }
      for (npy_int64 j = j0; j < j1 - 1; ++j) {
        num += 0.5 * (x[j + 1] - x[j]) *
               (y[i * n + j + 1] * wf[j + 1] + y[i * n + j] * wf[j]);
      }
      result[i * nfilters + f] = num / den[f];
    }
  }

  return result;
}
#endif

/**
 * @brief Batched weighted trapezoidal integration over final axis.
 *
 * Args are (xs, ys, weights_2d, denoms_1d, nthreads).
 */
static PyObject *weighted_trapz_last_axis_many_integration(PyObject *self,
                                                           PyObject *args) {
  (void)self;

  PyArrayObject *xs, *ys, *ws, *dens, *starts, *ends;
  int nthreads;

  if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!i", &PyArray_Type, &xs,
                        &PyArray_Type, &ys, &PyArray_Type, &ws, &PyArray_Type,
                        &dens, &PyArray_Type, &starts, &PyArray_Type, &ends,
                        &nthreads)) {
    return NULL;
  }

  if (PyArray_NDIM(xs) != 1) {
    PyErr_SetString(PyExc_ValueError, "xs must be a 1D array.");
    return NULL;
  }
  if (PyArray_NDIM(ws) != 2) {
    PyErr_SetString(PyExc_ValueError, "weights must be a 2D array.");
    return NULL;
  }
  if (PyArray_NDIM(dens) != 1) {
    PyErr_SetString(PyExc_ValueError, "denominators must be a 1D array.");
    return NULL;
  }
  if (PyArray_NDIM(starts) != 1 || PyArray_NDIM(ends) != 1) {
    PyErr_SetString(PyExc_ValueError,
                    "starts and ends must be 1D arrays.");
    return NULL;
  }

  npy_intp ndim = PyArray_NDIM(ys);
  npy_intp *shape = PyArray_SHAPE(ys);
  npy_intp n = shape[ndim - 1];
  npy_intp nfilters = PyArray_DIM(ws, 0);

  if (PyArray_DIM(xs, 0) != n) {
    PyErr_SetString(PyExc_ValueError,
                    "xs must match ys along the final axis.");
    return NULL;
  }
  if (PyArray_DIM(ws, 1) != n) {
    PyErr_SetString(PyExc_ValueError,
                    "weights second axis must match ys final axis.");
    return NULL;
  }
  if (PyArray_DIM(dens, 0) != nfilters) {
    PyErr_SetString(PyExc_ValueError,
                    "denominators must match number of filters.");
    return NULL;
  }
  if (PyArray_DIM(starts, 0) != nfilters || PyArray_DIM(ends, 0) != nfilters) {
    PyErr_SetString(PyExc_ValueError,
                    "starts/ends must match number of filters.");
    return NULL;
  }

  double *x = extract_data_double(xs, "xs");
  if (x == NULL) {
    return NULL;
  }
  double *y = (double *)PyArray_DATA(ys);
  double *w = extract_data_double(ws, "weights");
  if (w == NULL) {
    return NULL;
  }
  double *den = extract_data_double(dens, "denominators");
  if (den == NULL) {
    return NULL;
  }
  npy_int64 *jstarts = (npy_int64 *)PyArray_DATA(starts);
  npy_int64 *jends = (npy_int64 *)PyArray_DATA(ends);

  npy_intp num_elements = PyArray_SIZE(ys) / n;

  double *result_arr;
#ifdef WITH_OPENMP
  if (nthreads > 1) {
    result_arr = weighted_trapz_last_axis_many_parallel(
        x, y, w, den, jstarts, jends, n, num_elements, nfilters, nthreads);
  } else {
    result_arr = weighted_trapz_last_axis_many_serial(
        x, y, w, den, jstarts, jends, n, num_elements, nfilters);
  }
#else
  result_arr = weighted_trapz_last_axis_many_serial(
      x, y, w, den, jstarts, jends, n, num_elements, nfilters);
#endif

  npy_intp result_shape[NPY_MAXDIMS];
  for (npy_intp i = 0; i < ndim - 1; ++i) {
    result_shape[i] = shape[i];
  }
  result_shape[ndim - 1] = nfilters;

  PyArrayObject *result =
      wrap_array_to_numpy<double>(ndim, result_shape, result_arr);
  if (result == NULL) {
    free(result_arr);
    return NULL;
  }

  return (PyObject *)result;
}

/**
 * @brief Serial weighted Simpson integration over the final axis.
 *
 * Computes simps(y * w, x) / simps(w, x) for each spectrum in ys.
 */
static double *weighted_simps_last_axis_serial(double *x, double *y, double *w,
                                               npy_intp n,
                                               npy_intp num_elements) {
  double *result = (double *)calloc(num_elements, sizeof(double));

  if (n < 2) {
    return result;
  }

  /* Compute denominator once; it is shared by all spectra. */
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
 *
 * Computes simps(y * w, x) / simps(w, x) for each spectrum in ys.
 */
#ifdef WITH_OPENMP
static double *weighted_simps_last_axis_parallel(double *x, double *y, double *w,
                                                 npy_intp n,
                                                 npy_intp num_elements,
                                                 int nthreads) {
  double *result = (double *)calloc(num_elements, sizeof(double));

  if (n < 2) {
    return result;
  }

  /* Compute denominator once; it is shared by all spectra. */
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
 *
 * Computes simps(y * w, x) / simps(w, x) for each spectrum in ys.
 */
static PyObject *weighted_simps_last_axis_integration(PyObject *self,
                                                      PyObject *args) {
  (void)self; /* Unused variable */

  PyArrayObject *xs, *ys, *ws;
  int nthreads;

  /* Parse the input tuple */
  if (!PyArg_ParseTuple(args, "O!O!O!i", &PyArray_Type, &xs, &PyArray_Type,
                        &ys, &PyArray_Type, &ws, &nthreads)) {
    return NULL; /* Return NULL in case of parsing error */
  }

  if (PyArray_NDIM(xs) != 1) {
    PyErr_SetString(PyExc_ValueError, "xs must be a 1D array.");
    return NULL;
  }
  if (PyArray_NDIM(ws) != 1) {
    PyErr_SetString(PyExc_ValueError, "weights must be a 1D array.");
    return NULL;
  }

  /* Get the ys array dimensions. */
  npy_intp ndim = PyArray_NDIM(ys);
  npy_intp *shape = PyArray_SHAPE(ys);

  /* Number of elements along the last axis */
  npy_intp n = shape[ndim - 1];

  if (PyArray_DIM(xs, 0) != n || PyArray_DIM(ws, 0) != n) {
    PyErr_SetString(PyExc_ValueError,
                    "xs and weights must match ys along the final axis.");
    return NULL;
  }

  /* Get the data pointers. */
  double *x = extract_data_double(xs, "xs");
  if (x == NULL) {
    return NULL;
  }
  double *y = (double *)PyArray_DATA(ys);
  double *w = extract_data_double(ws, "weights");
  if (w == NULL) {
    return NULL;
  }

  /* Number of elements excluding the last axis */
  npy_intp num_elements = PyArray_SIZE(ys) / n;

  /* Compute the weighted result with the appropriate function. */
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

  /* Construct the output. */
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
    {"weighted_trapz_last_axis_many",
     weighted_trapz_last_axis_many_integration, METH_VARARGS,
     "Batched weighted trapezoidal integration with OpenMP"},
    {"weighted_simps_last_axis", weighted_simps_last_axis_integration,
     METH_VARARGS, "Weighted Simpson integration with OpenMP"},
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
