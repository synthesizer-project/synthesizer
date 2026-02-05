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
#include "data_types.h"
#include "numpy_helpers.h"
#include "property_funcs.h"
#include "timers.h"

/**
 * @brief Serial trapezoidal integration.
 *
 * @param xs 1D array of x values.
 * @param ys 1D array of y values.
 */
static Float *trapz_last_axis_serial(Float *x, Float *y, npy_intp n,
                                       npy_intp num_elements) {
  Float *integral = (Float *)calloc(num_elements, sizeof(Float));

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
static Float *trapz_last_axis_parallel(Float *x, Float *y, npy_intp n,
                                         npy_intp num_elements, int nthreads) {
  Float *integral = (Float *)calloc(num_elements, sizeof(Float));

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

/* --------------------------------------------------------------------------
 * Scaled variants: find max(abs(xs)), max(abs(ys)) from the Float inputs,
 * then integrate with all arithmetic and accumulation in double precision.
 * The output is a freshly-allocated double array; the caller multiplies by
 * the returned scale = xscale * yscale to recover physical units.
 *
 * Why double accumulation even in the SINGLE_PRECISION build?
 *   - The trapezoid kernel computes dx*(y0+y1) which can exceed float32
 *     range after normalisation when -ffast-math reassociates partial sums.
 *   - double accumulation is safe regardless of compiler flags and avoids
 *     a separate Python-side .astype(float64) on the result.
 *   - We still READ inputs as Float (float32 when SINGLE_PRECISION) so the
 *     memory-bandwidth saving from smaller input arrays is preserved.
 * -------------------------------------------------------------------------- */

/**
 * @brief Serial scaled trapezoidal integration (double accumulator).
 *
 * @param x             1D x values in Float (length n).
 * @param y             ND y values in Float, row-major.
 * @param n             Length of last axis.
 * @param num_elements  Product of all axes except the last.
 * @param out_scale     Written with xscale * yscale (both double).
 * @return              Freshly-allocated double array of length num_elements.
 */
static double *trapz_last_axis_scaled_serial(Float *x, Float *y, npy_intp n,
                                              npy_intp num_elements,
                                              double *out_scale) {
  /* --- find xscale (1D, serial) --- */
  double xscale = 0.0;
  for (npy_intp j = 0; j < n; ++j) {
    double ax = (double)x[j];
    if (ax < 0.0) ax = -ax;
    if (ax > xscale) xscale = ax;
  }

  /* --- find yscale (full array scan) --- */
  npy_intp total = num_elements * n;
  double yscale = 0.0;
  for (npy_intp k = 0; k < total; ++k) {
    double ay = (double)y[k];
    if (ay < 0.0) ay = -ay;
    if (ay > yscale) yscale = ay;
  }

  *out_scale = xscale * yscale;

  if (xscale == 0.0 || yscale == 0.0) {
    return (double *)calloc(num_elements, sizeof(double));
  }

  double inv_xs = 1.0 / xscale;
  double inv_ys = 1.0 / yscale;
  double *integral = (double *)calloc(num_elements, sizeof(double));

  for (npy_intp i = 0; i < num_elements; ++i) {
    double sum = 0.0;
    for (npy_intp j = 0; j < n - 1; ++j) {
      double dx = ((double)x[j + 1] - (double)x[j]) * inv_xs;
      double y0 = (double)y[i * n + j] * inv_ys;
      double y1 = (double)y[i * n + j + 1] * inv_ys;
      sum += 0.5 * dx * (y1 + y0);
    }
    integral[i] = sum;
  }

  return integral;
}

#ifdef WITH_OPENMP
/**
 * @brief Parallel scaled trapezoidal integration (double accumulator).
 *
 * Parallel reduction for yscale; then parallel-for over rows with a
 * thread-local double accumulator (no array reduction needed — each row
 * writes to a distinct output slot).
 */
static double *trapz_last_axis_scaled_parallel(Float *x, Float *y,
                                                npy_intp n,
                                                npy_intp num_elements,
                                                int nthreads,
                                                double *out_scale) {
  /* --- find xscale (1D, serial — negligible) --- */
  double xscale = 0.0;
  for (npy_intp j = 0; j < n; ++j) {
    double ax = (double)x[j];
    if (ax < 0.0) ax = -ax;
    if (ax > xscale) xscale = ax;
  }

  /* --- find yscale (parallel max reduction) --- */
  npy_intp total = num_elements * n;
  double yscale = 0.0;
#pragma omp parallel for num_threads(nthreads) reduction(max : yscale)
  for (npy_intp k = 0; k < total; ++k) {
    double ay = (double)y[k];
    if (ay < 0.0) ay = -ay;
    if (ay > yscale) yscale = ay;
  }

  *out_scale = xscale * yscale;

  if (xscale == 0.0 || yscale == 0.0) {
    return (double *)calloc(num_elements, sizeof(double));
  }

  double inv_xs = 1.0 / xscale;
  double inv_ys = 1.0 / yscale;
  double *integral = (double *)calloc(num_elements, sizeof(double));

  /* Each row writes to integral[i] — no race, no reduction needed. */
#pragma omp parallel for num_threads(nthreads)
  for (npy_intp i = 0; i < num_elements; ++i) {
    double sum = 0.0;
    for (npy_intp j = 0; j < n - 1; ++j) {
      double dx = ((double)x[j + 1] - (double)x[j]) * inv_xs;
      double y0 = (double)y[i * n + j] * inv_ys;
      double y1 = (double)y[i * n + j + 1] * inv_ys;
      sum += 0.5 * dx * (y1 + y0);
    }
    integral[i] = sum;
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

  if (!ensure_float_array(xs, "xs")) {
    return NULL;
  }
  if (!ensure_float_array(ys, "ys")) {
    return NULL;
  }

  /* Get the data pointer of the xs array */
  Float *x = extract_data_float(xs, "xs");
  if (x == NULL) {
    return NULL; /* Type error already set */
  }

  /* Get the data pointer of the ys array */
  Float *y = extract_data_float(ys, "ys");
  if (y == NULL) {
    return NULL; /* Type error already set */
  }

  /* Number of elements excluding the last axis */
  npy_intp num_elements = PyArray_SIZE(ys) / n;

  /* Compute the integral with the appropriate function. */
  Float *integral;
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
      wrap_array_to_numpy<Float>(ndim - 1, result_shape, integral);

  /* Create the output object. */
  if (result == NULL) {
    free(integral); /* Free the allocated memory on error */
    return NULL;    /* Return NULL in case of error */
  }
  PyObject *output = (PyObject *)result;

  return output;
}

/**
 * @brief Scaled trapezoidal integration over the final axis of an ND array.
 *
 * Fused version: finds max(|xs|) and max(|ys|) inside C, then integrates with
 * inline division.  Returns a 2-tuple (integral_array, scale) where
 * scale = xscale * yscale (as a Python float).  The caller multiplies the
 * result by scale to recover physical units.
 *
 * @param xs 1D array of x values (Float dtype, C-contiguous).
 * @param ys ND array of y values (Float dtype, C-contiguous).
 * @param num_threads Number of threads to use.
 */
static PyObject *trapz_last_axis_scaled_integration(PyObject *self,
                                                     PyObject *args) {
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

  if (!ensure_float_array(xs, "xs")) {
    return NULL;
  }
  if (!ensure_float_array(ys, "ys")) {
    return NULL;
  }

  Float *x = extract_data_float(xs, "xs");
  if (x == NULL) return NULL;
  Float *y = extract_data_float(ys, "ys");
  if (y == NULL) return NULL;

  npy_intp num_elements = PyArray_SIZE(ys) / n;

  double scale = 0.0;
  double *integral;
#ifdef WITH_OPENMP
  if (nthreads > 1) {
    integral = trapz_last_axis_scaled_parallel(x, y, n, num_elements,
                                               nthreads, &scale);
  } else {
    integral = trapz_last_axis_scaled_serial(x, y, n, num_elements, &scale);
  }
#else
  integral = trapz_last_axis_scaled_serial(x, y, n, num_elements, &scale);
#endif

  /* Wrap the double result into a NumPy float64 array. */
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

  /* Return (integral_array, scale) tuple. */
  PyObject *scale_obj = PyFloat_FromDouble(scale);
  if (scale_obj == NULL) {
    Py_DECREF(result);
    return NULL;
  }
  PyObject *out = PyTuple_Pack(2, (PyObject *)result, scale_obj);
  Py_DECREF(result);
  Py_DECREF(scale_obj);
  return out;
}

/* --------------------------------------------------------------------------
 * Weighted-trapz scaled variants: integrate ys[i,:] * w[:] over xs[:].
 * Same scaling / double-accumulation design as the plain scaled trapz,
 * but with a 1D weight vector fused into the inner loop.  This lets the
 * caller avoid three Python-level temporaries:
 *   arr_in_band = arr.compress(in_band)       // 370 MB copy
 *   transmission = arr_in_band * t_in_band    // 370 MB alloc
 *   integrand    = transmission / xs_in_band  // 370 MB alloc
 * All three passes are now a single fused read of arr + w.
 * -------------------------------------------------------------------------- */

/**
 * @brief Serial weighted scaled trapezoidal integration (double accumulator).
 *
 * @param x   1D x values (Float, length n).
 * @param y   ND y values (Float, row-major, last axis = n).
 * @param w   1D weight vector (Float, length n).  Integrand = y * w.
 * @param n   Length of last axis.
 * @param num_elements  Product of all axes except the last.
 * @param out_scale     Written with xscale * yscale * wscale.
 * @return    Freshly-allocated double array of length num_elements.
 */
static double *trapz_last_axis_weighted_serial(Float *x, Float *y, Float *w,
                                                npy_intp n,
                                                npy_intp num_elements,
                                                double *out_scale) {
  /* --- find xscale (1D) --- */
  double xscale = 0.0;
  for (npy_intp j = 0; j < n; ++j) {
    double ax = (double)x[j];
    if (ax < 0.0) ax = -ax;
    if (ax > xscale) xscale = ax;
  }

  /* --- find wscale (1D) --- */
  double wscale = 0.0;
  for (npy_intp j = 0; j < n; ++j) {
    double aw = (double)w[j];
    if (aw < 0.0) aw = -aw;
    if (aw > wscale) wscale = aw;
  }

  /* --- find yscale (full array) --- */
  npy_intp total = num_elements * n;
  double yscale = 0.0;
  for (npy_intp k = 0; k < total; ++k) {
    double ay = (double)y[k];
    if (ay < 0.0) ay = -ay;
    if (ay > yscale) yscale = ay;
  }

  *out_scale = xscale * yscale * wscale;

  if (xscale == 0.0 || yscale == 0.0 || wscale == 0.0) {
    return (double *)calloc(num_elements, sizeof(double));
  }

  double inv_xs = 1.0 / xscale;
  double inv_ys = 1.0 / yscale;
  double inv_ws = 1.0 / wscale;
  double *integral = (double *)calloc(num_elements, sizeof(double));

  for (npy_intp i = 0; i < num_elements; ++i) {
    double sum = 0.0;
    for (npy_intp j = 0; j < n - 1; ++j) {
      double dx  = ((double)x[j + 1] - (double)x[j]) * inv_xs;
      double yw0 = (double)y[i * n + j] * inv_ys * (double)w[j] * inv_ws;
      double yw1 = (double)y[i * n + j + 1] * inv_ys *
                   (double)w[j + 1] * inv_ws;
      sum += 0.5 * dx * (yw0 + yw1);
    }
    integral[i] = sum;
  }

  return integral;
}

#ifdef WITH_OPENMP
/**
 * @brief Parallel weighted scaled trapezoidal integration (double accumulator).
 */
static double *trapz_last_axis_weighted_parallel(Float *x, Float *y, Float *w,
                                                  npy_intp n,
                                                  npy_intp num_elements,
                                                  int nthreads,
                                                  double *out_scale) {
  /* --- find xscale (1D, serial) --- */
  double xscale = 0.0;
  for (npy_intp j = 0; j < n; ++j) {
    double ax = (double)x[j];
    if (ax < 0.0) ax = -ax;
    if (ax > xscale) xscale = ax;
  }

  /* --- find wscale (1D, serial) --- */
  double wscale = 0.0;
  for (npy_intp j = 0; j < n; ++j) {
    double aw = (double)w[j];
    if (aw < 0.0) aw = -aw;
    if (aw > wscale) wscale = aw;
  }

  /* --- find yscale (parallel max reduction) --- */
  npy_intp total = num_elements * n;
  double yscale = 0.0;
#pragma omp parallel for num_threads(nthreads) reduction(max : yscale)
  for (npy_intp k = 0; k < total; ++k) {
    double ay = (double)y[k];
    if (ay < 0.0) ay = -ay;
    if (ay > yscale) yscale = ay;
  }

  *out_scale = xscale * yscale * wscale;

  if (xscale == 0.0 || yscale == 0.0 || wscale == 0.0) {
    return (double *)calloc(num_elements, sizeof(double));
  }

  double inv_xs = 1.0 / xscale;
  double inv_ys = 1.0 / yscale;
  double inv_ws = 1.0 / wscale;
  double *integral = (double *)calloc(num_elements, sizeof(double));

  /* Each row writes to integral[i] — no race. */
#pragma omp parallel for num_threads(nthreads)
  for (npy_intp i = 0; i < num_elements; ++i) {
    double sum = 0.0;
    for (npy_intp j = 0; j < n - 1; ++j) {
      double dx  = ((double)x[j + 1] - (double)x[j]) * inv_xs;
      double yw0 = (double)y[i * n + j] * inv_ys * (double)w[j] * inv_ws;
      double yw1 = (double)y[i * n + j + 1] * inv_ys *
                   (double)w[j + 1] * inv_ws;
      sum += 0.5 * dx * (yw0 + yw1);
    }
    integral[i] = sum;
  }

  return integral;
}
#endif

/**
 * @brief Weighted trapezoidal integration over the final axis of an ND array.
 *
 * Fused version: computes ∫ y(x)*w(x) dx with inline scaling.
 * Returns a 2-tuple (float64 integral_array, scale) where
 * scale = xscale * yscale * wscale.
 *
 * @param xs 1D x values (Float, C-contiguous).
 * @param ys ND y values (Float, C-contiguous).
 * @param ws 1D weight vector (Float, C-contiguous, same length as xs).
 * @param nthreads Number of threads.
 */
static PyObject *trapz_last_axis_weighted_integration(PyObject *self,
                                                       PyObject *args) {
  (void)self;

  PyArrayObject *xs, *ys, *ws;
  int nthreads;

  if (!PyArg_ParseTuple(args, "O!O!O!i", &PyArray_Type, &xs, &PyArray_Type, &ys,
                        &PyArray_Type, &ws, &nthreads)) {
    return NULL;
  }

  npy_intp ndim = PyArray_NDIM(ys);
  npy_intp *shape = PyArray_SHAPE(ys);
  npy_intp n = shape[ndim - 1];

  if (!ensure_float_array(xs, "xs")) return NULL;
  if (!ensure_float_array(ys, "ys")) return NULL;
  if (!ensure_float_array(ws, "ws")) return NULL;

  Float *x = extract_data_float(xs, "xs");
  if (x == NULL) return NULL;
  Float *y = extract_data_float(ys, "ys");
  if (y == NULL) return NULL;
  Float *w = extract_data_float(ws, "ws");
  if (w == NULL) return NULL;

  npy_intp num_elements = PyArray_SIZE(ys) / n;

  double scale = 0.0;
  double *integral;
#ifdef WITH_OPENMP
  if (nthreads > 1) {
    integral = trapz_last_axis_weighted_parallel(x, y, w, n, num_elements,
                                                  nthreads, &scale);
  } else {
    integral = trapz_last_axis_weighted_serial(x, y, w, n, num_elements,
                                               &scale);
  }
#else
  integral = trapz_last_axis_weighted_serial(x, y, w, n, num_elements, &scale);
#endif

  /* Wrap the double result into a NumPy float64 array. */
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

  /* Return (integral_array, scale) tuple. */
  PyObject *scale_obj = PyFloat_FromDouble(scale);
  if (scale_obj == NULL) {
    Py_DECREF(result);
    return NULL;
  }
  PyObject *out = PyTuple_Pack(2, (PyObject *)result, scale_obj);
  Py_DECREF(result);
  Py_DECREF(scale_obj);
  return out;
}

/**
 * @brief Serial Simpson's integration.
 *
 * @param xs 1D array of x values.
 * @param ys ND array of y values.
 */
static Float *simps_last_axis_serial(Float *x, Float *y, npy_intp n,
                                       npy_intp num_elements) {
  Float *integral = (Float *)calloc(num_elements, sizeof(Float));

  for (npy_intp i = 0; i < num_elements; ++i) {
    if (n < 2) {
      continue;
    }
    /* n==2: only one interval, fall back to trapezoid (matches scipy). */
    if (n == 2) {
      integral[i] = (Float)(0.5 * ((double)x[1] - (double)x[0]) *
                           ((double)y[i * n] + (double)y[i * n + 1]));
      continue;
    }
    /* Generalised Simpson panels (correct for non-uniform spacing). */
    for (npy_intp j = 0; j < (n - 1) / 2; ++j) {
      npy_intp k  = 2 * j;
      double   h0 = (double)x[k + 1] - (double)x[k];
      double   h1 = (double)x[k + 2] - (double)x[k + 1];
      double   hs = h0 + h1;
      double   y0 = (double)y[i * n + k];
      double   y1 = (double)y[i * n + k + 1];
      double   y2 = (double)y[i * n + k + 2];
      integral[i] += (Float)(hs / 6.0 *
                     (y0 * (2.0 - h1 / h0) +
                      y1 * (hs * hs / (h0 * h1)) +
                      y2 * (2.0 - h0 / h1)));
    }
    /* Even n (odd intervals): Cartwright correction on last 3 points. */
    if (n % 2 == 0) {
      double h0 = (double)x[n - 2] - (double)x[n - 3];
      double h1 = (double)x[n - 1] - (double)x[n - 2];
      double alpha = (2.0 * h1 * h1 + 3.0 * h0 * h1) / (6.0 * (h0 + h1));
      double beta  = (h1 * h1 + 3.0 * h0 * h1) / (6.0 * h0);
      double eta   = (h1 * h1 * h1) / (6.0 * h0 * (h0 + h1));
      integral[i] += (Float)(alpha * (double)y[i * n + n - 1] +
                             beta  * (double)y[i * n + n - 2] -
                             eta   * (double)y[i * n + n - 3]);
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
static Float *simps_last_axis_parallel(Float *x, Float *y, npy_intp n,
                                        npy_intp num_elements, int nthreads) {
  Float *integral = (Float *)calloc(num_elements, sizeof(Float));

#pragma omp parallel for num_threads(nthreads)                                 \
    reduction(+ : integral[ : num_elements])
  for (npy_intp i = 0; i < num_elements; ++i) {
    if (n < 2) {
      continue;
    }
    /* n==2: only one interval, fall back to trapezoid (matches scipy). */
    if (n == 2) {
      integral[i] = (Float)(0.5 * ((double)x[1] - (double)x[0]) *
                           ((double)y[i * n] + (double)y[i * n + 1]));
      continue;
    }

    /* Generalised Simpson panels (correct for non-uniform spacing). */
    for (npy_intp j = 0; j < (n - 1) / 2; ++j) {
      npy_intp k  = 2 * j;
      double   h0 = (double)x[k + 1] - (double)x[k];
      double   h1 = (double)x[k + 2] - (double)x[k + 1];
      double   hs = h0 + h1;
      double   y0 = (double)y[i * n + k];
      double   y1 = (double)y[i * n + k + 1];
      double   y2 = (double)y[i * n + k + 2];
      integral[i] += (Float)(hs / 6.0 *
                     (y0 * (2.0 - h1 / h0) +
                      y1 * (hs * hs / (h0 * h1)) +
                      y2 * (2.0 - h0 / h1)));
    }
    /* Even n (odd intervals): Cartwright correction on last 3 points. */
    if (n % 2 == 0) {
      double h0 = (double)x[n - 2] - (double)x[n - 3];
      double h1 = (double)x[n - 1] - (double)x[n - 2];
      double alpha = (2.0 * h1 * h1 + 3.0 * h0 * h1) / (6.0 * (h0 + h1));
      double beta  = (h1 * h1 + 3.0 * h0 * h1) / (6.0 * h0);
      double eta   = (h1 * h1 * h1) / (6.0 * h0 * (h0 + h1));
      integral[i] += (Float)(alpha * (double)y[i * n + n - 1] +
                             beta  * (double)y[i * n + n - 2] -
                             eta   * (double)y[i * n + n - 3]);
    }
  }

  return integral;
}
#endif

/* --------------------------------------------------------------------------
 * Scaled Simpson's variants — same design as the scaled trapz kernels:
 * read Float* inputs, find scale in double, accumulate in double, return
 * double* + scale.  The odd-panel trapezoid tail uses the same pattern.
 * -------------------------------------------------------------------------- */

/**
 * @brief Serial scaled Simpson's integration (double accumulator).
 */
static double *simps_last_axis_scaled_serial(Float *x, Float *y, npy_intp n,
                                              npy_intp num_elements,
                                              double *out_scale) {
  /* --- find xscale (1D) --- */
  double xscale = 0.0;
  for (npy_intp j = 0; j < n; ++j) {
    double ax = (double)x[j];
    if (ax < 0.0) ax = -ax;
    if (ax > xscale) xscale = ax;
  }

  /* --- find yscale (full array) --- */
  npy_intp total = num_elements * n;
  double yscale = 0.0;
  for (npy_intp k = 0; k < total; ++k) {
    double ay = (double)y[k];
    if (ay < 0.0) ay = -ay;
    if (ay > yscale) yscale = ay;
  }

  *out_scale = xscale * yscale;

  if (xscale == 0.0 || yscale == 0.0) {
    return (double *)calloc(num_elements, sizeof(double));
  }

  double inv_xs = 1.0 / xscale;
  double inv_ys = 1.0 / yscale;
  double *integral = (double *)calloc(num_elements, sizeof(double));

  for (npy_intp i = 0; i < num_elements; ++i) {
    if (n < 2) continue;

    /* n==2: only one interval, fall back to trapezoid (matches scipy). */
    if (n == 2) {
      integral[i] = 0.5 * ((double)x[1] - (double)x[0]) * inv_xs *
                    ((double)y[i * n] + (double)y[i * n + 1]) * inv_ys;
      continue;
    }

    double sum = 0.0;

    /* Generalised Simpson panels (correct for non-uniform spacing). */
    for (npy_intp j = 0; j < (n - 1) / 2; ++j) {
      npy_intp k  = 2 * j;
      double   h0 = ((double)x[k + 1] - (double)x[k]) * inv_xs;
      double   h1 = ((double)x[k + 2] - (double)x[k + 1]) * inv_xs;
      double   hs = h0 + h1;
      double   y0 = (double)y[i * n + k] * inv_ys;
      double   y1 = (double)y[i * n + k + 1] * inv_ys;
      double   y2 = (double)y[i * n + k + 2] * inv_ys;
      sum += hs / 6.0 * (y0 * (2.0 - h1 / h0) +
                          y1 * (hs * hs / (h0 * h1)) +
                          y2 * (2.0 - h0 / h1));
    }

    /* Even n (odd intervals): Cartwright correction on last 3 points. */
    if (n % 2 == 0) {
      double h0 = ((double)x[n - 2] - (double)x[n - 3]) * inv_xs;
      double h1 = ((double)x[n - 1] - (double)x[n - 2]) * inv_xs;
      double alpha = (2.0 * h1 * h1 + 3.0 * h0 * h1) / (6.0 * (h0 + h1));
      double beta  = (h1 * h1 + 3.0 * h0 * h1) / (6.0 * h0);
      double eta   = (h1 * h1 * h1) / (6.0 * h0 * (h0 + h1));
      sum += alpha * (double)y[i * n + n - 1] * inv_ys +
             beta  * (double)y[i * n + n - 2] * inv_ys -
             eta   * (double)y[i * n + n - 3] * inv_ys;
    }

    integral[i] = sum;
  }

  return integral;
}

#ifdef WITH_OPENMP
/**
 * @brief Parallel scaled Simpson's integration (double accumulator).
 */
static double *simps_last_axis_scaled_parallel(Float *x, Float *y,
                                                npy_intp n,
                                                npy_intp num_elements,
                                                int nthreads,
                                                double *out_scale) {
  /* --- find xscale (1D, serial) --- */
  double xscale = 0.0;
  for (npy_intp j = 0; j < n; ++j) {
    double ax = (double)x[j];
    if (ax < 0.0) ax = -ax;
    if (ax > xscale) xscale = ax;
  }

  /* --- find yscale (parallel max reduction) --- */
  npy_intp total = num_elements * n;
  double yscale = 0.0;
#pragma omp parallel for num_threads(nthreads) reduction(max : yscale)
  for (npy_intp k = 0; k < total; ++k) {
    double ay = (double)y[k];
    if (ay < 0.0) ay = -ay;
    if (ay > yscale) yscale = ay;
  }

  *out_scale = xscale * yscale;

  if (xscale == 0.0 || yscale == 0.0) {
    return (double *)calloc(num_elements, sizeof(double));
  }

  double inv_xs = 1.0 / xscale;
  double inv_ys = 1.0 / yscale;
  double *integral = (double *)calloc(num_elements, sizeof(double));

  /* Each row writes to integral[i] — no race, no reduction needed. */
#pragma omp parallel for num_threads(nthreads)
  for (npy_intp i = 0; i < num_elements; ++i) {
    if (n < 2) continue;

    /* n==2: only one interval, fall back to trapezoid (matches scipy). */
    if (n == 2) {
      integral[i] = 0.5 * ((double)x[1] - (double)x[0]) * inv_xs *
                    ((double)y[i * n] + (double)y[i * n + 1]) * inv_ys;
      continue;
    }

    double sum = 0.0;

    /* Generalised Simpson panels (correct for non-uniform spacing). */
    for (npy_intp j = 0; j < (n - 1) / 2; ++j) {
      npy_intp k  = 2 * j;
      double   h0 = ((double)x[k + 1] - (double)x[k]) * inv_xs;
      double   h1 = ((double)x[k + 2] - (double)x[k + 1]) * inv_xs;
      double   hs = h0 + h1;
      double   y0 = (double)y[i * n + k] * inv_ys;
      double   y1 = (double)y[i * n + k + 1] * inv_ys;
      double   y2 = (double)y[i * n + k + 2] * inv_ys;
      sum += hs / 6.0 * (y0 * (2.0 - h1 / h0) +
                          y1 * (hs * hs / (h0 * h1)) +
                          y2 * (2.0 - h0 / h1));
    }

    /* Even n (odd intervals): Cartwright correction on last 3 points. */
    if (n % 2 == 0) {
      double h0 = ((double)x[n - 2] - (double)x[n - 3]) * inv_xs;
      double h1 = ((double)x[n - 1] - (double)x[n - 2]) * inv_xs;
      double alpha = (2.0 * h1 * h1 + 3.0 * h0 * h1) / (6.0 * (h0 + h1));
      double beta  = (h1 * h1 + 3.0 * h0 * h1) / (6.0 * h0);
      double eta   = (h1 * h1 * h1) / (6.0 * h0 * (h0 + h1));
      sum += alpha * (double)y[i * n + n - 1] * inv_ys +
             beta  * (double)y[i * n + n - 2] * inv_ys -
             eta   * (double)y[i * n + n - 3] * inv_ys;
    }

    integral[i] = sum;
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
  Float *x = extract_data_float(xs, "xs");
  if (x == NULL) {
    return NULL; /* Type error already set */
  }

  /* Get the data pointer of the ys array */
  Float *y = extract_data_float(ys, "ys");
  if (y == NULL) {
    return NULL; /* Type error already set */
  }

  /* Number of elements excluding the last axis */
  npy_intp num_elements = PyArray_SIZE(ys) / n;

  /* Compute the integral with the appropriate function. */
  Float *integral;
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
      wrap_array_to_numpy<Float>(ndim - 1, result_shape, integral);

  /* Create the output object. */
  if (result == NULL) {
    free(integral); /* Free the allocated memory on error */
    return NULL;    /* Return NULL in case of error */
  }
  PyObject *output = (PyObject *)result;

  return output;
}

/**
 * @brief Scaled Simpson's integration over the final axis of an ND array.
 *
 * Fused version: same contract as trapz_last_axis_scaled_integration.
 * Returns a 2-tuple (float64 integral_array, scale).
 */
static PyObject *simps_last_axis_scaled_integration(PyObject *self,
                                                     PyObject *args) {
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

  if (!ensure_float_array(xs, "xs")) {
    return NULL;
  }
  if (!ensure_float_array(ys, "ys")) {
    return NULL;
  }

  Float *x = extract_data_float(xs, "xs");
  if (x == NULL) return NULL;
  Float *y = extract_data_float(ys, "ys");
  if (y == NULL) return NULL;

  npy_intp num_elements = PyArray_SIZE(ys) / n;

  double scale = 0.0;
  double *integral;
#ifdef WITH_OPENMP
  if (nthreads > 1) {
    integral = simps_last_axis_scaled_parallel(x, y, n, num_elements,
                                               nthreads, &scale);
  } else {
    integral = simps_last_axis_scaled_serial(x, y, n, num_elements, &scale);
  }
#else
  integral = simps_last_axis_scaled_serial(x, y, n, num_elements, &scale);
#endif

  /* Wrap the double result into a NumPy float64 array. */
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

  /* Return (integral_array, scale) tuple. */
  PyObject *scale_obj = PyFloat_FromDouble(scale);
  if (scale_obj == NULL) {
    Py_DECREF(result);
    return NULL;
  }
  PyObject *out = PyTuple_Pack(2, (PyObject *)result, scale_obj);
  Py_DECREF(result);
  Py_DECREF(scale_obj);
  return out;
}

static PyMethodDef IntegrationMethods[] = {
    {"trapz_last_axis", trapz_last_axis_integration, METH_VARARGS,
     "Trapezoidal integration with OpenMP"},
    {"trapz_last_axis_scaled", trapz_last_axis_scaled_integration, METH_VARARGS,
     "Scaled trapezoidal integration (fused scale+integrate) with OpenMP"},
    {"simps_last_axis", simps_last_axis_integration, METH_VARARGS,
     "Simpson's integration with OpenMP"},
    {"simps_last_axis_scaled", simps_last_axis_scaled_integration, METH_VARARGS,
     "Scaled Simpson's integration (fused scale+integrate) with OpenMP"},
    {"trapz_last_axis_weighted", trapz_last_axis_weighted_integration,
     METH_VARARGS,
     "Weighted scaled trapezoidal integration (fused y*w+scale+integrate)"},
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
  return PyModule_Create(&integrationmodule);
}
