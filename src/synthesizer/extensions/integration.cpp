/******************************************************************************
 * C extension for numerical integration operations.
 *
 * This module provides optimized integration methods (trapezoidal and
 * Simpson's rules) for integrating over the last axis of N-dimensional
 * arrays. All implementations use scaled arithmetic with double-precision
 * accumulation to avoid overflow/underflow issues common in astrophysical
 * unit systems.
 *
 * Key features:
 *   - Scaled integration: automatically finds max(|x|) and max(|y|) and
 *     normalizes before integration to prevent overflow
 *   - Double accumulation: even in SINGLE_PRECISION builds, accumulation
 *     is done in double to avoid precision loss
 *   - Weighted integration: fused kernels for computing ∫ y(x)*w(x) dx
 *     without intermediate array allocations
 *   - OpenMP parallelization: all methods have serial and parallel versions
 *
 * Note: We READ inputs as Float (float32 or float64 depending on build) but
 * always ACCUMULATE in double precision and return double results.
 *****************************************************************************/

/* Python includes */
#define PY_ARRAY_UNIQUE_SYMBOL SYNTHESIZER_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy_init.h"
#include <Python.h>

/* Optional OpenMP include */
#ifdef WITH_OPENMP
#include <omp.h>
#endif

/* Local includes */
#include "cpp_to_python.h"
#include "data_types.h"
#include "numpy_helpers.h"
#include "property_funcs.h"
#include "timers.h"
#ifdef ATOMIC_TIMING
#include "timers_init.h"
#endif

/**
 * @brief Integrate over the last axis using the trapezoidal rule (serial).
 *
 * Finds max(|x|) and max(|y|), normalizes, then integrates with double
 * precision accumulation. Returns the integral and scale factor separately.
 *
 * @param x: 1D x values in Float (length n).
 * @param y: ND y values in Float, row-major.
 * @param n: Length of last axis.
 * @param num_elements: Product of all axes except the last.
 * @param out_scale: Output scale factor (xscale * yscale).
 *
 * @return: Freshly-allocated double array of length num_elements.
 */
static double *trapz_last_axis_scaled_serial(Float *x, Float *y, npy_intp n,
                                             npy_intp num_elements,
                                             double *out_scale) {

  /* Find the maximum absolute value in x for scaling (1D scan) */
  double xscale = 0.0;
  for (npy_intp j = 0; j < n; ++j) {
    double ax = (double)x[j];
    if (ax < 0.0)
      ax = -ax;
    if (ax > xscale)
      xscale = ax;
  }

  /* Find the maximum absolute value in y for scaling (full array scan) */
  npy_intp total = num_elements * n;
  double yscale = 0.0;
  for (npy_intp k = 0; k < total; ++k) {
    double ay = (double)y[k];
    if (ay < 0.0)
      ay = -ay;
    if (ay > yscale)
      yscale = ay;
  }

  /* Output the combined scale factor for caller's reference. */
  *out_scale = xscale * yscale;

  /* If either scale is zero, the result is zero so move along. */
  if (xscale == 0.0 || yscale == 0.0) {
    return (double *)calloc(num_elements, sizeof(double));
  }

  /* Otherwise, there's work to do. Get the result and scaling ready. */
  double inv_xs = 1.0 / xscale;
  double inv_ys = 1.0 / yscale;
  double *integral = (double *)calloc(num_elements, sizeof(double));

  /* Compute the trapezoidal integral with double casting for stability. */
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
 * @brief Integrate over the last axis using the trapezoidal rule (parallel).
 *
 * OpenMP-parallelized version. Uses parallel reduction for finding max(|y|),
 * then parallel-for over rows. Each row writes to a distinct output slot
 * (no array reduction needed).
 *
 * @param x: 1D x values in Float (length n).
 * @param y: ND y values in Float, row-major.
 * @param n: Length of last axis.
 * @param num_elements: Product of all axes except the last.
 * @param nthreads: Number of OpenMP threads to use.
 * @param out_scale: Output scale factor (xscale * yscale).
 *
 * @return: Freshly-allocated double array of length num_elements.
 */
static double *trapz_last_axis_scaled_parallel(Float *x, Float *y, npy_intp n,
                                               npy_intp num_elements,
                                               int nthreads,
                                               double *out_scale) {

  /* Find the maximum absolute value in x for scaling (1D scan). Sufficiently
   * cheap to do in a simple loop. */
  double xscale = 0.0;
  for (npy_intp j = 0; j < n; ++j) {
    double ax = (double)x[j];
    if (ax < 0.0)
      ax = -ax;
    if (ax > xscale)
      xscale = ax;
  }

  /* Find the maximum absolute value in y for scaling (parallel reduction) */
  npy_intp total = num_elements * n;
  double yscale = 0.0;
#pragma omp parallel for num_threads(nthreads) reduction(max : yscale)
  for (npy_intp k = 0; k < total; ++k) {
    double ay = (double)y[k];
    if (ay < 0.0)
      ay = -ay;
    if (ay > yscale)
      yscale = ay;
  }

  /* Output the combined scale factor for caller's reference. */
  *out_scale = xscale * yscale;

  /* If either scale is zero, the result is zero so move along. */
  if (xscale == 0.0 || yscale == 0.0) {
    return (double *)calloc(num_elements, sizeof(double));
  }

  /* Otherwise, there's work to do. Get the result and scaling ready. */
  double inv_xs = 1.0 / xscale;
  double inv_ys = 1.0 / yscale;
  double *integral = (double *)calloc(num_elements, sizeof(double));

  /* Each row writes to integral[i] — no race conditions. Parallelize over
   * rows with OpenMP and compute the trapezoidal integral with double casting
   * for stability. */
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
 * @brief Python binding for scaled trapezoidal integration.
 *
 * Integrates over the last axis of an ND array using the trapezoidal rule
 * with automatic scaling to prevent overflow. Returns (integral, scale)
 * where the physical result is integral * scale.
 *
 * @param xs: 1D array of x values (Float dtype, C-contiguous).
 * @param ys: ND array of y values (Float dtype, C-contiguous).
 * @param nthreads: Number of threads to use.
 *
 * @return: Tuple (integral_array, scale) where integral_array is float64
 *          and scale is a Python float.
 */
static PyObject *trapz_last_axis_scaled_integration(PyObject *self,
                                                    PyObject *args) {
  (void)self;

  /* Parse Python arguments: two arrays and an integer. */
  PyArrayObject *xs, *ys;
  int nthreads;
  if (!PyArg_ParseTuple(args, "O!O!i", &PyArray_Type, &xs, &PyArray_Type, &ys,
                        &nthreads)) {
    return NULL;
  }

  /* Validate inputs and extract data pointers. */
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
  if (x == NULL)
    return NULL;
  Float *y = extract_data_float(ys, "ys");
  if (y == NULL)
    return NULL;
  npy_intp num_elements = PyArray_SIZE(ys) / n;

  /* Perform the integration with automatic scaling. */
  double scale = 0.0;
  double *integral;
#ifdef WITH_OPENMP
  if (nthreads > 1) {
    integral = trapz_last_axis_scaled_parallel(x, y, n, num_elements, nthreads,
                                               &scale);
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

/**
 * @brief Weighted trapezoidal integration over the last axis (serial).
 *
 * Computes ∫ y(x) * w(x) dx where w is a weight vector. Uses scaled
 * arithmetic with double accumulation for numerical stability.
 *
 * @param x: 1D x values (Float, length n).
 * @param y: ND y values (Float, row-major, last axis = n).
 * @param w: 1D weight vector (Float, length n).
 * @param n: Length of last axis.
 * @param num_elements: Product of all axes except the last.
 * @param out_scale: Output scale factor (xscale * yscale * wscale).
 *
 * @return: Freshly-allocated double array of length num_elements.
 */
static double *trapz_last_axis_weighted_serial(Float *x, Float *y, Float *w,
                                               npy_intp n,
                                               npy_intp num_elements,
                                               double *out_scale) {

  /* Find the maximum absolute value in x for scaling (1D scan) */
  double xscale = 0.0;
  for (npy_intp j = 0; j < n; ++j) {
    double ax = (double)x[j];
    if (ax < 0.0)
      ax = -ax;
    if (ax > xscale)
      xscale = ax;
  }

  /* Find the maximum absolute value in w for scaling (1D scan) */
  double wscale = 0.0;
  for (npy_intp j = 0; j < n; ++j) {
    double aw = (double)w[j];
    if (aw < 0.0)
      aw = -aw;
    if (aw > wscale)
      wscale = aw;
  }

  /* Find the maximum absolute value in y for scaling (full array scan) */
  npy_intp total = num_elements * n;
  double yscale = 0.0;
  for (npy_intp k = 0; k < total; ++k) {
    double ay = (double)y[k];
    if (ay < 0.0)
      ay = -ay;
    if (ay > yscale)
      yscale = ay;
  }

  /* Output the combined scale factor for caller's reference. */
  *out_scale = xscale * yscale * wscale;

  /* If any scale is zero, the result is zero so move along. */
  if (xscale == 0.0 || yscale == 0.0 || wscale == 0.0) {
    return (double *)calloc(num_elements, sizeof(double));
  }

  /* Otherwise, there's work to do. Get the result and scaling ready. */
  double inv_xs = 1.0 / xscale;
  double inv_ys = 1.0 / yscale;
  double inv_ws = 1.0 / wscale;
  double *integral = (double *)calloc(num_elements, sizeof(double));

  /* Compute the weighted trapezoidal integral with double casting for
   * stability. The integrand is y(x) * w(x). */
  for (npy_intp i = 0; i < num_elements; ++i) {
    double sum = 0.0;
    for (npy_intp j = 0; j < n - 1; ++j) {
      double dx = ((double)x[j + 1] - (double)x[j]) * inv_xs;
      double yw0 = (double)y[i * n + j] * inv_ys * (double)w[j] * inv_ws;
      double yw1 =
          (double)y[i * n + j + 1] * inv_ys * (double)w[j + 1] * inv_ws;
      sum += 0.5 * dx * (yw0 + yw1);
    }
    integral[i] = sum;
  }

  return integral;
}

#ifdef WITH_OPENMP
/**
 * @brief Weighted trapezoidal integration over the last axis (parallel).
 *
 * OpenMP-parallelized version of weighted integration. Computes
 * ∫ y(x) * w(x) dx with automatic scaling and double accumulation.
 *
 * @param x: 1D x values (Float, length n).
 * @param y: ND y values (Float, row-major, last axis = n).
 * @param w: 1D weight vector (Float, length n).
 * @param n: Length of last axis.
 * @param num_elements: Product of all axes except the last.
 * @param nthreads: Number of OpenMP threads to use.
 * @param out_scale: Output scale factor (xscale * yscale * wscale).
 *
 * @return: Freshly-allocated double array of length num_elements.
 */
static double *trapz_last_axis_weighted_parallel(Float *x, Float *y, Float *w,
                                                 npy_intp n,
                                                 npy_intp num_elements,
                                                 int nthreads,
                                                 double *out_scale) {

  /* Find the maximum absolute value in x for scaling (1D scan). Sufficiently
   * cheap to do in a simple loop. */
  double xscale = 0.0;
  for (npy_intp j = 0; j < n; ++j) {
    double ax = (double)x[j];
    if (ax < 0.0)
      ax = -ax;
    if (ax > xscale)
      xscale = ax;
  }

  /* Find the maximum absolute value in w for scaling (1D scan). Sufficiently
   * cheap to do in a simple loop. */
  double wscale = 0.0;
  for (npy_intp j = 0; j < n; ++j) {
    double aw = (double)w[j];
    if (aw < 0.0)
      aw = -aw;
    if (aw > wscale)
      wscale = aw;
  }

  /* Find the maximum absolute value in y for scaling (parallel reduction) */
  npy_intp total = num_elements * n;
  double yscale = 0.0;
#pragma omp parallel for num_threads(nthreads) reduction(max : yscale)
  for (npy_intp k = 0; k < total; ++k) {
    double ay = (double)y[k];
    if (ay < 0.0)
      ay = -ay;
    if (ay > yscale)
      yscale = ay;
  }

  /* Output the combined scale factor for caller's reference. */
  *out_scale = xscale * yscale * wscale;

  /* If any scale is zero, the result is zero so move along. */
  if (xscale == 0.0 || yscale == 0.0 || wscale == 0.0) {
    return (double *)calloc(num_elements, sizeof(double));
  }

  /* Otherwise, there's work to do. Get the result and scaling ready. */
  double inv_xs = 1.0 / xscale;
  double inv_ys = 1.0 / yscale;
  double inv_ws = 1.0 / wscale;
  double *integral = (double *)calloc(num_elements, sizeof(double));

  /* Each row writes to integral[i] — no race conditions. Parallelize over
   * rows with OpenMP and compute the weighted trapezoidal integral with
   * double casting for stability. The integrand is y(x) * w(x). */
#pragma omp parallel for num_threads(nthreads)
  for (npy_intp i = 0; i < num_elements; ++i) {
    double sum = 0.0;
    for (npy_intp j = 0; j < n - 1; ++j) {
      double dx = ((double)x[j + 1] - (double)x[j]) * inv_xs;
      double yw0 = (double)y[i * n + j] * inv_ys * (double)w[j] * inv_ws;
      double yw1 =
          (double)y[i * n + j + 1] * inv_ys * (double)w[j + 1] * inv_ws;
      sum += 0.5 * dx * (yw0 + yw1);
    }
    integral[i] = sum;
  }

  return integral;
}
#endif

/**
 * @brief Weighted Simpson's integration over the last axis (serial).
 *
 * Computes ∫ y(x) * w(x) dx using Simpson's rule with Cartwright correction
 * for even-length arrays. Uses scaled arithmetic with double accumulation.
 *
 * @param x: 1D x values (Float, length n).
 * @param y: ND y values (Float, row-major, last axis = n).
 * @param w: 1D weight vector (Float, length n).
 * @param n: Length of last axis.
 * @param num_elements: Product of all axes except the last.
 * @param out_scale: Output scale factor (xscale * yscale * wscale).
 *
 * @return: Freshly-allocated double array of length num_elements.
 */
static double *simps_last_axis_weighted_serial(Float *x, Float *y, Float *w,
                                               npy_intp n,
                                               npy_intp num_elements,
                                               double *out_scale) {

  /* Find the maximum absolute value in x for scaling (1D scan) */
  double xscale = 0.0;
  for (npy_intp j = 0; j < n; ++j) {
    double ax = (double)x[j];
    if (ax < 0.0)
      ax = -ax;
    if (ax > xscale)
      xscale = ax;
  }

  /* Find the maximum absolute value in w for scaling (1D scan) */
  double wscale = 0.0;
  for (npy_intp j = 0; j < n; ++j) {
    double aw = (double)w[j];
    if (aw < 0.0)
      aw = -aw;
    if (aw > wscale)
      wscale = aw;
  }

  /* Find the maximum absolute value in y for scaling (full array scan) */
  npy_intp total = num_elements * n;
  double yscale = 0.0;
  for (npy_intp k = 0; k < total; ++k) {
    double ay = (double)y[k];
    if (ay < 0.0)
      ay = -ay;
    if (ay > yscale)
      yscale = ay;
  }

  /* Output the combined scale factor for caller's reference. */
  *out_scale = xscale * yscale * wscale;

  /* If any scale is zero, the result is zero so move along. */
  if (xscale == 0.0 || yscale == 0.0 || wscale == 0.0) {
    return (double *)calloc(num_elements, sizeof(double));
  }

  /* Otherwise, there's work to do. Get the result and scaling ready. */
  double inv_xs = 1.0 / xscale;
  double inv_ys = 1.0 / yscale;
  double inv_ws = 1.0 / wscale;
  double *integral = (double *)calloc(num_elements, sizeof(double));

  /* Compute the weighted Simpson's integral with double casting for stability.
   * The integrand is y(x) * w(x). */
  for (npy_intp i = 0; i < num_elements; ++i) {
    if (n < 2)
      continue;

    /* n==2: only one interval, fall back to trapezoid (matches scipy). */
    if (n == 2) {
      integral[i] = 0.5 * ((double)x[1] - (double)x[0]) * inv_xs *
                    ((double)y[i * n] * (double)w[0] +
                     (double)y[i * n + 1] * (double)w[1]) *
                    inv_ys * inv_ws;
      continue;
    }

    double sum = 0.0;

    /* Generalised Simpson panels (correct for non-uniform spacing). */
    for (npy_intp j = 0; j < (n - 1) / 2; ++j) {
      npy_intp k = 2 * j;
      double h0 = ((double)x[k + 1] - (double)x[k]) * inv_xs;
      double h1 = ((double)x[k + 2] - (double)x[k + 1]) * inv_xs;
      double hs = h0 + h1;
      double yw0 = (double)y[i * n + k] * inv_ys * (double)w[k] * inv_ws;
      double yw1 =
          (double)y[i * n + k + 1] * inv_ys * (double)w[k + 1] * inv_ws;
      double yw2 =
          (double)y[i * n + k + 2] * inv_ys * (double)w[k + 2] * inv_ws;
      sum += hs / 6.0 *
             (yw0 * (2.0 - h1 / h0) + yw1 * (hs * hs / (h0 * h1)) +
              yw2 * (2.0 - h0 / h1));
    }

    /* Even n (odd intervals): Cartwright correction on last 3 points. */
    if (n % 2 == 0) {
      double h0 = ((double)x[n - 2] - (double)x[n - 3]) * inv_xs;
      double h1 = ((double)x[n - 1] - (double)x[n - 2]) * inv_xs;
      double alpha = (2.0 * h1 * h1 + 3.0 * h0 * h1) / (6.0 * (h0 + h1));
      double beta = (h1 * h1 + 3.0 * h0 * h1) / (6.0 * h0);
      double eta = (h1 * h1 * h1) / (6.0 * h0 * (h0 + h1));
      sum +=
          alpha * (double)y[i * n + n - 1] * inv_ys * (double)w[n - 1] *
              inv_ws +
          beta * (double)y[i * n + n - 2] * inv_ys * (double)w[n - 2] * inv_ws -
          eta * (double)y[i * n + n - 3] * inv_ys * (double)w[n - 3] * inv_ws;
    }

    integral[i] = sum;
  }

  return integral;
}

#ifdef WITH_OPENMP
/**
 * @brief Weighted Simpson's integration over the last axis (parallel).
 *
 * OpenMP-parallelized version of weighted Simpson's integration.
 * Handles non-uniform grids with Cartwright correction for even-length arrays.
 *
 * @param x: 1D x values (Float, length n).
 * @param y: ND y values (Float, row-major, last axis = n).
 * @param w: 1D weight vector (Float, length n).
 * @param n: Length of last axis.
 * @param num_elements: Product of all axes except the last.
 * @param nthreads: Number of OpenMP threads to use.
 * @param out_scale: Output scale factor (xscale * yscale * wscale).
 *
 * @return: Freshly-allocated double array of length num_elements.
 */
static double *simps_last_axis_weighted_parallel(Float *x, Float *y, Float *w,
                                                 npy_intp n,
                                                 npy_intp num_elements,
                                                 int nthreads,
                                                 double *out_scale) {

  /* Find the maximum absolute value in x for scaling (1D scan). Sufficiently
   * cheap to do in a simple loop. */
  double xscale = 0.0;
  for (npy_intp j = 0; j < n; ++j) {
    double ax = (double)x[j];
    if (ax < 0.0)
      ax = -ax;
    if (ax > xscale)
      xscale = ax;
  }

  /* Find the maximum absolute value in w for scaling (1D scan). Sufficiently
   * cheap to do in a simple loop. */
  double wscale = 0.0;
  for (npy_intp j = 0; j < n; ++j) {
    double aw = (double)w[j];
    if (aw < 0.0)
      aw = -aw;
    if (aw > wscale)
      wscale = aw;
  }

  /* Find the maximum absolute value in y for scaling (parallel reduction) */
  npy_intp total = num_elements * n;
  double yscale = 0.0;
#pragma omp parallel for num_threads(nthreads) reduction(max : yscale)
  for (npy_intp k = 0; k < total; ++k) {
    double ay = (double)y[k];
    if (ay < 0.0)
      ay = -ay;
    if (ay > yscale)
      yscale = ay;
  }

  /* Output the combined scale factor for caller's reference. */
  *out_scale = xscale * yscale * wscale;

  /* If any scale is zero, the result is zero so move along. */
  if (xscale == 0.0 || yscale == 0.0 || wscale == 0.0) {
    return (double *)calloc(num_elements, sizeof(double));
  }

  /* Otherwise, there's work to do. Get the result and scaling ready. */
  double inv_xs = 1.0 / xscale;
  double inv_ys = 1.0 / yscale;
  double inv_ws = 1.0 / wscale;
  double *integral = (double *)calloc(num_elements, sizeof(double));

  /* Each row writes to integral[i] — no race conditions. Parallelize over
   * rows with OpenMP and compute the weighted Simpson's integral with double
   * casting for stability. The integrand is y(x) * w(x). */
#pragma omp parallel for num_threads(nthreads)
  for (npy_intp i = 0; i < num_elements; ++i) {
    if (n < 2)
      continue;

    /* n==2: only one interval, fall back to trapezoid (matches scipy). */
    if (n == 2) {
      integral[i] = 0.5 * ((double)x[1] - (double)x[0]) * inv_xs *
                    ((double)y[i * n] * (double)w[0] +
                     (double)y[i * n + 1] * (double)w[1]) *
                    inv_ys * inv_ws;
      continue;
    }

    double sum = 0.0;

    /* Generalised Simpson panels (correct for non-uniform spacing). */
    for (npy_intp j = 0; j < (n - 1) / 2; ++j) {
      npy_intp k = 2 * j;
      double h0 = ((double)x[k + 1] - (double)x[k]) * inv_xs;
      double h1 = ((double)x[k + 2] - (double)x[k + 1]) * inv_xs;
      double hs = h0 + h1;
      double yw0 = (double)y[i * n + k] * inv_ys * (double)w[k] * inv_ws;
      double yw1 =
          (double)y[i * n + k + 1] * inv_ys * (double)w[k + 1] * inv_ws;
      double yw2 =
          (double)y[i * n + k + 2] * inv_ys * (double)w[k + 2] * inv_ws;
      sum += hs / 6.0 *
             (yw0 * (2.0 - h1 / h0) + yw1 * (hs * hs / (h0 * h1)) +
              yw2 * (2.0 - h0 / h1));
    }

    /* Even n (odd intervals): Cartwright correction on last 3 points. */
    if (n % 2 == 0) {
      double h0 = ((double)x[n - 2] - (double)x[n - 3]) * inv_xs;
      double h1 = ((double)x[n - 1] - (double)x[n - 2]) * inv_xs;
      double alpha = (2.0 * h1 * h1 + 3.0 * h0 * h1) / (6.0 * (h0 + h1));
      double beta = (h1 * h1 + 3.0 * h0 * h1) / (6.0 * h0);
      double eta = (h1 * h1 * h1) / (6.0 * h0 * (h0 + h1));
      sum +=
          alpha * (double)y[i * n + n - 1] * inv_ys * (double)w[n - 1] *
              inv_ws +
          beta * (double)y[i * n + n - 2] * inv_ys * (double)w[n - 2] * inv_ws -
          eta * (double)y[i * n + n - 3] * inv_ys * (double)w[n - 3] * inv_ws;
    }

    integral[i] = sum;
  }

  return integral;
}
#endif

/**
 * @brief Python binding for weighted trapezoidal integration.
 *
 * Computes ∫ y(x) * w(x) dx over the last axis with automatic scaling.
 * Returns (integral, scale) where the physical result is integral * scale.
 *
 * @param xs: 1D array of x values (Float dtype, C-contiguous).
 * @param ys: ND array of y values (Float dtype, C-contiguous).
 * @param ws: 1D weight vector (Float dtype, C-contiguous, same length as xs).
 * @param nthreads: Number of threads to use.
 *
 * @return: Tuple (integral_array, scale) where integral_array is float64
 *          and scale is a Python float.
 */
static PyObject *trapz_last_axis_weighted_integration(PyObject *self,
                                                      PyObject *args) {
  (void)self;

  /* Parse Python arguments: three arrays and an integer. */
  PyArrayObject *xs, *ys, *ws;
  int nthreads;
  if (!PyArg_ParseTuple(args, "O!O!O!i", &PyArray_Type, &xs, &PyArray_Type, &ys,
                        &PyArray_Type, &ws, &nthreads)) {
    return NULL;
  }

  /* Validate inputs and extract data pointers. */
  npy_intp ndim = PyArray_NDIM(ys);
  npy_intp *shape = PyArray_SHAPE(ys);
  npy_intp n = shape[ndim - 1];
  if (!ensure_float_array(xs, "xs"))
    return NULL;
  if (!ensure_float_array(ys, "ys"))
    return NULL;
  if (!ensure_float_array(ws, "ws"))
    return NULL;
  Float *x = extract_data_float(xs, "xs");
  if (x == NULL)
    return NULL;
  Float *y = extract_data_float(ys, "ys");
  if (y == NULL)
    return NULL;
  Float *w = extract_data_float(ws, "ws");
  if (w == NULL)
    return NULL;
  npy_intp num_elements = PyArray_SIZE(ys) / n;

  /* Perform the weighted integration with automatic scaling. */
  double scale = 0.0;
  double *integral;
#ifdef WITH_OPENMP
  if (nthreads > 1) {
    integral = trapz_last_axis_weighted_parallel(x, y, w, n, num_elements,
                                                 nthreads, &scale);
  } else {
    integral =
        trapz_last_axis_weighted_serial(x, y, w, n, num_elements, &scale);
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
 * @brief Integrate over the last axis using Simpson's rule (serial).
 *
 * Uses generalized Simpson's rule for non-uniform grids with Cartwright
 * correction for even-length arrays. Automatic scaling and double accumulation.
 *
 * @param x: 1D x values (Float, length n).
 * @param y: ND y values (Float, row-major, last axis = n).
 * @param n: Length of last axis.
 * @param num_elements: Product of all axes except the last.
 * @param out_scale: Output scale factor (xscale * yscale).
 *
 * @return: Freshly-allocated double array of length num_elements.
 */
static double *simps_last_axis_scaled_serial(Float *x, Float *y, npy_intp n,
                                             npy_intp num_elements,
                                             double *out_scale) {

  /* Find the maximum absolute value in x for scaling (1D scan) */
  double xscale = 0.0;
  for (npy_intp j = 0; j < n; ++j) {
    double ax = (double)x[j];
    if (ax < 0.0)
      ax = -ax;
    if (ax > xscale)
      xscale = ax;
  }

  /* Find the maximum absolute value in y for scaling (full array scan) */
  npy_intp total = num_elements * n;
  double yscale = 0.0;
  for (npy_intp k = 0; k < total; ++k) {
    double ay = (double)y[k];
    if (ay < 0.0)
      ay = -ay;
    if (ay > yscale)
      yscale = ay;
  }

  /* Output the combined scale factor for caller's reference. */
  *out_scale = xscale * yscale;

  /* If either scale is zero, the result is zero so move along. */
  if (xscale == 0.0 || yscale == 0.0) {
    return (double *)calloc(num_elements, sizeof(double));
  }

  /* Otherwise, there's work to do. Get the result and scaling ready. */
  double inv_xs = 1.0 / xscale;
  double inv_ys = 1.0 / yscale;
  double *integral = (double *)calloc(num_elements, sizeof(double));

  /* Compute Simpson's integral with double casting for stability. */
  for (npy_intp i = 0; i < num_elements; ++i) {
    if (n < 2)
      continue;

    /* n==2: only one interval, fall back to trapezoid (matches scipy). */
    if (n == 2) {
      integral[i] = 0.5 * ((double)x[1] - (double)x[0]) * inv_xs *
                    ((double)y[i * n] + (double)y[i * n + 1]) * inv_ys;
      continue;
    }

    double sum = 0.0;

    /* Generalised Simpson panels (correct for non-uniform spacing). */
    for (npy_intp j = 0; j < (n - 1) / 2; ++j) {
      npy_intp k = 2 * j;
      double h0 = ((double)x[k + 1] - (double)x[k]) * inv_xs;
      double h1 = ((double)x[k + 2] - (double)x[k + 1]) * inv_xs;
      double hs = h0 + h1;
      double y0 = (double)y[i * n + k] * inv_ys;
      double y1 = (double)y[i * n + k + 1] * inv_ys;
      double y2 = (double)y[i * n + k + 2] * inv_ys;
      sum += hs / 6.0 *
             (y0 * (2.0 - h1 / h0) + y1 * (hs * hs / (h0 * h1)) +
              y2 * (2.0 - h0 / h1));
    }

    /* Even n (odd intervals): Cartwright correction on last 3 points. */
    if (n % 2 == 0) {
      double h0 = ((double)x[n - 2] - (double)x[n - 3]) * inv_xs;
      double h1 = ((double)x[n - 1] - (double)x[n - 2]) * inv_xs;
      double alpha = (2.0 * h1 * h1 + 3.0 * h0 * h1) / (6.0 * (h0 + h1));
      double beta = (h1 * h1 + 3.0 * h0 * h1) / (6.0 * h0);
      double eta = (h1 * h1 * h1) / (6.0 * h0 * (h0 + h1));
      sum += alpha * (double)y[i * n + n - 1] * inv_ys +
             beta * (double)y[i * n + n - 2] * inv_ys -
             eta * (double)y[i * n + n - 3] * inv_ys;
    }

    integral[i] = sum;
  }

  return integral;
}

#ifdef WITH_OPENMP
/**
 * @brief Integrate over the last axis using Simpson's rule (parallel).
 *
 * OpenMP-parallelized version of Simpson's integration. Handles non-uniform
 * grids with Cartwright correction for even-length arrays.
 *
 * @param x: 1D x values (Float, length n).
 * @param y: ND y values (Float, row-major, last axis = n).
 * @param n: Length of last axis.
 * @param num_elements: Product of all axes except the last.
 * @param nthreads: Number of OpenMP threads to use.
 * @param out_scale: Output scale factor (xscale * yscale).
 *
 * @return: Freshly-allocated double array of length num_elements.
 */
static double *simps_last_axis_scaled_parallel(Float *x, Float *y, npy_intp n,
                                               npy_intp num_elements,
                                               int nthreads,
                                               double *out_scale) {

  /* Find the maximum absolute value in x for scaling (1D scan). Sufficiently
   * cheap to do in a simple loop. */
  double xscale = 0.0;
  for (npy_intp j = 0; j < n; ++j) {
    double ax = (double)x[j];
    if (ax < 0.0)
      ax = -ax;
    if (ax > xscale)
      xscale = ax;
  }

  /* Find the maximum absolute value in y for scaling (parallel reduction) */
  npy_intp total = num_elements * n;
  double yscale = 0.0;
#pragma omp parallel for num_threads(nthreads) reduction(max : yscale)
  for (npy_intp k = 0; k < total; ++k) {
    double ay = (double)y[k];
    if (ay < 0.0)
      ay = -ay;
    if (ay > yscale)
      yscale = ay;
  }

  /* Output the combined scale factor for caller's reference. */
  *out_scale = xscale * yscale;

  /* If either scale is zero, the result is zero so move along. */
  if (xscale == 0.0 || yscale == 0.0) {
    return (double *)calloc(num_elements, sizeof(double));
  }

  /* Otherwise, there's work to do. Get the result and scaling ready. */
  double inv_xs = 1.0 / xscale;
  double inv_ys = 1.0 / yscale;
  double *integral = (double *)calloc(num_elements, sizeof(double));

  /* Each row writes to integral[i] — no race conditions. Parallelize over
   * rows with OpenMP and compute Simpson's integral with double casting
   * for stability. */
#pragma omp parallel for num_threads(nthreads)
  for (npy_intp i = 0; i < num_elements; ++i) {
    if (n < 2)
      continue;

    /* n==2: only one interval, fall back to trapezoid (matches scipy). */
    if (n == 2) {
      integral[i] = 0.5 * ((double)x[1] - (double)x[0]) * inv_xs *
                    ((double)y[i * n] + (double)y[i * n + 1]) * inv_ys;
      continue;
    }

    double sum = 0.0;

    /* Generalised Simpson panels (correct for non-uniform spacing). */
    for (npy_intp j = 0; j < (n - 1) / 2; ++j) {
      npy_intp k = 2 * j;
      double h0 = ((double)x[k + 1] - (double)x[k]) * inv_xs;
      double h1 = ((double)x[k + 2] - (double)x[k + 1]) * inv_xs;
      double hs = h0 + h1;
      double y0 = (double)y[i * n + k] * inv_ys;
      double y1 = (double)y[i * n + k + 1] * inv_ys;
      double y2 = (double)y[i * n + k + 2] * inv_ys;
      sum += hs / 6.0 *
             (y0 * (2.0 - h1 / h0) + y1 * (hs * hs / (h0 * h1)) +
              y2 * (2.0 - h0 / h1));
    }

    /* Even n (odd intervals): Cartwright correction on last 3 points. */
    if (n % 2 == 0) {
      double h0 = ((double)x[n - 2] - (double)x[n - 3]) * inv_xs;
      double h1 = ((double)x[n - 1] - (double)x[n - 2]) * inv_xs;
      double alpha = (2.0 * h1 * h1 + 3.0 * h0 * h1) / (6.0 * (h0 + h1));
      double beta = (h1 * h1 + 3.0 * h0 * h1) / (6.0 * h0);
      double eta = (h1 * h1 * h1) / (6.0 * h0 * (h0 + h1));
      sum += alpha * (double)y[i * n + n - 1] * inv_ys +
             beta * (double)y[i * n + n - 2] * inv_ys -
             eta * (double)y[i * n + n - 3] * inv_ys;
    }

    integral[i] = sum;
  }

  return integral;
}
#endif

/**
 * @brief Python binding for scaled Simpson's integration.
 *
 * Integrates over the last axis of an ND array using Simpson's rule
 * with automatic scaling. Returns (integral, scale) where the physical
 * result is integral * scale.
 *
 * @param xs: 1D array of x values (Float dtype, C-contiguous).
 * @param ys: ND array of y values (Float dtype, C-contiguous).
 * @param nthreads: Number of threads to use.
 *
 * @return: Tuple (integral_array, scale) where integral_array is float64
 *          and scale is a Python float.
 */
static PyObject *simps_last_axis_scaled_integration(PyObject *self,
                                                    PyObject *args) {
  (void)self;

  /* Parse Python arguments: two arrays and an integer. */
  PyArrayObject *xs, *ys;
  int nthreads;
  if (!PyArg_ParseTuple(args, "O!O!i", &PyArray_Type, &xs, &PyArray_Type, &ys,
                        &nthreads)) {
    return NULL;
  }

  /* Validate inputs and extract data pointers. */
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
  if (x == NULL)
    return NULL;
  Float *y = extract_data_float(ys, "ys");
  if (y == NULL)
    return NULL;
  npy_intp num_elements = PyArray_SIZE(ys) / n;

  /* Perform the integration with automatic scaling. */
  double scale = 0.0;
  double *integral;
#ifdef WITH_OPENMP
  if (nthreads > 1) {
    integral = simps_last_axis_scaled_parallel(x, y, n, num_elements, nthreads,
                                               &scale);
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

/**
 * @brief Python binding for weighted Simpson's integration.
 *
 * Computes ∫ y(x) * w(x) dx over the last axis using Simpson's rule with
 * automatic scaling. Returns (integral, scale) where the physical result
 * is integral * scale.
 *
 * @param xs: 1D array of x values (Float dtype, C-contiguous).
 * @param ys: ND array of y values (Float dtype, C-contiguous).
 * @param ws: 1D weight vector (Float dtype, C-contiguous, same length as xs).
 * @param nthreads: Number of threads to use.
 *
 * @return: Tuple (integral_array, scale) where integral_array is float64
 *          and scale is a Python float.
 */
static PyObject *simps_last_axis_weighted_integration(PyObject *self,
                                                      PyObject *args) {
  (void)self;

  /* Parse Python arguments: three arrays and an integer. */
  PyArrayObject *xs, *ys, *ws;
  int nthreads;
  if (!PyArg_ParseTuple(args, "O!O!O!i", &PyArray_Type, &xs, &PyArray_Type, &ys,
                        &PyArray_Type, &ws, &nthreads)) {
    return NULL;
  }

  /* Validate inputs and extract data pointers. */
  npy_intp ndim = PyArray_NDIM(ys);
  npy_intp *shape = PyArray_SHAPE(ys);
  npy_intp n = shape[ndim - 1];
  if (!ensure_float_array(xs, "xs"))
    return NULL;
  if (!ensure_float_array(ys, "ys"))
    return NULL;
  if (!ensure_float_array(ws, "ws"))
    return NULL;
  Float *x = extract_data_float(xs, "xs");
  if (x == NULL)
    return NULL;
  Float *y = extract_data_float(ys, "ys");
  if (y == NULL)
    return NULL;
  Float *w = extract_data_float(ws, "ws");
  if (w == NULL)
    return NULL;
  npy_intp num_elements = PyArray_SIZE(ys) / n;

  /* Perform the weighted integration with automatic scaling. */
  double scale = 0.0;
  double *integral;
#ifdef WITH_OPENMP
  if (nthreads > 1) {
    integral = simps_last_axis_weighted_parallel(x, y, w, n, num_elements,
                                                 nthreads, &scale);
  } else {
    integral =
        simps_last_axis_weighted_serial(x, y, w, n, num_elements, &scale);
  }
#else
  integral = simps_last_axis_weighted_serial(x, y, w, n, num_elements, &scale);
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

/* Below is all the gubbins needed to make the module importable in Python. */
static PyMethodDef IntegrationMethods[] = {
    {"trapz_last_axis_scaled", trapz_last_axis_scaled_integration, METH_VARARGS,
     "Scaled trapezoidal integration with OpenMP parallelization."},
    {"simps_last_axis_scaled", simps_last_axis_scaled_integration, METH_VARARGS,
     "Scaled Simpson's integration with OpenMP parallelization."},
    {"trapz_last_axis_weighted", trapz_last_axis_weighted_integration,
     METH_VARARGS,
     "Weighted scaled trapezoidal integration with OpenMP parallelization."},
    {"simps_last_axis_weighted", simps_last_axis_weighted_integration,
     METH_VARARGS,
     "Weighted scaled Simpson's integration with OpenMP parallelization."},
    {NULL, NULL, 0, NULL}};

/* Make this importable. */
static struct PyModuleDef integrationmodule = {
    PyModuleDef_HEAD_INIT,
    "integration",                                              /* m_name */
    "A module for optimized numerical integration operations.", /* m_doc */
    -1,                                                         /* m_size */
    IntegrationMethods,                                         /* m_methods */
    NULL,                                                       /* m_reload */
    NULL,                                                       /* m_traverse */
    NULL,                                                       /* m_clear */
    NULL,                                                       /* m_free */
};

PyMODINIT_FUNC PyInit_integration(void) {
  PyObject *m = PyModule_Create(&integrationmodule);
  if (m == NULL)
    return NULL;
  if (numpy_import() < 0) {
    Py_DECREF(m);
    PyErr_SetString(PyExc_RuntimeError, "Failed to import numpy.");
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
