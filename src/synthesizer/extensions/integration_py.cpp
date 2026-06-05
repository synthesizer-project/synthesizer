/******************************************************************************
 * Python bindings for the shared integration helpers.
 *
 * These wrappers validate NumPy inputs at the Python/C boundary, dispatch to
 * typed serial or OpenMP kernels, and return newly allocated NumPy arrays with
 * the requested floating-point output precision.
 *****************************************************************************/
#define PY_ARRAY_UNIQUE_SYMBOL SYNTHESIZER_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy_init.h"

#include <Python.h>
#include <cmath>
#include <new>

#ifdef WITH_OPENMP
#include <omp.h>
#endif

/* Local includes. */
#include "cpp_to_python.h"
#include "integration.h"
#include "python_to_cpp.h"
#include "timers.h"
#ifdef ATOMIC_TIMING
#include "timers_init.h"
#endif

/**
 * @brief Serial trapezoidal integration over the final axis.
 *
 * @tparam XReal The floating-point type of the x array.
 * @tparam YReal The floating-point type of the y array.
 * @tparam OutT The floating-point type stored in the returned buffer.
 *
 * @param x 1D integration grid shared by every row of y.
 * @param y Flattened ND input array to integrate over the final axis.
 * @param n Number of samples along the integrated axis.
 * @param num_elements Number of leading-axis entries to integrate.
 *
 * @return Newly allocated output buffer, or NULL on allocation failure.
 */
template <typename XReal, typename YReal, typename OutT>
static OutT *trapz_last_axis_serial(const XReal *x, const YReal *y, npy_intp n,
                                    npy_intp num_elements) {
  if (num_elements == 0) {
    return new (std::nothrow) OutT[1]();
  }

  OutT *integral = new (std::nothrow) OutT[num_elements]();
  if (integral == NULL) {
    return NULL;
  }

  for (npy_intp i = 0; i < num_elements; ++i) {
    integral[i] =
        trapz_1d<XReal, YReal, OutT>(x, y + i * n, static_cast<size_t>(n));
  }

  return integral;
}

/**
 * @brief Parallel trapezoidal integration over the final axis.
 *
 * The work is parallelised over the flattened leading axes so each thread owns
 * independent output elements.
 *
 * @tparam XReal The floating-point type of the x array.
 * @tparam YReal The floating-point type of the y array.
 * @tparam OutT The floating-point type stored in the returned buffer.
 *
 * @param x 1D integration grid shared by every row of y.
 * @param y Flattened ND input array to integrate over the final axis.
 * @param n Number of samples along the integrated axis.
 * @param num_elements Number of leading-axis entries to integrate.
 * @param nthreads Number of OpenMP threads to use.
 *
 * @return Newly allocated output buffer, or NULL on allocation failure.
 */
#ifdef WITH_OPENMP
template <typename XReal, typename YReal, typename OutT>
static OutT *trapz_last_axis_parallel(const XReal *x, const YReal *y,
                                      npy_intp n, npy_intp num_elements,
                                      int nthreads) {
  if (num_elements == 0) {
    return new (std::nothrow) OutT[1]();
  }

  OutT *integral = new (std::nothrow) OutT[num_elements]();
  if (integral == NULL) {
    return NULL;
  }

#pragma omp parallel for num_threads(nthreads)
  for (npy_intp i = 0; i < num_elements; ++i) {
    integral[i] =
        trapz_1d<XReal, YReal, OutT>(x, y + i * n, static_cast<size_t>(n));
  }
  return integral;
}
#endif

/**
 * @brief Wrap an integration result with the leading input shape.
 *
 * @tparam OutT: The floating-point type stored in the output array.
 *
 * @param result_arr: The raw output buffer.
 * @param ndim: The dimensionality of the input array.
 * @param shape: The shape of the input array.
 * @return The wrapped NumPy array, or NULL on failure.
 */
template <typename OutT>
static PyArrayObject *wrap_last_axis_result(OutT *result_arr, npy_intp ndim,
                                            npy_intp *shape) {
  /* Drop the final integrated axis when building the Python result shape. */
  npy_intp result_shape[NPY_MAXDIMS];
  for (npy_intp i = 0; i < ndim - 1; ++i) {
    result_shape[i] = shape[i];
  }

  PyArrayObject *result =
      wrap_array_to_numpy<OutT>(ndim - 1, result_shape, result_arr);
  if (result == NULL) {
    return NULL;
  }

  return result;
}

/**
 * @brief Execute trapezoidal integration after dtype dispatch.
 *
 * @tparam XReal The floating-point type of the x array.
 * @tparam YReal The floating-point type of the y array.
 * @tparam OutT The requested floating-point output type.
 *
 * @param xs 1D NumPy array containing the integration grid.
 * @param ys ND NumPy array whose final axis is integrated.
 * @param ndim Dimensionality of ys.
 * @param shape Shape of ys.
 * @param n Length of the final integrated axis.
 * @param num_elements Number of leading-axis entries to integrate.
 * @param nthreads Number of OpenMP threads to use when available.
 *
 * @return The wrapped NumPy output array, or NULL on failure.
 */
template <typename XReal, typename YReal, typename OutT>
static PyObject *trapz_last_axis(PyArrayObject *xs, PyArrayObject *ys,
                                 npy_intp ndim, npy_intp *shape, npy_intp n,
                                 npy_intp num_elements, int nthreads) {
  const XReal *x = data_ptr<const XReal>(xs);
  const YReal *y = data_ptr<const YReal>(ys);

  /* Select the serial or OpenMP kernel once the dtype family is known. */
  OutT *integral;
#ifdef WITH_OPENMP
  if (nthreads > 1) {
    integral = trapz_last_axis_parallel<XReal, YReal, OutT>(
        x, y, n, num_elements, nthreads);
  } else {
    integral =
        trapz_last_axis_serial<XReal, YReal, OutT>(x, y, n, num_elements);
  }
#else
  integral = trapz_last_axis_serial<XReal, YReal, OutT>(x, y, n, num_elements);
#endif

  if (integral == NULL) {
    PyErr_NoMemory();
    return NULL;
  }

  return (PyObject *)wrap_last_axis_result<OutT>(integral, ndim, shape);
}

/**
 * @brief Python wrapper for trapezoidal last-axis integration.
 *
 * Python passes NumPy arrays and a requested output dtype into this boundary
 * function. The wrapper validates shape, contiguity, and floating precision,
 * then dispatches to the matching typed kernel.
 *
 * @param self The module instance. (Unused)
 * @param args Python arguments containing xs, ys, nthreads, and out_dtype.
 *
 * @return A NumPy array with shape ys.shape[:-1], or NULL on failure.
 */
static PyObject *trapz_last_axis_integration(PyObject *self, PyObject *args) {

  (void)self;

  PyArrayObject *xs, *ys;
  int nthreads;
  PyObject *out_dtype;

  if (!PyArg_ParseTuple(args, "O!O!iO", &PyArray_Type, &xs, &PyArray_Type, &ys,
                        &nthreads, &out_dtype)) {
    return NULL;
  }

  if (PyArray_NDIM(xs) != 1) {
    PyErr_SetString(PyExc_ValueError, "xs must be a 1D array.");
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

  if (PyArray_DIM(xs, 0) != n) {
    PyErr_SetString(PyExc_ValueError,
                    "xs must match ys along the final axis.");
    return NULL;
  }

  const int xs_typenum = PyArray_TYPE(xs);
  const int ys_typenum = PyArray_TYPE(ys);
  if (!is_supported_float_typenum(xs_typenum) ||
      !is_supported_float_typenum(ys_typenum)) {
    PyErr_SetString(PyExc_TypeError, "xs and ys must be float32 or float64.");
    return NULL;
  }
  if (!is_c_contiguous(xs, "xs") || !is_c_contiguous(ys, "ys")) {
    return NULL;
  }

  /* Resolve the independently requested output dtype. */
  const int output_typenum = resolve_output_typenum(out_dtype, "out_dtype");
  if (output_typenum < 0) {
    return NULL;
  }

  /* Dispatch: encode xs/ys/output precision into a 3-bit key. */
  const npy_intp num_elements = PyArray_SIZE(ys) / n;
  int dispatch_key = ((xs_typenum == NPY_FLOAT64) << 2) |
                     ((ys_typenum == NPY_FLOAT64) << 1) |
                     (output_typenum == NPY_FLOAT64);

  /* Dispatch: call the matching typed kernel based on the dispatch key. */
  PyObject *result;
  switch (dispatch_key) {
    case 0:
      result = trapz_last_axis<float, float, float>(xs, ys, ndim, shape, n,
                                                    num_elements, nthreads);
      break;
    case 1:
      result = trapz_last_axis<float, float, double>(xs, ys, ndim, shape, n,
                                                     num_elements, nthreads);
      break;
    case 2:
      result = trapz_last_axis<float, double, float>(xs, ys, ndim, shape, n,
                                                     num_elements, nthreads);
      break;
    case 3:
      result = trapz_last_axis<float, double, double>(xs, ys, ndim, shape, n,
                                                      num_elements, nthreads);
      break;
    case 4:
      result = trapz_last_axis<double, float, float>(xs, ys, ndim, shape, n,
                                                     num_elements, nthreads);
      break;
    case 5:
      result = trapz_last_axis<double, float, double>(xs, ys, ndim, shape, n,
                                                      num_elements, nthreads);
      break;
    case 6:
      result = trapz_last_axis<double, double, float>(xs, ys, ndim, shape, n,
                                                      num_elements, nthreads);
      break;
    default:
      result = trapz_last_axis<double, double, double>(xs, ys, ndim, shape, n,
                                                       num_elements, nthreads);
      break;
  }
  return result;
}

/**
 * @brief Serial Simpson integration over the final axis.
 *
 * @tparam XReal The floating-point type of the x array.
 * @tparam YReal The floating-point type of the y array.
 * @tparam OutT The floating-point type stored in the returned buffer.
 *
 * @param x 1D integration grid shared by every row of y.
 * @param y Flattened ND input array to integrate over the final axis.
 * @param n Number of samples along the integrated axis.
 * @param num_elements Number of leading-axis entries to integrate.
 *
 * @return Newly allocated output buffer, or NULL on allocation failure.
 */
template <typename XReal, typename YReal, typename OutT>
static OutT *simps_last_axis_serial(const XReal *x, const YReal *y, npy_intp n,
                                    npy_intp num_elements) {
  if (num_elements == 0) {
    return new (std::nothrow) OutT[1]();
  }

  OutT *integral = new (std::nothrow) OutT[num_elements]();
  if (integral == NULL) {
    return NULL;
  }

  for (npy_intp i = 0; i < num_elements; ++i) {
    integral[i] =
        simps_1d<XReal, YReal, OutT>(x, y + i * n, static_cast<size_t>(n));
  }

  return integral;
}

/**
 * @brief Parallel Simpson integration over the final axis.
 *
 * @tparam XReal The floating-point type of the x array.
 * @tparam YReal The floating-point type of the y array.
 * @tparam OutT The floating-point type stored in the returned buffer.
 *
 * @param x 1D integration grid shared by every row of y.
 * @param y Flattened ND input array to integrate over the final axis.
 * @param n Number of samples along the integrated axis.
 * @param num_elements Number of leading-axis entries to integrate.
 * @param nthreads Number of OpenMP threads to use.
 *
 * @return Newly allocated output buffer, or NULL on allocation failure.
 */
#ifdef WITH_OPENMP
template <typename XReal, typename YReal, typename OutT>
static OutT *simps_last_axis_parallel(const XReal *x, const YReal *y,
                                      npy_intp n, npy_intp num_elements,
                                      int nthreads) {
  if (num_elements == 0) {
    return new (std::nothrow) OutT[1]();
  }

  OutT *integral = new (std::nothrow) OutT[num_elements]();
  if (integral == NULL) {
    return NULL;
  }

#pragma omp parallel for num_threads(nthreads)
  for (npy_intp i = 0; i < num_elements; ++i) {
    integral[i] =
        simps_1d<XReal, YReal, OutT>(x, y + i * n, static_cast<size_t>(n));
  }

  return integral;
}
#endif

/**
 * @brief Execute Simpson integration after dtype dispatch.
 *
 * @tparam XReal The floating-point type of the x array.
 * @tparam YReal The floating-point type of the y array.
 * @tparam OutT The requested floating-point output type.
 *
 * @param xs 1D NumPy array containing the integration grid.
 * @param ys ND NumPy array whose final axis is integrated.
 * @param ndim Dimensionality of ys.
 * @param shape Shape of ys.
 * @param n Length of the final integrated axis.
 * @param num_elements Number of leading-axis entries to integrate.
 * @param nthreads Number of OpenMP threads to use when available.
 *
 * @return The wrapped NumPy output array, or NULL on failure.
 */
template <typename XReal, typename YReal, typename OutT>
static PyObject *simps_last_axis(PyArrayObject *xs, PyArrayObject *ys,
                                 npy_intp ndim, npy_intp *shape, npy_intp n,
                                 npy_intp num_elements, int nthreads) {
  const XReal *x = data_ptr<const XReal>(xs);
  const YReal *y = data_ptr<const YReal>(ys);

  OutT *integral;
#ifdef WITH_OPENMP
  if (nthreads > 1) {
    integral = simps_last_axis_parallel<XReal, YReal, OutT>(
        x, y, n, num_elements, nthreads);
  } else {
    integral =
        simps_last_axis_serial<XReal, YReal, OutT>(x, y, n, num_elements);
  }
#else
  integral = simps_last_axis_serial<XReal, YReal, OutT>(x, y, n, num_elements);
#endif

  if (integral == NULL) {
    PyErr_NoMemory();
    return NULL;
  }

  return (PyObject *)wrap_last_axis_result<OutT>(integral, ndim, shape);
}

/**
 * @brief Python wrapper for Simpson last-axis integration.
 *
 * @param self The module instance. (Unused)
 * @param args Python arguments containing xs, ys, nthreads, and out_dtype.
 *
 * @return A NumPy array with shape ys.shape[:-1], or NULL on failure.
 */
static PyObject *simps_last_axis_integration(PyObject *self, PyObject *args) {
  (void)self;

  PyArrayObject *xs, *ys;
  int nthreads;
  PyObject *out_dtype;

  if (!PyArg_ParseTuple(args, "O!O!iO", &PyArray_Type, &xs, &PyArray_Type, &ys,
                        &nthreads, &out_dtype)) {
    return NULL;
  }

  if (PyArray_NDIM(xs) != 1) {
    PyErr_SetString(PyExc_ValueError, "xs must be a 1D array.");
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

  if (PyArray_DIM(xs, 0) != n) {
    PyErr_SetString(PyExc_ValueError,
                    "xs must match ys along the final axis.");
    return NULL;
  }

  const int xs_typenum = PyArray_TYPE(xs);
  const int ys_typenum = PyArray_TYPE(ys);
  if (!is_supported_float_typenum(xs_typenum) ||
      !is_supported_float_typenum(ys_typenum)) {
    PyErr_SetString(PyExc_TypeError, "xs and ys must be float32 or float64.");
    return NULL;
  }
  if (!is_c_contiguous(xs, "xs") || !is_c_contiguous(ys, "ys")) {
    return NULL;
  }

  /* Resolve the requested output dtype independently from the inputs. */
  const int output_typenum = resolve_output_typenum(out_dtype, "out_dtype");
  if (output_typenum < 0) {
    return NULL;
  }

  /* Dispatch: encode xs/ys/output precision into a 3-bit key. */
  const npy_intp num_elements = PyArray_SIZE(ys) / n;
  int dispatch_key = ((xs_typenum == NPY_FLOAT64) << 2) |
                     ((ys_typenum == NPY_FLOAT64) << 1) |
                     (output_typenum == NPY_FLOAT64);

  /* Dispatch: call the matching typed kernel based on the dispatch key. */
  PyObject *result;
  switch (dispatch_key) {
    case 0:
      result = simps_last_axis<float, float, float>(xs, ys, ndim, shape, n,
                                                    num_elements, nthreads);
      break;
    case 1:
      result = simps_last_axis<float, float, double>(xs, ys, ndim, shape, n,
                                                     num_elements, nthreads);
      break;
    case 2:
      result = simps_last_axis<float, double, float>(xs, ys, ndim, shape, n,
                                                     num_elements, nthreads);
      break;
    case 3:
      result = simps_last_axis<float, double, double>(xs, ys, ndim, shape, n,
                                                      num_elements, nthreads);
      break;
    case 4:
      result = simps_last_axis<double, float, float>(xs, ys, ndim, shape, n,
                                                     num_elements, nthreads);
      break;
    case 5:
      result = simps_last_axis<double, float, double>(xs, ys, ndim, shape, n,
                                                      num_elements, nthreads);
      break;
    case 6:
      result = simps_last_axis<double, double, float>(xs, ys, ndim, shape, n,
                                                      num_elements, nthreads);
      break;
    default:
      result = simps_last_axis<double, double, double>(xs, ys, ndim, shape, n,
                                                       num_elements, nthreads);
      break;
  }
  return result;
}

/**
 * @brief Serial weighted trapezoidal integration over the final axis.
 *
 * This computes the weighted mean over the final axis as
 *
 *   integral(y * w, x) / integral(w, x)
 *
 * using the trapezoidal rule for both numerator and denominator.
 *
 * @tparam Real The floating-point type of the input arrays.
 * @tparam OutT The floating-point type stored in the returned buffer.
 *
 * @param x 1D integration grid.
 * @param y Flattened ND values array.
 * @param w 1D weights defined on x.
 * @param n Number of samples along the integrated axis.
 * @param num_elements Number of leading-axis entries to integrate.
 *
 * @return Newly allocated output buffer, or NULL on allocation failure.
 */
template <typename Real, typename OutT>
static OutT *weighted_trapz_last_axis_serial(const Real *x, const Real *y,
                                             const Real *w, npy_intp n,
                                             npy_intp num_elements) {
  if (num_elements == 0) {
    return new (std::nothrow) OutT[1]();
  }

  OutT *result = new (std::nothrow) OutT[num_elements]();
  if (result == NULL) {
    return NULL;
  }

  /* Precompute the shared denominator once for all rows. */
  OutT den = static_cast<OutT>(0.0);
  for (npy_intp j = 0; j < n - 1; ++j) {
    den += static_cast<OutT>(0.5) * static_cast<OutT>(x[j + 1] - x[j]) *
           (static_cast<OutT>(w[j + 1]) + static_cast<OutT>(w[j]));
  }

  /* A zero denominator implies a zero-filled weighted average. */
  if (den == static_cast<OutT>(0.0)) {
    return result;
  }

  for (npy_intp i = 0; i < num_elements; ++i) {
    OutT num = static_cast<OutT>(0.0);
    for (npy_intp j = 0; j < n - 1; ++j) {
      num +=
          static_cast<OutT>(0.5) * static_cast<OutT>(x[j + 1] - x[j]) *
          (static_cast<OutT>(y[i * n + j + 1]) * static_cast<OutT>(w[j + 1]) +
           static_cast<OutT>(y[i * n + j]) * static_cast<OutT>(w[j]));
    }
    result[i] = num / den;
  }

  return result;
}

/**
 * @brief Parallel weighted trapezoidal integration over the final axis.
 *
 * The denominator is shared across all rows, so it is computed once before the
 * parallel loop over the flattened leading axes.
 *
 * @tparam Real The floating-point type of the input arrays.
 * @tparam OutT The floating-point type stored in the returned buffer.
 *
 * @param x 1D integration grid.
 * @param y Flattened ND values array.
 * @param w 1D weights defined on x.
 * @param n Number of samples along the integrated axis.
 * @param num_elements Number of leading-axis entries to integrate.
 * @param nthreads Number of OpenMP threads to use.
 *
 * @return Newly allocated output buffer, or NULL on allocation failure.
 */
#ifdef WITH_OPENMP
template <typename Real, typename OutT>
static OutT *weighted_trapz_last_axis_parallel(const Real *x, const Real *y,
                                               const Real *w, npy_intp n,
                                               npy_intp num_elements,
                                               int nthreads) {
  if (num_elements == 0) {
    return new (std::nothrow) OutT[1]();
  }

  OutT *result = new (std::nothrow) OutT[num_elements]();
  if (result == NULL) {
    return NULL;
  }

  OutT den = static_cast<OutT>(0.0);
  for (npy_intp j = 0; j < n - 1; ++j) {
    den += static_cast<OutT>(0.5) * static_cast<OutT>(x[j + 1] - x[j]) *
           (static_cast<OutT>(w[j + 1]) + static_cast<OutT>(w[j]));
  }

  if (den == static_cast<OutT>(0.0)) {
    return result;
  }

#pragma omp parallel for num_threads(nthreads)
  for (npy_intp i = 0; i < num_elements; ++i) {
    OutT num = static_cast<OutT>(0.0);
    for (npy_intp j = 0; j < n - 1; ++j) {
      num +=
          static_cast<OutT>(0.5) * static_cast<OutT>(x[j + 1] - x[j]) *
          (static_cast<OutT>(y[i * n + j + 1]) * static_cast<OutT>(w[j + 1]) +
           static_cast<OutT>(y[i * n + j]) * static_cast<OutT>(w[j]));
    }
    result[i] = num / den;
  }

  return result;
}
#endif

/**
 * @brief Execute weighted trapezoidal integration after dtype dispatch.
 *
 * @tparam Real The floating-point type of the validated input arrays.
 * @tparam OutT The requested floating-point output type.
 *
 * @param xs 1D NumPy integration grid.
 * @param ys ND NumPy values array.
 * @param ws 1D NumPy weights array.
 * @param ndim Dimensionality of ys.
 * @param shape Shape of ys.
 * @param n Length of the final integrated axis.
 * @param num_elements Number of leading-axis entries to integrate.
 * @param nthreads Number of OpenMP threads to use when available.
 *
 * @return The wrapped NumPy output array, or NULL on failure.
 */
template <typename Real, typename OutT>
static PyObject *weighted_trapz_last_axis(PyArrayObject *xs, PyArrayObject *ys,
                                          PyArrayObject *ws, npy_intp ndim,
                                          npy_intp *shape, npy_intp n,
                                          npy_intp num_elements,
                                          int nthreads) {
  const Real *x = data_ptr<const Real>(xs);
  const Real *y = data_ptr<const Real>(ys);
  const Real *w = data_ptr<const Real>(ws);

  OutT *result_arr;
#ifdef WITH_OPENMP
  if (nthreads > 1) {
    result_arr = weighted_trapz_last_axis_parallel<Real, OutT>(
        x, y, w, n, num_elements, nthreads);
  } else {
    result_arr =
        weighted_trapz_last_axis_serial<Real, OutT>(x, y, w, n, num_elements);
  }
#else
  result_arr =
      weighted_trapz_last_axis_serial<Real, OutT>(x, y, w, n, num_elements);
#endif

  if (result_arr == NULL) {
    PyErr_SetString(PyExc_MemoryError,
                    "Failed to allocate output for weighted trapz.");
    return NULL;
  }

  return (PyObject *)wrap_last_axis_result<OutT>(result_arr, ndim, shape);
}

/**
 * @brief Python wrapper for weighted trapezoidal last-axis integration.
 *
 * @param self The module instance. (Unused)
 * @param args Python arguments containing xs, ys, weights, nthreads, and
 *        out_dtype.
 *
 * @return A NumPy array with shape ys.shape[:-1], or NULL on failure.
 */
static PyObject *weighted_trapz_last_axis_integration(PyObject *self,
                                                      PyObject *args) {
  (void)self;

  PyArrayObject *xs, *ys, *ws;
  int nthreads;
  PyObject *out_dtype;

  if (!PyArg_ParseTuple(args, "O!O!O!iO", &PyArray_Type, &xs, &PyArray_Type,
                        &ys, &PyArray_Type, &ws, &nthreads, &out_dtype)) {
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

  const int xs_typenum = PyArray_TYPE(xs);
  const int ys_typenum = PyArray_TYPE(ys);
  const int ws_typenum = PyArray_TYPE(ws);
  if (!is_supported_float_typenum(xs_typenum) ||
      !is_supported_float_typenum(ys_typenum) ||
      !is_supported_float_typenum(ws_typenum)) {
    PyErr_SetString(PyExc_TypeError,
                    "xs, ys, and weights must be float32 or float64.");
    return NULL;
  }
  if (!is_c_contiguous(xs, "xs") || !is_c_contiguous(ys, "ys") ||
      !is_c_contiguous(ws, "weights")) {
    return NULL;
  }

  int input_typenum = promoted_float_typenum(xs_typenum, ys_typenum);
  input_typenum = promoted_float_typenum(input_typenum, ws_typenum);
  PyArrayObject *xs_cast = cast_float_array(xs, input_typenum);
  PyArrayObject *ys_cast = cast_float_array(ys, input_typenum);
  PyArrayObject *ws_cast = cast_float_array(ws, input_typenum);
  if (xs_cast == NULL || ys_cast == NULL || ws_cast == NULL) {
    Py_XDECREF(xs_cast);
    Py_XDECREF(ys_cast);
    Py_XDECREF(ws_cast);
    return NULL;
  }

  /* Resolve the independently requested output dtype. */
  const int output_typenum = resolve_output_typenum(out_dtype, "out_dtype");
  if (output_typenum < 0) {
    Py_DECREF(xs_cast);
    Py_DECREF(ys_cast);
    Py_DECREF(ws_cast);
    return NULL;
  }

  /* Dispatch: encode input/output precision into a 2-bit key. */
  const npy_intp num_elements = PyArray_SIZE(ys) / n;
  int dispatch_key =
      ((input_typenum == NPY_FLOAT64) << 1) | (output_typenum == NPY_FLOAT64);
  PyObject *result;
  switch (dispatch_key) {
    case 0:
      result = weighted_trapz_last_axis<float, float>(
          xs_cast, ys_cast, ws_cast, ndim, shape, n, num_elements, nthreads);
      break;
    case 1:
      result = weighted_trapz_last_axis<float, double>(
          xs_cast, ys_cast, ws_cast, ndim, shape, n, num_elements, nthreads);
      break;
    case 2:
      result = weighted_trapz_last_axis<double, float>(
          xs_cast, ys_cast, ws_cast, ndim, shape, n, num_elements, nthreads);
      break;
    default:
      result = weighted_trapz_last_axis<double, double>(
          xs_cast, ys_cast, ws_cast, ndim, shape, n, num_elements, nthreads);
      break;
  }

  Py_DECREF(xs_cast);
  Py_DECREF(ys_cast);
  Py_DECREF(ws_cast);
  return result;
}

/**
 * @brief Serial weighted Simpson integration over the final axis.
 *
 * This computes the same weighted mean as the trapezoidal path but uses the
 * composite Simpson rule, with a final trapezoidal tail when one interval is
 * left over.
 *
 * @tparam Real The floating-point type of the input arrays.
 * @tparam OutT The floating-point type stored in the returned buffer.
 *
 * @param x 1D integration grid.
 * @param y Flattened ND values array.
 * @param w 1D weights defined on x.
 * @param n Number of samples along the integrated axis.
 * @param num_elements Number of leading-axis entries to integrate.
 *
 * @return Newly allocated output buffer, or NULL on allocation failure.
 */
template <typename Real, typename OutT>
static OutT *weighted_simps_last_axis_serial(const Real *x, const Real *y,
                                             const Real *w, npy_intp n,
                                             npy_intp num_elements) {
  if (num_elements == 0) {
    return new (std::nothrow) OutT[1]();
  }

  OutT *result = new (std::nothrow) OutT[num_elements]();
  if (result == NULL) {
    return NULL;
  }

  /* With fewer than two samples there are no intervals to integrate. */
  if (n < 2) {
    return result;
  }

  /* Precompute the shared denominator once for all rows. */
  OutT den = static_cast<OutT>(0.0);
  for (npy_intp j = 0; j < (n - 1) / 2; ++j) {
    const npy_intp k = 2 * j;
    const OutT h0 = static_cast<OutT>(x[k + 1] - x[k]);
    const OutT h1 = static_cast<OutT>(x[k + 2] - x[k + 1]);

    if (h0 == static_cast<OutT>(0.0) || h1 == static_cast<OutT>(0.0)) {
      continue;
    }

    den += (h0 + h1) / static_cast<OutT>(6.0) *
           ((static_cast<OutT>(2.0) - h1 / h0) * static_cast<OutT>(w[k]) +
            ((h0 + h1) * (h0 + h1) / (h0 * h1)) * static_cast<OutT>(w[k + 1]) +
            (static_cast<OutT>(2.0) - h0 / h1) * static_cast<OutT>(w[k + 2]));
  }
  if ((n - 1) % 2 != 0) {
    den += static_cast<OutT>(0.5) * static_cast<OutT>(x[n - 1] - x[n - 2]) *
           (static_cast<OutT>(w[n - 1]) + static_cast<OutT>(w[n - 2]));
  }

  if (den == static_cast<OutT>(0.0)) {
    return result;
  }

  for (npy_intp i = 0; i < num_elements; ++i) {
    OutT num = static_cast<OutT>(0.0);
    for (npy_intp j = 0; j < (n - 1) / 2; ++j) {
      const npy_intp k = 2 * j;
      const OutT h0 = static_cast<OutT>(x[k + 1] - x[k]);
      const OutT h1 = static_cast<OutT>(x[k + 2] - x[k + 1]);

      if (h0 == static_cast<OutT>(0.0) || h1 == static_cast<OutT>(0.0)) {
        continue;
      }

      num += (h0 + h1) / static_cast<OutT>(6.0) *
             ((static_cast<OutT>(2.0) - h1 / h0) *
                  static_cast<OutT>(y[i * n + k]) * static_cast<OutT>(w[k]) +
              ((h0 + h1) * (h0 + h1) / (h0 * h1)) *
                  static_cast<OutT>(y[i * n + k + 1]) *
                  static_cast<OutT>(w[k + 1]) +
              (static_cast<OutT>(2.0) - h0 / h1) *
                  static_cast<OutT>(y[i * n + k + 2]) *
                  static_cast<OutT>(w[k + 2]));
    }
    if ((n - 1) % 2 != 0) {
      num +=
          static_cast<OutT>(0.5) * static_cast<OutT>(x[n - 1] - x[n - 2]) *
          (static_cast<OutT>(y[i * n + n - 1]) * static_cast<OutT>(w[n - 1]) +
           static_cast<OutT>(y[i * n + n - 2]) * static_cast<OutT>(w[n - 2]));
    }
    result[i] = num / den;
  }

  return result;
}

/**
 * @brief Parallel weighted Simpson integration over the final axis.
 *
 * @tparam Real The floating-point type of the input arrays.
 * @tparam OutT The floating-point type stored in the returned buffer.
 *
 * @param x 1D integration grid.
 * @param y Flattened ND values array.
 * @param w 1D weights defined on x.
 * @param n Number of samples along the integrated axis.
 * @param num_elements Number of leading-axis entries to integrate.
 * @param nthreads Number of OpenMP threads to use.
 *
 * @return Newly allocated output buffer, or NULL on allocation failure.
 */
#ifdef WITH_OPENMP
template <typename Real, typename OutT>
static OutT *weighted_simps_last_axis_parallel(const Real *x, const Real *y,
                                               const Real *w, npy_intp n,
                                               npy_intp num_elements,
                                               int nthreads) {
  if (num_elements == 0) {
    return new (std::nothrow) OutT[1]();
  }

  OutT *result = new (std::nothrow) OutT[num_elements]();
  if (result == NULL) {
    return NULL;
  }

  if (n < 2) {
    return result;
  }

  OutT den = static_cast<OutT>(0.0);
  for (npy_intp j = 0; j < (n - 1) / 2; ++j) {
    const npy_intp k = 2 * j;
    const OutT h0 = static_cast<OutT>(x[k + 1] - x[k]);
    const OutT h1 = static_cast<OutT>(x[k + 2] - x[k + 1]);

    if (h0 == static_cast<OutT>(0.0) || h1 == static_cast<OutT>(0.0)) {
      continue;
    }

    den += (h0 + h1) / static_cast<OutT>(6.0) *
           ((static_cast<OutT>(2.0) - h1 / h0) * static_cast<OutT>(w[k]) +
            ((h0 + h1) * (h0 + h1) / (h0 * h1)) * static_cast<OutT>(w[k + 1]) +
            (static_cast<OutT>(2.0) - h0 / h1) * static_cast<OutT>(w[k + 2]));
  }
  if ((n - 1) % 2 != 0) {
    den += static_cast<OutT>(0.5) * static_cast<OutT>(x[n - 1] - x[n - 2]) *
           (static_cast<OutT>(w[n - 1]) + static_cast<OutT>(w[n - 2]));
  }

  if (den == static_cast<OutT>(0.0)) {
    return result;
  }

#pragma omp parallel for num_threads(nthreads)
  for (npy_intp i = 0; i < num_elements; ++i) {
    OutT num = static_cast<OutT>(0.0);
    for (npy_intp j = 0; j < (n - 1) / 2; ++j) {
      const npy_intp k = 2 * j;
      const OutT h0 = static_cast<OutT>(x[k + 1] - x[k]);
      const OutT h1 = static_cast<OutT>(x[k + 2] - x[k + 1]);

      if (h0 == static_cast<OutT>(0.0) || h1 == static_cast<OutT>(0.0)) {
        continue;
      }

      num += (h0 + h1) / static_cast<OutT>(6.0) *
             ((static_cast<OutT>(2.0) - h1 / h0) *
                  static_cast<OutT>(y[i * n + k]) * static_cast<OutT>(w[k]) +
              ((h0 + h1) * (h0 + h1) / (h0 * h1)) *
                  static_cast<OutT>(y[i * n + k + 1]) *
                  static_cast<OutT>(w[k + 1]) +
              (static_cast<OutT>(2.0) - h0 / h1) *
                  static_cast<OutT>(y[i * n + k + 2]) *
                  static_cast<OutT>(w[k + 2]));
    }
    if ((n - 1) % 2 != 0) {
      num +=
          static_cast<OutT>(0.5) * static_cast<OutT>(x[n - 1] - x[n - 2]) *
          (static_cast<OutT>(y[i * n + n - 1]) * static_cast<OutT>(w[n - 1]) +
           static_cast<OutT>(y[i * n + n - 2]) * static_cast<OutT>(w[n - 2]));
    }
    result[i] = num / den;
  }

  return result;
}
#endif

/**
 * @brief Execute weighted Simpson integration after dtype dispatch.
 *
 * @tparam Real The floating-point type of the validated input arrays.
 * @tparam OutT The requested floating-point output type.
 *
 * @param xs 1D NumPy integration grid.
 * @param ys ND NumPy values array.
 * @param ws 1D NumPy weights array.
 * @param ndim Dimensionality of ys.
 * @param shape Shape of ys.
 * @param n Length of the final integrated axis.
 * @param num_elements Number of leading-axis entries to integrate.
 * @param nthreads Number of OpenMP threads to use when available.
 *
 * @return The wrapped NumPy output array, or NULL on failure.
 */
template <typename Real, typename OutT>
static PyObject *weighted_simps_last_axis(PyArrayObject *xs, PyArrayObject *ys,
                                          PyArrayObject *ws, npy_intp ndim,
                                          npy_intp *shape, npy_intp n,
                                          npy_intp num_elements,
                                          int nthreads) {
  const Real *x = data_ptr<const Real>(xs);
  const Real *y = data_ptr<const Real>(ys);
  const Real *w = data_ptr<const Real>(ws);

  OutT *result_arr;
#ifdef WITH_OPENMP
  if (nthreads > 1) {
    result_arr = weighted_simps_last_axis_parallel<Real, OutT>(
        x, y, w, n, num_elements, nthreads);
  } else {
    result_arr =
        weighted_simps_last_axis_serial<Real, OutT>(x, y, w, n, num_elements);
  }
#else
  result_arr =
      weighted_simps_last_axis_serial<Real, OutT>(x, y, w, n, num_elements);
#endif

  if (result_arr == NULL) {
    PyErr_SetString(PyExc_MemoryError,
                    "Failed to allocate output for weighted simps.");
    return NULL;
  }

  return (PyObject *)wrap_last_axis_result<OutT>(result_arr, ndim, shape);
}

/**
 * @brief Python wrapper for weighted Simpson last-axis integration.
 *
 * @param self The module instance. (Unused)
 * @param args Python arguments containing xs, ys, weights, nthreads, and
 *        out_dtype.
 *
 * @return A NumPy array with shape ys.shape[:-1], or NULL on failure.
 */
static PyObject *weighted_simps_last_axis_integration(PyObject *self,
                                                      PyObject *args) {
  (void)self;

  PyArrayObject *xs, *ys, *ws;
  int nthreads;
  PyObject *out_dtype;

  if (!PyArg_ParseTuple(args, "O!O!O!iO", &PyArray_Type, &xs, &PyArray_Type,
                        &ys, &PyArray_Type, &ws, &nthreads, &out_dtype)) {
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

  const int xs_typenum = PyArray_TYPE(xs);
  const int ys_typenum = PyArray_TYPE(ys);
  const int ws_typenum = PyArray_TYPE(ws);
  if (!is_supported_float_typenum(xs_typenum) ||
      !is_supported_float_typenum(ys_typenum) ||
      !is_supported_float_typenum(ws_typenum)) {
    PyErr_SetString(PyExc_TypeError,
                    "xs, ys, and weights must be float32 or float64.");
    return NULL;
  }
  if (!is_c_contiguous(xs, "xs") || !is_c_contiguous(ys, "ys") ||
      !is_c_contiguous(ws, "weights")) {
    return NULL;
  }

  int input_typenum = promoted_float_typenum(xs_typenum, ys_typenum);
  input_typenum = promoted_float_typenum(input_typenum, ws_typenum);
  PyArrayObject *xs_cast = cast_float_array(xs, input_typenum);
  PyArrayObject *ys_cast = cast_float_array(ys, input_typenum);
  PyArrayObject *ws_cast = cast_float_array(ws, input_typenum);
  if (xs_cast == NULL || ys_cast == NULL || ws_cast == NULL) {
    Py_XDECREF(xs_cast);
    Py_XDECREF(ys_cast);
    Py_XDECREF(ws_cast);
    return NULL;
  }

  /* Resolve the independently requested output dtype. */
  const int output_typenum = resolve_output_typenum(out_dtype, "out_dtype");
  if (output_typenum < 0) {
    Py_DECREF(xs_cast);
    Py_DECREF(ys_cast);
    Py_DECREF(ws_cast);
    return NULL;
  }

  /* Dispatch: encode input/output precision into a 2-bit key. */
  const npy_intp num_elements = PyArray_SIZE(ys) / n;
  int dispatch_key =
      ((input_typenum == NPY_FLOAT64) << 1) | (output_typenum == NPY_FLOAT64);

  /* Dispatch: call the matching typed kernel based on the dispatch key. */
  PyObject *result;
  switch (dispatch_key) {
    case 0:
      result = weighted_simps_last_axis<float, float>(
          xs_cast, ys_cast, ws_cast, ndim, shape, n, num_elements, nthreads);
      break;
    case 1:
      result = weighted_simps_last_axis<float, double>(
          xs_cast, ys_cast, ws_cast, ndim, shape, n, num_elements, nthreads);
      break;
    case 2:
      result = weighted_simps_last_axis<double, float>(
          xs_cast, ys_cast, ws_cast, ndim, shape, n, num_elements, nthreads);
      break;
    default:
      result = weighted_simps_last_axis<double, double>(
          xs_cast, ys_cast, ws_cast, ndim, shape, n, num_elements, nthreads);
      break;
  }

  Py_DECREF(xs_cast);
  Py_DECREF(ys_cast);
  Py_DECREF(ws_cast);
  return result;
}

static PyMethodDef IntegrationMethods[] = {
    {"trapz_last_axis", trapz_last_axis_integration, METH_VARARGS,
     "Integrate the final axis using the trapezoidal rule."},
    {"simps_last_axis", simps_last_axis_integration, METH_VARARGS,
     "Integrate the final axis using composite Simpson's rule."},
    {"weighted_trapz_last_axis", weighted_trapz_last_axis_integration,
     METH_VARARGS,
     "Compute a weighted mean over the final axis using the trapezoidal "
     "rule."},
    {"weighted_simps_last_axis", weighted_simps_last_axis_integration,
     METH_VARARGS,
     "Compute a weighted mean over the final axis using Simpson's rule."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef integrationmodule = {PyModuleDef_HEAD_INIT,
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
  if (m == NULL) return NULL;
#ifdef ATOMIC_TIMING
  if (import_toc_capsule() < 0) {
    Py_DECREF(m);
    return NULL;
  }
#endif
  return m;
}
