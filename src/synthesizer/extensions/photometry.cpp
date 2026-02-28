/******************************************************************************
 * C extension for batched photometry integration.
 *
 * This module computes broadband photometry for many spectra and many filters
 * in a single call, returning results in filter-first layout:
 *
 *     result shape = (nfilters, *leading_spectrum_shape)
 *
 * Two numerical integration methods are supported:
 *   - Trapezoidal rule ("trapz")
 *   - Composite Simpson's rule with trapezoidal tail ("simps")
 *
 * Serial and parallel (OpenMP) versions are provided for each method. The
 * parallel versions use OpenMP parallel-for over (entry, filter) pairs to
 * maximize available parallelism and improve load balancing, especially for
 * small numbers of entries. Each work item is independent so no synchronization
 * is required.
 *****************************************************************************/

/* Python includes */
#define PY_ARRAY_UNIQUE_SYMBOL SYNTHESIZER_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy_init.h"
#include <Python.h>

/* C/C++ includes */
#include <cstring>
#include <vector>

#ifdef WITH_OPENMP
#include <omp.h>
#endif

/* Local includes */
#include "cpp_to_python.h"
#include "property_funcs.h"

/**
 * @brief Compact per-filter metadata for use in tight inner loops.
 *
 * Rather than repeatedly indexing into the full weight matrix and checking
 * denominators during the hot integration loop, we precompute one of these
 * descriptors for each *active* filter (denominator != 0 and at least two
 * samples). This keeps the inner loop free of branches and indirect lookups.
 *
 * @param filter_index:    Position of this filter in the original filter list.
 * @param weights:         Pointer into the weight matrix row for this filter.
 * @param start:           First wavelength index with non-zero transmission.
 * @param end:             One-past-last wavelength index with non-zero
 *                         transmission.
 * @param inv_denominator: Precomputed 1.0 / denominator for this filter.
 */
struct FilterWork {
  npy_intp filter_index;
  const double *weights;
  npy_int64 start;
  npy_int64 end;
  double inv_denominator;
};

/**
 * @brief Build an array of compact work descriptors for active filters.
 *
 * A filter is considered "active" when:
 *   - Its denominator is non-zero (otherwise the integral is undefined).
 *   - Its support spans at least two wavelength samples (otherwise there
 *     are no intervals to integrate over).
 *
 * Inactive filters are skipped; their output slots remain zero.
 *
 * @param weight_matrix:    2D array of filter transmission weights,
 *                          shape (nfilters, wavelength_count).
 * @param denominators:     1D array of precomputed filter denominators.
 * @param starts:           1D array of first non-zero indices per filter.
 * @param ends:             1D array of one-past-last non-zero indices.
 * @param wavelength_count: Number of wavelength/frequency samples.
 * @param nfilters:         Total number of filters.
 *
 * @return A vector of FilterWork descriptors for active filters only.
 */
static std::vector<FilterWork>
build_filter_work(const double *weight_matrix, const double *denominators,
                  const npy_int64 *starts, const npy_int64 *ends,
                  npy_intp wavelength_count, npy_intp nfilters) {

  /* Pre-allocate space for up to nfilters entries. */
  std::vector<FilterWork> work;
  work.reserve((size_t)nfilters);

  /* Walk through every filter and keep only the active ones. */
  for (npy_intp filter_index = 0; filter_index < nfilters; ++filter_index) {
    const npy_int64 start = starts[filter_index];
    const npy_int64 end = ends[filter_index];
    const double denominator = denominators[filter_index];

    /* Skip filters with zero denominator or fewer than 2 samples. */
    if (denominator == 0.0 || end - start < 2) {
      continue;
    }

    /* Populate the work descriptor. */
    FilterWork item;
    item.filter_index = filter_index;
    item.weights = &weight_matrix[filter_index * wavelength_count];
    item.start = start;
    item.end = end;
    item.inv_denominator = 1.0 / denominator;
    work.push_back(item);
  }

  return work;
}

/**
 * @brief Compute batched photometry using the trapezoidal rule (serial).
 *
 * For each spectrum (entry) and each active filter, this evaluates:
 *
 *   photometry[f][entry] = (1 / den_f) * integral( spectrum * T_f, dlam )
 *
 * where T_f is the filter transmission and den_f is the precomputed
 * denominator. The trapezoidal rule approximates the integral as
 * sum of 0.5 * (x[j+1] - x[j]) * (y[j+1]*w[j+1] + y[j]*w[j]).
 *
 * Results are written directly in filter-major layout.
 *
 * @param x_values:         1D wavelength/frequency grid.
 * @param spectra_values:   Flattened spectra, shape (nentries, wavelength_count).
 * @param filter_work:      Vector of active filter descriptors.
 * @param wavelength_count: Number of wavelength/frequency samples.
 * @param nentries:         Number of spectra (entries).
 * @param nfilters:         Total number of filters (including inactive).
 *
 * @return Newly allocated filter-major result buffer, or NULL on failure.
 */
static double *compute_photometry_trapz_serial(
    const double *x_values, const double *spectra_values,
    const std::vector<FilterWork> &filter_work, npy_intp wavelength_count,
    npy_intp nentries, npy_intp nfilters) {

  /* Allocate output directly in filter-major layout. calloc zeroes the
   * memory so inactive filter slots are already 0. */
  double *filter_major =
      (double *)calloc((size_t)(nentries * nfilters), sizeof(double));
  if (filter_major == NULL) {
    return NULL;
  }

  /* Loop over entries (spectra). */
  for (npy_intp entry = 0; entry < nentries; ++entry) {

    /* Get a pointer to this entry's spectrum. */
    const double *spectrum = &spectra_values[entry * wavelength_count];

    /* Loop over active filters. */
    for (const FilterWork &work : filter_work) {

      /* Accumulate the trapezoidal quadrature numerator over the
       * filter's non-zero support range [start, end). */
      double numerator = 0.0;
      for (npy_int64 wavelength = work.start; wavelength < work.end - 1;
           ++wavelength) {
        numerator +=
            0.5 * (x_values[wavelength + 1] - x_values[wavelength]) *
            (spectrum[wavelength + 1] * work.weights[wavelength + 1] +
             spectrum[wavelength] * work.weights[wavelength]);
      }

      /* Divide by the precomputed denominator and store directly in
       * filter-major layout. */
      filter_major[work.filter_index * nentries + entry] =
          numerator * work.inv_denominator;
    }
  }

  return filter_major;
}

/**
 * @brief Compute batched photometry using composite Simpson's rule (serial).
 *
 * For each spectrum (entry) and each active filter, this evaluates the
 * same integral as the trapezoidal version but uses composite Simpson's
 * 1/3 rule for improved accuracy on smooth integrands. If the number of
 * intervals is odd, the last interval falls back to a single trapezoidal
 * step (the "tail").
 *
 * Results are written directly in filter-major layout.
 *
 * @param x_values:         1D wavelength/frequency grid.
 * @param spectra_values:   Flattened spectra, shape (nentries, wavelength_count).
 * @param filter_work:      Vector of active filter descriptors.
 * @param wavelength_count: Number of wavelength/frequency samples.
 * @param nentries:         Number of spectra (entries).
 * @param nfilters:         Total number of filters (including inactive).
 *
 * @return Newly allocated filter-major result buffer, or NULL on failure.
 */
static double *compute_photometry_simps_serial(
    const double *x_values, const double *spectra_values,
    const std::vector<FilterWork> &filter_work, npy_intp wavelength_count,
    npy_intp nentries, npy_intp nfilters) {

  /* Allocate output directly in filter-major layout. */
  double *filter_major =
      (double *)calloc((size_t)(nentries * nfilters), sizeof(double));
  if (filter_major == NULL) {
    return NULL;
  }

  /* Loop over entries (spectra). */
  for (npy_intp entry = 0; entry < nentries; ++entry) {

    /* Get a pointer to this entry's spectrum. */
    const double *spectrum = &spectra_values[entry * wavelength_count];

    /* Loop over active filters. */
    for (const FilterWork &work : filter_work) {

      /* How many wavelength samples does this filter span? */
      const npy_int64 sample_count = work.end - work.start;

      /* How many pairs of intervals can Simpson's rule cover?
       * Each pair consumes 3 points (2 intervals). */
      const npy_int64 npairs = (sample_count - 1) / 2;

      /* Is there a leftover single interval that must be handled
       * with a trapezoidal step? */
      const bool has_tail = ((sample_count - 1) % 2) != 0;

      /* Accumulate the Simpson's quadrature numerator. */
      double numerator = 0.0;
      for (npy_int64 pair_index = 0; pair_index < npairs; ++pair_index) {
        const npy_int64 k = work.start + 2 * pair_index;
        numerator +=
            (x_values[k + 2] - x_values[k]) *
            (spectrum[k] * work.weights[k] +
             4.0 * spectrum[k + 1] * work.weights[k + 1] +
             spectrum[k + 2] * work.weights[k + 2]) /
            6.0;
      }

      /* If there is a leftover interval, add a trapezoidal step. */
      if (has_tail) {
        const npy_int64 k0 = work.end - 2;
        const npy_int64 k1 = work.end - 1;
        numerator += 0.5 * (x_values[k1] - x_values[k0]) *
                     (spectrum[k1] * work.weights[k1] +
                      spectrum[k0] * work.weights[k0]);
      }

      /* Divide by the precomputed denominator and store directly in
       * filter-major layout. */
      filter_major[work.filter_index * nentries + entry] =
          numerator * work.inv_denominator;
    }
  }

  return filter_major;
}

#ifdef WITH_OPENMP
/**
 * @brief Compute batched photometry using the trapezoidal rule (parallel).
 *
 * This is the parallel version of compute_photometry_trapz_serial. Work is
 * parallelized over flattened (entry, active_filter) pairs, so even cases
 * with few entries and many wavelengths (e.g. integrated photometry) can
 * still exploit thread-level parallelism across filters.
 *
 * @param x_values:         1D wavelength/frequency grid.
 * @param spectra_values:   Flattened spectra, shape (nentries, wavelength_count).
 * @param filter_work:      Vector of active filter descriptors.
 * @param wavelength_count: Number of wavelength/frequency samples.
 * @param nentries:         Number of spectra (entries).
 * @param nfilters:         Total number of filters (including inactive).
 * @param nthreads:         Number of OpenMP threads to use.
 *
 * @return Newly allocated filter-major result buffer, or NULL on failure.
 */
static double *compute_photometry_trapz_parallel(
    const double *x_values, const double *spectra_values,
    const std::vector<FilterWork> &filter_work, npy_intp wavelength_count,
    npy_intp nentries, npy_intp nfilters, int nthreads) {

  /* Allocate the output directly in filter-major layout. calloc zeroes the
   * memory so inactive filter slots are already 0. */
  double *filter_major =
      (double *)calloc((size_t)(nentries * nfilters), sizeof(double));
  if (filter_major == NULL) {
    return NULL;
  }

  /* How many filters are active? */
  const npy_intp nactive = (npy_intp)filter_work.size();

  /* Parallelize over (entry, filter) pairs to maximize available parallelism.
   * With nentries * nactive work items, we get much better load balancing
   * especially for small nentries (e.g., 100 particles × 10 filters = 1000
   * work items vs just 100). OpenMP's parallel for handles scheduling. */
  const npy_intp total_work = nentries * nactive;

#pragma omp parallel for num_threads(nthreads) schedule(static)
  for (npy_intp work_idx = 0; work_idx < total_work; ++work_idx) {
    /* Decompose the linear work index into (entry, filter) indices. */
    const npy_intp entry = work_idx / nactive;
    const npy_intp af = work_idx % nactive;

    /* Get the filter work descriptor. */
    const FilterWork &work = filter_work[(size_t)af];

    /* Get a pointer to this entry's spectrum. */
    const double *spectrum = &spectra_values[entry * wavelength_count];

    /* Accumulate the trapezoidal quadrature numerator over the
     * filter's non-zero support range [start, end). */
    double numerator = 0.0;
    for (npy_int64 wavelength = work.start; wavelength < work.end - 1;
         ++wavelength) {
      numerator +=
          0.5 * (x_values[wavelength + 1] - x_values[wavelength]) *
          (spectrum[wavelength + 1] * work.weights[wavelength + 1] +
           spectrum[wavelength] * work.weights[wavelength]);
    }

    /* Write directly to the output. Each work_idx maps to a unique
     * (entry, filter) pair, so no synchronization is needed. */
    filter_major[work.filter_index * nentries + entry] =
        numerator * work.inv_denominator;
  }

  return filter_major;
}

/**
 * @brief Compute batched photometry using composite Simpson's rule (parallel).
 *
 * This is the parallel version of compute_photometry_simps_serial. The
 * parallelisation strategy is identical to the trapezoidal version:
 * flattened (entry, active_filter) pairs are distributed across threads and
 * each work item writes to a unique output location.
 *
 * @param x_values:         1D wavelength/frequency grid.
 * @param spectra_values:   Flattened spectra, shape (nentries, wavelength_count).
 * @param filter_work:      Vector of active filter descriptors.
 * @param wavelength_count: Number of wavelength/frequency samples.
 * @param nentries:         Number of spectra (entries).
 * @param nfilters:         Total number of filters (including inactive).
 * @param nthreads:         Number of OpenMP threads to use.
 *
 * @return Newly allocated filter-major result buffer, or NULL on failure.
 */
static double *compute_photometry_simps_parallel(
    const double *x_values, const double *spectra_values,
    const std::vector<FilterWork> &filter_work, npy_intp wavelength_count,
    npy_intp nentries, npy_intp nfilters, int nthreads) {

  /* Allocate the output directly in filter-major layout. */
  double *filter_major =
      (double *)calloc((size_t)(nentries * nfilters), sizeof(double));
  if (filter_major == NULL) {
    return NULL;
  }

  /* How many filters are active? */
  const npy_intp nactive = (npy_intp)filter_work.size();

  /* Parallelize over (entry, filter) pairs to maximize available parallelism.
   * With nentries * nactive work items, we get much better load balancing
   * especially for small nentries (e.g., 100 particles × 10 filters = 1000
   * work items vs just 100). OpenMP's parallel for handles scheduling. */
  const npy_intp total_work = nentries * nactive;

#pragma omp parallel for num_threads(nthreads) schedule(static)
  for (npy_intp work_idx = 0; work_idx < total_work; ++work_idx) {
    /* Decompose the linear work index into (entry, filter) indices. */
    const npy_intp entry = work_idx / nactive;
    const npy_intp af = work_idx % nactive;

    /* Get the filter work descriptor. */
    const FilterWork &work = filter_work[(size_t)af];

    /* Get a pointer to this entry's spectrum. */
    const double *spectrum = &spectra_values[entry * wavelength_count];

    /* How many wavelength samples does this filter span? */
    const npy_int64 sample_count = work.end - work.start;

    /* How many pairs of intervals can Simpson's rule cover? */
    const npy_int64 npairs = (sample_count - 1) / 2;

    /* Is there a leftover single interval for a trapezoidal tail? */
    const bool has_tail = ((sample_count - 1) % 2) != 0;

    /* Accumulate the Simpson's quadrature numerator. */
    double numerator = 0.0;
    for (npy_int64 pair_index = 0; pair_index < npairs; ++pair_index) {
      const npy_int64 k = work.start + 2 * pair_index;
      numerator +=
          (x_values[k + 2] - x_values[k]) *
          (spectrum[k] * work.weights[k] +
           4.0 * spectrum[k + 1] * work.weights[k + 1] +
           spectrum[k + 2] * work.weights[k + 2]) /
          6.0;
    }

    /* If there is a leftover interval, add a trapezoidal step. */
    if (has_tail) {
      const npy_int64 k0 = work.end - 2;
      const npy_int64 k1 = work.end - 1;
      numerator += 0.5 * (x_values[k1] - x_values[k0]) *
                   (spectrum[k1] * work.weights[k1] +
                    spectrum[k0] * work.weights[k0]);
    }

    /* Write directly to the output. Each work_idx maps to a unique
     * (entry, filter) pair, so no synchronization is needed. */
    filter_major[work.filter_index * nentries + entry] =
        numerator * work.inv_denominator;
  }

  return filter_major;
}
#endif /* WITH_OPENMP */

/**
 * @brief Python-facing entry point for batched photometry integration.
 *
 * This function is called from Python (via FilterCollection.apply_filters)
 * and dispatches to the appropriate serial or parallel kernel depending on
 * the requested thread count and whether OpenMP support was compiled in.
 *
 * @param self:  Unused (required by the Python C API).
 * @param args:  Positional arguments tuple containing:
 *   - xs      (ndarray):  1D wavelength/frequency grid, shape (nlam,).
 *   - ys      (ndarray):  ND spectra array; wavelength on the last axis.
 *   - ws      (ndarray):  2D filter weight matrix, shape (nfilters, nlam).
 *   - dens    (ndarray):  1D precomputed filter denominators, shape (nfilters,).
 *   - starts  (ndarray):  1D start indices per filter, shape (nfilters,).
 *   - ends    (ndarray):  1D end indices per filter, shape (nfilters,).
 *   - nthreads (int):     Number of OpenMP threads requested.
 *   - method   (str):     Integration method, "trapz" or "simps".
 *
 * @return A numpy array of shape (nfilters, *leading_spectrum_shape).
 */
static PyObject *compute_photometry_integration(PyObject *self, PyObject *args) {

  /* We don't need the self argument but it has to be there. Tell the
   * compiler we don't care. */
  (void)self;

  /* Declare the numpy array pointers we'll extract from args. */
  PyArrayObject *np_x_values, *np_spectra_values, *np_weight_matrix;
  PyArrayObject *np_denominators, *np_starts, *np_ends;
  int nthreads;
  const char *integration_method;

  /* Parse the positional arguments from the Python call. */
  if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!is", &PyArray_Type, &np_x_values,
                        &PyArray_Type, &np_spectra_values, &PyArray_Type,
                        &np_weight_matrix, &PyArray_Type, &np_denominators,
                        &PyArray_Type, &np_starts, &PyArray_Type, &np_ends,
                        &nthreads, &integration_method)) {
    return NULL;
  }

  /* Validate array dimensionalities. */
  if (PyArray_NDIM(np_x_values) != 1 || PyArray_NDIM(np_weight_matrix) != 2 ||
      PyArray_NDIM(np_denominators) != 1 || PyArray_NDIM(np_starts) != 1 ||
      PyArray_NDIM(np_ends) != 1) {
    PyErr_SetString(PyExc_ValueError,
                    "Invalid array ranks for photometry integration inputs.");
    return NULL;
  }

  /* The spectra array can be ND (e.g. (ngalaxies, nparticles, nlam));
   * the wavelength axis is always the last one. */
  const npy_intp spectra_ndim = PyArray_NDIM(np_spectra_values);
  if (spectra_ndim < 1) {
    PyErr_SetString(PyExc_ValueError,
                    "Input spectra array must have at least 1 dimension.");
    return NULL;
  }

  /* Extract shape information. */
  npy_intp *spectra_shape = PyArray_SHAPE(np_spectra_values);
  const npy_intp wavelength_count = spectra_shape[spectra_ndim - 1];
  const npy_intp nfilters = PyArray_DIM(np_weight_matrix, 0);

  if (wavelength_count == 0) {
    PyErr_SetString(PyExc_ValueError,
                    "Input spectra wavelength axis must be non-empty.");
    return NULL;
  }

  /* Validate that all shapes are consistent. */
  if (PyArray_DIM(np_x_values, 0) != wavelength_count ||
      PyArray_DIM(np_weight_matrix, 1) != wavelength_count ||
      PyArray_DIM(np_denominators, 0) != nfilters ||
      PyArray_DIM(np_starts, 0) != nfilters ||
      PyArray_DIM(np_ends, 0) != nfilters) {
    PyErr_SetString(PyExc_ValueError,
                    "Input shapes are inconsistent for photometry "
                    "integration.");
    return NULL;
  }

  /* Extract C pointers from the numpy arrays. */
  const double *x_values = extract_data_double(np_x_values, "xs");
  if (x_values == NULL) {
    return NULL;
  }

  const double *spectra_values =
      extract_data_double(np_spectra_values, "spectra");
  if (spectra_values == NULL) {
    return NULL;
  }

  const double *weight_matrix =
      extract_data_double(np_weight_matrix, "weights");
  if (weight_matrix == NULL) {
    return NULL;
  }

  const double *denominators =
      extract_data_double(np_denominators, "denominators");
  if (denominators == NULL) {
    return NULL;
  }

  const npy_int64 *starts = extract_index_array(np_starts, "starts");
  if (starts == NULL) {
    return NULL;
  }
  const npy_int64 *ends = extract_index_array(np_ends, "ends");
  if (ends == NULL) {
    return NULL;
  }

  for (npy_intp f = 0; f < nfilters; ++f) {
    if (starts[f] < 0 || ends[f] < 0 || starts[f] > ends[f] ||
        ends[f] > wavelength_count) {
      PyErr_SetString(PyExc_ValueError,
                      "Filter band indices must satisfy 0 <= start <= end <= "
                      "wavelength_count.");
      return NULL;
    }
  }

  /* The total number of spectra is the product of all leading dimensions.
   * We flatten them for the C kernel and reshape on return. */
  const npy_intp nentries = PyArray_SIZE(np_spectra_values) / wavelength_count;

  /* Which integration method was requested? */
  const bool use_simps = strcmp(integration_method, "simps") == 0;
  const bool use_trapz = strcmp(integration_method, "trapz") == 0;

  if (!use_simps && !use_trapz) {
    PyErr_SetString(PyExc_ValueError,
                    "method must be either 'trapz' or 'simps'.");
    return NULL;
  }

  /* Build the compact work descriptors for active filters. */
  const std::vector<FilterWork> filter_work =
      build_filter_work(weight_matrix, denominators, starts, ends,
                        wavelength_count, nfilters);

  /* Dispatch to the appropriate kernel. */
  double *result_array = NULL;

#ifdef WITH_OPENMP
  /* If we have multiple threads and OpenMP we can parallelise. */
  if (nthreads > 1) {
    if (use_trapz) {
      result_array = compute_photometry_trapz_parallel(
          x_values, spectra_values, filter_work, wavelength_count, nentries,
          nfilters, nthreads);
    } else {
      result_array = compute_photometry_simps_parallel(
          x_values, spectra_values, filter_work, wavelength_count, nentries,
          nfilters, nthreads);
    }
  }
  /* Otherwise there's no point paying the OpenMP overhead. */
  else {
    if (use_trapz) {
      result_array = compute_photometry_trapz_serial(
          x_values, spectra_values, filter_work, wavelength_count, nentries,
          nfilters);
    } else {
      result_array = compute_photometry_simps_serial(
          x_values, spectra_values, filter_work, wavelength_count, nentries,
          nfilters);
    }
  }
#else

  /* We don't have OpenMP, just call the serial version. */
  if (use_trapz) {
    result_array = compute_photometry_trapz_serial(
        x_values, spectra_values, filter_work, wavelength_count, nentries,
        nfilters);
  } else {
    result_array = compute_photometry_simps_serial(
        x_values, spectra_values, filter_work, wavelength_count, nentries,
        nfilters);
  }
#endif

  /* Check for allocation failure in the kernel. */
  if (result_array == NULL) {
    PyErr_SetString(PyExc_MemoryError,
                    "Failed to allocate photometry output arrays.");
    return NULL;
  }

  /* Build the output shape: (nfilters, *leading_spectrum_shape).
   * The leading dimensions of the input spectra become the trailing
   * dimensions of the output (the wavelength axis is consumed). */
  npy_intp result_shape[NPY_MAXDIMS];
  result_shape[0] = nfilters;
  for (npy_intp axis = 1; axis < spectra_ndim; ++axis) {
    result_shape[axis] = spectra_shape[axis - 1];
  }

  /* Wrap the raw C array into a numpy array (transfers ownership). */
  PyArrayObject *result =
      wrap_array_to_numpy<double>(spectra_ndim, result_shape, result_array);
  if (result == NULL) {
    free(result_array);
    return NULL;
  }

  return (PyObject *)result;
}

/* Below is all the gubbins needed to make the module importable in Python. */
static PyMethodDef PhotometryMethods[] = {
    {"compute_photometry", compute_photometry_integration, METH_VARARGS,
     "Compute batched photometry for many spectra and filters at once.\n"
     "\n"
     "Returns a numpy array in filter-first layout: "
     "(nfilters, *leading_spectrum_shape)."},
    {NULL, NULL, 0, NULL}};

/* Make this importable. */
static struct PyModuleDef photometrymodule = {
    PyModuleDef_HEAD_INIT,
    "photometry",                                          /* m_name */
    "A module for batched broadband photometry integration", /* m_doc */
    -1,                                                    /* m_size */
    PhotometryMethods,                                     /* m_methods */
    NULL,                                                  /* m_reload */
    NULL,                                                  /* m_traverse */
    NULL,                                                  /* m_clear */
    NULL,                                                  /* m_free */
};

PyMODINIT_FUNC PyInit_photometry(void) {
  PyObject *m = PyModule_Create(&photometrymodule);
  if (m == NULL)
    return NULL;
  if (numpy_import() < 0) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to import numpy.");
    Py_DECREF(m);
    return NULL;
  }
  return m;
}
