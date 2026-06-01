/******************************************************************************
 * C extension for reducing per-particle spectra.
 *
 * This module provides two closely related reduction paths:
 *
 *   1. A shared internal double-precision reduction kernel used by existing
 *      particle spectra extraction code.
 *   2. A Python-facing wrapper that validates NumPy inputs, dispatches to
 *      typed float32/float64 kernels, and returns a newly allocated reduced
 *      spectrum in the requested floating-point output precision.
 *
 * The core operation is a reduction over the leading particle axis:
 *
 *     result[lam] = sum_p part_spectra[p, lam]
 *
 * Serial and parallel (OpenMP) implementations are provided for both the
 * legacy internal double path and the Python-facing typed path.
 *****************************************************************************/

/* C/C++ includes */
#include <cmath>
#include <new>
#include <vector>

/* Python includes */
#define PY_ARRAY_UNIQUE_SYMBOL SYNTHESIZER_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy_init.h"
#include <Python.h>

/* Local includes */
#include "cpp_to_python.h"
#include "python_to_cpp.h"
#include "reductions.h"
#include "timers.h"
#ifdef ATOMIC_TIMING
#include "timers_init.h"
#endif

/**
 * @brief Reduce Npart spectra to integrated spectra.
 *
 * This is the legacy serial reduction path used internally by the particle
 * spectra extraction extensions, which still operate on double buffers.
 *
 * @param spectra: The output array to accumulate the spectra.
 * @param part_spectra: The per-particle spectra array.
 * @param nlam: The number of wavelengths in the spectra.
 * @param npart: The number of particles.
 */
static void reduce_spectra_serial(double *spectra, double *part_spectra,
                                  int nlam, int npart) {

  /* Cast npart to size_t for safety in the loop. */
  size_t npart_size = static_cast<size_t>(npart);

  /* Loop over particles. */
  for (size_t p = 0; p < npart_size; p++) {
    double *__restrict part_spectra_row = part_spectra + p * nlam;

    /* Loop over wavelengths. */
    for (int ilam = 0; ilam < nlam; ilam++) {
      /* Use fused multiply-add to accumulate with better precision.
       * Equivalent to: += spec_val * weight, but with a single rounding. */
      spectra[ilam] = std::fma(part_spectra_row[ilam], 1.0, spectra[ilam]);
    }
  }
}

/**
 * @brief Reduce Npart spectra to integrated spectra.
 *
 * This is the legacy parallel reduction path used internally by the particle
 * spectra extraction extensions, which still operate on double buffers.
 *
 * @param spectra: The output array to accumulate the spectra.
 * @param part_spectra: The per-particle spectra array.
 * @param nlam: The number of wavelengths in the spectra.
 * @param npart: The number of particles.
 * @param nthreads: The number of threads to use.
 */
#ifdef WITH_OPENMP
static void reduce_spectra_parallel(double *spectra, double *part_spectra,
                                    int nlam, int npart, int nthreads) {

  /* Cast npart to size_t for safety in the loop. */
  size_t npart_size = static_cast<size_t>(npart);

  /* Loop over particles in parallel. */
#if defined(_OPENMP) && _OPENMP >= 201511
#pragma omp parallel for num_threads(nthreads) reduction(+ : spectra[ : nlam])
  for (size_t p = 0; p < npart_size; p++) {
    double *__restrict part_spectra_row = part_spectra + p * nlam;
    for (int ilam = 0; ilam < nlam; ilam++) {
      spectra[ilam] += part_spectra_row[ilam];
    }
  }
#else // OpenMP < 4.5 or no array reduction support
#pragma omp parallel num_threads(nthreads)
  {
    // Thread-local accumulation to avoid false sharing and atomics
    std::vector<double> local(nlam, 0.0);
#pragma omp for nowait schedule(static)
    for (size_t p = 0; p < npart; p++) {
      double *__restrict part_spectra_row = part_spectra + p * nlam;
      for (int ilam = 0; ilam < nlam; ilam++) {
        local[ilam] += part_spectra_row[ilam];
      }
    }
    // Merge
#pragma omp critical
    {
      for (int ilam = 0; ilam < nlam; ilam++) {
        spectra[ilam] += local[ilam];
      }
    }
  }
#endif // WITH_OPENMP
}
#endif

/**
 * @brief Reduce Npart spectra to integrated spectra.
 *
 * This is the shared legacy wrapper that selects the serial or parallel double
 * implementation depending on the requested thread count and whether OpenMP is
 * available.
 *
 * @param spectra: The output array to accumulate the spectra.
 * @param part_spectra: The per-particle spectra array.
 * @param nlam: The number of wavelengths in the spectra.
 * @param npart: The number of particles.
 * @param nthreads: The number of threads to use.
 */
void reduce_spectra(double *spectra, double *part_spectra, int nlam, int npart,
                    int nthreads) {

  tic("reduce_spectra");
  if (nthreads > 1) {
#ifdef WITH_OPENMP
    reduce_spectra_parallel(spectra, part_spectra, nlam, npart, nthreads);
#else
    reduce_spectra_serial(spectra, part_spectra, nlam, npart);
#endif
  } else {
    reduce_spectra_serial(spectra, part_spectra, nlam, npart);
  }
  toc("reduce_spectra");
}

/**
 * @brief Reduce per-particle spectra to an integrated spectrum in serial.
 *
 * This typed path is used by the Python wrapper after dtype dispatch. It sums
 * over the leading particle axis and writes a one-dimensional spectrum of
 * length nlam.
 *
 * @tparam Real The floating-point type of the input per-particle spectra.
 * @tparam OutT The floating-point type stored in the reduced spectrum.
 *
 * @param spectra The output array to accumulate the spectra.
 * @param part_spectra The per-particle spectra array.
 * @param nlam The number of wavelengths in the spectra.
 * @param npart The number of particles.
 */
template <typename Real, typename OutT>
static void reduce_particle_spectra_serial_typed(OutT *spectra,
                                                 const Real *part_spectra,
                                                 npy_intp nlam,
                                                 npy_intp npart) {

  /* Cast once so the outer loop uses a size_t counter. */
  const size_t npart_size = static_cast<size_t>(npart);

  /* Walk over particles and accumulate directly into the reduced spectrum. */
  for (size_t p = 0; p < npart_size; p++) {
    const Real *__restrict part_spectra_row = part_spectra + p * nlam;

    /* Sum this particle's contribution across all wavelengths. */
    for (npy_intp ilam = 0; ilam < nlam; ilam++) {
      spectra[ilam] = std::fma(static_cast<OutT>(part_spectra_row[ilam]),
                               static_cast<OutT>(1.0), spectra[ilam]);
    }
  }
}

/**
 * @brief Reduce per-particle spectra to an integrated spectrum in parallel.
 *
 * The work is parallelised over particles while each thread accumulates into a
 * shared output reduction or a thread-local buffer, depending on available
 * OpenMP features.
 *
 * @tparam Real The floating-point type of the input per-particle spectra.
 * @tparam OutT The floating-point type stored in the reduced spectrum.
 *
 * @param spectra The output array to accumulate the spectra.
 * @param part_spectra The per-particle spectra array.
 * @param nlam The number of wavelengths in the spectra.
 * @param npart The number of particles.
 * @param nthreads The number of threads to use.
 */
#ifdef WITH_OPENMP
template <typename Real, typename OutT>
static void reduce_particle_spectra_parallel_typed(OutT *spectra,
                                                   const Real *part_spectra,
                                                   npy_intp nlam,
                                                   npy_intp npart,
                                                   int nthreads) {

  /* Cast once so the OpenMP loop uses a size_t counter. */
  const size_t npart_size = static_cast<size_t>(npart);

#if defined(_OPENMP) && _OPENMP >= 201511
  /* Use an OpenMP array reduction when the compiler supports it. */
#pragma omp parallel for num_threads(nthreads) reduction(+ : spectra[ : nlam])
  for (size_t p = 0; p < npart_size; p++) {
    const Real *__restrict part_spectra_row = part_spectra + p * nlam;
    for (npy_intp ilam = 0; ilam < nlam; ilam++) {
      spectra[ilam] += static_cast<OutT>(part_spectra_row[ilam]);
    }
  }
#else
  /* Fall back to thread-local accumulators when array reductions are not
   * available. */
#pragma omp parallel num_threads(nthreads)
  {
    std::vector<OutT> local((size_t)nlam, static_cast<OutT>(0.0));
#pragma omp for nowait schedule(static)
    for (size_t p = 0; p < npart_size; p++) {
      const Real *__restrict part_spectra_row = part_spectra + p * nlam;
      for (npy_intp ilam = 0; ilam < nlam; ilam++) {
        local[(size_t)ilam] += static_cast<OutT>(part_spectra_row[ilam]);
      }
    }

    /* Merge each thread-local spectrum into the shared output. */
#pragma omp critical
    {
      for (npy_intp ilam = 0; ilam < nlam; ilam++) {
        spectra[ilam] += local[(size_t)ilam];
      }
    }
  }
#endif
}
#endif

/**
 * @brief Execute typed particle spectra reduction after dtype dispatch.
 *
 * @tparam Real The floating-point type of the validated input spectra.
 * @tparam OutT The requested floating-point output type.
 *
 * @param np_part_spectra The validated 2D per-particle spectra array.
 * @param nthreads The number of threads to use.
 *
 * @return A one-dimensional NumPy array containing the reduced spectrum, or
 *         NULL on failure.
 */
template <typename Real, typename OutT>
static PyObject *reduce_particle_spectra_typed(PyArrayObject *np_part_spectra,
                                               int nthreads) {
  /* Extract the particle and wavelength dimensions from the validated input
   * array. The wrapper has already guaranteed a 2D contiguous float array. */
  const npy_intp *part_dims = PyArray_DIMS(np_part_spectra);
  const npy_intp npart = part_dims[0];
  const npy_intp nlam = part_dims[1];

  /* Allocate the reduced one-dimensional output spectrum in the requested
   * output precision. */
  OutT *spectra = new (std::nothrow) OutT[(size_t)nlam]();
  if (spectra == NULL) {
    PyErr_NoMemory();
    return NULL;
  }

  /* Grab a raw pointer once so the hot reduction loop stays free of NumPy API
   * access. */
  const Real *part_spectra = data_ptr<const Real>(np_part_spectra);

  /* Select the serial or OpenMP reduction once the input/output dtypes are
   * known. */
  if (nthreads > 1) {
#ifdef WITH_OPENMP
    reduce_particle_spectra_parallel_typed<Real, OutT>(spectra, part_spectra,
                                                       nlam, npart, nthreads);
#else
    reduce_particle_spectra_serial_typed<Real, OutT>(spectra, part_spectra,
                                                     nlam, npart);
#endif
  } else {
    reduce_particle_spectra_serial_typed<Real, OutT>(spectra, part_spectra,
                                                     nlam, npart);
  }

  /* Wrap the raw buffer as a one-dimensional NumPy array. */
  npy_intp out_dims[1] = {nlam};
  PyArrayObject *result = wrap_array_to_numpy<OutT>(1, out_dims, spectra);
  if (result == NULL) {
    delete[] spectra;
    return NULL;
  }

  /* Transfer ownership of the wrapped NumPy array back to Python. */
  return Py_BuildValue("N", result);
}

/**
 * @brief Reduce per-particle spectra to a single integrated spectrum.
 *
 * This exposes the shared C++ reduction used by the particle spectra
 * extensions directly to Python so other per-particle generation paths can
 * avoid falling back to NumPy reductions.
 *
 * Input arrays must already be NumPy arrays with supported floating-point
 * precision and C-contiguous layout. No implicit dtype conversion or copying
 * is performed in Python.
 *
 * Args:
 *   part_spectra (np.ndarray):
 *     A two-dimensional float32 or float64 NumPy array with shape
 *     (npart, nlam) containing per-particle spectra. The array must already
 *     be C-contiguous.
 *   nthreads (int):
 *     The number of threads to use for the reduction. If less than 1 the
 *     serial implementation is used.
 *   out_dtype (dtype):
 *     Requested output dtype, float32 or float64.
 *
 * Returns:
 *   np.ndarray:
 *     A one-dimensional NumPy array with shape (nlam) containing the
 *     integrated spectrum.
 */
PyObject *reduce_particle_spectra(PyObject *self, PyObject *args) {
  (void)self;

  /* Parse the Python-level inputs. */
  PyArrayObject *np_part_spectra;
  int nthreads;
  PyObject *out_dtype;

  if (!PyArg_ParseTuple(args, "O!iO", &PyArray_Type, &np_part_spectra,
                        &nthreads, &out_dtype)) {
    return NULL;
  }

  /* Validate that we have a two-dimensional array with shape
   * (npart, nlam). This helper is intentionally specialised to the common
   * per-particle spectra reduction case used by the Python operations layer. */
  if (PyArray_NDIM(np_part_spectra) != 2) {
    PyErr_SetString(PyExc_ValueError,
                    "part_spectra must be a 2D NumPy array.");
    return NULL;
  }

  /* Enforce one supported floating-point precision family for the input. */
  PyArrayObject *float_arrays[] = {np_part_spectra};
  const char *float_names[] = {"part_spectra"};
  int input_typenum = -1;
  if (!is_matching_float_dtypes(float_arrays, float_names, 1,
                                &input_typenum)) {
    return NULL;
  }

  /* Resolve the independently requested output dtype. */
  const int output_typenum = resolve_output_typenum(out_dtype, "out_dtype");
  if (output_typenum < 0) {
    return NULL;
  }

  /* Dispatch to the matching Real/OutT reduction path. */
  if (input_typenum == NPY_FLOAT32) {
    if (output_typenum == NPY_FLOAT32) {
      return reduce_particle_spectra_typed<float, float>(np_part_spectra,
                                                         nthreads);
    }

    return reduce_particle_spectra_typed<float, double>(np_part_spectra,
                                                        nthreads);
  }

  if (output_typenum == NPY_FLOAT32) {
    return reduce_particle_spectra_typed<double, float>(np_part_spectra,
                                                        nthreads);
  }

  return reduce_particle_spectra_typed<double, double>(np_part_spectra,
                                                       nthreads);
}

/* Python module definition. */
static PyMethodDef ReductionMethods[] = {
    {"reduce_particle_spectra", (PyCFunction)reduce_particle_spectra,
     METH_VARARGS,
     "Reduce per-particle spectra to a single integrated spectrum."},
    {NULL, NULL, 0, NULL}};

/* Make this importable. */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "reductions",                              /* m_name */
    "A module containing spectra reduction kernels", /* m_doc */
    -1,                                         /* m_size */
    ReductionMethods,                           /* m_methods */
    NULL,                                       /* m_reload */
    NULL,                                       /* m_traverse */
    NULL,                                       /* m_clear */
    NULL,                                       /* m_free */
};

PyMODINIT_FUNC PyInit_reductions(void) {
  /* Import the shared NumPy C API before exposing the module. */
  if (numpy_import() < 0) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to import numpy.");
    return NULL;
  }

  /* Create the Python module only after the NumPy API is ready. */
  PyObject *m = PyModule_Create(&moduledef);
  if (m == NULL)
    return NULL;
#ifdef ATOMIC_TIMING
  /* Import the shared timing capsule when atomic timing is enabled. */
  if (import_toc_capsule() < 0) {
    Py_DECREF(m);
    return NULL;
  }
#endif
  return m;
}
