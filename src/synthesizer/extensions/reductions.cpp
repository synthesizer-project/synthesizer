/******************************************************************************
 * C extension for reducing per-particle spectra.
 *
 * This module provides a shared templated reduction kernel used both by the
 * spectra extraction extensions and by the Python-facing wrapper.
 *
 * The core operation is a reduction over the leading particle axis:
 *
 *     result[lam] = sum_p part_spectra[p, lam]
 *
 * Serial and parallel (OpenMP) implementations are provided for the shared
 * templated path.
 *****************************************************************************/

/* C/C++ includes */
#include <cmath>
#include <new>
#include <vector>

#ifdef WITH_OPENMP
#include <omp.h>
#endif

/* Python includes */
#define PY_ARRAY_UNIQUE_SYMBOL SYNTHESIZER_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy_init.h"

#include <Python.h>

/* Local includes */
#include "cpp_to_python.h"
#include "floating_point_utils.h"
#include "python_to_cpp.h"
#include "reductions.h"
#include "timers.h"
#ifdef ATOMIC_TIMING
#include "timers_init.h"
#endif

/* The combine loop below is emitted once per specialised input count, so the
 * OpenMP directive has to come from a macro. It expands to nothing when the
 * build has no OpenMP support. */
#ifdef WITH_OPENMP
#define SYNTH_COMBINE_PRAGMA \
  _Pragma("omp parallel for num_threads(nthr) schedule(static)")
#else
#define SYNTH_COMBINE_PRAGMA
#endif

/**
 * @brief Return a value, or zero when it is NaN.
 *
 * Used instead of a branch so the combine loop stays vectorisable.
 *
 * @param value: The value to test.
 */
template <typename Real>
static inline Real nan_to_zero(Real value) {
  return is_nan_bits(value) ? static_cast<Real>(0) : value;
}

/* Rows summed in the input precision before the running total is widened.
 *
 * Adding float32 rows straight into a double accumulator forces a widening
 * conversion on every element and halves the usable vector width: measured
 * 21 GB/s against 52 GB/s for a same-width accumulation. Summing a block of
 * rows at the input precision and folding that block into the double total
 * keeps the inner loop at one width, while the rounding error grows with the
 * block length rather than with the particle count. At this block size the
 * result sits 4.3e-8 from the all-double answer for 100k float32 rows, which
 * is inside float32's own epsilon. */
static const size_t REDUCE_BLOCK = 128;

/**
 * @brief Sum a range of per-particle spectra into a double accumulator.
 *
 * @tparam Real The floating-point type of the input per-particle spectra.
 *
 * @param accum: The double precision accumulator, length nlam.
 * @param part_spectra: The per-particle spectra array.
 * @param start: The first particle in this range.
 * @param end: One past the last particle in this range.
 * @param nlam: The number of wavelengths in the spectra.
 * @param block: Scratch of length nlam used to sum a block of rows at the
 *     input precision, or NULL to sum straight into the accumulator.
 */
template <typename Real>
static void reduce_spectra_range(double *accum, const Real *part_spectra,
                                 size_t start, size_t end, int nlam,
                                 Real *block) {

  /* Without a scratch buffer, or when the input is already double precision
   * and there is no widening to avoid, sum straight into the accumulator. */
  if (block == NULL || sizeof(Real) == sizeof(double)) {
    for (size_t p = start; p < end; p++) {
      const Real *__restrict row = part_spectra + p * nlam;
      for (int ilam = 0; ilam < nlam; ilam++) {
        accum[ilam] += static_cast<double>(row[ilam]);
      }
    }
    return;
  }

  /* Otherwise sum a block of rows at the input precision. */
  for (size_t p0 = start; p0 < end; p0 += REDUCE_BLOCK) {
    const size_t p1 = (p0 + REDUCE_BLOCK < end) ? p0 + REDUCE_BLOCK : end;

    for (int ilam = 0; ilam < nlam; ilam++) {
      block[ilam] = static_cast<Real>(0);
    }

    for (size_t p = p0; p < p1; p++) {
      const Real *__restrict row = part_spectra + p * nlam;
      for (int ilam = 0; ilam < nlam; ilam++) {
        block[ilam] += row[ilam];
      }
    }

    for (int ilam = 0; ilam < nlam; ilam++) {
      accum[ilam] += static_cast<double>(block[ilam]);
    }
  }
}

/**
 * @brief Reduce Npart spectra to integrated spectra in serial.
 *
 * @tparam Real The floating-point type of the input per-particle spectra.
 *
 * @param accum: The double precision accumulator, length nlam.
 * @param part_spectra: The per-particle spectra array.
 * @param nlam: The number of wavelengths in the spectra.
 * @param npart: The number of particles.
 */
template <typename Real>
static void reduce_spectra_serial(double *accum, const Real *part_spectra,
                                  int nlam, int npart, bool blocked) {
  std::vector<Real> block(blocked ? nlam : 0);
  reduce_spectra_range<Real>(accum, part_spectra, 0,
                             static_cast<size_t>(npart), nlam,
                             blocked ? block.data() : NULL);
}

/**
 * @brief Reduce Npart spectra to integrated spectra in parallel.
 *
 * Each thread sums its own particle range into a private accumulator and the
 * accumulators are folded together once at the end, so there is no sharing on
 * the hot loop.
 *
 * @tparam Real The floating-point type of the input per-particle spectra.
 *
 * @param accum: The double precision accumulator, length nlam.
 * @param part_spectra: The per-particle spectra array.
 * @param nlam: The number of wavelengths in the spectra.
 * @param npart: The number of particles.
 * @param nthreads: The number of threads to use.
 */
#ifdef WITH_OPENMP
template <typename Real>
static void reduce_spectra_parallel(double *accum, const Real *part_spectra,
                                    int nlam, int npart, int nthreads,
                                    bool blocked) {

  const size_t npart_size = static_cast<size_t>(npart);

#pragma omp parallel num_threads(nthreads)
  {
    std::vector<double> local(nlam, 0.0);
    std::vector<Real> block(blocked ? nlam : 0);

    const int nthr = omp_get_num_threads();
    const int tid = omp_get_thread_num();
    const size_t per_thread = (npart_size + nthr - 1) / nthr;
    const size_t start = static_cast<size_t>(tid) * per_thread;
    const size_t end =
        (start + per_thread < npart_size) ? start + per_thread : npart_size;

    if (start < end) {
      reduce_spectra_range<Real>(local.data(), part_spectra, start, end, nlam,
                                 blocked ? block.data() : NULL);
    }

#pragma omp critical
    {
      for (int ilam = 0; ilam < nlam; ilam++) {
        accum[ilam] += local[ilam];
      }
    }
  }
}
#endif

/**
 * @brief Reduce Npart spectra to integrated spectra.
 *
 * @tparam Real The floating-point type of the input per-particle spectra.
 * @tparam OutT The floating-point type stored in the reduced spectrum.
 *
 * @param spectra: The output array to accumulate the spectra.
 * @param part_spectra: The per-particle spectra array.
 * @param nlam: The number of wavelengths in the spectra.
 * @param npart: The number of particles.
 * @param nthreads: The number of threads to use.
 */
template <typename Real, typename OutT>
void reduce_spectra(OutT *spectra, const Real *part_spectra, int nlam,
                    int npart, int nthreads) {

  tic("reduce_spectra");

  /* The running total is always double precision so a reduced precision
   * output doesn't accumulate error over many particles. */
  std::vector<double> accum(nlam, 0.0);

  /* Sum blocks of rows at the input precision only when the result is stored
   * at that precision too. The blocking error is then the same order as the
   * rounding the store already incurs, while a float64 output keeps the exact
   * double precision accumulation it asked for. */
  const bool blocked =
      sizeof(Real) == sizeof(float) && sizeof(OutT) == sizeof(float);

  if (nthreads > 1) {
#ifdef WITH_OPENMP
    reduce_spectra_parallel<Real>(accum.data(), part_spectra, nlam, npart,
                                  nthreads, blocked);
#else
    reduce_spectra_serial<Real>(accum.data(), part_spectra, nlam, npart,
                                blocked);
#endif
  } else {
    reduce_spectra_serial<Real>(accum.data(), part_spectra, nlam, npart,
                                blocked);
  }

  /* Fold the double precision accumulation into the output buffer. */
  for (int ilam = 0; ilam < nlam; ilam++) {
    spectra[ilam] += static_cast<OutT>(accum[ilam]);
  }

  toc("reduce_spectra");
}

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
static PyObject *reduce_particle_spectra_impl(PyArrayObject *np_part_spectra,
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

  reduce_spectra<Real, OutT>(spectra, part_spectra, (int)nlam, (int)npart,
                             nthreads);

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
   * per-particle spectra reduction case used by the Python operations layer.
   */
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

  /* Dispatch: call the matching typed kernel for the input/output dtypes. */
  return dispatch_float(input_typenum, [&](auto in) -> PyObject * {
    return dispatch_float(output_typenum, [&](auto o) -> PyObject * {
      using InReal = decltype(in);
      using OutT = decltype(o);
      return reduce_particle_spectra_impl<InReal, OutT>(np_part_spectra,
                                                        nthreads);
    });
  });
}

/**
 * @brief Combine equally shaped per-particle spectra, ignoring NaNs.
 *
 * Inputs must be C-contiguous 2D arrays sharing one supported floating-point
 * dtype. The output is allocated in that dtype and populated in one pass, so
 * no input conversion or temporary boolean-index arrays are needed.
 */
PyObject *combine_spectra_2d(PyObject *self, PyObject *args) {
  (void)self;

  PyObject *inputs_sequence;
  int nthreads;
  if (!PyArg_ParseTuple(args, "Oi", &inputs_sequence, &nthreads)) {
    return NULL;
  }

  PyObject *inputs_fast = PySequence_Fast(
      inputs_sequence, "inputs must be a sequence of 2D NumPy arrays.");
  if (inputs_fast == NULL) {
    return NULL;
  }

  const Py_ssize_t ninputs = PySequence_Fast_GET_SIZE(inputs_fast);
  if (ninputs == 0) {
    Py_DECREF(inputs_fast);
    PyErr_SetString(PyExc_ValueError,
                    "inputs must contain at least one array.");
    return NULL;
  }

  PyObject **items = PySequence_Fast_ITEMS(inputs_fast);
  std::vector<PyArrayObject *> arrays;
  arrays.reserve((size_t)ninputs);

  npy_intp nrow = -1;
  npy_intp nlam = -1;
  int input_typenum = -1;
  for (Py_ssize_t i = 0; i < ninputs; ++i) {
    if (!PyArray_Check(items[i])) {
      Py_DECREF(inputs_fast);
      PyErr_SetString(PyExc_TypeError, "all inputs must be NumPy arrays.");
      return NULL;
    }

    auto *array = reinterpret_cast<PyArrayObject *>(items[i]);
    if (PyArray_NDIM(array) != 2) {
      Py_DECREF(inputs_fast);
      PyErr_SetString(PyExc_ValueError, "all inputs must be 2D arrays.");
      return NULL;
    }
    if (!is_c_contiguous(array, "inputs") ||
        !is_float32_or_float64(array, "inputs")) {
      Py_DECREF(inputs_fast);
      return NULL;
    }

    if (i == 0) {
      nrow = PyArray_DIM(array, 0);
      nlam = PyArray_DIM(array, 1);
      input_typenum = PyArray_TYPE(array);
    } else if (PyArray_DIM(array, 0) != nrow ||
               PyArray_DIM(array, 1) != nlam) {
      Py_DECREF(inputs_fast);
      PyErr_SetString(PyExc_ValueError,
                      "all inputs must have the same shape.");
      return NULL;
    } else if (PyArray_TYPE(array) != input_typenum) {
      Py_DECREF(inputs_fast);
      PyErr_SetString(PyExc_TypeError,
                      "all inputs must have the same floating-point dtype.");
      return NULL;
    }

    arrays.push_back(array);
  }

  tic("combine_spectra_2d");
  PyObject *result =
      dispatch_float(input_typenum, [&](auto value) -> PyObject * {
        using Real = decltype(value);
        const size_t size = (size_t)PyArray_SIZE(arrays[0]);
        Real *output = new (std::nothrow) Real[size];
        if (output == NULL) {
          PyErr_NoMemory();
          return NULL;
        }

        std::vector<const Real *> input_ptrs;
        input_ptrs.reserve((size_t)ninputs);
        for (PyArrayObject *array : arrays) {
          input_ptrs.push_back(data_ptr<const Real>(array));
        }

        /* The input count is a runtime value and the NaN test was a
         * branch, which together stopped the loop vectorising. Specialising
         * the small counts and selecting rather than branching lets it
         * vectorise. */
        const Real *const *ins = input_ptrs.data();
        const int nin = (int)ninputs;
        const int nthr = nthreads > 1 ? nthreads : 1;
        (void)nthr; /* Only read by the OpenMP directive. */

#define SYNTH_COMBINE_INPUTS(N)                                       \
  case N: {                                                           \
    const Real *in[N];                                                \
    for (int k = 0; k < N; k++) in[k] = ins[k];                       \
    SYNTH_COMBINE_PRAGMA                                              \
    for (npy_intp index = 0; index < (npy_intp)size; ++index) {       \
      Real total = 0;                                                 \
      for (int k = 0; k < N; k++) total += nan_to_zero(in[k][index]); \
      output[index] = total;                                          \
    }                                                                 \
    break;                                                            \
  }

        switch (nin) {
          SYNTH_COMBINE_INPUTS(1)
          SYNTH_COMBINE_INPUTS(2)
          SYNTH_COMBINE_INPUTS(3)
          SYNTH_COMBINE_INPUTS(4)
          SYNTH_COMBINE_INPUTS(5)
          SYNTH_COMBINE_INPUTS(6)
          default: {
            SYNTH_COMBINE_PRAGMA
            for (npy_intp index = 0; index < (npy_intp)size; ++index) {
              Real total = 0;
              for (int k = 0; k < nin; k++) {
                total += nan_to_zero(ins[k][index]);
              }
              output[index] = total;
            }
            break;
          }
        }
#undef SYNTH_COMBINE_INPUTS

        npy_intp output_dims[2] = {nrow, nlam};
        return reinterpret_cast<PyObject *>(
            wrap_array_to_numpy<Real>(2, output_dims, output));
      });
  toc("combine_spectra_2d");

  Py_DECREF(inputs_fast);
  return result;
}

template void reduce_spectra<float, float>(float *, const float *, int, int,
                                           int);
template void reduce_spectra<float, double>(double *, const float *, int, int,
                                            int);
template void reduce_spectra<double, float>(float *, const double *, int, int,
                                            int);
template void reduce_spectra<double, double>(double *, const double *, int,
                                             int, int);

/* Python module definition. */
static PyMethodDef ReductionMethods[] = {
    {"reduce_particle_spectra", (PyCFunction)reduce_particle_spectra,
     METH_VARARGS,
     "Reduce per-particle spectra to a single integrated spectrum."},
    {"combine_spectra_2d", (PyCFunction)combine_spectra_2d, METH_VARARGS,
     "Combine 2D per-particle spectra without temporary arrays."},
    {NULL, NULL, 0, NULL}};

/* Make this importable. */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "reductions",                                    /* m_name */
    "A module containing spectra reduction kernels", /* m_doc */
    -1,                                              /* m_size */
    ReductionMethods,                                /* m_methods */
    NULL,                                            /* m_reload */
    NULL,                                            /* m_traverse */
    NULL,                                            /* m_clear */
    NULL,                                            /* m_free */
};

PyMODINIT_FUNC PyInit_reductions(void) {
  /* Import the shared NumPy C API before exposing the module. */
  if (numpy_import() < 0) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to import numpy.");
    return NULL;
  }

  /* Create the Python module only after the NumPy API is ready. */
  PyObject *m = PyModule_Create(&moduledef);
  if (m == NULL) return NULL;
#ifdef ATOMIC_TIMING
  /* Import the shared timing capsule when atomic timing is enabled. */
  if (import_toc_capsule() < 0) {
    Py_DECREF(m);
    return NULL;
  }
#endif
  return m;
}
