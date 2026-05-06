/* Standard includes */
#include <cmath>

/* Python includes */
#include <Python.h>

/* Local includes */
#include "cpp_to_python.h"
#include "reductions.h"
#include "timers.h"
#include "timers_init.h"

/**
 * @brief Reduce Npart spectra to integrated spectra.
 *
 * This is a serial version of the function.
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
    /* Loop over wavelengths. */
    for (int ilam = 0; ilam < nlam; ilam++) {
      size_t part_spec_ind = p * nlam + ilam;
      /* Use fused multiply-add to accumulate with better precision.
       * Equivalent to: += spec_val * weight, but with a single rounding. */
      spectra[ilam] = std::fma(part_spectra[part_spec_ind], 1.0, spectra[ilam]);
    }
  }
}

/**
 * @brief Reduce Npart spectra to integrated spectra.
 *
 * This is a parallel version of the function.
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
    for (int ilam = 0; ilam < nlam; ilam++) {
      size_t part_spec_ind = p * nlam + ilam;
      spectra[ilam] += part_spectra[part_spec_ind];
    }
  }
#else // OpenMP < 4.5 or no array reduction support
#pragma omp parallel num_threads(nthreads)
  {
    // Thread-local accumulation to avoid false sharing and atomics
    std::vector<double> local(nlam, 0.0);
#pragma omp for nowait schedule(static)
    for (size_t p = 0; p < npart; p++) {
      for (int ilam = 0; ilam < nlam; ilam++) {
        size_t part_spec_ind = p * nlam + ilam;
        local[ilam] += part_spectra[part_spec_ind];
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
 * This is a wrapper function that calls the serial or parallel version of the
 * function depending on the number of threads requested or whether OpenMP is
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
 * @brief Reduce per-particle spectra to a single integrated spectrum.
 *
 * This exposes the shared C++ reduction used by the particle spectra
 * extensions directly to Python so other per-particle generation paths can
 * avoid falling back to NumPy reductions.
 *
 * Args:
 *   part_spectra (np.ndarray):
 *     A two-dimensional float64 NumPy array with shape (npart, nlam)
 *     containing per-particle spectra.
 *   nthreads (int):
 *     The number of threads to use for the reduction. If less than 1 the
 *     serial implementation is used.
 *
 * Returns:
 *   np.ndarray:
 *     A one-dimensional float64 NumPy array with shape (nlam) containing the
 *     integrated spectrum.
 */
PyObject *reduce_particle_spectra(PyObject *self, PyObject *args) {
  /* We don't need the self argument but it has to be there. Tell the
   * compiler we don't care. */
  (void)self;

  /* Declare the Python-level inputs. We accept the per-particle spectra array
   * and the requested thread count. */
  PyObject *part_spectra_obj;
  int nthreads;

  /* Parse the Python arguments. The expected signature is
   * reduce_particle_spectra(part_spectra, nthreads). */
  if (!PyArg_ParseTuple(args, "Oi", &part_spectra_obj, &nthreads)) {
    return NULL;
  }

  /* Convert the input into a float64 NumPy array view. This guarantees that
   * the reduction kernel sees a NumPy array with the dtype it expects. */
  PyArrayObject *np_part_spectra = (PyArrayObject *)PyArray_FROM_OTF(
      part_spectra_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (np_part_spectra == NULL) {
    return NULL;
  }

  /* Validate that we have a two-dimensional array with shape
   * (npart, nlam). This helper is intentionally specialised to the common
   * per-particle spectra reduction case used by the Python operations layer. */
  if (PyArray_NDIM(np_part_spectra) != 2) {
    Py_DECREF(np_part_spectra);
    PyErr_SetString(PyExc_ValueError,
                    "part_spectra must be a 2D float64 NumPy array.");
    return NULL;
  }

  /* Extract the input dimensions so we can size the integrated output array
   * and call the shared reduction kernel. */
  const npy_intp *part_dims = PyArray_DIMS(np_part_spectra);
  int npart = static_cast<int>(part_dims[0]);
  int nlam = static_cast<int>(part_dims[1]);

  /* Allocate the one-dimensional integrated spectrum output array. */
  npy_intp out_dims[1] = {part_dims[1]};
  PyArrayObject *np_spectra =
      (PyArrayObject *)PyArray_ZEROS(1, out_dims, NPY_DOUBLE, 0);
  if (np_spectra == NULL) {
    Py_DECREF(np_part_spectra);
    return NULL;
  }

  /* Extract raw pointers to the NumPy buffers so the shared reduction code
   * can operate directly on contiguous double arrays. */
  double *spectra = static_cast<double *>(PyArray_DATA(np_spectra));
  double *part_spectra = static_cast<double *>(PyArray_DATA(np_part_spectra));

  /* Reduce the per-particle spectra onto the integrated spectrum. This reuses
   * the same C++ reduction kernel already used by the extraction path. */
  reduce_spectra(spectra, part_spectra, nlam, npart, nthreads);

  /* Drop our temporary reference to the input array view now that the
   * reduction has completed. */
  Py_DECREF(np_part_spectra);

  /* Return the newly created integrated spectrum array to Python. */
  return Py_BuildValue("N", np_spectra);
}

/* Below is all the gubbins needed to make the module importable in Python. */
static PyMethodDef ReductionMethods[] = {
    {"reduce_particle_spectra", (PyCFunction)reduce_particle_spectra,
     METH_VARARGS,
     "Method for reducing per-particle spectra to an integrated spectrum."},
    {NULL, NULL, 0, NULL}};

/* Make this importable. */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "reductions",                              /* m_name */
    "A module containing spectra reductions", /* m_doc */
    -1,                                         /* m_size */
    ReductionMethods,                           /* m_methods */
    NULL,                                       /* m_reload */
    NULL,                                       /* m_traverse */
    NULL,                                       /* m_clear */
    NULL,                                       /* m_free */
};

PyMODINIT_FUNC PyInit_reductions(void) {
  PyObject *m = PyModule_Create(&moduledef);
  if (m == NULL)
    return NULL;
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
