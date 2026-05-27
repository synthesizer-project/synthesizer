/* Standard includes */
#include <cmath>
#include <vector>

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
    double *__restrict part_spectra_row = part_spectra + p * nlam;

    /* Loop over wavelengths. */
    for (int ilam = 0; ilam < nlam; ilam++) {
      /* Use fused multiply-add to accumulate with better precision.
       * Equivalent to: += spec_val * weight, but with a single rounding. */
      spectra[ilam] = std::fma(part_spectra_row[ilam], 1.0, spectra[ilam]);
    }
  }
}

#ifdef WITH_OPENMP
/**
 * @brief Reduce Npart spectra to integrated spectra in parallel.
 *
 * @param spectra: The output array to accumulate the spectra.
 * @param part_spectra: The per-particle spectra array.
 * @param nlam: The number of wavelengths in the spectra.
 * @param npart: The number of particles.
 * @param nthreads: The number of threads to use.
 */
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
    std::vector<double> local(nlam, 0.0);
#pragma omp for nowait schedule(static)
    for (size_t p = 0; p < npart_size; p++) {
      double *__restrict part_spectra_row = part_spectra + p * nlam;
      for (int ilam = 0; ilam < nlam; ilam++) {
        local[ilam] += part_spectra_row[ilam];
      }
    }
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

/**
 * @brief Combine a sequence of same-shaped 2D spectra arrays, skipping NaNs.
 */
PyObject *combine_spectra_list_2d(PyObject *self, PyObject *args) {
  (void)self;

  PyObject *spectra_seq_obj;
  int nthreads;

  if (!PyArg_ParseTuple(args, "Oi", &spectra_seq_obj, &nthreads)) {
    return NULL;
  }

  PyObject *seq = PySequence_Fast(
      spectra_seq_obj, "spectra_list must be a sequence of 2D arrays.");
  if (seq == NULL) {
    return NULL;
  }

  const Py_ssize_t nspectra = PySequence_Fast_GET_SIZE(seq);
  if (nspectra < 1) {
    Py_DECREF(seq);
    PyErr_SetString(PyExc_ValueError,
                    "spectra_list must contain at least one array.");
    return NULL;
  }

  std::vector<PyArrayObject *> arrays;
  arrays.reserve(nspectra);

  for (Py_ssize_t i = 0; i < nspectra; i++) {
    PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
    PyArrayObject *arr = (PyArrayObject *)PyArray_FROM_OTF(
        item, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (arr == NULL) {
      for (PyArrayObject *loaded : arrays) {
        Py_DECREF(loaded);
      }
      Py_DECREF(seq);
      return NULL;
    }
    if (PyArray_NDIM(arr) != 2) {
      for (PyArrayObject *loaded : arrays) {
        Py_DECREF(loaded);
      }
      Py_DECREF(arr);
      Py_DECREF(seq);
      PyErr_SetString(PyExc_ValueError,
                      "All spectra arrays must be 2D float64 arrays.");
      return NULL;
    }
    arrays.push_back(arr);
  }

  const npy_intp *template_dims = PyArray_DIMS(arrays[0]);
  const int nspec = static_cast<int>(template_dims[0]);
  const int nlam = static_cast<int>(template_dims[1]);
  for (Py_ssize_t i = 1; i < nspectra; i++) {
    const npy_intp *dims = PyArray_DIMS(arrays[i]);
    if (dims[0] != template_dims[0] || dims[1] != template_dims[1]) {
      for (PyArrayObject *loaded : arrays) {
        Py_DECREF(loaded);
      }
      Py_DECREF(seq);
      PyErr_SetString(PyExc_ValueError,
                      "All spectra arrays must have identical 2D shapes.");
      return NULL;
    }
  }

  PyArrayObject *np_out = (PyArrayObject *)PyArray_ZEROS(
      2, const_cast<npy_intp *>(template_dims), NPY_DOUBLE, 0);
  if (np_out == NULL) {
    for (PyArrayObject *loaded : arrays) {
      Py_DECREF(loaded);
    }
    Py_DECREF(seq);
    return NULL;
  }

  std::vector<double *> spectra_ptrs;
  spectra_ptrs.reserve(nspectra);
  for (PyArrayObject *arr : arrays) {
    spectra_ptrs.push_back(static_cast<double *>(PyArray_DATA(arr)));
  }
  double *out = static_cast<double *>(PyArray_DATA(np_out));
  const int nelem = nspec * nlam;

  tic("combine_spectra_list_2d");

#ifdef WITH_OPENMP
#pragma omp parallel for if(nthreads > 1) num_threads(nthreads) schedule(static)
#endif
  for (int idx = 0; idx < nelem; idx++) {
    double total = 0.0;
    for (Py_ssize_t ispec = 0; ispec < nspectra; ispec++) {
      const double value = spectra_ptrs[ispec][idx];
      if (!std::isnan(value)) {
        total += value;
      }
    }
    out[idx] = total;
  }

  toc("combine_spectra_list_2d");

  for (PyArrayObject *arr : arrays) {
    Py_DECREF(arr);
  }
  Py_DECREF(seq);
  return Py_BuildValue("N", np_out);
}

static PyMethodDef ReductionMethods[] = {
    {"reduce_particle_spectra", (PyCFunction)reduce_particle_spectra,
     METH_VARARGS,
     "Method for reducing per-particle spectra to an integrated spectrum."},
    {"combine_spectra_list_2d", (PyCFunction)combine_spectra_list_2d,
     METH_VARARGS,
     "Combine same-shaped 2D spectra arrays while skipping NaNs."},
    {NULL, NULL, 0, NULL}};

/* Make this importable. */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "reductions",                     /* m_name */
    "A module containing spectra reductions", /* m_doc */
    -1,
    ReductionMethods,
    NULL,
    NULL,
    NULL,
    NULL,
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
