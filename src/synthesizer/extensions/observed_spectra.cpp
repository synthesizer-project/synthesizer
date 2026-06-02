/* ****************************************************************************
 * C extension to convert rest-frame luminosity density spectra into
 * observer-frame flux density spectra.
 *
 * This fills observer-frame wavelength, frequency, and flux buffers in place
 * so Python callers can avoid large temporary array allocations on the hot
 * path.
 * ************************************************************************** */
/* Standard includes */
#include <cmath>
#include <memory>

/* Python includes */
#include <Python.h>

/* Local includes */
#include "cpp_to_python.h"
#include "python_to_cpp.h"
#include "timers.h"
#ifdef ATOMIC_TIMING
#include "timers_init.h"
#endif

/**
 * @brief Populate observer-frame wavelength, frequency, and flux arrays.
 *
 * This expects preallocated NumPy output buffers provided by the Python layer
 * and fills them in place. The flux conversion is applied to the full spectra
 * array, while the observer-frame wavelength and frequency grids are populated
 * from the one-dimensional emitted grids.
 *
 * @param lnu: The input rest-frame luminosity density spectra, shape (...,
 * nlam).
 * @param lam: The input rest-frame wavelength grid, shape (nlam).
 * @param nu: The input rest-frame frequency grid, shape (nlam).
 * @param one_plus_z: The redshift factor (1 + z) to apply to the wavelength and
 * frequency grids.
 * @param conversion: The flux conversion factor to apply to the luminosity
 * density spectra to convert to flux density. This should already include the
 * (1 + z) factor for the bandwidth compression, so the kernel just applies it
 * as a simple scaling.
 * @param nthreads: The number of threads to use for the flux conversion. If
 * less than 1, the conversion is done in a single thread. The wavelength and
 * frequency grid population is not parallelised since it's typically much
 * smaller than the spectra arrays.
 * @param fnu_out: The preallocated output array for the observer-frame flux
 * density spectra, shape (..., nlam). Must be writable and have the same shape
 * and memory layout as lnu.
 * @param obslam_out: The preallocated output array for the observer-frame
 * wavelength grid, shape (nlam). Must be writable and have the same length as
 * lam. Can be NULL if the caller does not need this output.
 * @param obsnu_out: The preallocated output array for the observer-frame
 * frequency grid, shape (nlam). Must be writable and have the same length as
 * nu. Can be NULL if the caller does not need this output.
 * @param nelem: The total number of elements in the spectra arrays (the product
 * of all dimensions of lnu). This is used to drive the parallel loop for the
 * flux conversion.
 * @param nlam: The number of wavelength/frequency bins (the size of the last
 * dimension of lnu and the length of lam/nu). This is used to drive the loops
 * for populating the observer-frame grids.
 */
template <typename Real, typename OutT>
static void compute_fnu_typed(const Real *lnu, const Real *lam, const Real *nu,
                              Real one_plus_z, Real conversion, int nthreads,
                              OutT *fnu_out, OutT *obslam_out, OutT *obsnu_out,
                              npy_intp nelem, npy_intp nlam) {
#ifdef WITH_OPENMP
#pragma omp parallel for if (nthreads > 1) num_threads(nthreads)               \
    schedule(static)
#endif
  for (npy_intp idx = 0; idx < nelem; idx++) {
    fnu_out[idx] = static_cast<OutT>(lnu[idx] * conversion);
  }

  if (obslam_out != NULL && obsnu_out != NULL) {
    for (npy_intp ilam = 0; ilam < nlam; ilam++) {
      obslam_out[ilam] = static_cast<OutT>(lam[ilam] * one_plus_z);
      obsnu_out[ilam] = static_cast<OutT>(nu[ilam] / one_plus_z);
    }
  }
}

/*
 * @brief Python wrapper for compute_fnu_typed with dtype dispatch and
 * validation.
 *
 * This parses the Python-level inputs, validates them, and dispatches to the
 * correct typed kernel based on the input/output dtypes. The output arrays must
 * already be allocated by the caller and have the correct shape and memory
 * layout; no copying or allocation is done in Python.
 */
PyObject *compute_fnu(PyObject *self, PyObject *args) {
  (void)self;

  PyObject *lnu_obj;
  PyObject *lam_obj;
  PyObject *nu_obj;
  double one_plus_z;
  double conversion;
  int nthreads;
  PyObject *fnu_out_obj;
  PyObject *obslam_out_obj;
  PyObject *obsnu_out_obj;
  PyObject *out_dtype_obj = NULL;

  if (!PyArg_ParseTuple(args, "OOOddiOOO|O", &lnu_obj, &lam_obj, &nu_obj,
                        &one_plus_z, &conversion, &nthreads, &fnu_out_obj,
                        &obslam_out_obj, &obsnu_out_obj, &out_dtype_obj)) {
    return NULL;
  }

  /* Convert required inputs to array views. */
  PyArrayObject *np_lnu = (PyArrayObject *)PyArray_FromAny(
      lnu_obj, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
  PyArrayObject *np_lam = (PyArrayObject *)PyArray_FromAny(
      lam_obj, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
  PyArrayObject *np_nu = (PyArrayObject *)PyArray_FromAny(
      nu_obj, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
  PyArrayObject *np_fnu_out = (PyArrayObject *)PyArray_FromAny(
      fnu_out_obj, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
  if (!np_lnu || !np_lam || !np_nu || !np_fnu_out) {
    Py_XDECREF(np_lnu);
    Py_XDECREF(np_lam);
    Py_XDECREF(np_nu);
    Py_XDECREF(np_fnu_out);
    return NULL;
  }

  PyArrayObject *np_obslam_out =
      cpp_to_python::array_or_none(obslam_out_obj, "obslam_out");
  PyArrayObject *np_obsnu_out =
      cpp_to_python::array_or_none(obsnu_out_obj, "obsnu_out");
  if (PyErr_Occurred()) {
    Py_DECREF(np_lnu);
    Py_DECREF(np_lam);
    Py_DECREF(np_nu);
    Py_DECREF(np_fnu_out);
    Py_XDECREF(np_obslam_out);
    Py_XDECREF(np_obsnu_out);
    return NULL;
  }

  /* Basic shape checks. */
  if (PyArray_NDIM(np_lnu) < 1 || PyArray_NDIM(np_lam) != 1 ||
      PyArray_NDIM(np_nu) != 1) {
    Py_DECREF(np_lnu);
    Py_DECREF(np_lam);
    Py_DECREF(np_nu);
    Py_DECREF(np_fnu_out);
    Py_XDECREF(np_obslam_out);
    Py_XDECREF(np_obsnu_out);
    PyErr_SetString(PyExc_ValueError,
                    "lnu must be at least 1D and lam/nu must be 1D.");
    return NULL;
  }

  const int lnu_ndim = PyArray_NDIM(np_lnu);
  const npy_intp *lnu_dims = PyArray_DIMS(np_lnu);
  const npy_intp nlam = PyArray_DIMS(np_lam)[0];

  if (lnu_dims[lnu_ndim - 1] != nlam) {
    Py_DECREF(np_lnu);
    Py_DECREF(np_lam);
    Py_DECREF(np_nu);
    Py_DECREF(np_fnu_out);
    Py_XDECREF(np_obslam_out);
    Py_XDECREF(np_obsnu_out);
    PyErr_SetString(PyExc_ValueError,
                    "lam/nu length must match the last axis of lnu.");
    return NULL;
  }

  /* Resolve output typenum if provided. */
  int out_typenum = -1;
  if (out_dtype_obj != NULL && out_dtype_obj != Py_None) {
    out_typenum = resolve_output_typenum(out_dtype_obj, "out_dtype");
    if (out_typenum < 0) {
      Py_DECREF(np_lnu);
      Py_DECREF(np_lam);
      Py_DECREF(np_nu);
      Py_DECREF(np_fnu_out);
      Py_XDECREF(np_obslam_out);
      Py_XDECREF(np_obsnu_out);
      return NULL;
    }
  }

  /* Validate that the input arrays are float32 or float64 and match. */
  PyArrayObject *float_arrays[3] = {np_lnu, np_lam, np_nu};
  const char *float_names[3] = {"lnu", "lam", "nu"};
  int input_typenum = -1;
  if (!is_matching_float_dtypes(float_arrays, float_names, 3, &input_typenum)) {
    Py_DECREF(np_lnu);
    Py_DECREF(np_lam);
    Py_DECREF(np_nu);
    Py_DECREF(np_fnu_out);
    Py_XDECREF(np_obslam_out);
    Py_XDECREF(np_obsnu_out);
    return NULL;
  }

  if (out_typenum < 0)
    out_typenum = input_typenum;

  /* Ensure provided outputs (if any) match out_typenum and are C-contiguous. */
  if (PyArray_Check((PyObject *)np_fnu_out)) {
    if (PyArray_TYPE(np_fnu_out) != out_typenum ||
        !PyArray_ISCARRAY(np_fnu_out)) {
      Py_DECREF(np_lnu);
      Py_DECREF(np_lam);
      Py_DECREF(np_nu);
      Py_DECREF(np_fnu_out);
      Py_XDECREF(np_obslam_out);
      Py_XDECREF(np_obsnu_out);
      PyErr_SetString(
          PyExc_ValueError,
          "fnu_out must be a C-contiguous array with requested out_dtype.");
      return NULL;
    }
  }
  if (np_obslam_out && (PyArray_TYPE(np_obslam_out) != out_typenum ||
                        !PyArray_ISCARRAY(np_obslam_out))) {
    Py_DECREF(np_lnu);
    Py_DECREF(np_lam);
    Py_DECREF(np_nu);
    Py_DECREF(np_fnu_out);
    Py_XDECREF(np_obslam_out);
    Py_XDECREF(np_obsnu_out);
    PyErr_SetString(
        PyExc_ValueError,
        "obslam_out must be a C-contiguous array with requested out_dtype.");
    return NULL;
  }
  if (np_obsnu_out && (PyArray_TYPE(np_obsnu_out) != out_typenum ||
                       !PyArray_ISCARRAY(np_obsnu_out))) {
    Py_DECREF(np_lnu);
    Py_DECREF(np_lam);
    Py_DECREF(np_nu);
    Py_DECREF(np_fnu_out);
    Py_XDECREF(np_obslam_out);
    Py_XDECREF(np_obsnu_out);
    PyErr_SetString(
        PyExc_ValueError,
        "obsnu_out must be a C-contiguous array with requested out_dtype.");
    return NULL;
  }

  const npy_intp nelem = PyArray_SIZE(np_lnu);
  tic("compute_fnu");

  /* Dispatch on input and output typenum. */
  if (input_typenum == NPY_FLOAT32) {
    const float *lnu = data_ptr<float>(np_lnu);
    const float *lam = data_ptr<float>(np_lam);
    const float *nu = data_ptr<float>(np_nu);

    if (out_typenum == NPY_FLOAT32) {
      float *fnu_out = data_ptr<float>(np_fnu_out);
      float *obslam_out =
          np_obslam_out ? data_ptr<float>(np_obslam_out) : nullptr;
      float *obsnu_out = np_obsnu_out ? data_ptr<float>(np_obsnu_out) : nullptr;
      compute_fnu_typed<float, float>(
          lnu, lam, nu, static_cast<float>(one_plus_z),
          static_cast<float>(conversion), nthreads, fnu_out, obslam_out,
          obsnu_out, nelem, nlam);
    } else {
      double *fnu_out = data_ptr<double>(np_fnu_out);
      double *obslam_out =
          np_obslam_out ? data_ptr<double>(np_obslam_out) : nullptr;
      double *obsnu_out =
          np_obsnu_out ? data_ptr<double>(np_obsnu_out) : nullptr;
      compute_fnu_typed<float, double>(
          lnu, lam, nu, static_cast<float>(one_plus_z),
          static_cast<float>(conversion), nthreads, fnu_out, obslam_out,
          obsnu_out, nelem, nlam);
    }
  } else {
    const double *lnu = data_ptr<double>(np_lnu);
    const double *lam = data_ptr<double>(np_lam);
    const double *nu = data_ptr<double>(np_nu);

    if (out_typenum == NPY_FLOAT32) {
      float *fnu_out = data_ptr<float>(np_fnu_out);
      float *obslam_out =
          np_obslam_out ? data_ptr<float>(np_obslam_out) : nullptr;
      float *obsnu_out = np_obsnu_out ? data_ptr<float>(np_obsnu_out) : nullptr;
      compute_fnu_typed<double, float>(lnu, lam, nu, one_plus_z, conversion,
                                       nthreads, fnu_out, obslam_out, obsnu_out,
                                       nelem, nlam);
    } else {
      double *fnu_out = data_ptr<double>(np_fnu_out);
      double *obslam_out =
          np_obslam_out ? data_ptr<double>(np_obslam_out) : nullptr;
      double *obsnu_out =
          np_obsnu_out ? data_ptr<double>(np_obsnu_out) : nullptr;
      compute_fnu_typed<double, double>(lnu, lam, nu, one_plus_z, conversion,
                                        nthreads, fnu_out, obslam_out,
                                        obsnu_out, nelem, nlam);
    }
  }

  toc("compute_fnu");

  Py_DECREF(np_lnu);
  Py_DECREF(np_lam);
  Py_DECREF(np_nu);
  Py_DECREF(np_fnu_out);
  Py_XDECREF(np_obslam_out);
  Py_XDECREF(np_obsnu_out);
  Py_RETURN_NONE;
}

/* Below is all the gubbins needed to make the module importable in Python. */
static PyMethodDef ObservedSpectraMethods[] = {
    {"compute_fnu", (PyCFunction)compute_fnu, METH_VARARGS,
     "Populate observer-frame wavelength, frequency, and flux arrays."},
    {NULL, NULL, 0, NULL}};

/* Make this importable. */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "observed_spectra",
    "Observer-frame spectra helper kernels",
    -1,
    ObservedSpectraMethods,
    NULL,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC PyInit_observed_spectra(void) {
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
