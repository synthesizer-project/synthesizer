/* Standard includes */
#include <cmath>

/* Python includes */
#include <Python.h>

/* Local includes */
#include "cpp_to_python.h"
#include "observed_spectra.h"
#include "timers.h"
#include "timers_init.h"

/**
 * @brief Populate observer-frame wavelength, frequency, and flux arrays.
 *
 * This helper fills preallocated output buffers so the Python hot path can
 * avoid repeated temporary arrays when converting rest-frame luminosity
 * densities into observer-frame flux densities.
 */
PyObject *compute_fnu(PyObject *self, PyObject *args) {
  /* We don't need the self argument but it has to be there. Tell the
   * compiler we don't care. */
  (void)self;

  /* Declare the Python-level inputs. The caller owns the output buffers so we
   * can fill them in place on the hot path. */
  PyObject *lnu_obj;
  PyObject *lam_obj;
  PyObject *nu_obj;
  double one_plus_z;
  double conversion;
  int nthreads;
  PyObject *fnu_out_obj;
  PyObject *obslam_out_obj;
  PyObject *obsnu_out_obj;

  if (!PyArg_ParseTuple(args, "OOOddiOOO", &lnu_obj, &lam_obj, &nu_obj,
                        &one_plus_z, &conversion, &nthreads, &fnu_out_obj,
                        &obslam_out_obj, &obsnu_out_obj)) {
    return NULL;
  }

  /* Validate that the required inputs are NumPy arrays before casting them to
   * PyArrayObject pointers below. */
  if (!PyArray_Check(lnu_obj) || !PyArray_Check(lam_obj) ||
      !PyArray_Check(nu_obj) || !PyArray_Check(fnu_out_obj)) {
    PyErr_SetString(PyExc_TypeError,
                    "compute_fnu expects NumPy array inputs and fnu_out.");
    return NULL;
  }

  PyArrayObject *np_lnu = (PyArrayObject *)lnu_obj;
  PyArrayObject *np_lam = (PyArrayObject *)lam_obj;
  PyArrayObject *np_nu = (PyArrayObject *)nu_obj;
  PyArrayObject *np_fnu_out = (PyArrayObject *)fnu_out_obj;
  Py_INCREF(np_lnu);
  Py_INCREF(np_lam);
  Py_INCREF(np_nu);
  Py_INCREF(np_fnu_out);

  /* The observer-frame wavelength and frequency outputs are optional because
   * get_fnu0 reuses the emitted grids directly. */
  if (obslam_out_obj != Py_None && !PyArray_Check(obslam_out_obj)) {
    Py_DECREF(np_lnu);
    Py_DECREF(np_lam);
    Py_DECREF(np_nu);
    Py_DECREF(np_fnu_out);
    PyErr_SetString(PyExc_TypeError,
                    "obslam_out must be a NumPy array or None.");
    return NULL;
  }

  if (obsnu_out_obj != Py_None && !PyArray_Check(obsnu_out_obj)) {
    Py_DECREF(np_lnu);
    Py_DECREF(np_lam);
    Py_DECREF(np_nu);
    Py_DECREF(np_fnu_out);
    PyErr_SetString(PyExc_TypeError,
                    "obsnu_out must be a NumPy array or None.");
    return NULL;
  }

  PyArrayObject *np_obslam_out =
      obslam_out_obj == Py_None ? NULL : (PyArrayObject *)obslam_out_obj;
  PyArrayObject *np_obsnu_out =
      obsnu_out_obj == Py_None ? NULL : (PyArrayObject *)obsnu_out_obj;
  Py_XINCREF(np_obslam_out);
  Py_XINCREF(np_obsnu_out);

  /* Reject any inputs that would force an implicit conversion or copy in the
   * kernel so the hot path only ever sees float64 C-contiguous buffers. */
  if (PyArray_TYPE(np_lnu) != NPY_DOUBLE || PyArray_TYPE(np_lam) != NPY_DOUBLE ||
      PyArray_TYPE(np_nu) != NPY_DOUBLE ||
      PyArray_TYPE(np_fnu_out) != NPY_DOUBLE ||
      !PyArray_ISCARRAY_RO(np_lnu) || !PyArray_ISCARRAY_RO(np_lam) ||
      !PyArray_ISCARRAY_RO(np_nu) || !PyArray_ISCARRAY(np_fnu_out) ||
      (np_obslam_out != NULL &&
       (PyArray_TYPE(np_obslam_out) != NPY_DOUBLE ||
        !PyArray_ISCARRAY(np_obslam_out))) ||
      (np_obsnu_out != NULL &&
       (PyArray_TYPE(np_obsnu_out) != NPY_DOUBLE ||
        !PyArray_ISCARRAY(np_obsnu_out)))) {
    Py_DECREF(np_lnu);
    Py_DECREF(np_lam);
    Py_DECREF(np_nu);
    Py_DECREF(np_fnu_out);
    Py_XDECREF(np_obslam_out);
    Py_XDECREF(np_obsnu_out);
    PyErr_SetString(
        PyExc_ValueError,
        "compute_fnu requires float64 C-contiguous inputs and outputs.");
    return NULL;
  }

  if (PyArray_NDIM(np_lnu) < 1 || PyArray_NDIM(np_lam) != 1 ||
      PyArray_NDIM(np_nu) != 1 ||
      (np_obslam_out != NULL && PyArray_NDIM(np_obslam_out) != 1) ||
      (np_obsnu_out != NULL && PyArray_NDIM(np_obsnu_out) != 1)) {
    Py_DECREF(np_lnu);
    Py_DECREF(np_lam);
    Py_DECREF(np_nu);
    Py_DECREF(np_fnu_out);
    Py_XDECREF(np_obslam_out);
    Py_XDECREF(np_obsnu_out);
    PyErr_SetString(
        PyExc_ValueError,
        "lnu must be at least 1D and lam/nu/output grids must be 1D.");
    return NULL;
  }

  /* Ensure the output shapes match the input spectra shape and the shared
   * wavelength-axis length before entering the kernel loop. */
  const int lnu_ndim = PyArray_NDIM(np_lnu);
  const npy_intp *lnu_dims = PyArray_DIMS(np_lnu);
  const npy_intp *fnu_dims = PyArray_DIMS(np_fnu_out);
  if (PyArray_NDIM(np_fnu_out) != lnu_ndim) {
    Py_DECREF(np_lnu);
    Py_DECREF(np_lam);
    Py_DECREF(np_nu);
    Py_DECREF(np_fnu_out);
    Py_XDECREF(np_obslam_out);
    Py_XDECREF(np_obsnu_out);
    PyErr_SetString(PyExc_ValueError,
                    "fnu_out must have the same ndim as lnu.");
    return NULL;
  }

  for (int idim = 0; idim < lnu_ndim; idim++) {
    if (lnu_dims[idim] != fnu_dims[idim]) {
      Py_DECREF(np_lnu);
      Py_DECREF(np_lam);
      Py_DECREF(np_nu);
      Py_DECREF(np_fnu_out);
      Py_XDECREF(np_obslam_out);
      Py_XDECREF(np_obsnu_out);
      PyErr_SetString(PyExc_ValueError,
                      "fnu_out must match the shape of lnu.");
      return NULL;
    }
  }

  const npy_intp nlam = PyArray_DIMS(np_lam)[0];
  if (PyArray_DIMS(np_nu)[0] != nlam || lnu_dims[lnu_ndim - 1] != nlam ||
      (np_obslam_out != NULL && PyArray_DIMS(np_obslam_out)[0] != nlam) ||
      (np_obsnu_out != NULL && PyArray_DIMS(np_obsnu_out)[0] != nlam)) {
    Py_DECREF(np_lnu);
    Py_DECREF(np_lam);
    Py_DECREF(np_nu);
    Py_DECREF(np_fnu_out);
    Py_XDECREF(np_obslam_out);
    Py_XDECREF(np_obsnu_out);
    PyErr_SetString(PyExc_ValueError,
                    "lam/nu/output lengths must match the last lnu axis.");
    return NULL;
  }

  double *lnu = static_cast<double *>(PyArray_DATA(np_lnu));
  double *lam = static_cast<double *>(PyArray_DATA(np_lam));
  double *nu = static_cast<double *>(PyArray_DATA(np_nu));
  double *fnu_out = static_cast<double *>(PyArray_DATA(np_fnu_out));
  double *obslam_out = np_obslam_out == NULL
                           ? NULL
                           : static_cast<double *>(PyArray_DATA(np_obslam_out));
  double *obsnu_out = np_obsnu_out == NULL
                          ? NULL
                          : static_cast<double *>(PyArray_DATA(np_obsnu_out));
  const npy_intp nelem = PyArray_SIZE(np_lnu);

  /* Apply the scalar luminosity-to-flux conversion over the full spectra
   * buffer, then populate the observer-frame grids if requested. */
  tic("compute_fnu");

#ifdef WITH_OPENMP
#pragma omp parallel for if(nthreads > 1) num_threads(nthreads) schedule(static)
#endif
  for (npy_intp idx = 0; idx < nelem; idx++) {
    fnu_out[idx] = lnu[idx] * conversion;
  }

  if (obslam_out != NULL && obsnu_out != NULL) {
    for (npy_intp ilam = 0; ilam < nlam; ilam++) {
      obslam_out[ilam] = lam[ilam] * one_plus_z;
      obsnu_out[ilam] = nu[ilam] / one_plus_z;
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
    "observed_spectra",                       /* m_name */
    "Observer-frame spectra helper kernels", /* m_doc */
    -1,                                        /* m_size */
    ObservedSpectraMethods,                    /* m_methods */
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
