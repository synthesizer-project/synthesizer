/* Standard includes */
#include <cmath>

/* Python includes */
#include <Python.h>

/* Local includes */
#include "cpp_to_python.h"
#include "spectra_operations.h"
#include "timers.h"
#include "timers_init.h"

/**
 * @brief Scale a 2D spectra array by a per-spectrum factor.
 */
PyObject *scale_spectra_2d(PyObject *self, PyObject *args) {
  /* We don't need the self argument but it has to be there. Tell the
   * compiler we don't care. */
  (void)self;

  /* Declare the Python-level inputs for the spectra array, the per-spectrum
   * scaling vector, and the optional row/column masks. */
  PyObject *spectra_obj;
  PyObject *scaling_obj;
  PyObject *mask_obj = Py_None;
  PyObject *lam_mask_obj = Py_None;
  int nthreads;

  if (!PyArg_ParseTuple(args, "OOOOi", &spectra_obj, &scaling_obj, &mask_obj,
                        &lam_mask_obj, &nthreads)) {
    return NULL;
  }

  /* Convert the required inputs into float64 NumPy array views so the kernel
   * sees the dtype and layout it expects. */
  PyArrayObject *np_spectra = (PyArrayObject *)PyArray_FROM_OTF(
      spectra_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *np_scaling = (PyArrayObject *)PyArray_FROM_OTF(
      scaling_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *np_mask = nullptr;
  PyArrayObject *np_lam_mask = nullptr;

  if (np_spectra == NULL || np_scaling == NULL) {
    Py_XDECREF(np_spectra);
    Py_XDECREF(np_scaling);
    return NULL;
  }

  if (mask_obj != Py_None) {
    np_mask = (PyArrayObject *)PyArray_FROM_OTF(mask_obj, NPY_BOOL,
                                                NPY_ARRAY_IN_ARRAY);
    if (np_mask == NULL) {
      Py_DECREF(np_spectra);
      Py_DECREF(np_scaling);
      return NULL;
    }
  }

  if (lam_mask_obj != Py_None) {
    np_lam_mask = (PyArrayObject *)PyArray_FROM_OTF(lam_mask_obj, NPY_BOOL,
                                                    NPY_ARRAY_IN_ARRAY);
    if (np_lam_mask == NULL) {
      Py_DECREF(np_spectra);
      Py_DECREF(np_scaling);
      Py_XDECREF(np_mask);
      return NULL;
    }
  }

  /* Validate the spectra/scaling shapes and any optional masks before
   * allocating the output buffer. */
  if (PyArray_NDIM(np_spectra) != 2 || PyArray_NDIM(np_scaling) != 1) {
    PyErr_SetString(PyExc_ValueError,
                    "spectra must be 2D and scaling must be 1D.");
    Py_DECREF(np_spectra);
    Py_DECREF(np_scaling);
    Py_XDECREF(np_mask);
    Py_XDECREF(np_lam_mask);
    return NULL;
  }

  const npy_intp *spectra_dims = PyArray_DIMS(np_spectra);
  const npy_intp *scaling_dims = PyArray_DIMS(np_scaling);
  const int nspec = static_cast<int>(spectra_dims[0]);
  const int nlam = static_cast<int>(spectra_dims[1]);

  if (scaling_dims[0] != spectra_dims[0]) {
    PyErr_SetString(PyExc_ValueError,
                    "scaling length must match the spectra first dimension.");
    Py_DECREF(np_spectra);
    Py_DECREF(np_scaling);
    Py_XDECREF(np_mask);
    Py_XDECREF(np_lam_mask);
    return NULL;
  }

  if (np_mask != NULL &&
      (PyArray_NDIM(np_mask) != 1 || PyArray_DIMS(np_mask)[0] != spectra_dims[0])) {
    PyErr_SetString(PyExc_ValueError,
                    "mask must be a 1D boolean array matching spectra rows.");
    Py_DECREF(np_spectra);
    Py_DECREF(np_scaling);
    Py_XDECREF(np_mask);
    Py_XDECREF(np_lam_mask);
    return NULL;
  }

  if (np_lam_mask != NULL &&
      (PyArray_NDIM(np_lam_mask) != 1 || PyArray_DIMS(np_lam_mask)[0] != spectra_dims[1])) {
    PyErr_SetString(PyExc_ValueError,
                    "lam_mask must be a 1D boolean array matching spectra columns.");
    Py_DECREF(np_spectra);
    Py_DECREF(np_scaling);
    Py_XDECREF(np_mask);
    Py_XDECREF(np_lam_mask);
    return NULL;
  }

  /* Allocate an output array matching the spectra layout so the operation is
   * side-effect free on the input buffers. */
  PyArrayObject *np_out = (PyArrayObject *)PyArray_NewLikeArray(
      np_spectra, NPY_KEEPORDER, NULL, 0);
  if (np_out == NULL) {
    Py_DECREF(np_spectra);
    Py_DECREF(np_scaling);
    Py_XDECREF(np_mask);
    Py_XDECREF(np_lam_mask);
    return NULL;
  }

  double *spectra = static_cast<double *>(PyArray_DATA(np_spectra));
  double *scaling = static_cast<double *>(PyArray_DATA(np_scaling));
  double *out = static_cast<double *>(PyArray_DATA(np_out));
  npy_bool *mask = np_mask == NULL ? NULL : (npy_bool *)PyArray_DATA(np_mask);
  npy_bool *lam_mask =
      np_lam_mask == NULL ? NULL : (npy_bool *)PyArray_DATA(np_lam_mask);

  /* Apply the scaling in one pass over the spectra array while respecting the
   * optional row and wavelength masks. */
  tic("scale_spectra_2d");

#ifdef WITH_OPENMP
#pragma omp parallel for if(nthreads > 1) num_threads(nthreads) schedule(static)
#endif
  for (int ispec = 0; ispec < nspec; ispec++) {
    const double scale = scaling[ispec];
    const bool apply_spec = mask == NULL || mask[ispec];
    double *__restrict in_row = spectra + ispec * nlam;
    double *__restrict out_row = out + ispec * nlam;

    for (int ilam = 0; ilam < nlam; ilam++) {
      const bool apply_lam = lam_mask == NULL || lam_mask[ilam];
      out_row[ilam] = (apply_spec && apply_lam) ? in_row[ilam] * scale
                                                : in_row[ilam];
    }
  }

  toc("scale_spectra_2d");

  Py_DECREF(np_spectra);
  Py_DECREF(np_scaling);
  Py_XDECREF(np_mask);
  Py_XDECREF(np_lam_mask);
  return Py_BuildValue("N", np_out);
}

/**
 * @brief Apply exp(-tau_v * tau_x_v) attenuation to a 2D spectra array.
 */
PyObject *apply_separable_attenuation_2d(PyObject *self, PyObject *args) {
  /* We don't need the self argument but it has to be there. Tell the
   * compiler we don't care. */
  (void)self;

  /* Declare the Python-level inputs for the spectra buffer, the per-particle
   * optical depths, and the wavelength-dependent optical-depth curve. */
  PyObject *spectra_obj;
  PyObject *tau_v_obj;
  PyObject *tau_x_v_obj;
  PyObject *mask_obj = Py_None;
  int nthreads;

  if (!PyArg_ParseTuple(args, "OOOOi", &spectra_obj, &tau_v_obj, &tau_x_v_obj,
                        &mask_obj, &nthreads)) {
    return NULL;
  }

  /* Convert the required inputs into float64 NumPy array views and unpack the
   * optional row mask if one has been provided. */
  PyArrayObject *np_spectra = (PyArrayObject *)PyArray_FROM_OTF(
      spectra_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *np_tau_v = (PyArrayObject *)PyArray_FROM_OTF(
      tau_v_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *np_tau_x_v = (PyArrayObject *)PyArray_FROM_OTF(
      tau_x_v_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *np_mask = nullptr;

  if (np_spectra == NULL || np_tau_v == NULL || np_tau_x_v == NULL) {
    Py_XDECREF(np_spectra);
    Py_XDECREF(np_tau_v);
    Py_XDECREF(np_tau_x_v);
    return NULL;
  }

  if (mask_obj != Py_None) {
    np_mask = (PyArrayObject *)PyArray_FROM_OTF(mask_obj, NPY_BOOL,
                                                NPY_ARRAY_IN_ARRAY);
    if (np_mask == NULL) {
      Py_DECREF(np_spectra);
      Py_DECREF(np_tau_v);
      Py_DECREF(np_tau_x_v);
      return NULL;
    }
  }

  /* Validate that the row and column vectors match the 2D spectra shape so we
   * can fuse attenuation application without building a transmission matrix. */
  if (PyArray_NDIM(np_spectra) != 2 || PyArray_NDIM(np_tau_v) != 1 ||
      PyArray_NDIM(np_tau_x_v) != 1) {
    PyErr_SetString(PyExc_ValueError,
                    "spectra must be 2D and tau vectors must be 1D.");
    Py_DECREF(np_spectra);
    Py_DECREF(np_tau_v);
    Py_DECREF(np_tau_x_v);
    Py_XDECREF(np_mask);
    return NULL;
  }

  const npy_intp *spectra_dims = PyArray_DIMS(np_spectra);
  const int nrows = static_cast<int>(spectra_dims[0]);
  const int ncols = static_cast<int>(spectra_dims[1]);
  if (PyArray_DIMS(np_tau_v)[0] != spectra_dims[0] ||
      PyArray_DIMS(np_tau_x_v)[0] != spectra_dims[1]) {
    PyErr_SetString(PyExc_ValueError,
                    "tau vectors must match the spectra dimensions.");
    Py_DECREF(np_spectra);
    Py_DECREF(np_tau_v);
    Py_DECREF(np_tau_x_v);
    Py_XDECREF(np_mask);
    return NULL;
  }

  if (np_mask != NULL &&
      (PyArray_NDIM(np_mask) != 1 || PyArray_DIMS(np_mask)[0] != spectra_dims[0])) {
    PyErr_SetString(PyExc_ValueError,
                    "mask must be a 1D boolean array matching spectra rows.");
    Py_DECREF(np_spectra);
    Py_DECREF(np_tau_v);
    Py_DECREF(np_tau_x_v);
    Py_XDECREF(np_mask);
    return NULL;
  }

  /* Allocate an output buffer matching the input spectra layout so the fused
   * attenuation kernel never mutates the original spectra in place. */
  PyArrayObject *np_out = (PyArrayObject *)PyArray_NewLikeArray(
      np_spectra, NPY_KEEPORDER, NULL, 0);
  if (np_out == NULL) {
    Py_DECREF(np_spectra);
    Py_DECREF(np_tau_v);
    Py_DECREF(np_tau_x_v);
    Py_XDECREF(np_mask);
    return NULL;
  }

  double *spectra = static_cast<double *>(PyArray_DATA(np_spectra));
  double *tau_v = static_cast<double *>(PyArray_DATA(np_tau_v));
  double *tau_x_v = static_cast<double *>(PyArray_DATA(np_tau_x_v));
  double *out = static_cast<double *>(PyArray_DATA(np_out));
  npy_bool *mask = np_mask == NULL ? NULL : (npy_bool *)PyArray_DATA(np_mask);

  /* Apply the separable attenuation directly to the spectra buffer, avoiding
   * materialising the full exp(-tau_v * tau_x_v) matrix. */
  tic("apply_separable_attenuation_2d");

#ifdef WITH_OPENMP
#pragma omp parallel for if(nthreads > 1) num_threads(nthreads) schedule(static)
#endif
  for (int irow = 0; irow < nrows; irow++) {
    const bool apply = mask == NULL || mask[irow];
    const double row_tau = tau_v[irow];
    double *__restrict in_row = spectra + irow * ncols;
    double *__restrict out_row = out + irow * ncols;

    if (!apply) {
      for (int icol = 0; icol < ncols; icol++) {
        out_row[icol] = in_row[icol];
      }
      continue;
    }

    for (int icol = 0; icol < ncols; icol++) {
      out_row[icol] = in_row[icol] * std::exp(-row_tau * tau_x_v[icol]);
    }
  }

  toc("apply_separable_attenuation_2d");

  Py_DECREF(np_spectra);
  Py_DECREF(np_tau_v);
  Py_DECREF(np_tau_x_v);
  Py_XDECREF(np_mask);
  return Py_BuildValue("N", np_out);
}

/**
 * @brief Multiply a 2D array by a 1D row vector with optional row mask.
 */
PyObject *multiply_rows_by_vector_2d(PyObject *self, PyObject *args) {
  /* We don't need the self argument but it has to be there. Tell the
   * compiler we don't care. */
  (void)self;

  /* Declare the Python-level inputs for the 2D array, the row vector, and
   * the optional row mask. */
  PyObject *array_obj;
  PyObject *vector_obj;
  PyObject *mask_obj = Py_None;
  int nthreads;

  if (!PyArg_ParseTuple(args, "OOOi", &array_obj, &vector_obj, &mask_obj,
                        &nthreads)) {
    return NULL;
  }

  /* Convert the input buffers into float64 NumPy array views before running
   * the kernel. */
  PyArrayObject *np_array = (PyArrayObject *)PyArray_FROM_OTF(
      array_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *np_vector = (PyArrayObject *)PyArray_FROM_OTF(
      vector_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *np_mask = nullptr;

  if (np_array == NULL || np_vector == NULL) {
    Py_XDECREF(np_array);
    Py_XDECREF(np_vector);
    return NULL;
  }

  if (mask_obj != Py_None) {
    np_mask = (PyArrayObject *)PyArray_FROM_OTF(mask_obj, NPY_BOOL,
                                                NPY_ARRAY_IN_ARRAY);
    if (np_mask == NULL) {
      Py_DECREF(np_array);
      Py_DECREF(np_vector);
      return NULL;
    }
  }

  /* Validate the array/vector shape agreement and any optional row mask
   * before allocating the output array. */
  if (PyArray_NDIM(np_array) != 2 || PyArray_NDIM(np_vector) != 1) {
    PyErr_SetString(PyExc_ValueError,
                    "array must be 2D and vector must be 1D.");
    Py_DECREF(np_array);
    Py_DECREF(np_vector);
    Py_XDECREF(np_mask);
    return NULL;
  }

  const npy_intp *array_dims = PyArray_DIMS(np_array);
  if (PyArray_DIMS(np_vector)[0] != array_dims[0]) {
    PyErr_SetString(PyExc_ValueError,
                    "vector length must match the first array dimension.");
    Py_DECREF(np_array);
    Py_DECREF(np_vector);
    Py_XDECREF(np_mask);
    return NULL;
  }

  if (np_mask != NULL &&
      (PyArray_NDIM(np_mask) != 1 || PyArray_DIMS(np_mask)[0] != array_dims[0])) {
    PyErr_SetString(PyExc_ValueError,
                    "mask must be a 1D boolean array matching array rows.");
    Py_DECREF(np_array);
    Py_DECREF(np_vector);
    Py_XDECREF(np_mask);
    return NULL;
  }

  /* Allocate an output array matching the input layout so the operation keeps
   * the original array unchanged. */
  PyArrayObject *np_out = (PyArrayObject *)PyArray_NewLikeArray(
      np_array, NPY_KEEPORDER, NULL, 0);
  if (np_out == NULL) {
    Py_DECREF(np_array);
    Py_DECREF(np_vector);
    Py_XDECREF(np_mask);
    return NULL;
  }

  const int nrows = static_cast<int>(array_dims[0]);
  const int ncols = static_cast<int>(array_dims[1]);
  double *array = static_cast<double *>(PyArray_DATA(np_array));
  double *vector = static_cast<double *>(PyArray_DATA(np_vector));
  double *out = static_cast<double *>(PyArray_DATA(np_out));
  npy_bool *mask = np_mask == NULL ? NULL : (npy_bool *)PyArray_DATA(np_mask);

  /* Multiply each row by the matching vector entry in one pass over the
   * output buffer. */
  tic("multiply_rows_by_vector_2d");

#ifdef WITH_OPENMP
#pragma omp parallel for if(nthreads > 1) num_threads(nthreads) schedule(static)
#endif
  for (int irow = 0; irow < nrows; irow++) {
    const bool apply = mask == NULL || mask[irow];
    const double scale = vector[irow];
    double *__restrict in_row = array + irow * ncols;
    double *__restrict out_row = out + irow * ncols;
    for (int icol = 0; icol < ncols; icol++) {
      out_row[icol] = apply ? in_row[icol] * scale : in_row[icol];
    }
  }

  toc("multiply_rows_by_vector_2d");

  Py_DECREF(np_array);
  Py_DECREF(np_vector);
  Py_XDECREF(np_mask);
  return Py_BuildValue("N", np_out);
}

/**
 * @brief Multiply a 1D or 2D array by a 1D vector over the last axis.
 */
PyObject *multiply_array_by_vector_1d(PyObject *self, PyObject *args) {
  /* We don't need the self argument but it has to be there. Tell the
   * compiler we don't care. */
  (void)self;

  /* Declare the Python-level inputs for the array and the last-axis vector. */
  PyObject *array_obj;
  PyObject *vector_obj;
  int nthreads;

  if (!PyArg_ParseTuple(args, "OOi", &array_obj, &vector_obj, &nthreads)) {
    return NULL;
  }

  /* Convert the input buffers into float64 NumPy array views before running
   * the kernel. */
  PyArrayObject *np_array = (PyArrayObject *)PyArray_FROM_OTF(
      array_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *np_vector = (PyArrayObject *)PyArray_FROM_OTF(
      vector_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

  if (np_array == NULL || np_vector == NULL) {
    Py_XDECREF(np_array);
    Py_XDECREF(np_vector);
    return NULL;
  }

  /* Validate that the vector length matches the last array axis, then
   * allocate a same-shaped output array for the result. */
  if (PyArray_NDIM(np_vector) != 1 ||
      (PyArray_NDIM(np_array) != 1 && PyArray_NDIM(np_array) != 2)) {
    PyErr_SetString(PyExc_ValueError,
                    "array must be 1D or 2D and vector must be 1D.");
    Py_DECREF(np_array);
    Py_DECREF(np_vector);
    return NULL;
  }

  const npy_intp *array_dims = PyArray_DIMS(np_array);
  const int array_ndim = PyArray_NDIM(np_array);
  const int ncols = static_cast<int>(array_dims[array_ndim - 1]);
  if (PyArray_DIMS(np_vector)[0] != array_dims[array_ndim - 1]) {
    PyErr_SetString(PyExc_ValueError,
                    "vector length must match the last array dimension.");
    Py_DECREF(np_array);
    Py_DECREF(np_vector);
    return NULL;
  }

  PyArrayObject *np_out = (PyArrayObject *)PyArray_NewLikeArray(
      np_array, NPY_KEEPORDER, NULL, 0);
  if (np_out == NULL) {
    Py_DECREF(np_array);
    Py_DECREF(np_vector);
    return NULL;
  }

  double *array = static_cast<double *>(PyArray_DATA(np_array));
  double *vector = static_cast<double *>(PyArray_DATA(np_vector));
  double *out = static_cast<double *>(PyArray_DATA(np_out));
  const int nrows = array_ndim == 1 ? 1 : static_cast<int>(array_dims[0]);

  /* Apply the last-axis scaling in one pass across the output buffer. */
  tic("multiply_array_by_vector_1d");

#ifdef WITH_OPENMP
#pragma omp parallel for if(nthreads > 1) num_threads(nthreads) schedule(static)
#endif
  for (int irow = 0; irow < nrows; irow++) {
    double *__restrict in_row = array + irow * ncols;
    double *__restrict out_row = out + irow * ncols;
    for (int icol = 0; icol < ncols; icol++) {
      out_row[icol] = in_row[icol] * vector[icol];
    }
  }

  toc("multiply_array_by_vector_1d");

  Py_DECREF(np_array);
  Py_DECREF(np_vector);
  return Py_BuildValue("N", np_out);
}

/* Below is all the gubbins needed to make the module importable in Python. */
static PyMethodDef SpectraOperationMethods[] = {
    {"scale_spectra_2d", (PyCFunction)scale_spectra_2d, METH_VARARGS,
     "Scale a 2D spectra array by a 1D per-spectrum factor."},
    {"apply_separable_attenuation_2d",
     (PyCFunction)apply_separable_attenuation_2d, METH_VARARGS,
     "Apply exp(-tau_v * tau_x_v) attenuation to a 2D spectra array."},
    {"multiply_rows_by_vector_2d", (PyCFunction)multiply_rows_by_vector_2d,
     METH_VARARGS,
     "Multiply a 2D array by a 1D row vector with optional mask."},
    {"multiply_array_by_vector_1d", (PyCFunction)multiply_array_by_vector_1d,
     METH_VARARGS,
     "Multiply a 1D or 2D array by a 1D vector over the last axis."},
    {NULL, NULL, 0, NULL}};

/* Make this importable. */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "spectra_operations",                  /* m_name */
    "Generic spectra operation kernels",  /* m_doc */
    -1,                                     /* m_size */
    SpectraOperationMethods,                /* m_methods */
    NULL,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC PyInit_spectra_operations(void) {
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
