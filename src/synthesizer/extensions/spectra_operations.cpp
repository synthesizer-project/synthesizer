/******************************************************************************
 * Generic spectra operation kernels with split masked/unmasked backends.
 *
 * Provides scale_spectra_2d, apply_separable_attenuation_2d,
 * multiply_array_by_vector_1d, and scale_line_2d. Each arithmetic
 * operation is compiled as dedicated no-mask,
 * row-mask, lam-mask, and both-mask variants so the inner loops stay
 * branch-free.
 ******************************************************************************/

/* Standard includes */
#include <cmath>

/* Python includes */
#include <Python.h>

/* Local includes */
#include "cpp_to_python.h"
#include "python_to_cpp.h"
#include "timers.h"
#include "timers_init.h"

/* ------------------------------------------------------------------------ */
/*  scale_spectra_2d — per-spectrum row scaling with four mask combos       */
/* ------------------------------------------------------------------------ */
/*
 * Each mask combination is compiled as a separate function so the compiler
 * sees dedicated inner loops without runtime branch checks. The four
 * combinations (no mask, row mask, wavelength mask, both) are each emitted
 * in a serial and an OpenMP variant. */

/**
 * @brief This calculates per-spectrum row scaling with no masks applied.
 *
 * This is the serial version of the function for rows without a mask.
 *
 * @param spectra: The input 2D spectra array (nspec x nlam).
 * @param scaling: The per-spectrum scaling vector (nspec).
 * @param out: The pre-allocated output buffer (nspec x nlam).
 * @param nspec: The number of spectra rows.
 * @param nlam: The number of wavelength bins.
 */
template <typename Real, typename OutT>
static void scale_spectra_2d_no_mask_serial(const Real *__restrict__ spectra,
                                            const Real *__restrict__ scaling,
                                            OutT *out, int nspec, int nlam) {

  /* Loop over every spectrum in the grid. */
  for (int ispec = 0; ispec < nspec; ispec++) {

    /* Cache the row-scale factor so we only read it once. */
    const Real scale = scaling[ispec];
    const Real *in_row = spectra + ispec * nlam;
    OutT *out_row = out + ispec * nlam;

    /* Multiply every wavelength by the row factor. */
#pragma GCC ivdep
    for (int ilam = 0; ilam < nlam; ilam++) {
      out_row[ilam] = static_cast<OutT>(in_row[ilam] * scale);
    }
  }
}

/**
 * @brief This calculates per-spectrum row scaling with a 1D row mask.
 *
 * This is the serial version of the function for rows with a row mask.
 *
 * @param spectra: The input 2D spectra array (nspec x nlam).
 * @param scaling: The per-spectrum scaling vector (nspec).
 * @param mask: 1D boolean row mask (nspec).
 * @param out: The pre-allocated output buffer (nspec x nlam).
 * @param nspec: The number of spectra rows.
 * @param nlam: The number of wavelength bins.
 */
template <typename Real, typename OutT>
static void scale_spectra_2d_row_mask_serial(const Real *__restrict__ spectra,
                                             const Real *__restrict__ scaling,
                                             const npy_bool *mask, OutT *out,
                                             int nspec, int nlam) {

  /* Loop over every spectrum in the grid. */
  for (int ispec = 0; ispec < nspec; ispec++) {
    const Real *in_row = spectra + ispec * nlam;
    OutT *out_row = out + ispec * nlam;

    /* Scale masked rows and copy unmasked rows through unchanged. */
    if (mask[ispec]) {
      const Real scale = scaling[ispec];

#pragma GCC ivdep
      for (int ilam = 0; ilam < nlam; ilam++) {
        out_row[ilam] = static_cast<OutT>(in_row[ilam] * scale);
      }
    } else {
#pragma GCC ivdep
      for (int ilam = 0; ilam < nlam; ilam++) {
        out_row[ilam] = static_cast<OutT>(in_row[ilam]);
      }
    }
  }
}

/**
 * @brief This calculates per-spectrum row scaling with a 1D wavelength mask.
 *
 * This is the serial version of the function for rows with a wavelength mask.
 *
 * @param spectra: The input 2D spectra array (nspec x nlam).
 * @param scaling: The per-spectrum scaling vector (nspec).
 * @param lam_mask: 1D boolean wavelength mask (nlam).
 * @param out: The pre-allocated output buffer (nspec x nlam).
 * @param nspec: The number of spectra rows.
 * @param nlam: The number of wavelength bins.
 */
template <typename Real, typename OutT>
static void scale_spectra_2d_lam_mask_serial(const Real *__restrict__ spectra,
                                             const Real *__restrict__ scaling,
                                             const npy_bool *lam_mask,
                                             OutT *out, int nspec, int nlam) {
  /* Loop over every spectrum in the grid. */
  for (int ispec = 0; ispec < nspec; ispec++) {

    /* Cache the row-scale factor for this row. */
    const Real scale = scaling[ispec];
    const Real *in_row = spectra + ispec * nlam;
    OutT *out_row = out + ispec * nlam;

    /* Only scale wavelengths that pass the mask; keep others unchanged. */
    for (int ilam = 0; ilam < nlam; ilam++) {
      out_row[ilam] = lam_mask[ilam] ? static_cast<OutT>(in_row[ilam] * scale)
                                     : static_cast<OutT>(in_row[ilam]);
    }
  }
}

/**
 * @brief This calculates per-spectrum row scaling with both a row mask and
 *        a wavelength mask.
 *
 * This is the serial version of the function for rows with both masks.
 *
 * @param spectra: The input 2D spectra array (nspec x nlam).
 * @param scaling: The per-spectrum scaling vector (nspec).
 * @param mask: 1D boolean row mask (nspec).
 * @param lam_mask: 1D boolean wavelength mask (nlam).
 * @param out: The pre-allocated output buffer (nspec x nlam).
 * @param nspec: The number of spectra rows.
 * @param nlam: The number of wavelength bins.
 */
template <typename Real, typename OutT>
static void scale_spectra_2d_both_masks_serial(const Real *__restrict__ spectra,
                                               const Real *__restrict__ scaling,
                                               const npy_bool *mask,
                                               const npy_bool *lam_mask,
                                               OutT *out, int nspec, int nlam) {

  /* Loop over every spectrum in the grid. */
  for (int ispec = 0; ispec < nspec; ispec++) {
    const Real *in_row = spectra + ispec * nlam;
    OutT *out_row = out + ispec * nlam;

    /* For masked rows apply the wavelength-dependent scale; copy otherwise. */
    if (mask[ispec]) {
      const Real scale = scaling[ispec];

      for (int ilam = 0; ilam < nlam; ilam++) {
        out_row[ilam] = lam_mask[ilam] ? static_cast<OutT>(in_row[ilam] * scale)
                                       : static_cast<OutT>(in_row[ilam]);
      }
    } else {
      for (int ilam = 0; ilam < nlam; ilam++) {
        out_row[ilam] = static_cast<OutT>(in_row[ilam]);
      }
    }
  }
}

#ifdef WITH_OPENMP

/**
 * @brief This calculates per-spectrum row scaling with no masks using OpenMP.
 *
 * This is the parallel version of the function for rows without a mask.
 *
 * @param spectra: The input 2D spectra array (nspec x nlam).
 * @param scaling: The per-spectrum scaling vector (nspec).
 * @param out: The pre-allocated output buffer (nspec x nlam).
 * @param nspec: The number of spectra rows.
 * @param nlam: The number of wavelength bins.
 * @param nthreads: The number of OpenMP threads.
 */
template <typename Real, typename OutT>
static void scale_spectra_2d_no_mask_omp(const Real *__restrict__ spectra,
                                         const Real *__restrict__ scaling,
                                         OutT *out, int nspec, int nlam,
                                         int nthreads) {

  /* Split the spectra rows evenly across threads. */
#pragma omp parallel for num_threads(nthreads) schedule(static)

  for (int ispec = 0; ispec < nspec; ispec++) {

    /* Cache the row-scale factor for this row. */
    const Real scale = scaling[ispec];
    const Real *in_row = spectra + ispec * nlam;
    OutT *out_row = out + ispec * nlam;

    /* Multiply every wavelength by the row factor. */
#pragma omp simd
    for (int ilam = 0; ilam < nlam; ilam++) {
      out_row[ilam] = static_cast<OutT>(in_row[ilam] * scale);
    }
  }
}

/**
 * @brief This calculates per-spectrum row scaling with a row mask using
 *        OpenMP.
 *
 * This is the parallel version of the function for rows with a row mask.
 *
 * @param spectra: The input 2D spectra array (nspec x nlam).
 * @param scaling: The per-spectrum scaling vector (nspec).
 * @param mask: 1D boolean row mask (nspec).
 * @param out: The pre-allocated output buffer (nspec x nlam).
 * @param nspec: The number of spectra rows.
 * @param nlam: The number of wavelength bins.
 * @param nthreads: The number of OpenMP threads.
 */
template <typename Real, typename OutT>
static void scale_spectra_2d_row_mask_omp(const Real *__restrict__ spectra,
                                          const Real *__restrict__ scaling,
                                          const npy_bool *mask, OutT *out,
                                          int nspec, int nlam, int nthreads) {

  /* Split the spectra rows evenly across threads. */
#pragma omp parallel for num_threads(nthreads) schedule(static)

  for (int ispec = 0; ispec < nspec; ispec++) {
    const Real *in_row = spectra + ispec * nlam;
    OutT *out_row = out + ispec * nlam;

    /* Scale masked rows and copy unmasked rows through unchanged. */
    if (mask[ispec]) {
      const Real scale = scaling[ispec];

      for (int ilam = 0; ilam < nlam; ilam++) {
        out_row[ilam] = static_cast<OutT>(in_row[ilam] * scale);
      }
    } else {
      for (int ilam = 0; ilam < nlam; ilam++) {
        out_row[ilam] = static_cast<OutT>(in_row[ilam]);
      }
    }
  }
}

/**
 * @brief This calculates per-spectrum row scaling with a wavelength mask
 *        using OpenMP.
 *
 * This is the parallel version of the function for rows with a wavelength
 * mask.
 *
 * @param spectra: The input 2D spectra array (nspec x nlam).
 * @param scaling: The per-spectrum scaling vector (nspec).
 * @param lam_mask: 1D boolean wavelength mask (nlam).
 * @param out: The pre-allocated output buffer (nspec x nlam).
 * @param nspec: The number of spectra rows.
 * @param nlam: The number of wavelength bins.
 * @param nthreads: The number of OpenMP threads.
 */
template <typename Real, typename OutT>
static void scale_spectra_2d_lam_mask_omp(const Real *__restrict__ spectra,
                                          const Real *__restrict__ scaling,
                                          const npy_bool *lam_mask, OutT *out,
                                          int nspec, int nlam, int nthreads) {

  /* Split the spectra rows evenly across threads. */
#pragma omp parallel for num_threads(nthreads) schedule(static)
  for (int ispec = 0; ispec < nspec; ispec++) {

    /* Cache the row-scale factor for this row. */
    const Real scale = scaling[ispec];
    const Real *in_row = spectra + ispec * nlam;
    OutT *out_row = out + ispec * nlam;

    /* Only scale wavelengths that pass the mask; keep others unchanged. */
    for (int ilam = 0; ilam < nlam; ilam++) {
      out_row[ilam] = lam_mask[ilam] ? static_cast<OutT>(in_row[ilam] * scale)
                                     : static_cast<OutT>(in_row[ilam]);
    }
  }
}

/**
 * @brief This calculates per-spectrum row scaling with both masks using
 *        OpenMP.
 *
 * This is the parallel version of the function for rows with both masks.
 *
 * @param spectra: The input 2D spectra array (nspec x nlam).
 * @param scaling: The per-spectrum scaling vector (nspec).
 * @param mask: 1D boolean row mask (nspec).
 * @param lam_mask: 1D boolean wavelength mask (nlam).
 * @param out: The pre-allocated output buffer (nspec x nlam).
 * @param nspec: The number of spectra rows.
 * @param nlam: The number of wavelength bins.
 * @param nthreads: The number of OpenMP threads.
 */
template <typename Real, typename OutT>
static void scale_spectra_2d_both_masks_omp(const Real *__restrict__ spectra,
                                            const Real *__restrict__ scaling,
                                            const npy_bool *mask,
                                            const npy_bool *lam_mask, OutT *out,
                                            int nspec, int nlam, int nthreads) {

  /* Split the spectra rows evenly across threads. */
#pragma omp parallel for num_threads(nthreads) schedule(static)
  for (int ispec = 0; ispec < nspec; ispec++) {
    const Real *in_row = spectra + ispec * nlam;
    OutT *out_row = out + ispec * nlam;

    /* For masked rows apply the wavelength-dependent scale; copy otherwise. */
    if (mask[ispec]) {
      const Real scale = scaling[ispec];

      for (int ilam = 0; ilam < nlam; ilam++) {
        out_row[ilam] = lam_mask[ilam] ? static_cast<OutT>(in_row[ilam] * scale)
                                       : static_cast<OutT>(in_row[ilam]);
      }
    } else {
      for (int ilam = 0; ilam < nlam; ilam++) {
        out_row[ilam] = static_cast<OutT>(in_row[ilam]);
      }
    }
  }
}

#endif /* WITH_OPENMP */

/**
 * @brief Dispatch per-spectrum row scaling to the correct kernel.
 *
 * @param spectra: Input spectra array (nspec x nlam).
 * @param scaling: Per-spectrum scaling vector (nspec).
 * @param mask: Optional 1D row mask (nspec).
 * @param lam_mask: Optional 1D wavelength mask (nlam).
 * @param out: Output buffer (nspec x nlam).
 * @param nspec: The number of spectra rows.
 * @param nlam: The number of wavelength bins.
 * @param nthreads: The number of OpenMP threads.
 */
template <typename Real, typename OutT>
static void dispatch_scale_spectra_2d(const Real *spectra, const Real *scaling,
                                      const npy_bool *mask,
                                      const npy_bool *lam_mask, OutT *out,
                                      int nspec, int nlam, int nthreads) {

  /* Collapse the pointer checks once so the dispatch tree reads clearly. */
  const bool has_mask = (mask != NULL);
  const bool has_lam_mask = (lam_mask != NULL);

  if (nthreads > 1) {
#ifdef WITH_OPENMP
    /* When OpenMP is enabled, pick the parallel kernel that exactly matches
     * the caller's mask combination. */
    if (!has_mask && !has_lam_mask) {
      scale_spectra_2d_no_mask_omp(spectra, scaling, out, nspec, nlam,
                                   nthreads);
    } else if (has_mask && !has_lam_mask) {
      scale_spectra_2d_row_mask_omp(spectra, scaling, mask, out, nspec, nlam,
                                    nthreads);
    } else if (!has_mask && has_lam_mask) {
      scale_spectra_2d_lam_mask_omp(spectra, scaling, lam_mask, out, nspec,
                                    nlam, nthreads);
    } else {
      scale_spectra_2d_both_masks_omp(spectra, scaling, mask, lam_mask, out,
                                      nspec, nlam, nthreads);
    }
    return;
#else
    /* If this build has no OpenMP support, ignore the thread count and keep
     * going to the serial dispatch below. */
    (void)nthreads;
#endif
  }

  /* Single-threaded work, or OpenMP-disabled builds, both come through the
   * serial kernels. */
  if (!has_mask && !has_lam_mask) {
    scale_spectra_2d_no_mask_serial(spectra, scaling, out, nspec, nlam);
  } else if (has_mask && !has_lam_mask) {
    scale_spectra_2d_row_mask_serial(spectra, scaling, mask, out, nspec, nlam);
  } else if (!has_mask && has_lam_mask) {
    scale_spectra_2d_lam_mask_serial(spectra, scaling, lam_mask, out, nspec,
                                     nlam);
  } else {
    scale_spectra_2d_both_masks_serial(spectra, scaling, mask, lam_mask, out,
                                       nspec, nlam);
  }
}

/**
 * @brief Scale a 2D spectra array by a per-spectrum factor.
 *
 * Parses the Python arguments, validates shapes, allocates (or reuses) an
 * output buffer, then dispatches to the correct mask-specialised kernel.
 *
 * When @p out_obj is a writable ndarray of matching shape the result is
 * written directly into that buffer (safe for in-place when out_obj is the
 * same array as spectra). When @p out_obj is None a fresh array is allocated.
 *
 * @param spectra_obj: 2D float64 ndarray (nspec x nlam).
 * @param scaling_obj: 1D float64 ndarray (nspec).
 * @param mask_obj: Optional 1D boolean ndarray (nspec).
 * @param lam_mask_obj: Optional 1D boolean ndarray (nlam).
 * @param nthreads: Number of OpenMP threads.
 * @param out_obj: Optional output buffer, same shape and dtype as spectra.
 *
 * @return 2D float64 ndarray containing the scaled spectra.
 */
PyObject *scale_spectra_2d(PyObject *self, PyObject *args) {
  (void)self;

  PyObject *spectra_obj, *scaling_obj;
  PyObject *mask_obj = Py_None;
  PyObject *lam_mask_obj = Py_None;
  PyObject *out_obj = Py_None;
  PyObject *out_dtype_obj = NULL;
  int nthreads;

  if (!PyArg_ParseTuple(args, "OOOOi|OO", &spectra_obj, &scaling_obj, &mask_obj,
                        &lam_mask_obj, &nthreads, &out_obj, &out_dtype_obj)) {
    return NULL;
  }

  /* Convert required inputs to array views (no conversion yet).
   * TODO: Remove coercion by default. */
  PyArrayObject *np_spectra = (PyArrayObject *)PyArray_FromAny(
      spectra_obj, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
  PyArrayObject *np_scaling = (PyArrayObject *)PyArray_FromAny(
      scaling_obj, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
  if (!np_spectra || !np_scaling) {
    Py_XDECREF(np_spectra);
    Py_XDECREF(np_scaling);
    return NULL;
  }

  /* Optional masks handled as bool array views or NULL.
   * TODO: Remove coercion by default. */
  PyArrayObject *np_mask = nullptr;
  PyArrayObject *np_lam_mask = nullptr;
  if (mask_obj != Py_None) {
    np_mask = (PyArrayObject *)PyArray_FromAny(mask_obj, NULL, 0, 0,
                                               NPY_ARRAY_ENSUREARRAY, NULL);
    if (!np_mask) {
      Py_DECREF(np_spectra);
      Py_DECREF(np_scaling);
      return NULL;
    }
  }
  if (lam_mask_obj != Py_None) {
    np_lam_mask = (PyArrayObject *)PyArray_FromAny(lam_mask_obj, NULL, 0, 0,
                                                   NPY_ARRAY_ENSUREARRAY, NULL);
    if (!np_lam_mask) {
      Py_DECREF(np_spectra);
      Py_DECREF(np_scaling);
      Py_XDECREF(np_mask);
      return NULL;
    }
  }

  /* Basic shape checks. */
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
  const int nspec = static_cast<int>(spectra_dims[0]);
  const int nlam = static_cast<int>(spectra_dims[1]);

  if (PyArray_DIMS(np_scaling)[0] != spectra_dims[0]) {
    PyErr_SetString(PyExc_ValueError,
                    "scaling length must match the spectra first dimension.");
    Py_DECREF(np_spectra);
    Py_DECREF(np_scaling);
    Py_XDECREF(np_mask);
    Py_XDECREF(np_lam_mask);
    return NULL;
  }

  if (np_mask != NULL && (PyArray_NDIM(np_mask) != 1 ||
                          PyArray_DIMS(np_mask)[0] != spectra_dims[0])) {
    PyErr_SetString(PyExc_ValueError,
                    "mask must be a 1D bool array matching spectra rows.");
    Py_DECREF(np_spectra);
    Py_DECREF(np_scaling);
    Py_XDECREF(np_mask);
    Py_XDECREF(np_lam_mask);
    return NULL;
  }

  if (np_lam_mask != NULL &&
      (PyArray_NDIM(np_lam_mask) != 1 ||
       PyArray_DIMS(np_lam_mask)[0] != spectra_dims[1])) {
    PyErr_SetString(
        PyExc_ValueError,
        "lam_mask must be a 1D bool array matching spectra columns.");
    Py_DECREF(np_spectra);
    Py_DECREF(np_scaling);
    Py_XDECREF(np_mask);
    Py_XDECREF(np_lam_mask);
    return NULL;
  }

  /* Resolve input float family and requested out dtype. */
  PyArrayObject *float_arrays[2] = {np_spectra, np_scaling};
  const char *float_names[2] = {"spectra", "scaling"};
  int input_typenum = -1;
  if (!is_matching_float_dtypes(float_arrays, float_names, 2, &input_typenum)) {
    Py_DECREF(np_spectra);
    Py_DECREF(np_scaling);
    Py_XDECREF(np_mask);
    Py_XDECREF(np_lam_mask);
    return NULL;
  }

  int out_typenum = -1;
  if (out_dtype_obj != NULL && out_dtype_obj != Py_None) {
    out_typenum = resolve_output_typenum(out_dtype_obj, "out_dtype");
    if (out_typenum < 0) {
      Py_DECREF(np_spectra);
      Py_DECREF(np_scaling);
      Py_XDECREF(np_mask);
      Py_XDECREF(np_lam_mask);
      return NULL;
    }
  }
  if (out_typenum < 0)
    out_typenum = input_typenum;

  /* Prepare output array: reuse provided or allocate with requested dtype.
   * TODO: Remove coercion by default. */
  PyArrayObject *np_out = NULL;
  if (out_obj != Py_None) {
    np_out = (PyArrayObject *)PyArray_FromAny(
        out_obj, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_WRITEABLE, NULL);
    if (!np_out) {
      Py_DECREF(np_spectra);
      Py_DECREF(np_scaling);
      Py_XDECREF(np_mask);
      Py_XDECREF(np_lam_mask);
      return NULL;
    }
    if (PyArray_NDIM(np_out) != 2 ||
        PyArray_DIMS(np_out)[0] != spectra_dims[0] ||
        PyArray_DIMS(np_out)[1] != spectra_dims[1]) {
      PyErr_SetString(PyExc_ValueError,
                      "out must have shape (nspec, nlam) matching spectra.");
      Py_DECREF(np_spectra);
      Py_DECREF(np_scaling);
      Py_XDECREF(np_mask);
      Py_XDECREF(np_lam_mask);
      Py_DECREF(np_out);
      return NULL;
    }
    if (PyArray_TYPE(np_out) != out_typenum || !PyArray_ISCARRAY(np_out)) {
      PyErr_SetString(
          PyExc_ValueError,
          "out must be a C-contiguous array with the requested out_dtype.");
      Py_DECREF(np_spectra);
      Py_DECREF(np_scaling);
      Py_XDECREF(np_mask);
      Py_XDECREF(np_lam_mask);
      Py_DECREF(np_out);
      return NULL;
    }
  } else {
    npy_intp dims[2];
    dims[0] = spectra_dims[0];
    dims[1] = spectra_dims[1];
    np_out = (PyArrayObject *)PyArray_SimpleNew(2, dims, out_typenum);
    if (!np_out) {
      Py_DECREF(np_spectra);
      Py_DECREF(np_scaling);
      Py_XDECREF(np_mask);
      Py_XDECREF(np_lam_mask);
      return NULL;
    }
  }

  /* Dispatch to typed kernel. */
  tic("scale_spectra_2d");

  if (input_typenum == NPY_FLOAT32) {
    const float *spectra = data_ptr<const float>(np_spectra);
    const float *scaling = data_ptr<const float>(np_scaling);
    if (out_typenum == NPY_FLOAT32) {
      float *out = data_ptr<float>(np_out);
      dispatch_scale_spectra_2d<float, float>(
          spectra, scaling,
          np_mask ? data_ptr<const npy_bool>(np_mask) : nullptr,
          np_lam_mask ? data_ptr<const npy_bool>(np_lam_mask) : nullptr, out,
          nspec, nlam, nthreads);
    } else {
      double *out = data_ptr<double>(np_out);
      dispatch_scale_spectra_2d<float, double>(
          spectra, scaling,
          np_mask ? data_ptr<const npy_bool>(np_mask) : nullptr,
          np_lam_mask ? data_ptr<const npy_bool>(np_lam_mask) : nullptr, out,
          nspec, nlam, nthreads);
    }
  } else {
    const double *spectra = data_ptr<const double>(np_spectra);
    const double *scaling = data_ptr<const double>(np_scaling);
    if (out_typenum == NPY_FLOAT32) {
      float *out = data_ptr<float>(np_out);
      dispatch_scale_spectra_2d<double, float>(
          spectra, scaling,
          np_mask ? data_ptr<const npy_bool>(np_mask) : nullptr,
          np_lam_mask ? data_ptr<const npy_bool>(np_lam_mask) : nullptr, out,
          nspec, nlam, nthreads);
    } else {
      double *out = data_ptr<double>(np_out);
      dispatch_scale_spectra_2d<double, double>(
          spectra, scaling,
          np_mask ? data_ptr<const npy_bool>(np_mask) : nullptr,
          np_lam_mask ? data_ptr<const npy_bool>(np_lam_mask) : nullptr, out,
          nspec, nlam, nthreads);
    }
  }

  toc("scale_spectra_2d");

  Py_DECREF(np_spectra);
  Py_DECREF(np_scaling);
  Py_XDECREF(np_mask);
  Py_XDECREF(np_lam_mask);

  return Py_BuildValue("N", np_out);
}

/* ------------------------------------------------------------------------ */
/*  apply_separable_attenuation_2d — exp(-tau_v * tau_x_v) with mask split  */
/* ------------------------------------------------------------------------ */
/*
 * Separated into no-mask and row-mask variants so the inner expfma chain
 * in the no-mask case compiles without runtime branch checks. */

/**
 * @brief This applies separable attenuation with no row mask.
 *
 * This is the serial version of the function for rows without a mask.
 *
 * Computes out[irow, icol] = spectra[irow, icol] * exp(-tau_v[irow]
 * * tau_x_v[icol]) in a single fused pass.
 *
 * @param spectra: The input 2D spectra array (nrows x ncols).
 * @param tau_v: 1D V-band optical depth per row (nrows).
 * @param tau_x_v: 1D extinction curve per column (ncols).
 * @param out: The pre-allocated output buffer (nrows x ncols).
 * @param nrows: The number of spectra rows.
 * @param ncols: The number of spectral columns.
 */
template <typename Real, typename OutT>
static void attenuate_2d_no_mask_serial(const Real *__restrict__ spectra,
                                        const Real *__restrict__ tau_v,
                                        const Real *__restrict__ tau_x_v,
                                        OutT *out, int nrows, int ncols) {

  /* Loop over every row and apply the fused exponential-attenuation chain. */
  for (int irow = 0; irow < nrows; irow++) {

    /* Cache the V-band optical depth for this row so we read it once. */
    const Real row_tau = tau_v[irow];
    const Real *in_row = spectra + irow * ncols;
    OutT *out_row = out + irow * ncols;

    /* Attenuate every wavelength by exp(-tau_v * tau_x_v). */
    for (int icol = 0; icol < ncols; icol++) {
      out_row[icol] =
          static_cast<OutT>(in_row[icol] * std::exp(-row_tau * tau_x_v[icol]));
    }
  }
}

/**
 * @brief This applies separable attenuation with a 1D row mask.
 *
 * This is the serial version of the function for rows with a row mask.
 *
 * @param spectra: The input 2D spectra array (nrows x ncols).
 * @param tau_v: 1D V-band optical depth per row (nrows).
 * @param tau_x_v: 1D extinction curve per column (ncols).
 * @param mask: 1D boolean row mask (nrows).
 * @param out: The pre-allocated output buffer (nrows x ncols).
 * @param nrows: The number of spectra rows.
 * @param ncols: The number of spectral columns.
 */
template <typename Real, typename OutT>
static void attenuate_2d_with_mask_serial(const Real *__restrict__ spectra,
                                          const Real *__restrict__ tau_v,
                                          const Real *__restrict__ tau_x_v,
                                          const npy_bool *mask, OutT *out,
                                          int nrows, int ncols) {

  /* Loop over every row and dispatch based on the mask. */
  for (int irow = 0; irow < nrows; irow++) {
    const Real *in_row = spectra + irow * ncols;
    OutT *out_row = out + irow * ncols;

    /* Attenuate masked rows and copy unmasked rows through unchanged. */
    if (mask[irow]) {
      const Real row_tau = tau_v[irow];

      for (int icol = 0; icol < ncols; icol++) {
        out_row[icol] = static_cast<OutT>(in_row[icol] *
                                          std::exp(-row_tau * tau_x_v[icol]));
      }
    } else {
      for (int icol = 0; icol < ncols; icol++) {
        out_row[icol] = static_cast<OutT>(in_row[icol]);
      }
    }
  }
}

#ifdef WITH_OPENMP

/**
 * @brief This applies separable attenuation with no mask using OpenMP.
 *
 * This is the parallel version of the function for rows without a mask.
 *
 * @param spectra: The input 2D spectra array (nrows x ncols).
 * @param tau_v: 1D V-band optical depth per row (nrows).
 * @param tau_x_v: 1D extinction curve per column (ncols).
 * @param out: The pre-allocated output buffer (nrows x ncols).
 * @param nrows: The number of spectra rows.
 * @param ncols: The number of spectral columns.
 * @param nthreads: The number of OpenMP threads.
 */
template <typename Real, typename OutT>
static void attenuate_2d_no_mask_omp(const Real *__restrict__ spectra,
                                     const Real *__restrict__ tau_v,
                                     const Real *__restrict__ tau_x_v,
                                     OutT *out, int nrows, int ncols,
                                     int nthreads) {

  /* Split the rows evenly across threads. */
#pragma omp parallel for num_threads(nthreads) schedule(static)
  for (int irow = 0; irow < nrows; irow++) {
    /* Cache the V-band optical depth for this row. */
    const Real row_tau = tau_v[irow];
    const Real *in_row = spectra + irow * ncols;
    OutT *out_row = out + irow * ncols;

    /* Attenuate every wavelength by exp(-tau_v * tau_x_v). */
    for (int icol = 0; icol < ncols; icol++) {
      out_row[icol] =
          static_cast<OutT>(in_row[icol] * std::exp(-row_tau * tau_x_v[icol]));
    }
  }
}

/**
 * @brief This applies separable attenuation with a row mask using OpenMP.
 *
 * This is the parallel version of the function for rows with a row mask.
 *
 * @param spectra: The input 2D spectra array (nrows x ncols).
 * @param tau_v: 1D V-band optical depth per row (nrows).
 * @param tau_x_v: 1D extinction curve per column (ncols).
 * @param mask: 1D boolean row mask (nrows).
 * @param out: The pre-allocated output buffer (nrows x ncols).
 * @param nrows: The number of spectra rows.
 * @param ncols: The number of spectral columns.
 * @param nthreads: The number of OpenMP threads.
 */
template <typename Real, typename OutT>
static void attenuate_2d_with_mask_omp(const Real *__restrict__ spectra,
                                       const Real *__restrict__ tau_v,
                                       const Real *__restrict__ tau_x_v,
                                       const npy_bool *mask, OutT *out,
                                       int nrows, int ncols, int nthreads) {

  /* Split the rows evenly across threads. */
#pragma omp parallel for num_threads(nthreads) schedule(static)
  for (int irow = 0; irow < nrows; irow++) {
    const Real *in_row = spectra + irow * ncols;
    OutT *out_row = out + irow * ncols;

    /* Attenuate masked rows and copy unmasked rows through unchanged. */
    if (mask[irow]) {
      const Real row_tau = tau_v[irow];

      for (int icol = 0; icol < ncols; icol++) {
        out_row[icol] = static_cast<OutT>(in_row[icol] *
                                          std::exp(-row_tau * tau_x_v[icol]));
      }
    } else {
      for (int icol = 0; icol < ncols; icol++) {
        out_row[icol] = static_cast<OutT>(in_row[icol]);
      }
    }
  }
}

#endif /* WITH_OPENMP */

/**
 * @brief Dispatch separable attenuation to the correct kernel.
 *
 * @param spectra: Input spectra array (nrows x ncols).
 * @param tau_v: 1D V-band optical depth per row (nrows).
 * @param tau_x_v: 1D extinction curve per column (ncols).
 * @param mask: Optional 1D row mask (nrows).
 * @param out: Output buffer (nrows x ncols).
 * @param nrows: The number of spectra rows.
 * @param ncols: The number of spectral columns.
 * @param nthreads: The number of OpenMP threads.
 */
template <typename Real, typename OutT>
static void dispatch_attenuate_2d(const Real *spectra, const Real *tau_v,
                                  const Real *tau_x_v, const npy_bool *mask,
                                  OutT *out, int nrows, int ncols,
                                  int nthreads) {

  /* Attenuation only has one optional mask dimension, so the dispatch tree is
   * simpler than the scaling case. */
  const bool has_mask = (mask != NULL);

  if (nthreads > 1) {
#ifdef WITH_OPENMP
    /* Use the threaded kernels only when this build actually supports them. */
    if (!has_mask) {
      attenuate_2d_no_mask_omp(spectra, tau_v, tau_x_v, out, nrows, ncols,
                               nthreads);
    } else {
      attenuate_2d_with_mask_omp(spectra, tau_v, tau_x_v, mask, out, nrows,
                                 ncols, nthreads);
    }
    return;
#else
    /* OpenMP-free builds quietly fall back to the serial kernels below. */
    (void)nthreads;
#endif
  }

  /* The serial dispatch only needs to distinguish between masked and
   * unmasked rows. */
  if (!has_mask) {
    attenuate_2d_no_mask_serial(spectra, tau_v, tau_x_v, out, nrows, ncols);
  } else {
    attenuate_2d_with_mask_serial(spectra, tau_v, tau_x_v, mask, out, nrows,
                                  ncols);
  }
}

/**
 * @brief Apply exp(-tau_v * tau_x_v) attenuation to a 2D spectra array.
 *
 * Dispatches to a mask-specialised kernel so the inner expfma chain
 * is free of runtime branch checks.
 *
 * @param spectra_obj: 2D float64 ndarray (nrows x ncols).
 * @param tau_v_obj: 1D float64 ndarray (nrows), V-band optical depth.
 * @param tau_x_v_obj: 1D float64 ndarray (ncols), extinction curve.
 * @param mask_obj: Optional 1D boolean ndarray (nrows).
 * @param nthreads: Number of OpenMP threads.
 * @param out_obj: Optional output buffer, same shape as spectra.
 *
 * @return 2D float64 ndarray containing the attenuated spectra.
 */
PyObject *apply_separable_attenuation_2d(PyObject *self, PyObject *args) {
  (void)self;

  /* Declare inputs for spectra, per-particle optical depths,
   * wavelength-dependent extinction curve, and optional row mask. */
  PyObject *spectra_obj, *tau_v_obj, *tau_x_v_obj;
  PyObject *mask_obj = Py_None;
  PyObject *out_obj = Py_None;
  int nthreads;

  PyObject *out_dtype_obj = NULL;
  if (!PyArg_ParseTuple(args, "OOOOi|OO", &spectra_obj, &tau_v_obj,
                        &tau_x_v_obj, &mask_obj, &nthreads, &out_obj,
                        &out_dtype_obj)) {
    return NULL;
  }
  /* Convert inputs to array views (no conversion yet) so we can accept both
   * float32 and float64 inputs.
   * TODO: Remove coercion by default. */
  PyArrayObject *np_spectra = (PyArrayObject *)PyArray_FromAny(
      spectra_obj, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
  PyArrayObject *np_tau_v = (PyArrayObject *)PyArray_FromAny(
      tau_v_obj, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
  PyArrayObject *np_tau_x_v = (PyArrayObject *)PyArray_FromAny(
      tau_x_v_obj, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
  if (!np_spectra || !np_tau_v || !np_tau_x_v) {
    Py_XDECREF(np_spectra);
    Py_XDECREF(np_tau_v);
    Py_XDECREF(np_tau_x_v);
    return NULL;
  }

  /* Optional masks are also currently normalised through PyArray_FromAny.
   * TODO: Remove coercion by default. */
  PyArrayObject *np_mask = nullptr;
  if (mask_obj != Py_None) {
    np_mask = (PyArrayObject *)PyArray_FromAny(mask_obj, NULL, 0, 0,
                                               NPY_ARRAY_ENSUREARRAY, NULL);
    if (!np_mask) {
      Py_DECREF(np_spectra);
      Py_DECREF(np_tau_v);
      Py_DECREF(np_tau_x_v);
      return NULL;
    }
  }

  /* Validate shapes. */
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

  if (np_mask != NULL && (PyArray_NDIM(np_mask) != 1 ||
                          PyArray_DIMS(np_mask)[0] != spectra_dims[0])) {
    PyErr_SetString(PyExc_ValueError,
                    "mask must be a 1D bool array matching spectra rows.");
    Py_DECREF(np_spectra);
    Py_DECREF(np_tau_v);
    Py_DECREF(np_tau_x_v);
    Py_XDECREF(np_mask);
    return NULL;
  }

  /* Resolve input float family and out dtype. */
  PyArrayObject *float_arrays[3] = {np_spectra, np_tau_v, np_tau_x_v};
  const char *float_names[3] = {"spectra", "tau_v", "tau_x_v"};
  int input_typenum = -1;
  if (!is_matching_float_dtypes(float_arrays, float_names, 3, &input_typenum)) {
    Py_DECREF(np_spectra);
    Py_DECREF(np_tau_v);
    Py_DECREF(np_tau_x_v);
    Py_XDECREF(np_mask);
    return NULL;
  }

  int out_typenum = -1;
  if (out_dtype_obj != NULL && out_dtype_obj != Py_None) {
    out_typenum = resolve_output_typenum(out_dtype_obj, "out_dtype");
    if (out_typenum < 0) {
      Py_DECREF(np_spectra);
      Py_DECREF(np_tau_v);
      Py_DECREF(np_tau_x_v);
      Py_XDECREF(np_mask);
      return NULL;
    }
  }
  if (out_typenum < 0)
    out_typenum = input_typenum;

  /* Prepare output array: reuse provided or allocate with requested dtype.
   * TODO: Remove coercion by default. */
  PyArrayObject *np_out = NULL;
  if (out_obj != Py_None) {
    np_out = (PyArrayObject *)PyArray_FromAny(
        out_obj, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_WRITEABLE, NULL);
    if (!np_out) {
      Py_DECREF(np_spectra);
      Py_DECREF(np_tau_v);
      Py_DECREF(np_tau_x_v);
      Py_XDECREF(np_mask);
      return NULL;
    }
    if (PyArray_NDIM(np_out) != 2 ||
        PyArray_DIMS(np_out)[0] != spectra_dims[0] ||
        PyArray_DIMS(np_out)[1] != spectra_dims[1]) {
      PyErr_SetString(PyExc_ValueError,
                      "out must have shape (nrows, ncols) matching spectra.");
      Py_DECREF(np_spectra);
      Py_DECREF(np_tau_v);
      Py_DECREF(np_tau_x_v);
      Py_XDECREF(np_mask);
      Py_DECREF(np_out);
      return NULL;
    }
    if (PyArray_TYPE(np_out) != out_typenum || !PyArray_ISCARRAY(np_out)) {
      PyErr_SetString(
          PyExc_ValueError,
          "out must be a C-contiguous array with the requested out_dtype.");
      Py_DECREF(np_spectra);
      Py_DECREF(np_tau_v);
      Py_DECREF(np_tau_x_v);
      Py_XDECREF(np_mask);
      Py_DECREF(np_out);
      return NULL;
    }
  } else {
    npy_intp dims[2];
    dims[0] = spectra_dims[0];
    dims[1] = spectra_dims[1];
    np_out = (PyArrayObject *)PyArray_SimpleNew(2, dims, out_typenum);
    if (!np_out) {
      Py_DECREF(np_spectra);
      Py_DECREF(np_tau_v);
      Py_DECREF(np_tau_x_v);
      Py_XDECREF(np_mask);
      return NULL;
    }
  }

  /* Dispatch to typed kernel. */
  tic("apply_separable_attenuation_2d");
  if (input_typenum == NPY_FLOAT32) {
    const float *spectra = data_ptr<const float>(np_spectra);
    const float *tau_v = data_ptr<const float>(np_tau_v);
    const float *tau_x_v = data_ptr<const float>(np_tau_x_v);
    if (out_typenum == NPY_FLOAT32) {
      float *out = data_ptr<float>(np_out);
      dispatch_attenuate_2d<float, float>(
          spectra, tau_v, tau_x_v,
          np_mask ? data_ptr<const npy_bool>(np_mask) : nullptr, out, nrows,
          ncols, nthreads);
    } else {
      double *out = data_ptr<double>(np_out);
      dispatch_attenuate_2d<float, double>(
          spectra, tau_v, tau_x_v,
          np_mask ? data_ptr<const npy_bool>(np_mask) : nullptr, out, nrows,
          ncols, nthreads);
    }
  } else {
    const double *spectra = data_ptr<const double>(np_spectra);
    const double *tau_v = data_ptr<const double>(np_tau_v);
    const double *tau_x_v = data_ptr<const double>(np_tau_x_v);
    if (out_typenum == NPY_FLOAT32) {
      float *out = data_ptr<float>(np_out);
      dispatch_attenuate_2d<double, float>(
          spectra, tau_v, tau_x_v,
          np_mask ? data_ptr<const npy_bool>(np_mask) : nullptr, out, nrows,
          ncols, nthreads);
    } else {
      double *out = data_ptr<double>(np_out);
      dispatch_attenuate_2d<double, double>(
          spectra, tau_v, tau_x_v,
          np_mask ? data_ptr<const npy_bool>(np_mask) : nullptr, out, nrows,
          ncols, nthreads);
    }
  }

  toc("apply_separable_attenuation_2d");

  Py_DECREF(np_spectra);
  Py_DECREF(np_tau_v);
  Py_DECREF(np_tau_x_v);
  Py_XDECREF(np_mask);

  return Py_BuildValue("N", np_out);
}

/* ------------------------------------------------------------------------ */
/*  multiply_array_by_vector_1d — last-axis multiply with optional out       */
/* ------------------------------------------------------------------------ */
/*
 * Already branch-free (no mask dimension). The only addition is an optional
 * out buffer for in-place support. */

/**
 * @brief Multiply a 1D or 2D array by a 1D vector over the last axis.
 *
 * @param array_obj: 1D or 2D float64 ndarray.
 * @param vector_obj: 1D float64 ndarray matching the last array dimension.
 * @param nthreads: Number of OpenMP threads.
 * @param out_obj: Optional output buffer, same shape as array.
 *
 * @return Array with each element multiplied by the corresponding vector
 *     entry on the last axis.
 */
template <typename Real, typename OutT>
static PyObject *multiply_array_by_vector_1d_typed(PyArrayObject *np_array,
                                                   PyArrayObject *np_vector,
                                                   int nthreads,
                                                   PyArrayObject *np_out) {
  const npy_intp *array_dims = PyArray_DIMS(np_array);
  const int array_ndim = PyArray_NDIM(np_array);
  const int ncols = static_cast<int>(array_dims[array_ndim - 1]);
  const int nrows = (array_ndim == 1) ? 1 : static_cast<int>(array_dims[0]);

  const Real *array = data_ptr<const Real>(np_array);
  const Real *vector = data_ptr<const Real>(np_vector);
  OutT *out = data_ptr<OutT>(np_out);

  tic("multiply_array_by_vector_1d");

#ifdef WITH_OPENMP
#pragma omp parallel for if (nthreads > 1) num_threads(nthreads)               \
    schedule(static)
#endif
  for (int irow = 0; irow < nrows; irow++) {
    const Real *in_row = array + (npy_intp)irow * ncols;
    OutT *out_row = out + (npy_intp)irow * ncols;
    for (int icol = 0; icol < ncols; icol++) {
      out_row[icol] = static_cast<OutT>(in_row[icol] * vector[icol]);
    }
  }

  toc("multiply_array_by_vector_1d");

  return Py_BuildValue("N", np_out);
}

PyObject *multiply_array_by_vector_1d(PyObject *self, PyObject *args) {
  (void)self;

  PyObject *array_obj, *vector_obj;
  PyObject *out_obj = Py_None;
  PyObject *out_dtype_obj = NULL;
  int nthreads;

  if (!PyArg_ParseTuple(args, "OOi|OO", &array_obj, &vector_obj, &nthreads,
                        &out_obj, &out_dtype_obj)) {
    return NULL;
  }

  /* Convert inputs to array views (no conversion yet).
   * TODO: Remove coercion by default. */
  PyArrayObject *np_array = (PyArrayObject *)PyArray_FromAny(
      array_obj, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
  PyArrayObject *np_vector = (PyArrayObject *)PyArray_FromAny(
      vector_obj, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
  if (!np_array || !np_vector) {
    Py_XDECREF(np_array);
    Py_XDECREF(np_vector);
    return NULL;
  }

  /* Validate dimensions. */
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

  /* Resolve input float family. */
  PyArrayObject *float_arrays[2] = {np_array, np_vector};
  const char *float_names[2] = {"array", "vector"};
  int input_typenum = -1;
  if (!is_matching_float_dtypes(float_arrays, float_names, 2, &input_typenum)) {
    Py_DECREF(np_array);
    Py_DECREF(np_vector);
    return NULL;
  }

  int out_typenum = -1;
  if (out_dtype_obj != NULL && out_dtype_obj != Py_None) {
    out_typenum = resolve_output_typenum(out_dtype_obj, "out_dtype");
    if (out_typenum < 0) {
      Py_DECREF(np_array);
      Py_DECREF(np_vector);
      return NULL;
    }
  }
  if (out_typenum < 0)
    out_typenum = input_typenum;

  /* Prepare output array: reuse provided or allocate a new one with
   * out_typenum.
   * TODO: Remove coercion by default. */
  PyArrayObject *np_out = NULL;
  if (out_obj != Py_None) {
    np_out = (PyArrayObject *)PyArray_FromAny(
        out_obj, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_WRITEABLE, NULL);
    if (!np_out) {
      Py_DECREF(np_array);
      Py_DECREF(np_vector);
      return NULL;
    }
    /* Verify shape */
    if (PyArray_NDIM(np_out) != array_ndim) {
      PyErr_SetString(PyExc_ValueError,
                      "out array must have the same shape as input array.");
      Py_DECREF(np_array);
      Py_DECREF(np_vector);
      Py_DECREF(np_out);
      return NULL;
    }
    for (int d = 0; d < array_ndim; d++) {
      if (PyArray_DIMS(np_out)[d] != array_dims[d]) {
        PyErr_SetString(PyExc_ValueError,
                        "out array must have the same shape as input array.");
        Py_DECREF(np_array);
        Py_DECREF(np_vector);
        Py_DECREF(np_out);
        return NULL;
      }
    }
    if (PyArray_TYPE(np_out) != out_typenum || !PyArray_ISCARRAY(np_out)) {
      PyErr_SetString(
          PyExc_ValueError,
          "out array must be C-contiguous with the requested out_dtype.");
      Py_DECREF(np_array);
      Py_DECREF(np_vector);
      Py_DECREF(np_out);
      return NULL;
    }
  } else {
    /* Allocate a new array with the same shape but requested dtype. */
    npy_intp dims[NPY_MAXDIMS];
    for (int d = 0; d < array_ndim; d++)
      dims[d] = array_dims[d];
    np_out = (PyArrayObject *)PyArray_SimpleNew(array_ndim, dims, out_typenum);
    if (!np_out) {
      Py_DECREF(np_array);
      Py_DECREF(np_vector);
      return NULL;
    }
  }

  /* Dispatch on input/output precision. */
  if (input_typenum == NPY_FLOAT32) {
    if (out_typenum == NPY_FLOAT32) {
      PyObject *res = multiply_array_by_vector_1d_typed<float, float>(
          np_array, np_vector, nthreads, np_out);
      Py_DECREF(np_array);
      Py_DECREF(np_vector);
      return res;
    }
    PyObject *res = multiply_array_by_vector_1d_typed<float, double>(
        np_array, np_vector, nthreads, np_out);
    Py_DECREF(np_array);
    Py_DECREF(np_vector);
    return res;
  } else {
    if (out_typenum == NPY_FLOAT32) {
      PyObject *res = multiply_array_by_vector_1d_typed<double, float>(
          np_array, np_vector, nthreads, np_out);
      Py_DECREF(np_array);
      Py_DECREF(np_vector);
      return res;
    }
    PyObject *res = multiply_array_by_vector_1d_typed<double, double>(
        np_array, np_vector, nthreads, np_out);
    Py_DECREF(np_array);
    Py_DECREF(np_vector);
    return res;
  }
}

/* ------------------------------------------------------------------------ */
/*  scale_line_2d — fused lum+cont scaling with four mask combos            */
/* ------------------------------------------------------------------------ */
/*
 * Processes both arrays in one loop so the scaling vectors are read once
 * instead of twice. The four mask combinations mirror scale_spectra_2d. */

/**
 * @brief This calculates fused lum+cont row scaling with no masks.
 *
 * This is the serial version of the function for rows without a mask.
 *
 * @param lum: Input luminosity array (nspec x nlam).
 * @param cont: Input continuum array (nspec x nlam).
 * @param scaling_lum: Per-spectrum luminosity factor (nspec).
 * @param scaling_cont: Per-spectrum continuum factor (nspec).
 * @param out_lum: Output luminosity buffer (nspec x nlam).
 * @param out_cont: Output continuum buffer (nspec x nlam).
 * @param nspec: The number of spectra rows.
 * @param nlam: The number of wavelength bins.
 */
template <typename Real, typename OutT>
static void scale_line_2d_no_mask_serial(const Real *__restrict__ lum,
                                         const Real *__restrict__ cont,
                                         const Real *scaling_lum,
                                         const Real *scaling_cont,
                                         OutT *out_lum, OutT *out_cont,
                                         int nspec, int nlam) {

  /* Loop over every spectrum and scale both lum and cont in one pass. */
  for (int i = 0; i < nspec; i++) {

    /* Cache both scale factors so we read each once. */
    const Real sl = scaling_lum[i];
    const Real sc = scaling_cont[i];
    const Real *in_lum_row = lum + i * nlam;
    const Real *in_cont_row = cont + i * nlam;
    OutT *out_lum_row = out_lum + i * nlam;
    OutT *out_cont_row = out_cont + i * nlam;

#pragma GCC ivdep
    for (int j = 0; j < nlam; j++) {
      out_lum_row[j] = static_cast<OutT>(in_lum_row[j] * sl);
      out_cont_row[j] = static_cast<OutT>(in_cont_row[j] * sc);
    }
  }
}

/**
 * @brief This calculates fused lum+cont row scaling with a 1D row mask.
 *
 * This is the serial version of the function for rows with a row mask.
 *
 * @param lum: Input luminosity array (nspec x nlam).
 * @param cont: Input continuum array (nspec x nlam).
 * @param scaling_lum: Per-spectrum luminosity factor (nspec).
 * @param scaling_cont: Per-spectrum continuum factor (nspec).
 * @param mask: 1D boolean row mask (nspec).
 * @param out_lum: Output luminosity buffer (nspec x nlam).
 * @param out_cont: Output continuum buffer (nspec x nlam).
 * @param nspec: The number of spectra rows.
 * @param nlam: The number of wavelength bins.
 */
template <typename Real, typename OutT>
static void scale_line_2d_row_mask_serial(const Real *__restrict__ lum,
                                          const Real *__restrict__ cont,
                                          const Real *scaling_lum,
                                          const Real *scaling_cont,
                                          const npy_bool *mask, OutT *out_lum,
                                          OutT *out_cont, int nspec, int nlam) {

  /* Loop over every spectrum and dispatch based on the mask. */
  for (int i = 0; i < nspec; i++) {
    const Real *in_lum_row = lum + i * nlam;
    const Real *in_cont_row = cont + i * nlam;
    OutT *out_lum_row = out_lum + i * nlam;
    OutT *out_cont_row = out_cont + i * nlam;

    /* Scale masked rows and copy unmasked rows through unchanged. */
    if (mask[i]) {
      const Real sl = scaling_lum[i];
      const Real sc = scaling_cont[i];
#pragma GCC ivdep
      for (int j = 0; j < nlam; j++) {
        out_lum_row[j] = static_cast<OutT>(in_lum_row[j] * sl);
        out_cont_row[j] = static_cast<OutT>(in_cont_row[j] * sc);
      }
    } else {
#pragma GCC ivdep
      for (int j = 0; j < nlam; j++) {
        out_lum_row[j] = static_cast<OutT>(in_lum_row[j]);
        out_cont_row[j] = static_cast<OutT>(in_cont_row[j]);
      }
    }
  }
}

/**
 * @brief This calculates fused lum+cont row scaling with a 1D wavelength
 *        mask.
 *
 * This is the serial version of the function for rows with a wavelength mask.
 *
 * @param lum: Input luminosity array (nspec x nlam).
 * @param cont: Input continuum array (nspec x nlam).
 * @param scaling_lum: Per-spectrum luminosity factor (nspec).
 * @param scaling_cont: Per-spectrum continuum factor (nspec).
 * @param lam_mask: 1D boolean wavelength mask (nlam).
 * @param out_lum: Output luminosity buffer (nspec x nlam).
 * @param out_cont: Output continuum buffer (nspec x nlam).
 * @param nspec: The number of spectra rows.
 * @param nlam: The number of wavelength bins.
 */
template <typename Real, typename OutT>
static void scale_line_2d_lam_mask_serial(
    const Real *__restrict__ lum, const Real *__restrict__ cont,
    const Real *scaling_lum, const Real *scaling_cont, const npy_bool *lam_mask,
    OutT *out_lum, OutT *out_cont, int nspec, int nlam) {

  /* Loop over every spectrum. */
  for (int i = 0; i < nspec; i++) {

    /* Cache both scale factors for this row. */
    const Real sl = scaling_lum[i];
    const Real sc = scaling_cont[i];
    const Real *in_lum_row = lum + i * nlam;
    const Real *in_cont_row = cont + i * nlam;
    OutT *out_lum_row = out_lum + i * nlam;
    OutT *out_cont_row = out_cont + i * nlam;

    /* Only scale wavelengths that pass the mask; keep others unchanged. */
    for (int j = 0; j < nlam; j++) {
      const bool apply = lam_mask[j];
      out_lum_row[j] = apply ? static_cast<OutT>(in_lum_row[j] * sl)
                             : static_cast<OutT>(in_lum_row[j]);
      out_cont_row[j] = apply ? static_cast<OutT>(in_cont_row[j] * sc)
                              : static_cast<OutT>(in_cont_row[j]);
    }
  }
}

/**
 * @brief This calculates fused lum+cont row scaling with both a row mask
 *        and a wavelength mask.
 *
 * This is the serial version of the function for rows with both masks.
 *
 * @param lum: Input luminosity array (nspec x nlam).
 * @param cont: Input continuum array (nspec x nlam).
 * @param scaling_lum: Per-spectrum luminosity factor (nspec).
 * @param scaling_cont: Per-spectrum continuum factor (nspec).
 * @param mask: 1D boolean row mask (nspec).
 * @param lam_mask: 1D boolean wavelength mask (nlam).
 * @param out_lum: Output luminosity buffer (nspec x nlam).
 * @param out_cont: Output continuum buffer (nspec x nlam).
 * @param nspec: The number of spectra rows.
 * @param nlam: The number of wavelength bins.
 */
template <typename Real, typename OutT>
static void scale_line_2d_both_masks_serial(
    const Real *__restrict__ lum, const Real *__restrict__ cont,
    const Real *scaling_lum, const Real *scaling_cont, const npy_bool *mask,
    const npy_bool *lam_mask, OutT *out_lum, OutT *out_cont, int nspec,
    int nlam) {

  /* Loop over every spectrum and dispatch based on the row mask. */
  for (int i = 0; i < nspec; i++) {
    const Real *in_lum_row = lum + i * nlam;
    const Real *in_cont_row = cont + i * nlam;
    OutT *out_lum_row = out_lum + i * nlam;
    OutT *out_cont_row = out_cont + i * nlam;

    /* For masked rows apply the wavelength-dependent scale; copy otherwise. */
    if (mask[i]) {
      const Real sl = scaling_lum[i];
      const Real sc = scaling_cont[i];
      for (int j = 0; j < nlam; j++) {
        const bool apply = lam_mask[j];
        out_lum_row[j] = apply ? static_cast<OutT>(in_lum_row[j] * sl)
                               : static_cast<OutT>(in_lum_row[j]);
        out_cont_row[j] = apply ? static_cast<OutT>(in_cont_row[j] * sc)
                                : static_cast<OutT>(in_cont_row[j]);
      }
    } else {
      for (int j = 0; j < nlam; j++) {
        out_lum_row[j] = static_cast<OutT>(in_lum_row[j]);
        out_cont_row[j] = static_cast<OutT>(in_cont_row[j]);
      }
    }
  }
}

#ifdef WITH_OPENMP

/**
 * @brief This calculates fused lum+cont scaling with no masks using OpenMP.
 *
 * This is the parallel version of the function for rows without a mask.
 *
 * @param lum: Input luminosity array (nspec x nlam).
 * @param cont: Input continuum array (nspec x nlam).
 * @param scaling_lum: Per-spectrum luminosity factor (nspec).
 * @param scaling_cont: Per-spectrum continuum factor (nspec).
 * @param out_lum: Output luminosity buffer (nspec x nlam).
 * @param out_cont: Output continuum buffer (nspec x nlam).
 * @param nspec: The number of spectra rows.
 * @param nlam: The number of wavelength bins.
 * @param nthreads: The number of OpenMP threads.
 */
template <typename Real, typename OutT>
static void scale_line_2d_no_mask_omp(const Real *__restrict__ lum,
                                      const Real *__restrict__ cont,
                                      const Real *scaling_lum,
                                      const Real *scaling_cont, OutT *out_lum,
                                      OutT *out_cont, int nspec, int nlam,
                                      int nthreads) {

  /* Split the spectra rows evenly across threads. */
#pragma omp parallel for num_threads(nthreads) schedule(static)
  for (int i = 0; i < nspec; i++) {
    /* Cache both scale factors for this row. */
    const Real sl = scaling_lum[i];
    const Real sc = scaling_cont[i];
    const Real *in_lum_row = lum + i * nlam;
    const Real *in_cont_row = cont + i * nlam;
    OutT *out_lum_row = out_lum + i * nlam;
    OutT *out_cont_row = out_cont + i * nlam;

#pragma omp simd
    for (int j = 0; j < nlam; j++) {
      out_lum_row[j] = static_cast<OutT>(in_lum_row[j] * sl);
      out_cont_row[j] = static_cast<OutT>(in_cont_row[j] * sc);
    }
  }
}

/**
 * @brief This calculates fused lum+cont scaling with a row mask using OpenMP.
 *
 * This is the parallel version of the function for rows with a row mask.
 *
 * @param lum: Input luminosity array (nspec x nlam).
 * @param cont: Input continuum array (nspec x nlam).
 * @param scaling_lum: Per-spectrum luminosity factor (nspec).
 * @param scaling_cont: Per-spectrum continuum factor (nspec).
 * @param mask: 1D boolean row mask (nspec).
 * @param out_lum: Output luminosity buffer (nspec x nlam).
 * @param out_cont: Output continuum buffer (nspec x nlam).
 * @param nspec: The number of spectra rows.
 * @param nlam: The number of wavelength bins.
 * @param nthreads: The number of OpenMP threads.
 */
template <typename Real, typename OutT>
static void scale_line_2d_row_mask_omp(
    const Real *__restrict__ lum, const Real *__restrict__ cont,
    const Real *scaling_lum, const Real *scaling_cont, const npy_bool *mask,
    OutT *out_lum, OutT *out_cont, int nspec, int nlam, int nthreads) {

  /* Split the spectra rows evenly across threads. */
#pragma omp parallel for num_threads(nthreads) schedule(static)
  for (int i = 0; i < nspec; i++) {
    const Real *in_lum_row = lum + i * nlam;
    const Real *in_cont_row = cont + i * nlam;
    OutT *out_lum_row = out_lum + i * nlam;
    OutT *out_cont_row = out_cont + i * nlam;

    /* Scale masked rows and copy unmasked rows through unchanged. */
    if (mask[i]) {
      const Real sl = scaling_lum[i];
      const Real sc = scaling_cont[i];
#pragma omp simd
      for (int j = 0; j < nlam; j++) {
        out_lum_row[j] = static_cast<OutT>(in_lum_row[j] * sl);
        out_cont_row[j] = static_cast<OutT>(in_cont_row[j] * sc);
      }
    } else {
#pragma omp simd
      for (int j = 0; j < nlam; j++) {
        out_lum_row[j] = static_cast<OutT>(in_lum_row[j]);
        out_cont_row[j] = static_cast<OutT>(in_cont_row[j]);
      }
    }
  }
}

/**
 * @brief This calculates fused lum+cont scaling with a wavelength mask using
 *        OpenMP.
 *
 * This is the parallel version of the function for rows with a wavelength
 * mask.
 *
 * @param lum: Input luminosity array (nspec x nlam).
 * @param cont: Input continuum array (nspec x nlam).
 * @param scaling_lum: Per-spectrum luminosity factor (nspec).
 * @param scaling_cont: Per-spectrum continuum factor (nspec).
 * @param lam_mask: 1D boolean wavelength mask (nlam).
 * @param out_lum: Output luminosity buffer (nspec x nlam).
 * @param out_cont: Output continuum buffer (nspec x nlam).
 * @param nspec: The number of spectra rows.
 * @param nlam: The number of wavelength bins.
 * @param nthreads: The number of OpenMP threads.
 */
template <typename Real, typename OutT>
static void scale_line_2d_lam_mask_omp(
    const Real *__restrict__ lum, const Real *__restrict__ cont,
    const Real *scaling_lum, const Real *scaling_cont, const npy_bool *lam_mask,
    OutT *out_lum, OutT *out_cont, int nspec, int nlam, int nthreads) {

  /* Split the spectra rows evenly across threads. */
#pragma omp parallel for num_threads(nthreads) schedule(static)
  for (int i = 0; i < nspec; i++) {
    /* Cache both scale factors for this row. */
    const Real sl = scaling_lum[i];
    const Real sc = scaling_cont[i];
    const Real *in_lum_row = lum + i * nlam;
    const Real *in_cont_row = cont + i * nlam;
    OutT *out_lum_row = out_lum + i * nlam;
    OutT *out_cont_row = out_cont + i * nlam;

    /* Only scale wavelengths that pass the mask; keep others unchanged. */
    for (int j = 0; j < nlam; j++) {
      const bool apply = lam_mask[j];
      out_lum_row[j] = apply ? static_cast<OutT>(in_lum_row[j] * sl)
                             : static_cast<OutT>(in_lum_row[j]);
      out_cont_row[j] = apply ? static_cast<OutT>(in_cont_row[j] * sc)
                              : static_cast<OutT>(in_cont_row[j]);
    }
  }
}

/**
 * @brief This calculates fused lum+cont scaling with both masks using OpenMP.
 *
 * This is the parallel version of the function for rows with both masks.
 *
 * @param lum: Input luminosity array (nspec x nlam).
 * @param cont: Input continuum array (nspec x nlam).
 * @param scaling_lum: Per-spectrum luminosity factor (nspec).
 * @param scaling_cont: Per-spectrum continuum factor (nspec).
 * @param mask: 1D boolean row mask (nspec).
 * @param lam_mask: 1D boolean wavelength mask (nlam).
 * @param out_lum: Output luminosity buffer (nspec x nlam).
 * @param out_cont: Output continuum buffer (nspec x nlam).
 * @param nspec: The number of spectra rows.
 * @param nlam: The number of wavelength bins.
 * @param nthreads: The number of OpenMP threads.
 */
template <typename Real, typename OutT>
static void scale_line_2d_both_masks_omp(
    const Real *__restrict__ lum, const Real *__restrict__ cont,
    const Real *scaling_lum, const Real *scaling_cont, const npy_bool *mask,
    const npy_bool *lam_mask, OutT *out_lum, OutT *out_cont, int nspec,
    int nlam, int nthreads) {

  /* Split the spectra rows evenly across threads. */
#pragma omp parallel for num_threads(nthreads) schedule(static)
  for (int i = 0; i < nspec; i++) {
    const Real *in_lum_row = lum + i * nlam;
    const Real *in_cont_row = cont + i * nlam;
    OutT *out_lum_row = out_lum + i * nlam;
    OutT *out_cont_row = out_cont + i * nlam;

    /* For masked rows apply the wavelength-dependent scale; copy otherwise. */
    if (mask[i]) {
      const Real sl = scaling_lum[i];
      const Real sc = scaling_cont[i];
      for (int j = 0; j < nlam; j++) {
        const bool apply = lam_mask[j];
        out_lum_row[j] = apply ? static_cast<OutT>(in_lum_row[j] * sl)
                               : static_cast<OutT>(in_lum_row[j]);
        out_cont_row[j] = apply ? static_cast<OutT>(in_cont_row[j] * sc)
                                : static_cast<OutT>(in_cont_row[j]);
      }
    } else {
      for (int j = 0; j < nlam; j++) {
        out_lum_row[j] = static_cast<OutT>(in_lum_row[j]);
        out_cont_row[j] = static_cast<OutT>(in_cont_row[j]);
      }
    }
  }
}

#endif /* WITH_OPENMP */

template <typename Real, typename OutT>
static void dispatch_scale_line_2d(
    const Real *lum, const Real *cont, const Real *scaling_lum,
    const Real *scaling_cont, const npy_bool *mask, const npy_bool *lam_mask,
    OutT *out_lum, OutT *out_cont, int nspec, int nlam, int nthreads) {

  const bool has_mask = (mask != NULL);
  const bool has_lam_mask = (lam_mask != NULL);

#ifdef WITH_OPENMP
  if (nthreads > 1) {
    if (!has_mask && !has_lam_mask) {
      scale_line_2d_no_mask_omp<Real, OutT>(lum, cont, scaling_lum,
                                            scaling_cont, out_lum, out_cont,
                                            nspec, nlam, nthreads);
    } else if (has_mask && !has_lam_mask) {
      scale_line_2d_row_mask_omp<Real, OutT>(lum, cont, scaling_lum,
                                             scaling_cont, mask, out_lum,
                                             out_cont, nspec, nlam, nthreads);
    } else if (!has_mask && has_lam_mask) {
      scale_line_2d_lam_mask_omp<Real, OutT>(lum, cont, scaling_lum,
                                             scaling_cont, lam_mask, out_lum,
                                             out_cont, nspec, nlam, nthreads);
    } else {
      scale_line_2d_both_masks_omp<Real, OutT>(
          lum, cont, scaling_lum, scaling_cont, mask, lam_mask, out_lum,
          out_cont, nspec, nlam, nthreads);
    }
    return;
  }
#endif

  if (!has_mask && !has_lam_mask) {
    scale_line_2d_no_mask_serial<Real, OutT>(
        lum, cont, scaling_lum, scaling_cont, out_lum, out_cont, nspec, nlam);
  } else if (has_mask && !has_lam_mask) {
    scale_line_2d_row_mask_serial<Real, OutT>(lum, cont, scaling_lum,
                                              scaling_cont, mask, out_lum,
                                              out_cont, nspec, nlam);
  } else if (!has_mask && has_lam_mask) {
    scale_line_2d_lam_mask_serial<Real, OutT>(lum, cont, scaling_lum,
                                              scaling_cont, lam_mask, out_lum,
                                              out_cont, nspec, nlam);
  } else {
    scale_line_2d_both_masks_serial<Real, OutT>(lum, cont, scaling_lum,
                                                scaling_cont, mask, lam_mask,
                                                out_lum, out_cont, nspec, nlam);
  }
}

/**
 * @brief Fused lum+cont scaling with per-spectrum factors.
 *
 * Processes both arrays in one parallel loop so the scaling vectors and masks
 * are read once instead of twice. Accepts separate factors for luminosity
 * and continuum (relevant when unyt unit conversion differs).
 *
 * @param lum_obj: 2D float64 ndarray (nspec x nlam).
 * @param cont_obj: 2D float64 ndarray (nspec x nlam), same shape as lum.
 * @param scaling_lum_obj: 1D float64 ndarray (nspec).
 * @param scaling_cont_obj: 1D float64 ndarray (nspec).
 * @param mask_obj: Optional 1D boolean (nspec).
 * @param lam_mask_obj: Optional 1D boolean (nlam).
 * @param nthreads: Number of OpenMP threads.
 * @param out_lum_obj: Optional output buffer for lum.
 * @param out_cont_obj: Optional output buffer for cont.
 *
 * @return Tuple (scaled_lum, scaled_cont).
 */
PyObject *scale_line_2d(PyObject *self, PyObject *args) {
  (void)self;

  /* Parse the argument tuple. */
  PyObject *lum_obj, *cont_obj, *scaling_lum_obj, *scaling_cont_obj;
  PyObject *mask_obj = Py_None;
  PyObject *lam_mask_obj = Py_None;
  PyObject *out_lum_obj = Py_None;
  PyObject *out_cont_obj = Py_None;
  int nthreads;

  PyObject *out_dtype_obj = NULL;
  if (!PyArg_ParseTuple(args, "OOOOOOi|OOO", &lum_obj, &cont_obj,
                        &scaling_lum_obj, &scaling_cont_obj, &mask_obj,
                        &lam_mask_obj, &nthreads, &out_lum_obj, &out_cont_obj,
                        &out_dtype_obj)) {
    return NULL;
  }

  /* Convert inputs to array views (accept float32 or float64).
   * TODO: Remove coercion by default. */
  PyArrayObject *np_lum = (PyArrayObject *)PyArray_FromAny(
      lum_obj, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
  PyArrayObject *np_cont = (PyArrayObject *)PyArray_FromAny(
      cont_obj, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
  PyArrayObject *np_scaling_lum = (PyArrayObject *)PyArray_FromAny(
      scaling_lum_obj, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
  PyArrayObject *np_scaling_cont = (PyArrayObject *)PyArray_FromAny(
      scaling_cont_obj, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);

  if (!np_lum || !np_cont || !np_scaling_lum || !np_scaling_cont) {
    Py_XDECREF(np_lum);
    Py_XDECREF(np_cont);
    Py_XDECREF(np_scaling_lum);
    Py_XDECREF(np_scaling_cont);
    return NULL;
  }

  /* Optional masks start out absent and only become arrays when the caller
   * explicitly passed them in.
   * TODO: Remove coercion by default. */
  PyArrayObject *np_mask = nullptr;
  PyArrayObject *np_lam_mask = nullptr;

  if (mask_obj != Py_None) {
    np_mask = (PyArrayObject *)PyArray_FromAny(mask_obj, NULL, 0, 0,
                                               NPY_ARRAY_ENSUREARRAY, NULL);
    if (!np_mask) {
      Py_DECREF(np_lum);
      Py_DECREF(np_cont);
      Py_DECREF(np_scaling_lum);
      Py_DECREF(np_scaling_cont);
      return NULL;
    }
  }

  if (lam_mask_obj != Py_None) {
    np_lam_mask = (PyArrayObject *)PyArray_FromAny(lam_mask_obj, NULL, 0, 0,
                                                   NPY_ARRAY_ENSUREARRAY, NULL);
    if (!np_lam_mask) {
      Py_DECREF(np_lum);
      Py_DECREF(np_cont);
      Py_DECREF(np_scaling_lum);
      Py_DECREF(np_scaling_cont);
      Py_XDECREF(np_mask);
      return NULL;
    }
  }

  /* Validate shapes. */
  if (PyArray_NDIM(np_lum) != 2 || PyArray_NDIM(np_cont) != 2 ||
      PyArray_NDIM(np_scaling_lum) != 1 || PyArray_NDIM(np_scaling_cont) != 1) {
    PyErr_SetString(PyExc_ValueError,
                    "lum and cont must be 2D, scalings must be 1D.");
    Py_DECREF(np_lum);
    Py_DECREF(np_cont);
    Py_DECREF(np_scaling_lum);
    Py_DECREF(np_scaling_cont);
    Py_XDECREF(np_mask);
    Py_XDECREF(np_lam_mask);
    return NULL;
  }

  /* Cache the common shape information once so the later checks stay simple. */
  const npy_intp *lum_dims = PyArray_DIMS(np_lum);
  const npy_intp *cont_dims = PyArray_DIMS(np_cont);
  const int nspec = static_cast<int>(lum_dims[0]);
  const int nlam = static_cast<int>(lum_dims[1]);

  if (cont_dims[0] != lum_dims[0] || cont_dims[1] != lum_dims[1]) {
    PyErr_SetString(PyExc_ValueError, "lum and cont must have the same shape.");
    Py_DECREF(np_lum);
    Py_DECREF(np_cont);
    Py_DECREF(np_scaling_lum);
    Py_DECREF(np_scaling_cont);
    Py_XDECREF(np_mask);
    Py_XDECREF(np_lam_mask);
    return NULL;
  }

  if (PyArray_DIMS(np_scaling_lum)[0] != lum_dims[0] ||
      PyArray_DIMS(np_scaling_cont)[0] != lum_dims[0]) {
    PyErr_SetString(PyExc_ValueError,
                    "scaling lengths must match the first array dimension.");
    Py_DECREF(np_lum);
    Py_DECREF(np_cont);
    Py_DECREF(np_scaling_lum);
    Py_DECREF(np_scaling_cont);
    Py_XDECREF(np_mask);
    Py_XDECREF(np_lam_mask);
    return NULL;
  }

  if (np_mask != NULL &&
      (PyArray_NDIM(np_mask) != 1 || PyArray_DIMS(np_mask)[0] != lum_dims[0])) {
    PyErr_SetString(PyExc_ValueError,
                    "mask must be a 1D bool array matching rows.");
    Py_DECREF(np_lum);
    Py_DECREF(np_cont);
    Py_DECREF(np_scaling_lum);
    Py_DECREF(np_scaling_cont);
    Py_XDECREF(np_mask);
    Py_XDECREF(np_lam_mask);
    return NULL;
  }

  if (np_lam_mask != NULL && (PyArray_NDIM(np_lam_mask) != 1 ||
                              PyArray_DIMS(np_lam_mask)[0] != lum_dims[1])) {
    PyErr_SetString(PyExc_ValueError,
                    "lam_mask must be a 1D bool array matching columns.");
    Py_DECREF(np_lum);
    Py_DECREF(np_cont);
    Py_DECREF(np_scaling_lum);
    Py_DECREF(np_scaling_cont);
    Py_XDECREF(np_mask);
    Py_XDECREF(np_lam_mask);
    return NULL;
  }

  /* Resolve input float family and requested out dtype. */
  PyArrayObject *float_arrays[4] = {np_lum, np_cont, np_scaling_lum,
                                    np_scaling_cont};
  const char *float_names[4] = {"lum", "cont", "scaling_lum", "scaling_cont"};
  int input_typenum = -1;
  if (!is_matching_float_dtypes(float_arrays, float_names, 4, &input_typenum)) {
    Py_DECREF(np_lum);
    Py_DECREF(np_cont);
    Py_DECREF(np_scaling_lum);
    Py_DECREF(np_scaling_cont);
    Py_XDECREF(np_mask);
    Py_XDECREF(np_lam_mask);
    return NULL;
  }

  int out_typenum = -1;
  if (out_dtype_obj != NULL && out_dtype_obj != Py_None) {
    out_typenum = resolve_output_typenum(out_dtype_obj, "out_dtype");
    if (out_typenum < 0) {
      Py_DECREF(np_lum);
      Py_DECREF(np_cont);
      Py_DECREF(np_scaling_lum);
      Py_DECREF(np_scaling_cont);
      Py_XDECREF(np_mask);
      Py_XDECREF(np_lam_mask);
      return NULL;
    }
  }
  if (out_typenum < 0)
    out_typenum = input_typenum;

  /* Allocate or reuse output buffers with the requested dtype.
   * TODO: Remove coercion by default. */
  PyArrayObject *np_out_lum = nullptr;
  PyArrayObject *np_out_cont = nullptr;
  if (out_lum_obj != Py_None && out_cont_obj != Py_None) {
    np_out_lum = (PyArrayObject *)PyArray_FromAny(
        out_lum_obj, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_WRITEABLE,
        NULL);
    np_out_cont = (PyArrayObject *)PyArray_FromAny(
        out_cont_obj, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_WRITEABLE,
        NULL);
    if (!np_out_lum || !np_out_cont) {
      Py_XDECREF(np_out_lum);
      Py_XDECREF(np_out_cont);
      Py_DECREF(np_lum);
      Py_DECREF(np_cont);
      Py_DECREF(np_scaling_lum);
      Py_DECREF(np_scaling_cont);
      Py_XDECREF(np_mask);
      Py_XDECREF(np_lam_mask);
      return NULL;
    }
    if (PyArray_NDIM(np_out_lum) != 2 ||
        PyArray_DIMS(np_out_lum)[0] != lum_dims[0] ||
        PyArray_DIMS(np_out_lum)[1] != lum_dims[1] ||
        PyArray_NDIM(np_out_cont) != 2 ||
        PyArray_DIMS(np_out_cont)[0] != lum_dims[0] ||
        PyArray_DIMS(np_out_cont)[1] != lum_dims[1]) {
      PyErr_SetString(PyExc_ValueError,
                      "out arrays must match lum/cont shape.");
      Py_DECREF(np_out_lum);
      Py_DECREF(np_out_cont);
      Py_DECREF(np_lum);
      Py_DECREF(np_cont);
      Py_DECREF(np_scaling_lum);
      Py_DECREF(np_scaling_cont);
      Py_XDECREF(np_mask);
      Py_XDECREF(np_lam_mask);
      return NULL;
    }
    if (PyArray_TYPE(np_out_lum) != out_typenum ||
        PyArray_TYPE(np_out_cont) != out_typenum ||
        !PyArray_ISCARRAY(np_out_lum) || !PyArray_ISCARRAY(np_out_cont)) {
      PyErr_SetString(
          PyExc_ValueError,
          "out arrays must be C-contiguous with the requested out_dtype.");
      Py_DECREF(np_out_lum);
      Py_DECREF(np_out_cont);
      Py_DECREF(np_lum);
      Py_DECREF(np_cont);
      Py_DECREF(np_scaling_lum);
      Py_DECREF(np_scaling_cont);
      Py_XDECREF(np_mask);
      Py_XDECREF(np_lam_mask);
      return NULL;
    }
  } else {
    npy_intp dims[2];
    dims[0] = lum_dims[0];
    dims[1] = lum_dims[1];
    np_out_lum = (PyArrayObject *)PyArray_SimpleNew(2, dims, out_typenum);
    np_out_cont = (PyArrayObject *)PyArray_SimpleNew(2, dims, out_typenum);
    if (!np_out_lum || !np_out_cont) {
      Py_XDECREF(np_out_lum);
      Py_XDECREF(np_out_cont);
      Py_DECREF(np_lum);
      Py_DECREF(np_cont);
      Py_DECREF(np_scaling_lum);
      Py_DECREF(np_scaling_cont);
      Py_XDECREF(np_mask);
      Py_XDECREF(np_lam_mask);
      return NULL;
    }
  }

  /* Extract raw pointers and dispatch to the typed fused kernel. */
  tic("scale_line_2d");
  if (input_typenum == NPY_FLOAT32) {
    const float *lum = data_ptr<const float>(np_lum);
    const float *cont = data_ptr<const float>(np_cont);
    const float *scaling_lum = data_ptr<const float>(np_scaling_lum);
    const float *scaling_cont = data_ptr<const float>(np_scaling_cont);
    const npy_bool *mask =
        np_mask ? data_ptr<const npy_bool>(np_mask) : nullptr;
    const npy_bool *lam_mask =
        np_lam_mask ? data_ptr<const npy_bool>(np_lam_mask) : nullptr;
    if (out_typenum == NPY_FLOAT32) {
      float *out_lum = data_ptr<float>(np_out_lum);
      float *out_cont = data_ptr<float>(np_out_cont);
      dispatch_scale_line_2d<float, float>(lum, cont, scaling_lum, scaling_cont,
                                           mask, lam_mask, out_lum, out_cont,
                                           nspec, nlam, nthreads);
    } else {
      double *out_lum = data_ptr<double>(np_out_lum);
      double *out_cont = data_ptr<double>(np_out_cont);
      dispatch_scale_line_2d<float, double>(
          lum, cont, scaling_lum, scaling_cont, mask, lam_mask, out_lum,
          out_cont, nspec, nlam, nthreads);
    }
  } else {
    const double *lum = data_ptr<const double>(np_lum);
    const double *cont = data_ptr<const double>(np_cont);
    const double *scaling_lum = data_ptr<const double>(np_scaling_lum);
    const double *scaling_cont = data_ptr<const double>(np_scaling_cont);
    const npy_bool *mask =
        np_mask ? data_ptr<const npy_bool>(np_mask) : nullptr;
    const npy_bool *lam_mask =
        np_lam_mask ? data_ptr<const npy_bool>(np_lam_mask) : nullptr;
    if (out_typenum == NPY_FLOAT32) {
      float *out_lum = data_ptr<float>(np_out_lum);
      float *out_cont = data_ptr<float>(np_out_cont);
      dispatch_scale_line_2d<double, float>(
          lum, cont, scaling_lum, scaling_cont, mask, lam_mask, out_lum,
          out_cont, nspec, nlam, nthreads);
    } else {
      double *out_lum = data_ptr<double>(np_out_lum);
      double *out_cont = data_ptr<double>(np_out_cont);
      dispatch_scale_line_2d<double, double>(
          lum, cont, scaling_lum, scaling_cont, mask, lam_mask, out_lum,
          out_cont, nspec, nlam, nthreads);
    }
  }

  toc("scale_line_2d");

  Py_DECREF(np_lum);
  Py_DECREF(np_cont);
  Py_DECREF(np_scaling_lum);
  Py_DECREF(np_scaling_cont);
  Py_XDECREF(np_mask);
  Py_XDECREF(np_lam_mask);

  return Py_BuildValue("(NN)", np_out_lum, np_out_cont);
}

/* Module initialisation — required for Python to import this extension. */
static PyMethodDef SpectraOperationMethods[] = {
    {"scale_spectra_2d", (PyCFunction)scale_spectra_2d, METH_VARARGS,
     "Scale a 2D spectra array by a 1D per-spectrum factor with mask-"
     "specialised kernels. Accepts an optional out array for in-place use."},
    {"apply_separable_attenuation_2d",
     (PyCFunction)apply_separable_attenuation_2d, METH_VARARGS,
     "Apply exp(-tau_v * tau_x_v) attenuation to a 2D spectra array. "
     "Accepts an optional out array for in-place use."},
    {"multiply_array_by_vector_1d", (PyCFunction)multiply_array_by_vector_1d,
     METH_VARARGS,
     "Multiply a 1D or 2D array by a 1D vector over the last axis. "
     "Accepts an optional out array for in-place use."},
    {"scale_line_2d", (PyCFunction)scale_line_2d, METH_VARARGS,
     "Fused lum+cont scaling with per-spectrum factors. Processes both "
     "arrays in one parallel loop to halve dispatch overhead. "
     "Accepts optional out arrays for in-place use."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "spectra_operations",
    "Generic spectra operation kernels with split masked/unmasked backends",
    -1,
    SpectraOperationMethods,
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
