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
 * @brief Populate observer-frame wavelength, frequency, and flux arrays.
 *
 * Args:
 *   lnu (np.ndarray):
 *     Float64 spectra array with wavelength on the last axis.
 *   lam (np.ndarray):
 *     One-dimensional emitted wavelength grid.
 *   nu (np.ndarray):
 *     One-dimensional emitted frequency grid.
 *   one_plus_z (float):
 *     Observer-frame wavelength scaling factor.
 *   conversion (float):
 *     Scalar conversion from luminosity density to flux density.
 *   nthreads (int):
 *     Number of threads to use for the bulk flux conversion loop.
 *   fnu_out (np.ndarray):
 *     Preallocated float64 output array matching ``lnu``.
 *   obslam_out (np.ndarray):
 *     Preallocated float64 wavelength output array matching ``lam``.
 *   obsnu_out (np.ndarray):
 *     Preallocated float64 frequency output array matching ``nu``.
 *
 * Returns:
 *   None:
 *     The output arrays are populated in place.
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

  if (!PyArg_ParseTuple(args, "OOOddiOOO", &lnu_obj, &lam_obj, &nu_obj,
                        &one_plus_z, &conversion, &nthreads, &fnu_out_obj,
                        &obslam_out_obj, &obsnu_out_obj)) {
    return NULL;
  }

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

  if (PyArray_TYPE(np_lnu) != NPY_DOUBLE || PyArray_TYPE(np_lam) != NPY_DOUBLE ||
      PyArray_TYPE(np_nu) != NPY_DOUBLE || PyArray_TYPE(np_fnu_out) != NPY_DOUBLE ||
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
  double *obslam_out =
      np_obslam_out == NULL ? NULL : static_cast<double *>(PyArray_DATA(np_obslam_out));
  double *obsnu_out =
      np_obsnu_out == NULL ? NULL : static_cast<double *>(PyArray_DATA(np_obsnu_out));
  const npy_intp nelem = PyArray_SIZE(np_lnu);

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

/**
 * @brief Scale a 2D spectra array by a per-spectrum factor.
 *
 * The common hot-path shape is ``(nspec, nlam)`` scaled by a 1D array of
 * length ``nspec`` with optional per-spectrum and wavelength masks. This
 * helper keeps the work in one threaded C++ loop and returns a new array.
 */
PyObject *scale_spectra_2d(PyObject *self, PyObject *args) {
  (void)self;

  PyObject *spectra_obj;
  PyObject *scaling_obj;
  PyObject *mask_obj = Py_None;
  PyObject *lam_mask_obj = Py_None;
  int nthreads;

  if (!PyArg_ParseTuple(args, "OOOOi", &spectra_obj, &scaling_obj, &mask_obj,
                        &lam_mask_obj, &nthreads)) {
    return NULL;
  }

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

  if (np_mask != NULL) {
    if (PyArray_NDIM(np_mask) != 1 || PyArray_DIMS(np_mask)[0] != spectra_dims[0]) {
      PyErr_SetString(PyExc_ValueError,
                      "mask must be a 1D boolean array matching spectra rows.");
      Py_DECREF(np_spectra);
      Py_DECREF(np_scaling);
      Py_XDECREF(np_mask);
      Py_XDECREF(np_lam_mask);
      return NULL;
    }
  }

  if (np_lam_mask != NULL) {
    if (PyArray_NDIM(np_lam_mask) != 1 ||
        PyArray_DIMS(np_lam_mask)[0] != spectra_dims[1]) {
      PyErr_SetString(
          PyExc_ValueError,
          "lam_mask must be a 1D boolean array matching spectra columns.");
      Py_DECREF(np_spectra);
      Py_DECREF(np_scaling);
      Py_XDECREF(np_mask);
      Py_XDECREF(np_lam_mask);
      return NULL;
    }
  }

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

/**
 * @brief Multiply a 2D array by a 1D row vector with optional row mask.
 */
PyObject *multiply_rows_by_vector_2d(PyObject *self, PyObject *args) {
  (void)self;

  PyObject *array_obj;
  PyObject *vector_obj;
  PyObject *mask_obj = Py_None;
  int nthreads;

  if (!PyArg_ParseTuple(args, "OOOi", &array_obj, &vector_obj, &mask_obj,
                        &nthreads)) {
    return NULL;
  }

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
  (void)self;

  PyObject *array_obj;
  PyObject *vector_obj;
  int nthreads;

  if (!PyArg_ParseTuple(args, "OOi", &array_obj, &vector_obj, &nthreads)) {
    return NULL;
  }

  PyArrayObject *np_array = (PyArrayObject *)PyArray_FROM_OTF(
      array_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *np_vector = (PyArrayObject *)PyArray_FROM_OTF(
      vector_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

  if (np_array == NULL || np_vector == NULL) {
    Py_XDECREF(np_array);
    Py_XDECREF(np_vector);
    return NULL;
  }

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
static PyMethodDef ReductionMethods[] = {
    {"reduce_particle_spectra", (PyCFunction)reduce_particle_spectra,
     METH_VARARGS,
     "Method for reducing per-particle spectra to an integrated spectrum."},
    {"compute_fnu", (PyCFunction)compute_fnu, METH_VARARGS,
     "Populate observer-frame wavelength, frequency, and flux arrays."},
    {"scale_spectra_2d", (PyCFunction)scale_spectra_2d, METH_VARARGS,
     "Scale a 2D spectra array by a 1D per-spectrum factor."},
    {"combine_spectra_list_2d", (PyCFunction)combine_spectra_list_2d,
     METH_VARARGS,
     "Combine same-shaped 2D spectra arrays while skipping NaNs."},
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
