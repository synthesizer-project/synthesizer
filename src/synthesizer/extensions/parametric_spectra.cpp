/******************************************************************************
 * C extension to calculate SEDs for parametric stellar populations.
 *
 * A parametric population is a set of N dimensional bins (e.g. age and
 * metallicity bins) each holding a mass. The masses are spread onto the grid
 * using the bin in cell (BIC) approach (see weights.h) and the resulting
 * grid weights are contracted with the grid spectra.
 *
 * This is the parametric sibling of integrated_spectra (summed over all
 * populations) and particle_spectra (one spectrum per population).
 *****************************************************************************/
/* C includes */
#include <algorithm>
#include <array>
#include <memory>
#include <new>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <type_traits>
#include <vector>

/* Python includes */
#define PY_ARRAY_UNIQUE_SYMBOL SYNTHESIZER_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy_init.h"

#include <Python.h>

/* Local includes */
#include "cpp_to_python.h"
#include "grid_props.h"
#include "macros.h"
#include "pop_props.h"
#include "property_funcs.h"
#include "python_to_cpp.h"
#include "timers.h"
#ifdef ATOMIC_TIMING
#include "timers_init.h"
#endif
#include "weights.h"

/* Optional openmp include. */
#ifdef WITH_OPENMP
#include <omp.h>
#endif

/**
 * @brief Validate the Python population inputs and fill a Populations struct.
 *
 * @param edges_tuple: A tuple of 1D bin edge arrays, one per grid axis (in
 *                     linear units).
 * @param mask_obj: The mask (bool, same shape as the masses) or None.
 * @param np_masses: The masses with shape (npop, nbins_0, ..., nbins_N).
 * @param log_flags: A sequence of booleans, one per grid axis, flagging
 *                   axes whose grid coordinates are log10 values.
 * @param grid_props: The grid properties (used for the number of axes).
 * @param pops: The struct to fill.
 *
 * @return True on success, false (with a Python error set) otherwise.
 */
static bool build_populations(PyObject *edges_tuple, PyArrayObject *np_masses,
                              PyObject *log_flags, PyObject *mask_obj,
                              GridProps *grid_props, Populations &pops) {

  const int ndim = grid_props->ndim;
  pops.ndim = ndim;

  /* We need one set of edges and one log flag per grid axis. */
  if (!PyTuple_Check(edges_tuple) || PyTuple_Size(edges_tuple) != ndim) {
    PyErr_Format(PyExc_ValueError,
                 "edges must be a tuple with one array per grid axis (%d).",
                 ndim);
    return false;
  }
  if (!PySequence_Check(log_flags) || PySequence_Size(log_flags) != ndim) {
    PyErr_Format(PyExc_ValueError,
                 "log_flags must be a sequence with one entry per grid axis "
                 "(%d).",
                 ndim);
    return false;
  }

  /* The masses must have a leading population axis then one per grid axis.
   */
  if (!PyArray_Check(reinterpret_cast<PyObject *>(np_masses)) ||
      PyArray_NDIM(np_masses) != ndim + 1) {
    PyErr_Format(PyExc_ValueError,
                 "masses must be an array of shape (npop, nbins_0, ..., "
                 "nbins_%d).",
                 ndim - 1);
    return false;
  }
  pops.npop = static_cast<int>(PyArray_DIM(np_masses, 0));

  /* Collect the float arrays so we can check they share one dtype. */
  PyArrayObject *float_arrays[MAX_GRID_NDIM + 1] = {NULL};
  const char *float_names[MAX_GRID_NDIM + 1] = {NULL};
  float_arrays[0] = np_masses;
  float_names[0] = "masses";

  pops.ncells = 1;
  for (int idim = 0; idim < ndim; idim++) {

    /* Get the edges along this axis. */
    PyObject *edges_obj = PyTuple_GetItem(edges_tuple, idim);
    if (edges_obj == NULL || !PyArray_Check(edges_obj) ||
        PyArray_NDIM(reinterpret_cast<PyArrayObject *>(edges_obj)) != 1) {
      PyErr_Format(PyExc_ValueError,
                   "The edges for axis %d must be a 1D array.", idim);
      return false;
    }
    PyArrayObject *np_edges = reinterpret_cast<PyArrayObject *>(edges_obj);
    float_arrays[idim + 1] = np_edges;
    float_names[idim + 1] = "bin edges";

    /* There is one fewer bin than there are edges. */
    const npy_intp nedges = PyArray_DIM(np_edges, 0);
    if (nedges < 2) {
      PyErr_Format(PyExc_ValueError,
                   "The edges for axis %d must define at least one bin.",
                   idim);
      return false;
    }
    pops.nbins[idim] = static_cast<int>(nedges - 1);
    if (PyArray_DIM(np_masses, idim + 1) != pops.nbins[idim]) {
      PyErr_Format(PyExc_ValueError,
                   "masses has %ld bins along axis %d but the edges define "
                   "%d.",
                   static_cast<long>(PyArray_DIM(np_masses, idim + 1)), idim,
                   pops.nbins[idim]);
      return false;
    }
    pops.ncells *= pops.nbins[idim];

    /* Get the log flag for this axis. */
    PyObject *flag = PySequence_GetItem(log_flags, idim);
    if (flag == NULL) {
      return false;
    }
    const int is_log = PyObject_IsTrue(flag);
    Py_DECREF(flag);
    if (is_log < 0) {
      return false;
    }
    pops.log_axis[idim] = is_log == 1;
  }

  /* Check the dtypes match and the arrays are contiguous. */
  if (!is_matching_float_dtypes(float_arrays, float_names, ndim + 1,
                                &pops.float_typenum)) {
    return false;
  }
  if (PyArray_TYPE(np_masses) != pops.float_typenum) {
    PyErr_SetString(PyExc_TypeError,
                    "masses and bin edges must share the same dtype.");
    return false;
  }

  /* Store the pointers and check the edges are non-decreasing. */
  pops.masses = PyArray_DATA(np_masses);
  for (int idim = 0; idim < ndim; idim++) {
    PyArrayObject *np_edges =
        reinterpret_cast<PyArrayObject *>(PyTuple_GetItem(edges_tuple, idim));
    if (PyArray_TYPE(np_edges) != pops.float_typenum) {
      PyErr_SetString(PyExc_TypeError,
                      "masses and bin edges must share the same dtype.");
      return false;
    }
    pops.edges[idim] = PyArray_DATA(np_edges);
    bool sorted = dispatch_float(pops.float_typenum, [&](auto r) {
      using Real = decltype(r);
      const Real *edges = pops.get_edges<Real>(idim);
      for (int i = 0; i < pops.nbins[idim]; i++) {
        if (!(edges[i + 1] >= edges[i])) {
          return false;
        }
      }
      return true;
    });
    if (!sorted) {
      PyErr_Format(PyExc_ValueError,
                   "The edges for axis %d must be non-decreasing.", idim);
      return false;
    }
  }

  /* Attach the mask if we have one. */
  pops.mask = nullptr;
  if (mask_obj != NULL && mask_obj != Py_None) {
    PyArrayObject *np_mask = array_or_none(mask_obj, "mask");
    if (np_mask == NULL) {
      return false;
    }
    if (PyArray_TYPE(np_mask) != NPY_BOOL ||
        !PyArray_SAMESHAPE(np_mask, np_masses)) {
      PyErr_SetString(PyExc_ValueError,
                      "mask must be a boolean array with the same shape as "
                      "the masses.");
      return false;
    }
    if (!is_c_contiguous(np_mask, "mask")) {
      return false;
    }
    pops.mask = static_cast<const npy_bool *>(PyArray_DATA(np_mask));
  }

  return true;
}

/**
 * @brief Get the number of wavelength elements from the grid spectra.
 *
 * @param spectra_obj: The grid spectra (grid axes followed by wavelength).
 *
 * @return The number of wavelength elements, or -1 (with a Python error set).
 */
static int get_nlam(PyObject *spectra_obj) {
  if (!PyArray_Check(spectra_obj) ||
      PyArray_NDIM(reinterpret_cast<PyArrayObject *>(spectra_obj)) < 2) {
    PyErr_SetString(PyExc_ValueError,
                    "grid_spectra must be an array with the grid axes "
                    "followed by a wavelength axis.");
    return -1;
  }
  PyArrayObject *np_spectra = reinterpret_cast<PyArrayObject *>(spectra_obj);
  return static_cast<int>(
      PyArray_DIM(np_spectra, PyArray_NDIM(np_spectra) - 1));
}

/* Fortran BLAS gemm signatures, as exported by scipy.linalg.cython_blas. */
typedef void (*dgemm_t)(char *, char *, int *, int *, int *, double *,
                        double *, int *, double *, int *, double *, double *,
                        int *);
typedef void (*sgemm_t)(char *, char *, int *, int *, int *, float *, float *,
                        int *, float *, int *, float *, float *, int *);

/* The BLAS routines, fetched from SciPy when the module is imported. */
static dgemm_t blas_dgemm = NULL;
static sgemm_t blas_sgemm = NULL;

/**
 * @brief Fetch a BLAS routine from scipy.linalg.cython_blas.
 *
 * SciPy exports C pointers to the BLAS it links against, so we can use an
 * optimised BLAS without linking one ourselves at build time.
 *
 * @param capi: The cython_blas __pyx_capi__ dictionary.
 * @param name: The routine name (e.g. "dgemm").
 *
 * @return The routine's address, or NULL (with a Python error set).
 */
static void *get_blas_routine(PyObject *capi, const char *name) {
  PyObject *capsule = PyDict_GetItemString(capi, name);
  if (capsule == NULL) {
    PyErr_Format(PyExc_ImportError,
                 "scipy.linalg.cython_blas does not export %s.", name);
    return NULL;
  }
  return PyCapsule_GetPointer(capsule, PyCapsule_GetName(capsule));
}

/**
 * @brief Load the BLAS gemm routines from SciPy.
 *
 * @return 0 on success, -1 (with a Python error set) otherwise.
 */
static int load_blas(void) {
  PyObject *module = PyImport_ImportModule("scipy.linalg.cython_blas");
  if (module == NULL) {
    return -1;
  }
  PyObject *capi = PyObject_GetAttrString(module, "__pyx_capi__");
  Py_DECREF(module);
  if (capi == NULL) {
    return -1;
  }
  blas_dgemm = reinterpret_cast<dgemm_t>(get_blas_routine(capi, "dgemm"));
  blas_sgemm = reinterpret_cast<sgemm_t>(get_blas_routine(capi, "sgemm"));
  Py_DECREF(capi);
  return (blas_dgemm == NULL || blas_sgemm == NULL) ? -1 : 0;
}

/**
 * @brief Call the BLAS gemm matching the precision of Real.
 *
 * Computes C = A B for column major A (m, k), B (k, n) and C (m, n).
 */
template <typename Real>
static void gemm(int m, int n, int k, const Real *a, int lda, const Real *b,
                 int ldb, Real *c, int ldc) {
  char trans = 'N';
  Real alpha = 1;
  Real beta = 0;
  if constexpr (std::is_same_v<Real, double>) {
    blas_dgemm(&trans, &trans, &m, &n, &k, &alpha, const_cast<double *>(a),
               &lda, const_cast<double *>(b), &ldb, &beta, c, &ldc);
  } else {
    blas_sgemm(&trans, &trans, &m, &n, &k, &alpha, const_cast<float *>(a),
               &lda, const_cast<float *>(b), &ldb, &beta, c, &ldc);
  }
}

/**
 * @brief Contract each population's grid weights with the grid spectra.
 *
 * This is the dense product out (npop, nlam) = weights (npop, grid size) x
 * spectra (grid size, nlam). Unlike particles, which each touch only 2^ndim
 * grid points, a population can touch every grid point, so this is done with
 * an optimised BLAS gemm.
 *
 * Read as column major, the row major arrays are the transposes, so we
 * compute out^T = spectra^T weights^T. A wavelength mask is handled by
 * calling gemm once per contiguous run of unmasked wavelengths, which the
 * leading dimensions let us address without copying. Masked wavelengths are
 * left untouched in the (zero initialised) output.
 *
 * The product is computed at the grid precision and cast to OutT if needed.
 *
 * NOTE: BLAS threading is controlled by the BLAS library, not nthreads. The
 * Python callers limit it to the requested number of threads with
 * threadpoolctl (see synthesizer.utils.blas). Apple's Accelerate can't be
 * limited at runtime.
 *
 * @tparam GridReal The floating-point type of the grid spectra and weights.
 * @tparam OutT The floating-point type stored in the output buffer.
 *
 * @param grid_props: The grid properties.
 * @param weights: The (npop, grid size) per population grid weights.
 * @param npop: The number of populations.
 * @param out: The zero initialised (npop, nlam) output array.
 */
template <typename GridReal, typename OutT>
static void population_spectra_loop(GridProps *grid_props,
                                    const GridReal *weights, const int npop,
                                    OutT *out) {

  tic("population_spectra_loop");

  const GridReal *grid_spectra = grid_props->get_spectra<GridReal>();
  const int nlam = grid_props->nlam;
  const int grid_size = grid_props->size;

  /* gemm writes at the grid precision, so stage the result if the output
   * precision differs. */
  std::vector<GridReal> staging;
  GridReal *result;
  if constexpr (std::is_same_v<GridReal, OutT>) {
    result = out;
  } else {
    staging.assign(static_cast<size_t>(npop) * nlam, 0);
    result = staging.data();
  }

  /* Call gemm on each contiguous run of unmasked wavelengths. */
  int ilam = 0;
  while (ilam < nlam) {
    if (grid_props->lam_is_masked(ilam)) {
      ilam++;
      continue;
    }
    const int start = ilam;
    while (ilam < nlam && !grid_props->lam_is_masked(ilam)) {
      ilam++;
    }
    gemm<GridReal>(/*m*/ ilam - start, /*n*/ npop, /*k*/ grid_size,
                   grid_spectra + start, /*lda*/ nlam, weights,
                   /*ldb*/ grid_size, result + start, /*ldc*/ nlam);
  }

  /* Cast the staged result into the output. */
  if constexpr (!std::is_same_v<GridReal, OutT>) {
    for (size_t i = 0; i < staging.size(); i++) {
      out[i] = static_cast<OutT>(staging[i]);
    }
  }

  toc("population_spectra_loop");
}

/**
 * @brief Contract a single set of grid weights with the grid spectra.
 *
 * This is the integrated (all populations summed) case. With a single set of
 * weights the product is memory bound, so rather than a BLAS gemv we walk
 * the grid spectra row by row, skipping grid points with zero weight. Many
 * SFZHs only touch a small part of the grid (e.g. a single metallicity or a
 * young burst), which a dense BLAS product can't exploit. With OpenMP the
 * wavelengths are split into one block per thread.
 *
 * Accumulation happens in double precision before being written to the
 * output at OutT. Masked wavelengths are skipped and left untouched in the
 * (zero initialised) output.
 *
 * @tparam GridReal The floating-point type of the grid spectra and weights.
 * @tparam OutT The floating-point type stored in the output buffer.
 *
 * @param grid_props: The grid properties.
 * @param weights: The grid weights.
 * @param out: The zero initialised (nlam,) output array.
 * @param nthreads: The number of threads to use.
 */
template <typename GridReal, typename OutT>
static void integrated_spectra_loop(GridProps *grid_props,
                                    const GridReal *weights, OutT *out,
                                    const int nthreads) {

  tic("integrated_spectra_loop");

  const GridReal *__restrict grid_spectra =
      grid_props->get_spectra<GridReal>();
  const size_t nlam = static_cast<size_t>(grid_props->nlam);
  const int grid_size = grid_props->size;

  /* Collect the wavelengths we actually need to compute. */
  std::vector<int> good_lams;
  good_lams.reserve(nlam);
  for (size_t ilam = 0; ilam < nlam; ilam++) {
    if (!grid_props->lam_is_masked(static_cast<int>(ilam))) {
      good_lams.push_back(static_cast<int>(ilam));
    }
  }
  const bool has_lam_mask = good_lams.size() != nlam;
  const int ngood = static_cast<int>(good_lams.size());

  /* Split the wavelengths into one block per thread. */
  const int nblocks = std::max(1, std::min(ngood, nthreads));
  const int block_len = (ngood + nblocks - 1) / nblocks;

#ifdef WITH_OPENMP
#pragma omp parallel for num_threads(nthreads) if (nthreads > 1) \
    schedule(static)
#endif
  for (int iblock = 0; iblock < nblocks; iblock++) {
    const int start = iblock * block_len;
    const int len = std::min(ngood, start + block_len) - start;
    if (len <= 0) {
      continue;
    }
    std::vector<double> accum(len, 0.0);
    double *__restrict acc = accum.data();

    /* Add each grid point's spectrum weighted by its grid weight. */
    for (int grid_ind = 0; grid_ind < grid_size; grid_ind++) {
      const double weight = static_cast<double>(weights[grid_ind]);
      if (weight <= 0.0) {
        continue;
      }
      const GridReal *__restrict row =
          grid_spectra + static_cast<size_t>(grid_ind) * nlam;
      if (has_lam_mask) {
        for (int j = 0; j < len; j++) {
          acc[j] += weight * static_cast<double>(row[good_lams[start + j]]);
        }
      } else {
        const GridReal *__restrict seg = row + start;
#ifdef WITH_OPENMP
#pragma omp simd
#endif
        for (int j = 0; j < len; j++) {
          acc[j] += weight * static_cast<double>(seg[j]);
        }
      }
    }

    /* Write this block of the spectrum. */
    for (int j = 0; j < len; j++) {
      out[good_lams[start + j]] = static_cast<OutT>(acc[j]);
    }
  }
#ifndef WITH_OPENMP
  (void)nthreads;
#endif

  toc("integrated_spectra_loop");
}

/**
 * @brief Computes the integrated SED of a set of parametric populations.
 *
 * Every population is spread onto the grid with the bin in cell (BIC)
 * approach and summed into a single set of grid weights, which are then
 * contracted with the grid spectra.
 *
 * @param np_grid_spectra: The SPS spectra array.
 * @param grid_tuple: The tuple of grid axis arrays (log10 where flagged).
 * @param edges_tuple: The tuple of bin edge arrays (linear units), in the
 *                     same order as grid_tuple.
 * @param np_masses: The masses with shape (npop, nbins_0, ..., nbins_N).
 * @param log_flags: One boolean per axis flagging log10 grid axes.
 * @param nthreads: The number of threads to use.
 * @param np_grid_weights: Precomputed grid weights or None.
 * @param mask: A boolean mask with the shape of the masses or None.
 * @param np_lam_mask: A wavelength mask or None.
 * @param out_dtype: Requested floating-point dtype for the returned spectrum.
 * @param prop_names: Optional names for the grid axes (for error messages).
 *
 * @return A tuple containing the integrated spectrum and grid weights.
 */
PyObject *compute_integrated_parametric_sed(PyObject *self, PyObject *args) {

  tic("compute_integrated_parametric_sed");

  /* We don't need the self argument but it has to be there. */
  (void)self;

  int nthreads;
  PyObject *spectra_obj, *grid_tuple, *edges_tuple, *log_flags;
  PyObject *weights_obj, *mask_obj, *lam_mask_obj, *out_dtype;
  PyObject *prop_names = NULL;
  PyArrayObject *np_masses;

  if (!PyArg_ParseTuple(args, "OOOOOiOOOO|O", &spectra_obj, &grid_tuple,
                        &edges_tuple, &np_masses, &log_flags, &nthreads,
                        &weights_obj, &mask_obj, &lam_mask_obj, &out_dtype,
                        &prop_names)) {
    return NULL;
  }

  /* Get the number of wavelengths from the spectra. */
  const int nlam = get_nlam(spectra_obj);
  if (nlam < 0) {
    return NULL;
  }

  /* Extract the grid struct. */
  PyArrayObject *np_grid_weights =
      weights_obj == Py_None ? NULL
                             : reinterpret_cast<PyArrayObject *>(weights_obj);
  PyArrayObject *np_lam_mask =
      lam_mask_obj == Py_None
          ? NULL
          : reinterpret_cast<PyArrayObject *>(lam_mask_obj);
  auto grid_props = std::unique_ptr<GridProps>(new GridProps(
      reinterpret_cast<PyArrayObject *>(spectra_obj), grid_tuple,
      /*np_lam*/ NULL, np_lam_mask, nlam, np_grid_weights, prop_names));
  RETURN_IF_PYERR();

  /* Extract the populations. */
  Populations pops;
  if (!build_populations(edges_tuple, np_masses, log_flags, mask_obj,
                         grid_props.get(), pops)) {
    return NULL;
  }

  /* Resolve the dtypes. */
  int grid_typenum = grid_props->get_float_typenum();
  if (grid_typenum == -1) {
    grid_typenum = NPY_FLOAT64;
  }
  const int output_typenum = resolve_output_typenum(out_dtype, "out_dtype");
  if (output_typenum < 0) {
    return NULL;
  }

  /* Compute the grid weights (unless we were given them) and contract them
   * with the grid spectra. */
  PyArrayObject *np_spectra = NULL;
  dispatch_float(pops.float_typenum, [&](auto p) {
    dispatch_float(grid_typenum, [&](auto g) {
      dispatch_float(output_typenum, [&](auto o) {
        using PopReal = decltype(p);
        using GridReal = decltype(g);
        using OutT = decltype(o);
        GridReal *grid_weights = grid_props->get_grid_weights<GridReal>();
        if (grid_weights == NULL || PyErr_Occurred()) {
          return;
        }
        if (grid_props->need_grid_weights()) {
          weight_loop_bic<PopReal, GridReal, GridReal>(grid_props.get(), &pops,
                                                       grid_props->size,
                                                       grid_weights, nthreads);
          if (PyErr_Occurred()) {
            return;
          }
        }
        npy_intp np_dims[1] = {nlam};
        np_spectra =
            (PyArrayObject *)PyArray_ZEROS(1, np_dims, output_typenum, 0);
        if (np_spectra == NULL) {
          return;
        }
        integrated_spectra_loop<GridReal, OutT>(
            grid_props.get(), grid_weights,
            static_cast<OutT *>(PyArray_DATA(np_spectra)), nthreads);
      });
    });
  });

  if (np_spectra == NULL) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_RuntimeError,
                      "Could not compute integrated parametric SED.");
    }
    return NULL;
  }
  RETURN_IF_PYERR();

  /* Extract the output grid weights before we free the grid object. */
  PyArrayObject *np_out_weights = grid_props->get_np_grid_weights();

  toc("compute_integrated_parametric_sed");

  return Py_BuildValue("NN", np_spectra, np_out_weights);
}

/**
 * @brief Computes the SED of each parametric population.
 *
 * Each population is spread onto the grid with the bin in cell (BIC)
 * approach and its grid weights contracted with the grid spectra.
 *
 * @param np_grid_spectra: The SPS spectra array.
 * @param grid_tuple: The tuple of grid axis arrays (log10 where flagged).
 * @param edges_tuple: The tuple of bin edge arrays (linear units), in the
 *                     same order as grid_tuple.
 * @param np_masses: The masses with shape (npop, nbins_0, ..., nbins_N).
 * @param log_flags: One boolean per axis flagging log10 grid axes.
 * @param nthreads: The number of threads to use.
 * @param np_pop_weights: Precomputed (npop, *grid_shape) weights at the grid
 *                        dtype, or None.
 * @param mask: A boolean mask with the shape of the masses or None.
 * @param np_lam_mask: A wavelength mask or None.
 * @param out_dtype: Requested floating-point dtype for the returned spectra.
 * @param prop_names: Optional names for the grid axes (for error messages).
 *
 * @return A tuple of the (npop, nlam) per population spectra and the
 *         (npop, *grid_shape) per population grid weights.
 */
PyObject *compute_population_seds(PyObject *self, PyObject *args) {

  tic("compute_population_seds");

  /* We don't need the self argument but it has to be there. */
  (void)self;

  int nthreads;
  PyObject *spectra_obj, *grid_tuple, *edges_tuple, *log_flags;
  PyObject *weights_obj, *mask_obj, *lam_mask_obj, *out_dtype;
  PyObject *prop_names = NULL;
  PyArrayObject *np_masses;

  if (!PyArg_ParseTuple(args, "OOOOOiOOOO|O", &spectra_obj, &grid_tuple,
                        &edges_tuple, &np_masses, &log_flags, &nthreads,
                        &weights_obj, &mask_obj, &lam_mask_obj, &out_dtype,
                        &prop_names)) {
    return NULL;
  }

  /* Get the number of wavelengths from the spectra. */
  const int nlam = get_nlam(spectra_obj);
  if (nlam < 0) {
    return NULL;
  }

  /* Extract the grid struct. */
  PyArrayObject *np_lam_mask =
      lam_mask_obj == Py_None
          ? NULL
          : reinterpret_cast<PyArrayObject *>(lam_mask_obj);
  auto grid_props = std::unique_ptr<GridProps>(
      new GridProps(reinterpret_cast<PyArrayObject *>(spectra_obj), grid_tuple,
                    /*np_lam*/ NULL, np_lam_mask, nlam,
                    /*np_grid_weights*/ NULL, prop_names));
  RETURN_IF_PYERR();

  /* Extract the populations. */
  Populations pops;
  if (!build_populations(edges_tuple, np_masses, log_flags, mask_obj,
                         grid_props.get(), pops)) {
    return NULL;
  }

  /* Resolve the dtypes. */
  int grid_typenum = grid_props->get_float_typenum();
  if (grid_typenum == -1) {
    grid_typenum = NPY_FLOAT64;
  }
  const int output_typenum = resolve_output_typenum(out_dtype, "out_dtype");
  if (output_typenum < 0) {
    return NULL;
  }

  /* Allocate the output. */
  npy_intp np_dims[2] = {pops.npop, nlam};
  PyArrayObject *np_pop_spectra =
      (PyArrayObject *)PyArray_ZEROS(2, np_dims, output_typenum, 0);
  if (np_pop_spectra == NULL) {
    return NULL;
  }

  /* Get the per population weights (at the grid precision, since they feed
   * gemm alongside the grid spectra), either the ones we were given or new
   * ones we compute below. */
  npy_intp np_weight_dims[MAX_GRID_NDIM + 1];
  np_weight_dims[0] = pops.npop;
  for (int idim = 0; idim < grid_props->ndim; idim++) {
    np_weight_dims[idim + 1] = grid_props->dims[idim];
  }
  PyArrayObject *np_pop_weights;
  const bool have_weights = weights_obj != Py_None;
  if (have_weights) {
    np_pop_weights = array_or_none(weights_obj, "pop_weights");
    if (np_pop_weights == NULL) {
      Py_DECREF(np_pop_spectra);
      return NULL;
    }
    if (PyArray_TYPE(np_pop_weights) != grid_typenum ||
        PyArray_NDIM(np_pop_weights) != grid_props->ndim + 1 ||
        !PyArray_IS_C_CONTIGUOUS(np_pop_weights) ||
        PyArray_SIZE(np_pop_weights) !=
            static_cast<npy_intp>(pops.npop) * grid_props->size) {
      Py_DECREF(np_pop_spectra);
      PyErr_SetString(PyExc_ValueError,
                      "pop_weights must be a contiguous (npop, *grid_shape) "
                      "array with the grid's dtype.");
      return NULL;
    }
    Py_INCREF(np_pop_weights);
  } else {
    np_pop_weights = (PyArrayObject *)PyArray_ZEROS(
        grid_props->ndim + 1, np_weight_dims, grid_typenum, 0);
    if (np_pop_weights == NULL) {
      Py_DECREF(np_pop_spectra);
      return NULL;
    }
  }

  /* Spread the populations onto the grid (unless we were given the weights)
   * and contract with the spectra. */
  dispatch_float(pops.float_typenum, [&](auto p) {
    dispatch_float(grid_typenum, [&](auto g) {
      dispatch_float(output_typenum, [&](auto o) {
        using PopReal = decltype(p);
        using GridReal = decltype(g);
        using OutT = decltype(o);
        GridReal *weights =
            static_cast<GridReal *>(PyArray_DATA(np_pop_weights));
        if (!have_weights) {
          weight_loop_bic_per_population<PopReal, GridReal, GridReal>(
              grid_props.get(), &pops, weights, nthreads);
          if (PyErr_Occurred()) {
            return;
          }
        }
        population_spectra_loop<GridReal, OutT>(
            grid_props.get(), weights, pops.npop,
            static_cast<OutT *>(PyArray_DATA(np_pop_spectra)));
      });
    });
  });
  if (PyErr_Occurred()) {
    Py_DECREF(np_pop_spectra);
    Py_DECREF(np_pop_weights);
    return NULL;
  }

  toc("compute_population_seds");

  return Py_BuildValue("NN", np_pop_spectra, np_pop_weights);
}

/**
 * @brief Computes the grid weights of a set of parametric populations.
 *
 * Every population is spread onto the grid with the bin in cell (BIC)
 * approach and summed into a single set of grid weights. No spectra are
 * involved, so this can map populations onto any set of axes (e.g. to view a
 * population's SFZH or remap it onto new axes).
 *
 * @param grid_tuple: The tuple of grid axis arrays (log10 where flagged).
 * @param edges_tuple: The tuple of bin edge arrays (linear units), in the
 *                     same order as grid_tuple.
 * @param np_masses: The masses with shape (npop, nbins_0, ..., nbins_N).
 * @param log_flags: One boolean per axis flagging log10 grid axes.
 * @param nthreads: The number of threads to use.
 * @param mask: A boolean mask with the shape of the masses or None.
 * @param prop_names: Optional names for the grid axes (for error messages).
 *
 * @return The grid weights (with the grid's shape and dtype).
 */
PyObject *compute_parametric_weights(PyObject *self, PyObject *args) {

  tic("compute_parametric_weights");

  /* We don't need the self argument but it has to be there. */
  (void)self;

  int nthreads;
  PyObject *grid_tuple, *edges_tuple, *log_flags, *mask_obj;
  PyObject *prop_names = NULL;
  PyArrayObject *np_masses;

  if (!PyArg_ParseTuple(args, "OOOOiO|O", &grid_tuple, &edges_tuple,
                        &np_masses, &log_flags, &nthreads, &mask_obj,
                        &prop_names)) {
    return NULL;
  }

  /* Extract the grid struct (without any spectra). */
  auto grid_props = std::unique_ptr<GridProps>(
      new GridProps(/*np_spectra*/ NULL, grid_tuple, /*np_lam*/ NULL,
                    /*np_lam_mask*/ NULL, /*nlam*/ 0,
                    /*np_grid_weights*/ NULL, prop_names));
  RETURN_IF_PYERR();

  /* Extract the populations. */
  Populations pops;
  if (!build_populations(edges_tuple, np_masses, log_flags, mask_obj,
                         grid_props.get(), pops)) {
    return NULL;
  }

  /* Resolve the grid dtype. */
  int grid_typenum = grid_props->get_float_typenum();
  if (grid_typenum == -1) {
    grid_typenum = NPY_FLOAT64;
  }

  /* Spread the populations onto the grid. */
  dispatch_float(pops.float_typenum, [&](auto p) {
    dispatch_float(grid_typenum, [&](auto g) {
      using PopReal = decltype(p);
      using GridReal = decltype(g);
      GridReal *grid_weights = grid_props->get_grid_weights<GridReal>();
      if (grid_weights == NULL || PyErr_Occurred()) {
        return;
      }
      weight_loop_bic<PopReal, GridReal, GridReal>(
          grid_props.get(), &pops, grid_props->size, grid_weights, nthreads);
    });
  });
  RETURN_IF_PYERR();

  toc("compute_parametric_weights");

  return (PyObject *)grid_props->get_np_grid_weights();
}

/* Below is all the gubbins needed to make the module importable in Python. */
static PyMethodDef ParametricSedMethods[] = {
    {"compute_integrated_parametric_sed",
     (PyCFunction)compute_integrated_parametric_sed, METH_VARARGS,
     "Method for calculating the integrated spectra of parametric "
     "populations."},
    {"compute_parametric_weights", (PyCFunction)compute_parametric_weights,
     METH_VARARGS,
     "Method for calculating the grid weights of parametric populations."},
    {"compute_population_seds", (PyCFunction)compute_population_seds,
     METH_VARARGS,
     "Method for calculating the spectra of each parametric population."},
    {NULL, NULL, 0, NULL}};

/* Make this importable. */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "parametric_spectra",                                  /* m_name */
    "A module to calculate parametric population spectra", /* m_doc */
    -1,                                                    /* m_size */
    ParametricSedMethods,                                  /* m_methods */
    NULL,                                                  /* m_reload */
    NULL,                                                  /* m_traverse */
    NULL,                                                  /* m_clear */
    NULL,                                                  /* m_free */
};

PyMODINIT_FUNC PyInit_parametric_spectra(void) {
  PyObject *m = PyModule_Create(&moduledef);
  if (m == NULL) return NULL;
  if (numpy_import() < 0) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to import numpy.");
    Py_DECREF(m);
    return NULL;
  }
  if (load_blas() < 0) {
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
