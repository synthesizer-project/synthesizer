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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
 * @brief Get the number of wavelength elements from the grid spectra.
 *
 * @param spectra_obj: The grid spectra (grid axes followed by wavelength).
 *
 * @return The number of wavelength elements, or -1 (with a Python error set).
 */
static int get_nlam(PyObject *spectra_obj) {
  /* We need an array with at least one grid axis and a wavelength axis. */
  if (!PyArray_Check(spectra_obj) ||
      PyArray_NDIM(reinterpret_cast<PyArrayObject *>(spectra_obj)) < 2) {
    PyErr_SetString(PyExc_ValueError,
                    "grid_spectra must be an array with the grid axes "
                    "followed by a wavelength axis.");
    return -1;
  }

  /* The wavelength axis is the last axis. */
  PyArrayObject *np_spectra = reinterpret_cast<PyArrayObject *>(spectra_obj);
  return static_cast<int>(
      PyArray_DIM(np_spectra, PyArray_NDIM(np_spectra) - 1));
}

/**
 * @brief Contract each population's grid weights with one chunk of the grid
 *        spectra.
 *
 * We walk the grid spectra row by row, skipping grid points no population
 * touches, and add this chunk of each row to the spectrum of every
 * population with a non-zero weight there. Each row is read once for all the
 * populations, which (for the few populations a parametric Stars typically
 * holds) keeps the product memory bound, and many SFZHs only touch a small
 * part of the grid.
 *
 * Accumulation happens in double precision before being written to the
 * output at OutT. Masked wavelengths are skipped and left untouched in the
 * (zero initialised) output.
 *
 * @tparam GridReal The floating-point type of the grid spectra.
 * @tparam OutT The floating-point type stored in the output buffer.
 *
 * @param grid_props: The grid properties.
 * @param row_start: The first non-zero weight of each grid point (compressed
 *                   rows, grid size + 1 entries).
 * @param pop_inds: The population of each non-zero weight.
 * @param pop_weights: The non-zero weights.
 * @param good_lams: The indices of the unmasked wavelengths.
 * @param npop: The number of populations.
 * @param start: The first unmasked wavelength of the chunk.
 * @param len: The number of unmasked wavelengths in the chunk.
 * @param out: The zero initialised (npop, nlam) output array.
 */
template <typename GridReal, typename OutT>
static void population_spectra_chunk(GridProps *grid_props,
                                     const std::vector<int> &row_start,
                                     const std::vector<int> &pop_inds,
                                     const std::vector<double> &pop_weights,
                                     const std::vector<int> &good_lams,
                                     const int npop, const int start,
                                     const int len, OutT *out) {

  /* Unpack the grid properties. */
  const GridReal *__restrict grid_spectra =
      grid_props->get_spectra<GridReal>();
  const size_t nlam = static_cast<size_t>(grid_props->nlam);
  const int grid_size = grid_props->size;
  const bool has_lam_mask = good_lams.size() != nlam;

  /* Set up the accumulators for this chunk of every population's spectrum
   * and somewhere to gather a row's unmasked wavelengths. */
  std::vector<double> accum(static_cast<size_t>(npop) * len, 0.0);
  std::vector<double> seg_buf(has_lam_mask ? len : 0);

  /* Add each grid point's spectrum to every population weighting it. */
  for (int grid_ind = 0; grid_ind < grid_size; grid_ind++) {

    /* Skip grid points no population touches. */
    const int first = row_start[grid_ind];
    const int last = row_start[grid_ind + 1];
    if (first == last) {
      continue;
    }
    const GridReal *__restrict row =
        grid_spectra + static_cast<size_t>(grid_ind) * nlam;

    /* Gather this chunk of the row if wavelengths are masked. */
    if (has_lam_mask) {
      for (int j = 0; j < len; j++) {
        seg_buf[j] = static_cast<double>(row[good_lams[start + j]]);
      }
    }

    /* Loop over the populations weighting this grid point. */
    for (int k = first; k < last; k++) {
      const double weight = pop_weights[k];
      double *__restrict acc =
          accum.data() + static_cast<size_t>(pop_inds[k]) * len;

      /* Add the weighted chunk (gathered or contiguous) to the
       * population's accumulator. */
      if (has_lam_mask) {
        const double *__restrict seg = seg_buf.data();
#ifdef WITH_OPENMP
#pragma omp simd
#endif
        for (int j = 0; j < len; j++) {
          acc[j] += weight * seg[j];
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
  }

  /* Write this chunk of every population's spectrum. */
  for (int ipop = 0; ipop < npop; ipop++) {
    const double *acc = accum.data() + static_cast<size_t>(ipop) * len;
    OutT *pop_out = out + static_cast<size_t>(ipop) * nlam;
    for (int j = 0; j < len; j++) {
      pop_out[good_lams[start + j]] = static_cast<OutT>(acc[j]);
    }
  }
}

/**
 * @brief Contract each population's grid weights with the grid spectra.
 *
 * This is the serial version of the function.
 *
 * @tparam GridReal The floating-point type of the grid spectra.
 * @tparam OutT The floating-point type stored in the output buffer.
 *
 * @param grid_props: The grid properties.
 * @param row_start: The first non-zero weight of each grid point.
 * @param pop_inds: The population of each non-zero weight.
 * @param pop_weights: The non-zero weights.
 * @param good_lams: The indices of the unmasked wavelengths.
 * @param npop: The number of populations.
 * @param chunk_len: The number of unmasked wavelengths in each chunk.
 * @param out: The zero initialised (npop, nlam) output array.
 */
template <typename GridReal, typename OutT>
static void population_spectra_loop_serial(
    GridProps *grid_props, const std::vector<int> &row_start,
    const std::vector<int> &pop_inds, const std::vector<double> &pop_weights,
    const std::vector<int> &good_lams, const int npop, const int chunk_len,
    OutT *out) {

  /* Loop over the chunks of wavelengths. */
  const int ngood = static_cast<int>(good_lams.size());
  for (int start = 0; start < ngood; start += chunk_len) {
    population_spectra_chunk<GridReal, OutT>(
        grid_props, row_start, pop_inds, pop_weights, good_lams, npop, start,
        std::min(chunk_len, ngood - start), out);
  }
}

/**
 * @brief Contract each population's grid weights with the grid spectra.
 *
 * This is the parallel version of the function. The chunks of wavelengths
 * are shared between the threads, each writing only its own wavelengths.
 *
 * @tparam GridReal The floating-point type of the grid spectra.
 * @tparam OutT The floating-point type stored in the output buffer.
 *
 * @param grid_props: The grid properties.
 * @param row_start: The first non-zero weight of each grid point.
 * @param pop_inds: The population of each non-zero weight.
 * @param pop_weights: The non-zero weights.
 * @param good_lams: The indices of the unmasked wavelengths.
 * @param npop: The number of populations.
 * @param chunk_len: The number of unmasked wavelengths in each chunk.
 * @param out: The zero initialised (npop, nlam) output array.
 * @param nthreads: The number of threads to use.
 */
#ifdef WITH_OPENMP
template <typename GridReal, typename OutT>
static void population_spectra_loop_omp(GridProps *grid_props,
                                        const std::vector<int> &row_start,
                                        const std::vector<int> &pop_inds,
                                        const std::vector<double> &pop_weights,
                                        const std::vector<int> &good_lams,
                                        const int npop, const int chunk_len,
                                        OutT *out, const int nthreads) {

  /* Loop over the chunks of wavelengths, sharing them between threads. */
  const int ngood = static_cast<int>(good_lams.size());
#pragma omp parallel for num_threads(nthreads) schedule(static)
  for (int start = 0; start < ngood; start += chunk_len) {
    population_spectra_chunk<GridReal, OutT>(
        grid_props, row_start, pop_inds, pop_weights, good_lams, npop, start,
        std::min(chunk_len, ngood - start), out);
  }
}
#endif /* WITH_OPENMP */

/**
 * @brief Contract each population's grid weights with the grid spectra.
 *
 * This is out (npop, nlam) = weights (npop, grid size) x spectra (grid size,
 * nlam). The wavelengths are split into chunks small enough that a chunk of
 * every population's spectrum stays in cache, which are contracted one at a
 * time (see population_spectra_chunk).
 *
 * This is a wrapper which calls the correct function based on the number of
 * threads requested and whether OpenMP is available.
 *
 * @tparam GridReal The floating-point type of the grid spectra and weights.
 * @tparam OutT The floating-point type stored in the output buffer.
 *
 * @param grid_props: The grid properties.
 * @param weights: The (npop, grid size) per population grid weights.
 * @param npop: The number of populations.
 * @param out: The zero initialised (npop, nlam) output array.
 * @param nthreads: The number of threads to use.
 */
template <typename GridReal, typename OutT>
static void population_spectra_loop(GridProps *grid_props,
                                    const GridReal *weights, const int npop,
                                    OutT *out, const int nthreads) {

  tic("population_spectra_loop");

  const int nlam = grid_props->nlam;
  const int grid_size = grid_props->size;

  /* Collect the non-zero weights of each grid point (as compressed rows
   * of population indices and weights) so the loops only visit the grid
   * points and populations that contribute. */
  std::vector<int> row_start(grid_size + 1, 0);
  std::vector<int> pop_inds;
  std::vector<double> pop_weights;
  for (int grid_ind = 0; grid_ind < grid_size; grid_ind++) {
    for (int ipop = 0; ipop < npop; ipop++) {
      const double weight = static_cast<double>(
          weights[static_cast<size_t>(ipop) * grid_size + grid_ind]);
      if (weight > 0.0) {
        pop_inds.push_back(ipop);
        pop_weights.push_back(weight);
      }
    }
    row_start[grid_ind + 1] = static_cast<int>(pop_inds.size());
  }

  /* Collect the wavelengths we actually need to compute. */
  std::vector<int> good_lams;
  good_lams.reserve(nlam);
  for (int ilam = 0; ilam < nlam; ilam++) {
    if (!grid_props->lam_is_masked(ilam)) {
      good_lams.push_back(ilam);
    }
  }
  const int ngood = static_cast<int>(good_lams.size());

  /* Split the wavelengths into chunks, at least one per thread and short
   * enough that every population's accumulator for a chunk (~256 KB) stays
   * in cache. */
  const int max_len = std::max(64, 32768 / std::max(npop, 1));
  const int nchunks = std::max(
      {1, std::min(ngood, nthreads), (ngood + max_len - 1) / max_len});
  const int chunk_len = std::max(1, (ngood + nchunks - 1) / nchunks);

  /* Call the correct function for the configuration/number of threads. */

#ifdef WITH_OPENMP

  /* If we have multiple threads and OpenMP we can parallelise. */
  if (nthreads > 1) {
    population_spectra_loop_omp<GridReal, OutT>(
        grid_props, row_start, pop_inds, pop_weights, good_lams, npop,
        chunk_len, out, nthreads);
  }
  /* Otherwise there's no point paying the OpenMP overhead. */
  else {
    population_spectra_loop_serial<GridReal, OutT>(
        grid_props, row_start, pop_inds, pop_weights, good_lams, npop,
        chunk_len, out);
  }

#else

  (void)nthreads;

  /* We don't have OpenMP, just call the serial version. */
  population_spectra_loop_serial<GridReal, OutT>(
      grid_props, row_start, pop_inds, pop_weights, good_lams, npop, chunk_len,
      out);

#endif

  toc("population_spectra_loop");
}

/**
 * @brief Contract the grid weights with one block of the grid spectra.
 *
 * We walk the grid spectra row by row, skipping grid points with zero
 * weight. Many SFZHs only touch a small part of the grid (e.g. a single
 * metallicity or a young burst), which a dense BLAS product can't exploit.
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
 * @param good_lams: The indices of the unmasked wavelengths.
 * @param start: The first entry of good_lams in this block.
 * @param len: The number of entries of good_lams in this block.
 * @param out: The zero initialised (nlam,) output array.
 */
template <typename GridReal, typename OutT>
static inline void integrated_spectra_block(GridProps *grid_props,
                                            const GridReal *weights,
                                            const std::vector<int> &good_lams,
                                            const int start, const int len,
                                            OutT *out) {

  /* Unpack the grid spectra and their dimensions. */
  const GridReal *__restrict grid_spectra =
      grid_props->get_spectra<GridReal>();
  const size_t nlam = static_cast<size_t>(grid_props->nlam);
  const int grid_size = grid_props->size;

  /* Without a wavelength mask the block is a contiguous segment. */
  const bool has_lam_mask = good_lams.size() != nlam;

  /* Accumulate this block in double precision. */
  std::vector<double> accum(len, 0.0);
  double *__restrict acc = accum.data();

  /* Add each grid point's spectrum weighted by its grid weight. */
  for (int grid_ind = 0; grid_ind < grid_size; grid_ind++) {
    /* Skip grid points the populations put no mass on. */
    const double weight = static_cast<double>(weights[grid_ind]);
    if (weight <= 0.0) {
      continue;
    }

    /* Get this grid point's spectrum. */
    const GridReal *__restrict row =
        grid_spectra + static_cast<size_t>(grid_ind) * nlam;

    /* With a wavelength mask gather the good wavelengths, otherwise the
     * block is a contiguous segment we can vectorise over. */
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

/**
 * @brief Contract a single set of grid weights with the grid spectra.
 *
 * This is the serial version of the function.
 *
 * @tparam GridReal The floating-point type of the grid spectra and weights.
 * @tparam OutT The floating-point type stored in the output buffer.
 *
 * @param grid_props: The grid properties.
 * @param weights: The grid weights.
 * @param good_lams: The indices of the unmasked wavelengths.
 * @param out: The zero initialised (nlam,) output array.
 */
template <typename GridReal, typename OutT>
static void integrated_spectra_loop_serial(GridProps *grid_props,
                                           const GridReal *weights,
                                           const std::vector<int> &good_lams,
                                           OutT *out) {

  /* Every unmasked wavelength is a single block. */
  integrated_spectra_block<GridReal, OutT>(
      grid_props, weights, good_lams, /*start=*/0,
      static_cast<int>(good_lams.size()), out);
}

/**
 * @brief Contract a single set of grid weights with the grid spectra.
 *
 * This is the parallel version of the function.
 *
 * The unmasked wavelengths are split into one block per thread. Each block
 * writes to its own wavelengths of the output so there are no races.
 *
 * @tparam GridReal The floating-point type of the grid spectra and weights.
 * @tparam OutT The floating-point type stored in the output buffer.
 *
 * @param grid_props: The grid properties.
 * @param weights: The grid weights.
 * @param good_lams: The indices of the unmasked wavelengths.
 * @param out: The zero initialised (nlam,) output array.
 * @param nthreads: The number of threads to use.
 */
#ifdef WITH_OPENMP
template <typename GridReal, typename OutT>
static void integrated_spectra_loop_omp(GridProps *grid_props,
                                        const GridReal *weights,
                                        const std::vector<int> &good_lams,
                                        OutT *out, const int nthreads) {

  /* Split the wavelengths into one block per thread. */
  const int ngood = static_cast<int>(good_lams.size());
  const int nblocks = std::max(1, std::min(ngood, nthreads));
  const int block_len = (ngood + nblocks - 1) / nblocks;

#pragma omp parallel for num_threads(nthreads) schedule(static)
  for (int iblock = 0; iblock < nblocks; iblock++) {
    /* Get this block's range of good wavelengths (the last may be short,
     * or empty). */
    const int start = iblock * block_len;
    const int len = std::min(ngood, start + block_len) - start;
    if (len <= 0) {
      continue;
    }

    /* Contract the weights with this block of the spectra. */
    integrated_spectra_block<GridReal, OutT>(grid_props, weights, good_lams,
                                             start, len, out);
  }
}
#endif /* WITH_OPENMP */

/**
 * @brief Contract a single set of grid weights with the grid spectra.
 *
 * This is the integrated (all populations summed) case. It is a wrapper
 * which calls the correct function based on the number of threads requested
 * and whether OpenMP is available.
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

  /* Collect the wavelengths we actually need to compute. */
  std::vector<int> good_lams;
  good_lams.reserve(grid_props->nlam);
  for (int ilam = 0; ilam < grid_props->nlam; ilam++) {
    if (!grid_props->lam_is_masked(ilam)) {
      good_lams.push_back(ilam);
    }
  }

  /* Call the correct function for the configuration/number of threads. */

#ifdef WITH_OPENMP

  /* If we have multiple threads and OpenMP we can parallelise. */
  if (nthreads > 1) {
    integrated_spectra_loop_omp<GridReal, OutT>(grid_props, weights, good_lams,
                                                out, nthreads);
  }
  /* Otherwise there's no point paying the OpenMP overhead. */
  else {
    integrated_spectra_loop_serial<GridReal, OutT>(grid_props, weights,
                                                   good_lams, out);
  }

#else

  (void)nthreads;

  /* We don't have OpenMP, just call the serial version. */
  integrated_spectra_loop_serial<GridReal, OutT>(grid_props, weights,
                                                 good_lams, out);

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
 * @param edges_tuple: The tuple of bin edge arrays (linear units), one per
 *                     grid axis.
 * @param np_masses: The masses with shape (npop, nbins_0, ..., nbins_N).
 * @param log_flags: One boolean per grid axis flagging log10 grid axes.
 * @param bin_axes: The input (masses/mask) axis matching each grid axis.
 * @param unit_scales: The factor converting each grid axis' edges into the
 *                     grid axis' units.
 * @param nthreads: The number of threads to use.
 * @param np_grid_weights: Precomputed grid weights or None.
 * @param mask: The allowed (nmask, 2) range along the first nmask input
 *              axes or None.
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

  /* Parse the arguments (prop_names is optional). */
  int nthreads;
  PyObject *spectra_obj, *grid_tuple, *edges_tuple, *log_flags, *bin_axes;
  PyObject *unit_scales;
  PyObject *weights_obj, *mask_obj, *lam_mask_obj, *out_dtype;
  PyObject *prop_names = NULL;
  PyArrayObject *np_masses;

  if (!PyArg_ParseTuple(args, "OOOOOOOiOOOO|O", &spectra_obj, &grid_tuple,
                        &edges_tuple, &np_masses, &log_flags, &bin_axes,
                        &unit_scales, &nthreads, &weights_obj, &mask_obj,
                        &lam_mask_obj, &out_dtype, &prop_names)) {
    return NULL;
  }

  /* Precomputed weights are used as they are, so a mask can't be applied. */
  if (weights_obj != Py_None && mask_obj != Py_None) {
    PyErr_SetString(PyExc_ValueError,
                    "A mask can't be applied to precomputed grid weights.");
    return NULL;
  }

  /* Get the number of wavelengths from the spectra. */
  const int nlam = get_nlam(spectra_obj);
  if (nlam < 0) {
    return NULL;
  }

  /* Extract the grid struct, treating None weights and wavelength mask as
   * absent. */
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

  /* Create the object that holds the population properties. */
  auto pops = std::unique_ptr<Populations>(
      new Populations(edges_tuple, np_masses, log_flags, bin_axes, unit_scales,
                      mask_obj, prop_names, grid_props->ndim));
  RETURN_IF_PYERR();

  /* Resolve the dtypes, defaulting the grid to float64. */
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
  dispatch_float(pops->get_float_typenum(), [&](auto p) {
    dispatch_float(grid_typenum, [&](auto g) {
      dispatch_float(output_typenum, [&](auto o) {
        using PopReal = decltype(p);
        using GridReal = decltype(g);
        using OutT = decltype(o);

        /* Get the grid weights buffer (either the precomputed weights or a
         * new zeroed array). */
        GridReal *grid_weights = grid_props->get_grid_weights<GridReal>();
        if (grid_weights == NULL || PyErr_Occurred()) {
          return;
        }

        /* Spread the populations onto the grid if we weren't given the
         * weights. */
        if (grid_props->need_grid_weights()) {
          weight_loop_bic<PopReal, GridReal, GridReal>(
              grid_props.get(), pops.get(), grid_props->size, grid_weights,
              nthreads);
          if (PyErr_Occurred()) {
            return;
          }
        }

        /* Allocate the zeroed output spectrum and contract the weights with
         * the grid spectra into it. */
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

  /* Make sure a failure inside the dispatch always surfaces as an error. */
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
 * @param edges_tuple: The tuple of bin edge arrays (linear units), one per
 *                     grid axis.
 * @param np_masses: The masses with shape (npop, nbins_0, ..., nbins_N).
 * @param log_flags: One boolean per grid axis flagging log10 grid axes.
 * @param bin_axes: The input (masses/mask) axis matching each grid axis.
 * @param unit_scales: The factor converting each grid axis' edges into the
 *                     grid axis' units.
 * @param nthreads: The number of threads to use.
 * @param np_pop_weights: Precomputed (npop, *grid_shape) weights at the grid
 *                        dtype, or None.
 * @param mask: The allowed (nmask, 2) range along the first nmask input
 *              axes or None.
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

  /* Parse the arguments (prop_names is optional). */
  int nthreads;
  PyObject *spectra_obj, *grid_tuple, *edges_tuple, *log_flags, *bin_axes;
  PyObject *unit_scales;
  PyObject *weights_obj, *mask_obj, *lam_mask_obj, *out_dtype;
  PyObject *prop_names = NULL;
  PyArrayObject *np_masses;

  if (!PyArg_ParseTuple(args, "OOOOOOOiOOOO|O", &spectra_obj, &grid_tuple,
                        &edges_tuple, &np_masses, &log_flags, &bin_axes,
                        &unit_scales, &nthreads, &weights_obj, &mask_obj,
                        &lam_mask_obj, &out_dtype, &prop_names)) {
    return NULL;
  }

  /* Precomputed weights are used as they are, so a mask can't be applied. */
  if (weights_obj != Py_None && mask_obj != Py_None) {
    PyErr_SetString(PyExc_ValueError,
                    "A mask can't be applied to precomputed grid weights.");
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

  /* Create the object that holds the population properties. */
  auto pops = std::unique_ptr<Populations>(
      new Populations(edges_tuple, np_masses, log_flags, bin_axes, unit_scales,
                      mask_obj, prop_names, grid_props->ndim));
  RETURN_IF_PYERR();

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
  npy_intp np_dims[2] = {pops->npop, nlam};
  PyArrayObject *np_pop_spectra =
      (PyArrayObject *)PyArray_ZEROS(2, np_dims, output_typenum, 0);
  if (np_pop_spectra == NULL) {
    return NULL;
  }

  /* Get the per population weights (at the grid precision, like the grid
   * spectra they are contracted with), either the ones we were given or new
   * ones we compute below. */
  npy_intp np_weight_dims[MAX_GRID_NDIM + 1];
  np_weight_dims[0] = pops->npop;
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
            static_cast<npy_intp>(pops->npop) * grid_props->size) {
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
  dispatch_float(pops->get_float_typenum(), [&](auto p) {
    dispatch_float(grid_typenum, [&](auto g) {
      dispatch_float(output_typenum, [&](auto o) {
        using PopReal = decltype(p);
        using GridReal = decltype(g);
        using OutT = decltype(o);
        GridReal *weights =
            static_cast<GridReal *>(PyArray_DATA(np_pop_weights));
        if (!have_weights) {
          weight_loop_bic_per_population<PopReal, GridReal, GridReal>(
              grid_props.get(), pops.get(), weights, nthreads);
          if (PyErr_Occurred()) {
            return;
          }
        }
        population_spectra_loop<GridReal, OutT>(
            grid_props.get(), weights, pops->npop,
            static_cast<OutT *>(PyArray_DATA(np_pop_spectra)), nthreads);
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
 * @param edges_tuple: The tuple of bin edge arrays (linear units), one per
 *                     grid axis.
 * @param np_masses: The masses with shape (npop, nbins_0, ..., nbins_N).
 * @param log_flags: One boolean per grid axis flagging log10 grid axes.
 * @param bin_axes: The input (masses/mask) axis matching each grid axis.
 * @param unit_scales: The factor converting each grid axis' edges into the
 *                     grid axis' units.
 * @param nthreads: The number of threads to use.
 * @param mask: The allowed (nmask, 2) range along the first nmask input
 *              axes or None.
 * @param prop_names: Optional names for the grid axes (for error messages).
 *
 * @return The grid weights (with the grid's shape and dtype).
 */
PyObject *compute_parametric_weights(PyObject *self, PyObject *args) {

  tic("compute_parametric_weights");

  /* We don't need the self argument but it has to be there. */
  (void)self;

  /* Parse the arguments (prop_names is optional). */
  int nthreads;
  PyObject *grid_tuple, *edges_tuple, *log_flags, *bin_axes, *unit_scales;
  PyObject *mask_obj;
  PyObject *prop_names = NULL;
  PyArrayObject *np_masses;

  if (!PyArg_ParseTuple(args, "OOOOOOiO|O", &grid_tuple, &edges_tuple,
                        &np_masses, &log_flags, &bin_axes, &unit_scales,
                        &nthreads, &mask_obj, &prop_names)) {
    return NULL;
  }

  /* Extract the grid struct (without any spectra). */
  auto grid_props = std::unique_ptr<GridProps>(
      new GridProps(/*np_spectra*/ NULL, grid_tuple, /*np_lam*/ NULL,
                    /*np_lam_mask*/ NULL, /*nlam*/ 0,
                    /*np_grid_weights*/ NULL, prop_names));
  RETURN_IF_PYERR();

  /* Create the object that holds the population properties. */
  auto pops = std::unique_ptr<Populations>(
      new Populations(edges_tuple, np_masses, log_flags, bin_axes, unit_scales,
                      mask_obj, prop_names, grid_props->ndim));
  RETURN_IF_PYERR();

  /* Resolve the grid dtype, defaulting to float64. */
  int grid_typenum = grid_props->get_float_typenum();
  if (grid_typenum == -1) {
    grid_typenum = NPY_FLOAT64;
  }

  /* Spread the populations onto the grid. */
  dispatch_float(pops->get_float_typenum(), [&](auto p) {
    dispatch_float(grid_typenum, [&](auto g) {
      using PopReal = decltype(p);
      using GridReal = decltype(g);

      /* Get a new zeroed grid weights buffer. */
      GridReal *grid_weights = grid_props->get_grid_weights<GridReal>();
      if (grid_weights == NULL || PyErr_Occurred()) {
        return;
      }

      /* Accumulate every population's bins into it. */
      weight_loop_bic<PopReal, GridReal, GridReal>(
          grid_props.get(), pops.get(), grid_props->size, grid_weights,
          nthreads);
    });
  });
  RETURN_IF_PYERR();

  toc("compute_parametric_weights");

  /* Hand the weights array back to Python. */
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

/**
 * @brief Initialise the parametric_spectra module.
 *
 * @return The module object, or NULL (with a Python error set) on failure.
 */
PyMODINIT_FUNC PyInit_parametric_spectra(void) {
  /* Create the module. */
  PyObject *m = PyModule_Create(&moduledef);
  if (m == NULL) return NULL;

  /* Declare the module safe to use without the GIL so importing it doesn't
   * re-enable the GIL on free-threaded builds of Python. */
#ifdef Py_GIL_DISABLED
  PyUnstable_Module_SetGIL(m, Py_MOD_GIL_NOT_USED);
#endif

  /* Import numpy's C API. */
  if (numpy_import() < 0) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to import numpy.");
    Py_DECREF(m);
    return NULL;
  }

  /* Hook up the shared atomic timers if they are enabled. */
#ifdef ATOMIC_TIMING
  if (import_toc_capsule() < 0) {
    Py_DECREF(m);
    return NULL;
  }
#endif
  return m;
}
