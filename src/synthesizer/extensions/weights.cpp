/******************************************************************************
 * A C module containing all the weights functions common to all particle
 * spectra extensions.
 *
 * NOTE: This file serves a dual role. It is both a standalone extension
 * module (PyInit_weights) AND compiled as a source file into
 * integrated_spectra, particle_spectra, doppler_particle_spectra, and sfzh.
 * When compiled into another extension, PyInit_weights is dead code.
 *****************************************************************************/
/* C includes */
#include <array>
#include <math.h>
#include <new>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

/* Python includes */
#include <Python.h>

/* Local includes */
#include "cpp_to_python.h"
#include "index_utils.h"
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
 * @brief This calculates the grid weights in each grid cell using a cloud
 *        in cell (CIC) approach.
 *
 * This is the serial version of the function. Each particle distributes its
 * weight across 2^ndim neighboring grid cells based on its fractional
 * distance along each axis.
 *
 * Accumulation always happens in double precision so reduced precision
 * inputs/outputs don't degrade the summation over many particles.
 *
 * @tparam PartReal The floating-point type of the particle arrays.
 * @tparam GridReal The floating-point type of the grid axis arrays.
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A class containing the particle properties.
 * @param out_arr: The double precision accumulation buffer (grid sized).
 */
template <typename PartReal, typename GridReal>
static void weight_loop_cic_serial(GridProps *grid_props, Particles *parts,
                                   double *out_arr) {
  /* Unpack the grid properties. */
  const std::array<int, MAX_GRID_NDIM> dims = grid_props->dims;
  const int ndim = grid_props->ndim;
  const int num_sub_cells = 1 << ndim;

  /* Build sub_dims = [2,2,...,2] once */
  std::array<int, MAX_GRID_NDIM> sub_dims;
  for (int i = 0; i < ndim; ++i) {
    sub_dims[i] = 2;
  }

  /* Precompute for each sub-cell:
   *  - the per-dim offsets (0 or 1)
   *  - the linear offset = sum(offsets[d] * stride[d])
   */
  struct SubCell {
    std::array<int, MAX_GRID_NDIM> offs;
    int linoff;
  };
  std::vector<SubCell> subcells(num_sub_cells);
  {
    std::array<int, MAX_GRID_NDIM> tmp{};
    for (int ic = 0; ic < num_sub_cells; ++ic) {
      get_indices_from_flat(ic, ndim, sub_dims, tmp);
      subcells[ic].offs = tmp;
      /* by passing tmp (0/1) into get_flat_index we get
         exactly ∑ tmp[d] * stride[d] */
      subcells[ic].linoff = get_flat_index(tmp, dims.data(), ndim);
    }
  }

  /* Loop over particles. */
  for (int p = 0; p < parts->npart; ++p) {
    /* Skip if this particle is masked. */
    if (parts->part_is_masked(p)) {
      continue;
    }

    /* Get this particle's weight and base cell info. */
    const double weight =
        static_cast<double>(parts->get_weight_at<PartReal>(p));

    std::array<int, MAX_GRID_NDIM> part_idx;
    std::array<GridReal, MAX_GRID_NDIM> axis_frac;
    get_part_ind_frac_cic<PartReal, GridReal>(part_idx, axis_frac, grid_props,
                                              parts, p);

    /* Compute linear index of the “low” corner once */
    const int base_lin = get_flat_index(part_idx, dims.data(), ndim);

    /* Now distribute into each of the 2^ndim subcells. */
    for (int ic = 0; ic < num_sub_cells; ++ic) {
      const auto &sc = subcells[ic];

      /* Compute the CIC fraction for this corner */
      GridReal frac = static_cast<GridReal>(1);
      for (int d = 0; d < ndim; ++d) {
        frac *= sc.offs[d] ? axis_frac[d]
                           : (static_cast<GridReal>(1) - axis_frac[d]);
      }
      if (frac == static_cast<GridReal>(0)) {
        continue;
      }

      /* Final flat index = base + precomputed offset */
      const int flat_ind = base_lin + sc.linoff;
      out_arr[flat_ind] += static_cast<double>(frac) * weight;
    }
  }
}

/**
 * @brief This calculates the grid weights in each grid cell using a cloud
 *        in cell approach.
 *
 * This is the parallel version of the function.
 *
 * Each thread accumulates weights into a private local buffer, which is added
 * into the global output array at the end of the thread’s execution.
 *
 * @tparam PartReal The floating-point type of the particle arrays.
 * @tparam GridReal The floating-point type of the grid axis arrays.
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A class containing the particle properties.
 * @param out_size: The size of the output array.
 * @param out_arr: The double precision accumulation buffer (grid sized).
 * @param nthreads: The number of threads to use.
 */
#ifdef WITH_OPENMP
template <typename PartReal, typename GridReal>
static void weight_loop_cic_omp(GridProps *grid_props, Particles *parts,
                                int out_size, double *out_arr, int nthreads) {

  /* Unpack the grid properties. */
  const std::array<int, MAX_GRID_NDIM> dims = grid_props->dims;
  const int ndim = grid_props->ndim;

  /* Set the sub cell constants we'll use below. */
  const int num_sub_cells = 1 << ndim;  // 2^ndim
  std::array<int, MAX_GRID_NDIM> sub_dims;
  for (int i = 0; i < ndim; i++) {
    sub_dims[i] = 2;
  }

  /* Precompute sub-cell offsets and linear offsets once */
  struct SubCell {
    std::array<int, MAX_GRID_NDIM> offs;
    int linoff;
  };
  std::vector<SubCell> subcells(num_sub_cells);
  {
    std::array<int, MAX_GRID_NDIM> tmp{};
    for (int ic = 0; ic < num_sub_cells; ic++) {
      get_indices_from_flat(ic, ndim, sub_dims, tmp);
      subcells[ic].offs = tmp;
      subcells[ic].linoff = get_flat_index(tmp, dims.data(), ndim);
    }
  }

#pragma omp parallel num_threads(nthreads)
  {
    /* Slice up the particles between threads. */
    const int npart_per_thread = (parts->npart + nthreads - 1) / nthreads;
    const int tid = omp_get_thread_num();
    const int start = tid * npart_per_thread;
    int end = start + npart_per_thread;
    if (end > parts->npart) end = parts->npart;

    /* Allocate a local output array to avoid races. */
    std::vector<double> local_out_arr(out_size, 0.0);

    /* Loop over the particles assigned to this thread. */
    for (int p = start; p < end; p++) {

      /* Skip if this particle is masked. */
      if (parts->part_is_masked(p)) {
        continue;
      }

      /* Get this particle's weight. */
      const double weight =
          static_cast<double>(parts->get_weight_at<PartReal>(p));

      /* Setup the base cell indices and axis fractions. */
      std::array<int, MAX_GRID_NDIM> part_indices;
      std::array<GridReal, MAX_GRID_NDIM> axis_fracs;
      get_part_ind_frac_cic<PartReal, GridReal>(part_indices, axis_fracs,
                                                grid_props, parts, p);

      /* Compute base linear index for the “low” corner once */
      const int base_lin = get_flat_index(part_indices, dims.data(), ndim);

      /* Now loop over each of the 2^ndim sub-cells. */
      for (int ic = 0; ic < num_sub_cells; ic++) {
        const auto &sc = subcells[ic];

        /* Compute the CIC fraction for this corner */
        GridReal frac = static_cast<GridReal>(1);
        for (int d = 0; d < ndim; d++) {
          frac *= sc.offs[d] ? axis_fracs[d]
                             : (static_cast<GridReal>(1) - axis_fracs[d]);
        }
        if (frac == static_cast<GridReal>(0)) {
          continue;
        }

        /* Accumulate into the thread-local buffer using precomputed offset */
        const int flat_ind = base_lin + sc.linoff;
        local_out_arr[flat_ind] += static_cast<double>(frac) * weight;
      }
    }

    /* Merge local buffer into global array */
#pragma omp critical
    {
      for (int i = 0; i < out_size; i++) {
        out_arr[i] += local_out_arr[i];
      }
    }
  }
}
#endif /* WITH_OPENMP */

/**
 * @brief This calculates the grid weights in each grid cell using a cloud
 *        in cell approach.
 *
 * This is a wrapper which calls the correct function based on the number of
 * threads requested and whether OpenMP is available.
 *
 * All accumulation happens in a double precision scratch buffer which is
 * cast into the output buffer once at the end, so reduced precision outputs
 * don't suffer float32 accumulation error over many particles.
 *
 * @tparam PartReal The floating-point type of the particle arrays.
 * @tparam GridReal The floating-point type of the grid axis arrays.
 * @tparam OutT The floating-point type stored in the output buffer.
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param out_size: The size of the output array.
 * @param out: The output array.
 * @param nthreads: The number of threads to use.
 */
template <typename PartReal, typename GridReal, typename OutT>
void weight_loop_cic(GridProps *grid_props, Particles *parts, int out_size,
                     OutT *out, const int nthreads) {

  tic("weight_loop_cic");

  /* Accumulate in double regardless of the requested output precision. */
  std::vector<double> accum(out_size, 0.0);

  /* Call the correct function for the configuration/number of threads. */

#ifdef WITH_OPENMP

  /* If we have multiple threads and OpenMP we can parallelise. */
  if (nthreads > 1) {
    weight_loop_cic_omp<PartReal, GridReal>(grid_props, parts, out_size,
                                            accum.data(), nthreads);
  }
  /* Otherwise there's no point paying the OpenMP overhead. */
  else {
    weight_loop_cic_serial<PartReal, GridReal>(grid_props, parts,
                                               accum.data());
  }

#else

  (void)nthreads;

  /* We don't have OpenMP, just call the serial version. */
  weight_loop_cic_serial<PartReal, GridReal>(grid_props, parts, accum.data());

#endif

  /* Fold the double precision accumulation into the output buffer. */
  for (int i = 0; i < out_size; i++) {
    out[i] += static_cast<OutT>(accum[i]);
  }

  toc("weight_loop_cic");
}

/**
 * @brief This calculates the grid weights in each grid cell using a nearest
 *        grid point approach.
 *
 * This is the serial version of the function.
 *
 * @tparam PartReal The floating-point type of the particle arrays.
 * @tparam GridReal The floating-point type of the grid axis arrays.
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param out_arr: The double precision accumulation buffer (grid sized).
 */
template <typename PartReal, typename GridReal>
static void weight_loop_ngp_serial(GridProps *grid_props, Particles *parts,
                                   double *out_arr) {

  /* Unpack the grid properties. */
  std::array<int, MAX_GRID_NDIM> dims = grid_props->dims;
  const int ndim = grid_props->ndim;

  /* Loop over particles. */
  for (int p = 0; p < parts->npart; p++) {

    /* Skip masked particles. */
    if (parts->part_is_masked(p)) {
      continue;
    }

    /* Get this particle's weight. */
    const double weight =
        static_cast<double>(parts->get_weight_at<PartReal>(p));

    /* Setup the index array. */
    std::array<int, MAX_GRID_NDIM> part_indices;

    /* Get the grid indices for the particle. */
    get_part_inds_ngp<PartReal, GridReal>(part_indices, grid_props, parts, p);

    /* Unravel the indices. */
    int flat_ind = get_flat_index(part_indices, dims.data(), ndim);

    /* Store the weight. */
    out_arr[flat_ind] += weight;
  }
}

/**
 * @brief This calculates the grid weights in each grid cell using a nearest
 *        grid point approach.
 *
 * This is the parallel version of the function.
 *
 * Each thread accumulates weights into a private local buffer, which is added
 * into the global output array at the end of the thread’s execution.
 *
 * @tparam PartReal The floating-point type of the particle arrays.
 * @tparam GridReal The floating-point type of the grid axis arrays.
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param out_size: The size of the output array.
 * @param out_arr: The double precision accumulation buffer (grid sized).
 * @param nthreads: The number of threads to use.
 */
#ifdef WITH_OPENMP
template <typename PartReal, typename GridReal>
static void weight_loop_ngp_omp(GridProps *grid_props, Particles *parts,
                                int out_size, double *out_arr, int nthreads) {

  /* Unpack the grid properties. */
  std::array<int, MAX_GRID_NDIM> dims = grid_props->dims;
  const int ndim = grid_props->ndim;

#pragma omp parallel num_threads(nthreads)
  {

    /* First let's slice up the particles between the threads. */
    const int npart_per_thread = (parts->npart + nthreads - 1) / nthreads;

    /* Get the thread id. */
    const int tid = omp_get_thread_num();

    /* Get the start and end particle indices for this thread. */
    const int start = tid * npart_per_thread;
    int end = start + npart_per_thread;
    if (end > parts->npart) {
      end = parts->npart;
    }

    /* Allocate a local output array. This avoids race conditions and false
     * sharing. */
    std::vector<double> local_out_arr(out_size, 0.0);

    /* Loop over the assigned particle range. */
    for (int p = start; p < end; ++p) {

      /* Skip masked particles. */
      if (parts->part_is_masked(p)) {
        continue;
      }

      /* Get this particle's weight. */
      const double weight =
          static_cast<double>(parts->get_weight_at<PartReal>(p));

      /* Setup the index array. */
      std::array<int, MAX_GRID_NDIM> part_indices;

      /* Get the grid indices for the particle. */
      get_part_inds_ngp<PartReal, GridReal>(part_indices, grid_props, parts,
                                            p);

      /* Unravel the indices. */
      int flat_ind = get_flat_index(part_indices, dims.data(), ndim);

      /* Store the weight in the thread-local output array. */
      local_out_arr[flat_ind] += weight;
    }

    /* Update the global output array. This is the only critical section. */
#pragma omp critical
    {
      for (int i = 0; i < out_size; i++) {
        out_arr[i] += local_out_arr[i];
      }
    }
  }
}
#endif

/**
 * @brief This calculates the grid weights in each grid cell using a nearest
 *        grid point approach.
 *
 * This is a wrapper which calls the correct function based on the number of
 * threads requested and whether OpenMP is available.
 *
 * All accumulation happens in a double precision scratch buffer which is
 * cast into the output buffer once at the end, so reduced precision outputs
 * don't suffer float32 accumulation error over many particles.
 *
 * @tparam PartReal The floating-point type of the particle arrays.
 * @tparam GridReal The floating-point type of the grid axis arrays.
 * @tparam OutT The floating-point type stored in the output buffer.
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param out_size: The size of the output array.
 * @param out: The output array.
 * @param nthreads: The number of threads to use.
 */
template <typename PartReal, typename GridReal, typename OutT>
void weight_loop_ngp(GridProps *grid_props, Particles *parts, int out_size,
                     OutT *out, const int nthreads) {

  tic("weight_loop_ngp");

  /* Accumulate in double regardless of the requested output precision. */
  std::vector<double> accum(out_size, 0.0);

  /* Call the correct function for the configuration/number of threads. */

#ifdef WITH_OPENMP

  /* If we have multiple threads and OpenMP we can parallelise. */
  if (nthreads > 1) {
    weight_loop_ngp_omp<PartReal, GridReal>(grid_props, parts, out_size,
                                            accum.data(), nthreads);
  }
  /* Otherwise there's no point paying the OpenMP overhead. */
  else {
    weight_loop_ngp_serial<PartReal, GridReal>(grid_props, parts,
                                               accum.data());
  }

#else

  (void)nthreads;

  /* We don't have OpenMP, just call the serial version. */
  weight_loop_ngp_serial<PartReal, GridReal>(grid_props, parts, accum.data());

#endif

  /* Fold the double precision accumulation into the output buffer. */
  for (int i = 0; i < out_size; i++) {
    out[i] += static_cast<OutT>(accum[i]);
  }

  toc("weight_loop_ngp");
}

/* Explicit instantiations to satisfy multi-extension linking. All
 * (PartReal, GridReal, OutT) combinations are required by the dtype
 * dispatchers in sfzh and integrated_spectra. */
#define INSTANTIATE_WEIGHT_LOOPS(PartReal, GridReal, OutT) \
  template void weight_loop_cic<PartReal, GridReal, OutT>( \
      GridProps *, Particles *, int, OutT *, const int);   \
  template void weight_loop_ngp<PartReal, GridReal, OutT>( \
      GridProps *, Particles *, int, OutT *, const int);

INSTANTIATE_WEIGHT_LOOPS(float, float, float)
INSTANTIATE_WEIGHT_LOOPS(float, float, double)
INSTANTIATE_WEIGHT_LOOPS(float, double, float)
INSTANTIATE_WEIGHT_LOOPS(float, double, double)
INSTANTIATE_WEIGHT_LOOPS(double, float, float)
INSTANTIATE_WEIGHT_LOOPS(double, float, double)
INSTANTIATE_WEIGHT_LOOPS(double, double, float)
INSTANTIATE_WEIGHT_LOOPS(double, double, double)

#undef INSTANTIATE_WEIGHT_LOOPS
