/******************************************************************************
 * A C module containing all the weights functions common to all particle
 * spectra extensions.
 *****************************************************************************/
#ifndef WEIGHTS_H_
#define WEIGHTS_H_
/* C includes */
#include <cmath>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

/* Local includes */
#include "grid_props.h"
#include "index_utils.h"
#include "macros.h"
#include "part_props.h"
#include "pop_props.h"
#include "property_funcs.h"

/**
 * @brief Performs a binary search for the index of an array corresponding to
 * a value.
 *
 * @tparam Real The floating-point type.
 * @param low: The initial low index (probably beginning of array).
 * @param high: The initial high index (probably size of array).
 * @param arr: The array to search in.
 * @param val: The value to search for.
 */
template <typename Real>
static inline int binary_search(int low, int high, const Real *arr,
                                const Real val) {

  /* While we don't have a pair of adjacent indices. */
  int diff = high - low;
  while (diff > 1) {

    /* Define the midpoint. */
    int mid = low + (int)floor(diff / 2);

    /* Where is the midpoint relative to the value? */
    if (val >= arr[mid]) {
      low = mid;
    } else {
      high = mid;
    }

    /* Compute the new range. */
    diff = high - low;
  }

  return high;
}

/**
 * @brief Get the grid indices of a particle based on its properties.
 *
 * This will also calculate the fractions of the particle's mass in each grid
 * cell. (Unnecessary for NGP, but required for CIC.)
 *
 * The grid axis arrays are read at the grid dtype and the particle value at
 * the particle dtype, so mixed precision inputs never reinterpret a buffer
 * at the wrong width. All fraction arithmetic happens at the grid dtype.
 *
 * @tparam PartReal The floating-point type of the particle arrays.
 * @tparam GridReal The floating-point type of the grid axis arrays.
 * @param part_indices: The output array of base (lower) grid indices.
 * @param axis_fracs: The output array of fractional distances to upper grid
 * cell.
 * @param grid_props: The properties of the grid.
 * @param part_props: The properties of the particle.
 * @param p: The particle index.
 */
template <typename PartReal, typename GridReal>
static inline void get_part_ind_frac_cic(
    std::array<int, MAX_GRID_NDIM> &part_indices,
    std::array<GridReal, MAX_GRID_NDIM> &axis_fracs, GridProps *grid_props,
    Particles *parts, int p) {

  /* Loop over dimensions, finding the mass weightings and indices. */
  for (int dim = 0; dim < grid_props->ndim; dim++) {

    /* Get the array of grid coordinates for this dimension. */
    const GridReal *grid_axis = grid_props->get_axis<GridReal>(dim);
    const int dim_size = grid_props->dims[dim];

    /* Get the particle's value along this dimension. */
    const GridReal part_val =
        static_cast<GridReal>(parts->get_part_prop_at<PartReal>(dim, p));

    int lower, upper;
    GridReal frac;

    /* Handle values outside the grid bounds. Clamp to edges. */
    if (part_val <= grid_axis[0]) {

      /* Particle lies below the lowest grid edge. Clamp to first cell. */
      lower = 0;
      upper = 1;
      frac = static_cast<GridReal>(0);

    } else if (part_val >= grid_axis[dim_size - 1]) {

      /* Particle lies beyond the last grid edge. Clamp to final cell. */
      lower = dim_size - 2;
      upper = dim_size - 1;
      frac = static_cast<GridReal>(1);

    } else {

      /* Find the upper cell index such that:
       *   grid_axis[lower] <= part_val < grid_axis[upper]
       */
      upper =
          binary_search(/*low=*/0, /*high=*/dim_size - 1, grid_axis, part_val);
      lower = upper - 1;

      /* Compute the linear fraction between the two grid points. */
      const GridReal low = grid_axis[lower];
      const GridReal high = grid_axis[upper];
      frac = (part_val - low) / (high - low);
    }

    /* Set the base (lower) index for CIC. */
    part_indices[dim] = lower;

    /* Set the fraction toward the upper cell. */
    axis_fracs[dim] = frac;
  }
}

/**
 * @brief Get the nearest grid indices of a particle based on its properties.
 *
 * For each axis, this finds the grid point closest to the particle's position.
 *
 * @tparam PartReal The floating-point type of the particle arrays.
 * @tparam GridReal The floating-point type of the grid axis arrays.
 * @param part_indices: The output array of nearest grid point indices.
 * @param grid_props: The properties of the grid.
 * @param part_props: The properties of the particle.
 * @param p: The particle index.
 */
template <typename PartReal, typename GridReal>
static inline void get_part_inds_ngp(
    std::array<int, MAX_GRID_NDIM> &part_indices, GridProps *grid_props,
    Particles *parts, int p) {

  /* Loop over dimensions finding the indices. */
  for (int dim = 0; dim < grid_props->ndim; dim++) {

    /* Get this array of grid coordinate values for this dimension. */
    const GridReal *grid_axis = grid_props->get_axis<GridReal>(dim);
    const int dim_size = grid_props->dims[dim];

    /* Get the particle's coordinate along this axis. */
    const GridReal part_val =
        static_cast<GridReal>(parts->get_part_prop_at<PartReal>(dim, p));

    int part_cell;

    /* Handle pathological grids with only 1 point along this axis. */
    if (dim_size == 1) {
      part_indices[dim] = 0;
      continue;
    }

    /* Clamp particle to the grid range if outside bounds. */
    if (part_val <= grid_axis[0]) {
      part_cell = 0;

    } else if (part_val >= grid_axis[dim_size - 1]) {
      part_cell = dim_size - 1;

    } else {
      /* Find the upper bounding grid cell. */
      part_cell =
          binary_search(/*low=*/0, /*high=*/dim_size - 1, grid_axis, part_val);
    }

    /* Choose the closest grid point (lower or upper) based on distance. */
    if (part_cell == 0) {
      /* Handle lower edge: can't access part_cell - 1 */
      part_indices[dim] = 0;

    } else if ((part_val - grid_axis[part_cell - 1]) <
               (grid_axis[part_cell] - part_val)) {
      part_indices[dim] = part_cell - 1;
    } else {
      part_indices[dim] = part_cell;
    }
  }
}

/**
 * @brief Compute the mean of log10(x) over [lo, hi] for x uniform in [lo, hi].
 *
 * This is (hi log10(hi) - lo log10(lo)) / (hi - lo) - 1 / ln(10), written in
 * terms of r = hi / lo so it stays accurate for narrow intervals, where the
 * direct form cancels catastrophically.
 *
 * @param lo: The lower bound (must be > 0).
 * @param hi: The upper bound (must be > lo).
 *
 * @return The mean of log10(x) over the interval.
 */
static inline double mean_log10_uniform(double lo, double hi) {
  const double d = hi / lo - 1.0;
  double excess;
  if (d < 1e-4) {
    /* Series expansion of (r ln r - d) / d for small d. */
    excess = d / 2.0 - d * d / 6.0 + d * d * d / 12.0;
  } else {
    excess = ((1.0 + d) * std::log1p(d) - d) / d;
  }
  return std::log10(lo) + excess / M_LN10;
}

/**
 * @brief Add a contribution to a grid point in a bin's fraction list.
 *
 * The contributions arrive in ascending grid index order, so a repeated index
 * can only be the last one added and is merged into it.
 *
 * @param inds: The grid indices this bin contributes to.
 * @param fracs: The fraction of the bin's mass given to each index.
 * @param ind: The grid index to add.
 * @param frac: The fraction to add.
 */
static inline void add_bin_frac(std::vector<int> &inds,
                                std::vector<double> &fracs, int ind,
                                double frac) {
  if (frac <= 0.0) {
    return;
  }
  if (!inds.empty() && inds.back() == ind) {
    fracs.back() += frac;
  } else {
    inds.push_back(ind);
    fracs.push_back(frac);
  }
}

/**
 * @brief Get the grid points and mass fractions of a bin along one axis
 *        using a bin in cell (BIC) approach.
 *
 * BIC is the extension of cloud in cell (CIC) to bins of finite width. Each
 * grid point carries a hat function, linear in the grid's coordinate (log10
 * or linear), equal to 1 at the point and 0 at its neighbours, so the
 * spectra are assumed to vary linearly between neighbouring points exactly as
 * in CIC. A bin's mass is spread uniformly over its extent in linear
 * (physical) units, and the fraction given to each grid point is the mean of
 * that point's hat function over the bin. A zero width bin therefore reduces
 * exactly to CIC.
 *
 * Mass outside the grid is clamped onto the edge points, as in CIC.
 *
 * NOTE: The uniform distribution within each bin could be generalised to
 * other distributions in the future, but there is no current need.
 *
 * @tparam GridReal The floating-point type of the grid axis arrays.
 * @param grid_axis: The grid point coordinates along this axis.
 * @param dim_size: The number of grid points along this axis.
 * @param log_axis: Are the grid coordinates log10 of the bin coordinates?
 * @param lo: The lower edge of the bin (linear units).
 * @param hi: The upper edge of the bin (linear units).
 * @param inds: The output grid indices (cleared first).
 * @param fracs: The output mass fractions for each index (cleared first).
 */
template <typename GridReal>
static inline void get_pop_bin_fracs_bic(const GridReal *grid_axis,
                                         const int dim_size,
                                         const bool log_axis, const double lo,
                                         const double hi,
                                         std::vector<int> &inds,
                                         std::vector<double> &fracs) {
  inds.clear();
  fracs.clear();

  /* Handle pathological grids with only 1 point along this axis. */
  if (dim_size == 1) {
    inds.push_back(0);
    fracs.push_back(1.0);
    return;
  }

  /* Convert a grid coordinate into linear units. */
  auto to_linear = [&](int k) {
    const double g = static_cast<double>(grid_axis[k]);
    return log_axis ? std::pow(10.0, g) : g;
  };

  /* A zero width bin is a point, so this is just CIC. */
  if (hi <= lo) {

    /* Points at or below zero on a log axis are below the grid. */
    if (log_axis && lo <= 0.0) {
      inds.push_back(0);
      fracs.push_back(1.0);
      return;
    }
    const double u = log_axis ? std::log10(lo) : lo;
    if (u <= static_cast<double>(grid_axis[0])) {
      inds.push_back(0);
      fracs.push_back(1.0);
    } else if (u >= static_cast<double>(grid_axis[dim_size - 1])) {
      inds.push_back(dim_size - 1);
      fracs.push_back(1.0);
    } else {
      const int upper = binary_search(/*low=*/0, /*high=*/dim_size - 1,
                                      grid_axis, static_cast<GridReal>(u));
      const double g_lo = static_cast<double>(grid_axis[upper - 1]);
      const double g_hi = static_cast<double>(grid_axis[upper]);
      const double frac = (u - g_lo) / (g_hi - g_lo);
      add_bin_frac(inds, fracs, upper - 1, 1.0 - frac);
      add_bin_frac(inds, fracs, upper, frac);
    }
    return;
  }

  const double width = hi - lo;
  const double x_first = to_linear(0);
  const double x_last = to_linear(dim_size - 1);

  /* Mass below the first grid point is clamped onto it. */
  if (lo < x_first) {
    add_bin_frac(inds, fracs, 0, (std::min(hi, x_first) - lo) / width);
  }

  /* Find the first grid interval overlapping the bin. */
  int k = 0;
  if (lo > x_first) {
    const double u_lo = log_axis ? std::log10(lo) : lo;
    k = binary_search(/*low=*/0, /*high=*/dim_size - 1, grid_axis,
                      static_cast<GridReal>(u_lo)) -
        1;
  }

  /* Split the mass between the two points bounding each overlapped grid
   * interval, using the mean of the hat function over the overlap. */
  double x_k = to_linear(k);
  while (k < dim_size - 1 && x_k < hi) {
    const double x_k1 = to_linear(k + 1);
    const double s = std::max(lo, x_k);
    const double t = std::min(hi, x_k1);
    if (t > s) {
      const double g_k = static_cast<double>(grid_axis[k]);
      const double g_k1 = static_cast<double>(grid_axis[k + 1]);
      const double u_mean =
          log_axis ? mean_log10_uniform(s, t) : 0.5 * (s + t);
      double frac = (u_mean - g_k) / (g_k1 - g_k);
      frac = std::min(1.0, std::max(0.0, frac));
      const double overlap = (t - s) / width;
      add_bin_frac(inds, fracs, k, overlap * (1.0 - frac));
      add_bin_frac(inds, fracs, k + 1, overlap * frac);
    }
    x_k = x_k1;
    k++;
  }

  /* Mass above the last grid point is clamped onto it. */
  if (hi > x_last) {
    add_bin_frac(inds, fracs, dim_size - 1,
                 (hi - std::max(lo, x_last)) / width);
  }
}

/* Typed kernel entry points. Callers must provide validated buffers and
 * dispatch on the particle/grid/output dtypes (see dispatch_float in
 * python_to_cpp.h). Accumulation always happens in double precision
 * internally; the result is folded into the output buffer at OutT. */
template <typename PartReal, typename GridReal, typename OutT>
void weight_loop_cic(GridProps *grid, Particles *parts, int out_size,
                     OutT *out, const int nthreads);

template <typename PartReal, typename GridReal, typename OutT>
void weight_loop_ngp(GridProps *grid, Particles *parts, int out_size,
                     OutT *out, const int nthreads);

/* Typed bin in cell (BIC) entry points for parametric populations. The
 * integrated version sums every population into a single grid sized output,
 * the per population version writes one grid sized row per population. */
template <typename PopReal, typename GridReal, typename OutT>
void weight_loop_bic(GridProps *grid, Populations *pops, int out_size,
                     OutT *out, const int nthreads);

template <typename PopReal, typename GridReal, typename OutT>
void weight_loop_bic_per_population(GridProps *grid, Populations *pops,
                                    OutT *out, const int nthreads);

#endif  // WEIGHTS_H_
