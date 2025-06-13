/******************************************************************************
 * A C module containing all the weights functions common to all particle
 * spectra extensions.
 *****************************************************************************/
#ifndef WEIGHTS_H_
#define WEIGHTS_H_
/* C includes */
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Local includes */
#include "grid_props.h"
#include "index_utils.h"
#include "macros.h"
#include "part_props.h"
#include "property_funcs.h"

/**
 * @brief Performs a binary search for the index of an array corresponding to
 * a value.
 *
 * @param low: The initial low index (probably beginning of array).
 * @param high: The initial high index (probably size of array).
 * @param arr: The array to search in.
 * @param val: The value to search for.
 */
static inline int binary_search(int low, int high, const double *arr,
                                const double val) {

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
 * @param part_indices: The output array of base (lower) grid indices.
 * @param axis_fracs: The output array of fractional distances to upper grid
 * cell.
 * @param grid_props: The properties of the grid.
 * @param part_props: The properties of the particle.
 * @param p: The particle index.
 */
static inline void
get_part_ind_frac_cic(std::array<int, MAX_GRID_NDIM> &part_indices,
                      std::array<double, MAX_GRID_NDIM> &axis_fracs,
                      GridProps *grid_props, Particles *parts, int p) {

  /* Loop over dimensions, finding the mass weightings and indices. */
  for (int dim = 0; dim < grid_props->ndim; dim++) {

    /* Get the array of grid coordinates for this dimension. */
    const double *grid_axis = grid_props->get_axis(dim);
    const int dim_size = grid_props->dims[dim];

    /* Get the particle's value along this dimension. */
    const double part_val = parts->get_part_prop_at(dim, p);

    int lower, upper;
    double frac;

    /* Handle values outside the grid bounds. Clamp to edges. */
    if (part_val <= grid_axis[0]) {

      /* Particle lies below the lowest grid edge. Clamp to first cell. */
      lower = 0;
      upper = 1;
      frac = 0.0;

    } else if (part_val >= grid_axis[dim_size - 1]) {

      /* Particle lies beyond the last grid edge. Clamp to final cell. */
      lower = dim_size - 2;
      upper = dim_size - 1;
      frac = 1.0;

    } else {

      /* Find the upper cell index such that:
       *   grid_axis[lower] <= part_val < grid_axis[upper]
       */
      upper =
          binary_search(/*low=*/0, /*high=*/dim_size - 1, grid_axis, part_val);
      lower = upper - 1;

      /* Compute the linear fraction between the two grid points. */
      const double low = grid_axis[lower];
      const double high = grid_axis[upper];
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
 * @param part_indices: The output array of nearest grid point indices.
 * @param grid_props: The properties of the grid.
 * @param part_props: The properties of the particle.
 * @param p: The particle index.
 */
static inline void
get_part_inds_ngp(std::array<int, MAX_GRID_NDIM> &part_indices,
                  GridProps *grid_props, Particles *parts, int p) {

  /* Loop over dimensions finding the indices. */
  for (int dim = 0; dim < grid_props->ndim; dim++) {

    /* Get this array of grid coordinate values for this dimension. */
    const double *grid_axis = grid_props->get_axis(dim);
    const int dim_size = grid_props->dims[dim];

    /* Get the particle's coordinate along this axis. */
    const double part_val = parts->get_part_prop_at(dim, p);

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

/* Prototypes */
void weight_loop_cic(GridProps *grid, Particles *parts, int out_size, void *out,
                     const int nthreads);
void weight_loop_ngp(GridProps *grid, Particles *parts, int out_size, void *out,
                     const int nthreads);

#endif // WEIGHTS_H_
