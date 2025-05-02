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
#include "macros.h"
#include "property_funcs.h"

/**
 * @brief Compute an ndimensional index from a flat index.
 *
 * @param flat_ind: The flattened index to unravel.
 * @param ndim: The number of dimensions for the unraveled index.
 * @param dims: The size of each dimension.
 * @param indices: The output N-dimensional indices.
 */
static inline void get_indices_from_flat(int flat_ind, int ndim,
                                         const int *dims, int *indices) {

  /* Loop over indices calculating each one. */
  for (int i = ndim - 1; i > -1; i--) {
    indices[i] = flat_ind % dims[i];
    flat_ind /= dims[i];
  }
}

/**
 * @brief Compute a flat grid index based on the grid dimensions.
 *
 * @param multi_index: An array of N-dimensional indices.
 * @param dims: The length of each dimension.
 * @param ndim: The number of dimensions.
 */
static inline int get_flat_index(const int *multi_index, const int *dims,
                                 const int ndims) {
  int index = 0, stride = 1;
  for (int i = ndims - 1; i >= 0; i--) {
    index += stride * multi_index[i];
    stride *= dims[i];
  }

  return index;
}

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
 * @brief Get the grid indices of a particle based on it's properties.
 *
 * This will also calculate the fractions of the particle's mass in each grid
 * cell. (uncessary for NGP below)
 *
 * @param part_indices: The output array of indices.
 * @param axis_fracs: The output array of fractions.
 * @param dims: The size of each dimension.
 * @param ndim: The number of dimensions.
 * @param grid_props: The properties of the grid.
 * @param part_props: The properties of the particle.
 * @param p: The particle index.
 */
static inline void get_part_ind_frac_cic(int *part_indices, double *axis_fracs,
                                         int *dims, int ndim,
                                         double **grid_props,
                                         double **part_props, int p) {

  /* Loop over dimensions finding the mass weightings and indicies. */
  for (int dim = 0; dim < ndim; dim++) {

    /* Get this array of grid properties for this dimension */
    const double *grid_prop = grid_props[dim];

    /* Get this particle property. */
    const double part_val = part_props[dim][p];

    /* Here we need to handle if we are outside the range of values. If so
     * there's no point in searching and we return the edge nearest to the
     * value. */
    int part_cell;
    double frac;
    if (part_val <= grid_prop[0]) {

      /* Use the grid edge. */
      part_cell = 0;
      frac = 0;

    } else if (part_val > grid_prop[dims[dim] - 1]) {

      /* Use the grid edge. */
      part_cell = dims[dim] - 1;
      frac = 1;

    } else {

      /* Find the grid index corresponding to this particle property. */
      part_cell =
          binary_search(/*low*/ 0, /*high*/ dims[dim] - 1, grid_prop, part_val);

      /* Calculate the fraction. Note, here we do the "low" cell, the cell
       * above is calculated from this fraction. */
      frac = (grid_prop[part_cell] - part_val) /
             (grid_prop[part_cell] - grid_prop[part_cell - 1]);
    }

    /* Set the fraction for this dimension. */
    axis_fracs[dim] = (1 - frac);

    /* Set this index. */
    part_indices[dim] = part_cell;
  }
}

/**
 * @brief Get the grid indices of a particle based on it's properties.
 *
 * @param part_indices: The output array of indices.
 * @param dims: The size of each dimension.
 * @param ndim: The number of dimensions.
 * @param grid_props: The properties of the grid.
 * @param part_props: The properties of the particle.
 * @param p: The particle index.
 */
static inline void get_part_inds_ngp(int *part_indices, int *dims, int ndim,
                                     double **grid_props, double **part_props,
                                     int p) {

  /* Loop over dimensions finding the indicies. */
  for (int dim = 0; dim < ndim; dim++) {

    /* Get this array of grid properties for this dimension */
    const double *grid_prop = grid_props[dim];

    /* Get this particle property. */
    const double part_val = part_props[dim][p];

    /* Handle weird grids with only 1 grid cell on a particuar axis.  */
    int part_cell;
    if (dims[dim] == 1) {
      part_cell = 0;
    }

    /* Here we need to handle if we are outside the range of values. If so
     * there's no point in searching and we return the edge nearest to the
     * value. */
    else if (part_val <= grid_prop[0]) {

      /* Use the grid edge. */
      part_cell = 0;

    } else if (part_val > grid_prop[dims[dim] - 1]) {

      /* Use the grid edge. */
      part_cell = dims[dim] - 1;

    } else {

      /* Find the grid index corresponding to this particle property. */
      part_cell =
          binary_search(/*low*/ 0, /*high*/ dims[dim] - 1, grid_prop, part_val);
    }

    /* Set the index to the closest grid point either side of part_val. */
    if (part_cell == 0) {
      /* Handle the case where part_cell - 1 doesn't exist. */
      part_indices[dim] = part_cell;
    } else if ((part_val - grid_prop[part_cell - 1]) <
               (grid_prop[part_cell] - part_val)) {
      part_indices[dim] = part_cell - 1;
    } else {
      part_indices[dim] = part_cell;
    }
  }
}

/* Prototypes */
void weight_loop_cic(struct grid *grid, struct particles *parts, int out_size,
                     void *out, const int nthreads);
void weight_loop_ngp(struct grid *grid, struct particles *parts, int out_size,
                     void *out, const int nthreads);

#endif // WEIGHTS_H_
