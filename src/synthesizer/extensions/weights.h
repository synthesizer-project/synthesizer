/******************************************************************************
 * A C module containing all the weights functions common to all particle
 * spectra extensions.
 *****************************************************************************/
#ifndef WEIGHTS_H_
#define WEIGHTS_H_
/* C includes */
#include <math.h>
#include <string.h>

/* Local includes */
#include "macros.h"
#include "property_funcs.h"

/* Define a struct to hold the useful data a generic callback function will
 * need. */
struct callback_data {
  int *indices;
  int *dims;
  int ndim;
  int particle_index;
  int nlam;
  int npart;
  double fesc;
  double *grid_spectra;
  double *grid_lines;
  double *grid_continuum;
};

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

/* Prototypes */
void get_part_ind_frac_cic(int *part_indices, double *axis_fracs, int *dims,
                           int ndim, double **grid_props, double **part_props,
                           int p);
void weight_loop_cic(struct grid *grid, struct particles *parts, int out_size,
                     void *out, const int nthreads);
void get_part_inds_ngp(int *part_indices, int *dims, int ndim,
                       double **grid_props, double **part_props, int p);
void weight_loop_ngp(struct grid *grid, struct particles *parts, int out_size,
                     void *out, const int nthreads);

#endif // WEIGHTS_H_
