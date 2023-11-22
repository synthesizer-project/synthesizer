/******************************************************************************
 * A C module containing all the weights functions common to all particle
 * spectra extensions.
 *****************************************************************************/
/* C includes */
#include <math.h>

/**
 * @brief Compute an ndimensional index from a flat index.
 *
 * @param flat_ind: The flattened index to unravel.
 * @param ndim: The number of dimensions for the unraveled index.
 * @param dims: The size of each dimension.
 * @param indices: The output N-dimensional indices.
 */
void get_indices_from_flat(int flat_ind, int ndim, const int *dims,
                           int *indices) {

  /* Loop over indices calculating each one. */
  for (int i = 0; i < ndim; i++) {
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
int get_flat_index(const int *multi_index, const int *dims, const int ndims) {
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
int binary_search(int low, int high, const double *arr, const double val) {

  /* While we don't have a pair of adjacent indices. */
  int diff = high - low;
  while (diff > 1) {

    /* Define the midpoint. */
    int mid = low + floor(diff / 2);

    /* Where is the midpoint relative to the value? */
    if (val > arr[mid]) {
      low = mid;
    } else {
      high = mid;
    }

    /* Compute the new range. */
    diff = high - low;
  }

  /* Return the nearest grid point. */
  if ((val - arr[low]) < (arr[high] - val)) {
    return high;
  } else {
    return low;
  }
}

/**
 * @brief This calculates the grid weights in each grid cell using a cloud
 *        in cell approach.
 *
 * @param grid_props: An array of the properties along each grid axis.
 * @param part_props: An array of the particle properties, in the same property
 *                    order as grid props.
 * @param mass: The mass of the current particle.
 * @param weights: The weight of each grid point.
 * @param dims: The length of each grid dimension.
 * @param ndim: The number of grid dimensions.
 * @param p: Index of the current particle.
 */
void weight_loop_cic(const double **grid_props, const double **part_props,
                     const double mass, double *weights, const int *dims,
                     const int ndim, const int p) {

  /* Setup the index and mass fraction arrays. */
  int frac_indices[(int)pow(2, (double)ndim)][ndim];
  double fracs[(int)pow(2, (double)ndim)];

  /* Set up the fractions. */
  for (int icell = 0; icell < (int)pow(2, (double)ndim); icell++) {
    fracs[icell] = 1;
  }

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
      frac = 1;

    } else if (part_val > grid_prop[dims[dim] - 1]) {

      /* Use the grid edge. */
      part_cell = dims[dim] - 1;
      frac = 1;

    } else {

      /* Find the grid index corresponding to this particle property. */
      part_cell =
          binary_search(/*low*/ 0, /*high*/ dims[dim] - 1, grid_prop, part_val);

      /* Make sure we have the cell containing the particle. */
      if (grid_prop[part_cell] < part_val) {
        part_cell += 1;
      }

      /* Calculate the fraction. Note, here we do the cell containing the
       * particle, the cell below is calculated from this fraction. */
      frac = (part_val - grid_prop[part_cell]) /
             (grid_prop[part_cell] - grid_prop[part_cell - 1]);
    }

    /* Set the fractions. */
    fracs[dim * 2 + 1] *= frac;
    fracs[dim * 2] *= 1 - frac;

    /* Set these indices. */
    for (int jdim = 0; jdim < ndim; jdim++) {
      frac_indices[jdim * 2][dim] = part_cell - 1;
      frac_indices[jdim * 2 + 1][dim] = part_cell;
    }
  }

  /* Normalise the fractions. */
  double sum = 0;
  for (int icell = 0; icell < (int)pow(2, (double)ndim); icell++) {
    sum += fracs[icell];
  }
  for (int icell = 0; icell < (int)pow(2, (double)ndim); icell++) {
    fracs[icell] /= sum;
  }

  /* Now loop over this collection of cells collecting and setting their
   * weights. */
  for (int icell = 0; icell < (int)pow(2, (double)ndim); icell++) {

    /* Early skip for cells contributing a 0 fraction. */
    if (fracs[icell] <= 0)
      continue;

    /* We have a contribution, get the flattened index into the grid array. */
    const int weight_ind = get_flat_index(frac_indices[icell], dims, ndim);

    /* Add the weight. */
    weights[weight_ind] += mass * fracs[icell];
  }
}

/**
 * @brief This calculates the grid weights in each grid cell using a nearest
 *        grid point approach.
 *
 * @param grid_props: An array of the properties along each grid axis.
 * @param part_props: An array of the particle properties, in the same property
 *                    order as grid props.
 * @param mass: The mass of the current particle.
 * @param weights: The weight of each grid point.
 * @param dims: The length of each grid dimension.
 * @param ndim: The number of grid dimensions.
 * @param p: Index of the current particle.
 */
void weight_loop_ngp(const double **grid_props, const double **part_props,
                     const double mass, double *weights, const int *dims,
                     const int ndim, const int p) {

  /* Setup the index array. */
  int part_indices[ndim];

  /* Loop over dimensions finding the indicies. */
  for (int dim = 0; dim < ndim; dim++) {

    /* Get this array of grid properties for this dimension */
    const double *grid_prop = grid_props[dim];

    /* Get this particle property. */
    const double part_val = part_props[dim][p];

    /* Here we need to handle if we are outside the range of values. If so
     * there's no point in searching and we return the edge nearest to the
     * value. */
    int part_cell;
    if (part_val <= grid_prop[0]) {

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

    /* Set the index. */
    part_indices[dim] = part_cell;
  }

  /* Get the weight's index. */
  const int weight_ind = get_flat_index(part_indices, dims, ndim);

  /* Add the weight. */
  weights[weight_ind] += mass;
}
