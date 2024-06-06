/******************************************************************************
 * A C module containing all the weights functions common to all particle
 * spectra extensions.
 *****************************************************************************/
/* C includes */
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Local includes */
#include "weights.h"

/**
 * @brief Initialise the weights struct.
 *
 * @param ndim: The number of dimensions.
 * @param dims: The size of each dimension.
 * @param size: The number of non-zero elements.
 */
Weights *init_weights(int ndim, const int *dims, int size) {

  /* Allocate the struct. */
  Weights *weights = malloc(sizeof(Weights));
  if (weights == NULL) {
    return NULL;
  }

  /* Allocate the axis size array. */
  weights->axis_size = malloc(ndim * sizeof(int));
  if (weights->axis_size == NULL) {
    free(weights);
    return NULL;
  }
  memcpy(weights->axis_size, dims, ndim * sizeof(int));

  /* Allocate the indices array. */
  weights->indices = malloc(size * sizeof(int *));
  if (weights->indices == NULL) {
    free(weights->axis_size);
    free(weights);
    return NULL;
  }
  for (int i = 0; i < size; i++) {
    weights->indices[i] = malloc(ndim * sizeof(int));
  }

  /* Allocate the values array. */
  weights->values = malloc(size * sizeof(double));
  if (weights->values == NULL) {
    for (int i = 0; i < size; i++) {
      free(weights->indices[i]);
    }
    free(weights->indices);
    free(weights->axis_size);
    free(weights);
    return NULL;
  }

  /* Set the size. */
  weights->size = size;
  if (weights->values == NULL) {
    for (int i = 0; i < size; i++) {
      free(weights->indices[i]);
    }
    free(weights->indices);
    free(weights->axis_size);
    free(weights);
    return NULL;
  }

  /* Allocate the part indices array. */
  weights->part_indices = malloc(size * sizeof(int));
  if (weights->part_indices == NULL) {
    for (int i = 0; i < size; i++) {
      free(weights->indices[i]);
    }
    free(weights->indices);
    free(weights->axis_size);
    free(weights->values);
    free(weights);
    return NULL;
  }

  return weights;
}

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
 * @brief This calculates the grid weights in each grid cell using a cloud
 *        in cell approach.
 *
 * @param grid_props: An array of the properties along each grid axis.
 * @param part_props: An array of the particle properties, in the same property
 *                    order as grid props.
 * @param part_masses: The mass of the particles.
 * @param dims: The length of each grid dimension.
 * @param ndim: The number of grid dimensions.
 * @param npart: The number of particles.
 */
Weights *weight_loop_cic(const double **grid_props, const double **part_props,
                         const double *part_masses, const int *dims,
                         const int ndim, const int npart) {

  /* Get a weights struct to store what we find. */
  Weights *weights =
      init_weights(ndim, dims, npart * (int)pow(2, (double)ndim));
  if (weights == NULL) {
    return NULL;
  }

  /* Loop over particles. */
  for (int p = 0; p < npart; p++) {

    /* Get this particle's mass. */
    const double mass = part_masses[p];

    /* Setup the index and mass fraction arrays. */
    int part_indices[ndim];
    double axis_fracs[ndim];

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
        part_cell = binary_search(/*low*/ 0, /*high*/ dims[dim] - 1, grid_prop,
                                  part_val);

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

    /* To combine fractions we will need an array of dimensions for the
     * subset. These are always two in size, one for the low and one for high
     * grid point. */
    int sub_dims[ndim];
    for (int idim = 0; idim < ndim; idim++) {
      sub_dims[idim] = 2;
    }

    /* Now loop over this collection of cells collecting and setting their
     * weights. */
    for (int icell = 0; icell < (int)pow(2, (double)ndim); icell++) {

      /* Set up some index arrays we'll need. */
      int subset_ind[ndim];
      int frac_ind[ndim];

      /* Get the multi-dimensional version of icell. */
      get_indices_from_flat(icell, ndim, sub_dims, subset_ind);

      /* Multiply all contributing fractions and get the fractions index
       * in the grid. */
      double frac = 1;
      for (int idim = 0; idim < ndim; idim++) {
        if (subset_ind[idim] == 0) {
          frac *= (1 - axis_fracs[idim]);
          frac_ind[idim] = part_indices[idim] - 1;
        } else {
          frac *= axis_fracs[idim];
          frac_ind[idim] = part_indices[idim];
        }
      }

      /* Early skip for cells contributing a 0 fraction. */
      if (frac <= 0)
        continue;

      /* Store the grid indices. */
      for (int idim = 0; idim < ndim; idim++) {
        weights->indices[p * (int)pow(2, (double)ndim) + icell][idim] =
            frac_ind[idim];
      }

      /* Store the weight. */
      weights->values[p * (int)pow(2, (double)ndim) + icell] = mass * frac;

      /* Store the particle index. */
      weights->part_indices[p * (int)pow(2, (double)ndim) + icell] = p;
    }
  }

  return weights;
}

/**
 * @brief This calculates the grid weights in each grid cell using a nearest
 *        grid point approach.
 *
 * @param grid_props: An array of the properties along each grid axis.
 * @param part_props: An array of the particle properties, in the same property
 *                    order as grid props.
 * @param part_masses: The mass of the particles.
 * @param dims: The length of each grid dimension.
 * @param ndim: The number of grid dimensions.
 * @param npart: The number of particles.
 */
Weights *weight_loop_ngp(const double **grid_props, const double **part_props,
                         const double *part_masses, const int *dims,
                         const int ndim, const int npart) {

  /* Get a weights struct to store what we find. */
  Weights *weights = init_weights(ndim, dims, npart);
  if (weights == NULL) {
    return NULL;
  }

  /* Loop over particles. */
  for (int p = 0; p < npart; p++) {

    /* Get this particle's mass. */
    const double mass = part_masses[p];

    /* Setup the index array. */
    int part_indices[ndim];

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
        part_cell = binary_search(/*low*/ 0, /*high*/ dims[dim] - 1, grid_prop,
                                  part_val);
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

    /* Store the grid indices. */
    for (int idim = 0; idim < ndim; idim++) {
      weights->indices[p][idim] = part_indices[idim];
    }

    /* Store the weight. */
    weights->values[p] = mass;

    /* Store the particle index. */
    weights->part_indices[p] = p;
  }

  return weights;
}
