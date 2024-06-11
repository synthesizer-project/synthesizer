/******************************************************************************
 * A C module containing all the weights functions common to all particle
 * spectra extensions.
 *****************************************************************************/
/* C includes */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Python includes */
#include <Python.h>

/* Local includes */
#include "weights.h"

/* Optional openmp include. */
#ifdef WITH_OPENMP
#include <omp.h>
#endif

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
 * This is the serial version of the function.
 *
 * @param grid: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param out_size: The size of the output array. (This will be allocated within
 *                  this function.)
 * @param func: The callback function to be called.
 */
static double *weight_loop_cic_serial(struct grid *grid,
                                      struct particles *parts, int out_size,
                                      WeightFunc func) {

  /* Unpack the grid properties. */
  const int *dims = grid->dims;
  const int ndim = grid->ndim;
  const double **grid_props = grid->props;

  /* Unpack the particles properties. */
  const double *part_masses = parts->mass;
  const double **part_props = parts->props;
  const double *fesc = parts->fesc;
  const int npart = parts->npart;

  /* Allocate the output. */
  double *out = (double *)malloc(out_size * sizeof(double));
  if (out == NULL) {
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for output.");
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

      /* Define the callback data. */
      struct callback_data data = {
          .indices = frac_ind,
          .dims = dims,
          .ndim = ndim,
          .particle_index = p,
          .nlam = grid->nlam,
          .npart = npart,
          .fesc = fesc != NULL ? fesc[p] : 0,
          .grid_spectra = grid->spectra,
          .grid_lines = grid->lines,
          .grid_continuum = grid->continuum,
      };

      /* Call the callback function if we have something to do. */
      func(frac * mass, &data, out);
    }
  }
  return out;
}

/**
 * @brief This calculates the grid weights in each grid cell using a cloud
 *        in cell approach.
 *
 * This is the parallel version of the function.
 *
 * @param grid: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param out_size: The size of the output array. (This will be allocated within
 *                  this function.)
 * @param func: The callback function to be called.
 */
static double *weight_loop_cic_omp(struct grid *grid, struct particles *parts,
                                   int out_size, WeightFunc func) {
#ifdef WITH_OPENMP

  /* Define the output array we'll unpack everything into. */
  double *out = (double *)malloc(out_size * sizeof(double));
  if (out == NULL) {
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for output.");
    return NULL;
  }
  bzero(out, out_size * sizeof(double));

#pragma omp parallel
  {
    /* Unpack the grid properties. */
    const int *dims = grid->dims;
    const int ndim = grid->ndim;
    const double **grid_props = grid->props;

    /* Unpack the particles properties. */
    const double *part_masses = parts->mass;
    const double **part_props = parts->props;
    const double *fesc = parts->fesc;
    const int npart = parts->npart;

    /* Allocate the output. */
    double *out_per_thread = (double *)malloc(out_size * sizeof(double));
    if (out == NULL) {
      PyErr_SetString(PyExc_MemoryError,
                      "Failed to allocate memory for output.");
      return NULL;
    }
    bzero(out_per_thread, out_size * sizeof(double));

    /* Loop over particles. */
#pragma omp for
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
          part_cell = binary_search(/*low*/ 0, /*high*/ dims[dim] - 1,
                                    grid_prop, part_val);

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

        /* Define the callback data. */
        struct callback_data data = {
            .indices = frac_ind,
            .dims = dims,
            .ndim = ndim,
            .particle_index = p,
            .nlam = grid->nlam,
            .npart = npart,
            .fesc = fesc != NULL ? fesc[p] : 0,
            .grid_spectra = grid->spectra,
            .grid_lines = grid->lines,
            .grid_continuum = grid->continuum,
        };

        /* Call the callback function if we have something to do. */
        func(frac * mass, &data, out_per_thread);
      }
    }

    /* Use reduction to collect everything into the output array */
#pragma omp for reduction(+ : out[ : out_size])
    for (int i = 0; i < out_size; i++) {
      out[i] += out_per_thread[i];
    }

    /* Free the per-thread output. */
    free(out_per_thread);
  }
  return out;
#else
  /* We shouldn't be able to get here. */
  PyErr_SetString(PyExc_RuntimeError,
                  "OpenMP not enabled but in an OpenMP function.");
  return NULL;
#endif
}

/**
 * @brief This calculates the grid weights in each grid cell using a cloud
 *        in cell approach.
 *
 * This is a wrapper which calls the correct function based on the number of
 * threads requested and whether OpenMP is available.
 *
 * @param grid: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param out_size: The size of the output array. (This will be allocated
 * within this function.)
 * @param func: The callback function to be called.
 * @param nthreads: The number of threads to use.
 */
double *weight_loop_cic(struct grid *grid, struct particles *parts,
                        int out_size, WeightFunc func, const int nthreads) {

  /* Declare the output. This will be allocated when needed. */
  double *out;

  /* Call the correct function for the configuration/number of threads. */

#ifdef WITH_OPENMP

  /* If we have multiple threads and OpenMP we can parallelise. */
  if (nthreads > 1) {
    omp_set_num_threads(nthreads);
    out = weight_loop_cic_omp(grid, parts, out_size, func);
  }
  /* Otherwise there's no point paying the OpenMP overhead. */
  else {
    out = weight_loop_cic_serial(grid, parts, out, func);
  }

#else

  /* We don't have OpenMP, just call the serial version. */
  out = weight_loop_cic_serial(grid, parts, out, func);

#endif
  /* Return what we've done, any error message will have been set at the
   * point of the error. */
  if (out == NULL) {
    return NULL;
  } else {
    return out;
  }
}

/**
 * @brief This calculates the grid weights in each grid cell using a nearest
 *       grid point approach.
 *
 * This is the serial version of the function.
 *
 * @param grid: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param out_size: The size of the output array. (This will be allocated
 * within this function.)
 * @param func: The callback function to be called.
 */
static double *weight_loop_ngp_serial(struct grid *grid,
                                      struct particles *parts, int out_size,
                                      WeightFunc func) {

  /* Allocate the output. */
  double *out = (double *)malloc(out_size * sizeof(double));
  if (out == NULL) {
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for output.");
    return NULL;
  }

  /* Unpack the grid properties. */
  const int *dims = grid->dims;
  const int ndim = grid->ndim;
  const double **grid_props = grid->props;

  /* Unpack the particles properties. */
  const double *part_masses = parts->mass;
  const double **part_props = parts->props;
  const double *fesc = parts->fesc;
  const int npart = parts->npart;

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

    /* Define the callback data. */
    struct callback_data data = {
        .indices = part_indices,
        .dims = dims,
        .ndim = ndim,
        .particle_index = p,
        .nlam = grid->nlam,
        .npart = npart,
        .fesc = fesc != NULL ? fesc[p] : 0,
        .grid_spectra = grid->spectra,
        .grid_lines = grid->lines,
        .grid_continuum = grid->continuum,
    };

    /* Call the callback function if we have something to do. */
    func(mass, &data, out);
  }
  return out;
}

/**
 * @brief This calculates the grid weights in each grid cell using a nearest
 *       grid point approach.
 *
 * This is the serial version of the function.
 *
 * @param grid: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param out_size: The size of the output array. (This will be allocated
 * within this function.)
 * @param func: The callback function to be called.
 */
static double *weight_loop_ngp_omp(struct grid *grid, struct particles *parts,
                                   int out_size, WeightFunc func) {

#ifdef WITH_OPENMP

  /* Define the output array we'll unpack everything into. */
  double *out = (double *)malloc(out_size * sizeof(double));
  if (out == NULL) {
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for output.");
    return NULL;
  }
  bzero(out, out_size * sizeof(double));

#pragma omp parallel
  {
    /* Unpack the grid properties. */
    const int *dims = grid->dims;
    const int ndim = grid->ndim;
    const double **grid_props = grid->props;

    /* Unpack the particles properties. */
    const double *part_masses = parts->mass;
    const double **part_props = parts->props;
    const double *fesc = parts->fesc;
    const int npart = parts->npart;

    /* Allocate the output. */
    double *out_per_thread = (double *)malloc(out_size * sizeof(double));
    if (out == NULL) {
      PyErr_SetString(PyExc_MemoryError,
                      "Failed to allocate memory for output.");
      return NULL;
    }
    bzero(out_per_thread, out_size * sizeof(double));

    /* Loop over particles. */
#pragma omp for
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
          part_cell = binary_search(/*low*/ 0, /*high*/ dims[dim] - 1,
                                    grid_prop, part_val);
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

      /* Define the callback data. */
      struct callback_data data = {
          .indices = part_indices,
          .dims = dims,
          .ndim = ndim,
          .particle_index = p,
          .nlam = grid->nlam,
          .npart = npart,
          .fesc = fesc != NULL ? fesc[p] : 0,
          .grid_spectra = grid->spectra,
          .grid_lines = grid->lines,
          .grid_continuum = grid->continuum,
      };

      /* Call the callback function if we have something to do. */
      func(mass, &data, out_per_thread);
    }

    /* Use reduction to collect everything into the output array */
#pragma omp for reduction(+ : out[ : out_size])
    for (int i = 0; i < out_size; i++) {
      out[i] += out_per_thread[i];
    }

    /* Free the per-thread output. */
    free(out_per_thread);
  }

  return out;

#else
  /* We shouldn't be able to get here. */
  PyErr_SetString(PyExc_RuntimeError,
                  "OpenMP not enabled but in an OpenMP function.");
  return NULL;
#endif
}

/**
 * @brief This calculates the grid weights in each grid cell using a nearest
 *        grid point approach.
 *
 * This is a wrapper which calls the correct function based on the number of
 * threads requested and whether OpenMP is available.
 *
 * @param grid: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param out_size: The size of the output array. (This will be allocated
 * within this function.)
 * @param func: The callback function to be called.
 * @param nthreads: The number of threads to use.
 */
double *weight_loop_ngp(struct grid *grid, struct particles *parts,
                        int out_size, WeightFunc func, const int nthreads) {

  /* Declare the output. This will be allocated when needed. */
  double *out;

  /* Call the correct function for the configuration/number of threads. */

#ifdef WITH_OPENMP

  /* If we have multiple threads and OpenMP we can parallelise. */
  if (nthreads > 1) {
    omp_set_num_threads(nthreads);
    out = weight_loop_ngp_omp(grid, parts, out_size, func);
  }
  /* Otherwise there's no point paying the OpenMP overhead. */
  else {
    out = weight_loop_ngp_serial(grid, parts, out, func);
  }

#else

  /* We don't have OpenMP, just call the serial version. */
  out = weight_loop_ngp_serial(grid, parts, out, func);

#endif
  /* Return what we've done, any error message will have been set at the
   * point of the error. */
  if (out == NULL) {
    return NULL;
  } else {
    return out;
  }
}
