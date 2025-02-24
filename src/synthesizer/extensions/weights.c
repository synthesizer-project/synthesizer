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
#include "timers.h"
#include "weights.h"

/* Optional openmp include. */
#ifdef WITH_OPENMP
#include <omp.h>
#endif

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
void get_part_ind_frac_cic(int *part_indices, double *axis_fracs, int *dims,
                           int ndim, double **grid_props, double **part_props,
                           int p) {

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
 * @brief This calculates the grid weights in each grid cell using a cloud
 *        in cell approach.
 *
 * This is the serial version of the function.
 *
 * @param grid: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param out: The output array.
 */
static void weight_loop_cic_serial(struct grid *grid, struct particles *parts,
                                   void *out) {

  /* Unpack the grid properties. */
  int *dims = grid->dims;
  int ndim = grid->ndim;
  double **grid_props = grid->props;

  /* Unpack the particles properties. */
  double *part_masses = parts->mass;
  double **part_props = parts->props;
  double *fesc = parts->fesc;
  int npart = parts->npart;

  /* Set the sub cell constants we'll use below. */
  const int num_sub_cells = 1 << ndim; /* 2^ndim */
  int sub_dims[ndim];
  for (int i = 0; i < ndim; i++) {
    sub_dims[i] = 2;
  }

  /* Convert out. */
  double *out_arr = (double *)out;

  /* Loop over particles. */
  for (int p = 0; p < npart; p++) {

    /* Get this particle's mass. */
    const double mass = part_masses[p];

    /* Setup the index and mass fraction arrays. */
    int part_indices[ndim];
    double axis_fracs[ndim];

    /* Get the grid indices and cell fractions for the particle. */
    get_part_ind_frac_cic(part_indices, axis_fracs, dims, ndim, grid_props,
                          part_props, p);

    /* Now loop over this collection of cells collecting and setting their
     * weights. */
    for (int icell = 0; icell < num_sub_cells; icell++) {

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

      /* Nothing to do if fraction is 0. */
      if (frac == 0) {
        continue;
      }

      /* Unravel the indices. */
      int flat_ind = get_flat_index(frac_ind, dims, ndim);

      /* Store the weight. */
      out_arr[flat_ind] += frac * mass * (1.0 - fesc[p]);
    }
  }
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
 * @param out: The output array.
 * @param nthreads: The number of threads to use.
 */
#ifdef WITH_OPENMP
static void weight_loop_cic_omp(struct grid *grid, struct particles *parts,
                                int out_size, void *out, int nthreads) {

  /* Convert out. */
  double *out_arr = (double *)out;

  /* Unpack the grid properties. */
  int *dims = grid->dims;
  int ndim = grid->ndim;
  double **grid_props = grid->props;

  /* Unpack the particles properties. */
  double *part_masses = parts->mass;
  double **part_props = parts->props;
  double *fesc = parts->fesc;
  int npart = parts->npart;

  /* Set the sub cell constants we'll use below. */
  const int num_sub_cells = 1 << ndim; // 2^ndim
  int sub_dims[ndim];
  for (int i = 0; i < ndim; i++) {
    sub_dims[i] = 2;
  }

#pragma omp parallel num_threads(nthreads)
  {

    /* First lets slice up the particles between the threads. */
    int npart_per_thread = (npart + nthreads - 1) / nthreads;

    /* Get the thread id. */
    int tid = omp_get_thread_num();

    /* Get the start and end particle indices for this thread. */
    int start = tid * npart_per_thread;
    int end = start + npart_per_thread;
    if (end >= npart) {
      end = npart;
    }

    /* Get local pointers to the particle properties. */
    double *local_part_masses = part_masses + start;
    double *local_fesc = fesc + start;

    /* Allocate a local output array. */
    double *local_out_arr = calloc(out_size, sizeof(double));
    if (local_out_arr == NULL) {
      PyErr_SetString(PyExc_MemoryError,
                      "Failed to allocate memory for output.");
    }

    /* Parallel loop with atomic updates. */
    for (int p = 0; p < end - start; p++) {

      /* Get this particle's mass. */
      const double mass = local_part_masses[p];

      /* Setup the index and mass fraction arrays. */
      int part_indices[ndim];
      double axis_fracs[ndim];

      /* Get the grid indices and cell fractions for the particle. */
      get_part_ind_frac_cic(part_indices, axis_fracs, dims, ndim, grid_props,
                            part_props, p + start);

      /* Now loop over this collection of cells collecting and setting their
       * weights. */
      for (int icell = 0; icell < num_sub_cells; icell++) {

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

        if (frac == 0) {
          continue;
        }

        /* Unravel the indices. */
        int flat_ind = get_flat_index(frac_ind, dims, ndim);

        /* Store the weight. */
        local_out_arr[flat_ind] += frac * mass * (1.0 - local_fesc[p]);
      }
    }

    /* Update the global output array */
#pragma omp critical
    {
      for (int i = 0; i < out_size; i++) {
        out_arr[i] += local_out_arr[i];
      }
    }

    /* Clean up. */
    free(local_out_arr);
  }
}
#endif

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
 * @param out: The output array.
 * @param nthreads: The number of threads to use.
 */
void weight_loop_cic(struct grid *grid, struct particles *parts, int out_size,
                     void *out, const int nthreads) {

  double start_time = tic();

  /* Call the correct function for the configuration/number of threads. */

#ifdef WITH_OPENMP

  /* If we have multiple threads and OpenMP we can parallelise. */
  if (nthreads > 1) {
    weight_loop_cic_omp(grid, parts, out_size, out, nthreads);
  }
  /* Otherwise there's no point paying the OpenMP overhead. */
  else {
    weight_loop_cic_serial(grid, parts, out);
  }

#else

  (void)nthreads;

  /* We don't have OpenMP, just call the serial version. */
  weight_loop_cic_serial(grid, parts, out);

#endif
  toc("Cloud in Cell weight loop", start_time);
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
void get_part_inds_ngp(int *part_indices, int *dims, int ndim,
                       double **grid_props, double **part_props, int p) {

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

/**
 * @brief This calculates the grid weights in each grid cell using a nearest
 *       grid point approach.
 *
 * This is the serial version of the function.
 *
 * @param grid: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param out: The output array.
 */
static void weight_loop_ngp_serial(struct grid *grid, struct particles *parts,
                                   void *out) {

  /* Unpack the grid properties. */
  int *dims = grid->dims;
  int ndim = grid->ndim;
  double **grid_props = grid->props;

  /* Unpack the particles properties. */
  double *part_masses = parts->mass;
  double **part_props = parts->props;
  double *fesc = parts->fesc;
  int npart = parts->npart;

  /* Convert out. */
  double *out_arr = (double *)out;

  /* Loop over particles. */
  for (int p = 0; p < npart; p++) {

    /* Get this particle's mass. */
    const double mass = part_masses[p];

    /* Setup the index array. */
    int part_indices[ndim];

    /* Get the grid indices for the particle */
    get_part_inds_ngp(part_indices, dims, ndim, grid_props, part_props, p);

    /* Unravel the indices. */
    int flat_ind = get_flat_index(part_indices, dims, ndim);

    /* Store the weight. */
    out_arr[flat_ind] += mass * (1.0 - fesc[p]);
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
 * @param out: The output array.
 * @param nthreads: The number of threads to use.
 */
#ifdef WITH_OPENMP
static void weight_loop_ngp_omp(struct grid *grid, struct particles *parts,
                                int out_size, void *out, int nthreads) {
  /* Convert out. */
  double *out_arr = (double *)out;

  /* Unpack the grid properties. */
  int *dims = grid->dims;
  int ndim = grid->ndim;
  double **grid_props = grid->props;

  /* Unpack the particles properties. */
  double *part_masses = parts->mass;
  double **part_props = parts->props;
  double *fesc = parts->fesc;
  int npart = parts->npart;

#pragma omp parallel num_threads(nthreads)
  {

    /* First lets slice up the particles between the threads. */
    int npart_per_thread = (npart + nthreads - 1) / nthreads;

    /* Get the thread id. */
    int tid = omp_get_thread_num();

    /* Get the start and end particle indices for this thread. */
    int start = tid * npart_per_thread;
    int end = start + npart_per_thread;
    if (end >= npart) {
      end = npart;
    }

    /* Get local pointers to the particle properties. */
    double *local_part_masses = part_masses + start;
    double *local_fesc = fesc + start;

    /* Allocate a local output array. */
    double *local_out_arr = calloc(out_size, sizeof(double));
    if (local_out_arr == NULL) {
      PyErr_SetString(PyExc_MemoryError,
                      "Failed to allocate memory for output.");
    }

    /* Parallel loop with atomic updates. */
    for (int p = 0; p < end - start; p++) {

      /* Get the grid indices for the particle */
      int part_indices[ndim];
      get_part_inds_ngp(part_indices, dims, ndim, grid_props, part_props,
                        p + start);

      /* Unravel the indices. */
      int flat_ind = get_flat_index(part_indices, dims, ndim);

      /* Calculate this particles contribution to the grid cell. */
      double contribution = local_part_masses[p] * (1.0 - local_fesc[p]);

      /* Update the shared output array atomically */
      local_out_arr[flat_ind] += contribution;
    }

    /* Update the global output array */
#pragma omp critical
    {
      for (int i = 0; i < out_size; i++) {
        out_arr[i] += local_out_arr[i];
      }
    }

    /* Clean up. */
    free(local_out_arr);
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
 * @param grid: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param out_size: The size of the output array. (This will be allocated
 * within this function.)
 * @param out: The output array.
 * @param nthreads: The number of threads to use.
 */
void weight_loop_ngp(struct grid *grid, struct particles *parts, int out_size,
                     void *out, const int nthreads) {

  double start_time = tic();

  /* Call the correct function for the configuration/number of threads. */

#ifdef WITH_OPENMP

  /* If we have multiple threads and OpenMP we can parallelise. */
  if (nthreads > 1) {
    weight_loop_ngp_omp(grid, parts, out_size, out, nthreads);
  }
  /* Otherwise there's no point paying the OpenMP overhead. */
  else {
    weight_loop_ngp_serial(grid, parts, out);
  }

#else

  (void)nthreads;

  /* We don't have OpenMP, just call the serial version. */
  weight_loop_ngp_serial(grid, parts, out);

#endif
  toc("Nearest Grid Point weight loop", start_time);
}
