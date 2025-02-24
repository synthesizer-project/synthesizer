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
      out_arr[flat_ind] += frac * mass;
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
        local_out_arr[flat_ind] += frac * mass;
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
    out_arr[flat_ind] += mass;
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
      double contribution = local_part_masses[p];

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

/**
 * @brief Compute the weight in each grid cell based on the particles.
 *
 * @param grid: The Grid object.
 * @param parts: The object containing the particle properties.
 * @param method: The method to use for assigning weights.
 * @param nthreads: The number of threads to use.
 *
 * @return The weights in each grid cell.
 */
PyObject *compute_grid_weights(PyObject *self, PyObject *args) {

  double start_time = tic();
  double setup_start = tic();

  /* We don't need the self argument but it has to be there. Tell the compiler
   * we don't care. */
  (void)self;

  int ndim, npart, nthreads;
  PyObject *grid_tuple, *part_tuple;
  PyArrayObject *np_part_mass, *np_ndims;
  char *method;

  if (!PyArg_ParseTuple(args, "OOOOiisi", &grid_tuple, &part_tuple,
                        &np_part_mass, &np_ndims, &ndim, &npart, &method,
                        &nthreads))
    return NULL;

  /* Extract the grid struct. */
  struct grid *grid_props =
      get_spectra_grid_struct(grid_tuple, np_ndims, /*np_grid_spectra*/ NULL,
                              /*np_lam*/ NULL, ndim, /*nlam*/ 1);
  if (grid_props == NULL) {
    return NULL;
  }

  /* Extract the particle struct. */
  struct particles *part_props = get_part_struct(
      part_tuple, np_part_mass, /*np_velocities*/ NULL, npart, ndim);
  if (part_props == NULL) {
    return NULL;
  }
  /* Allocate the sfzh array to output. */
  double *grid_weights = calloc(grid_props->size, sizeof(double));
  if (grid_weights == NULL) {
    PyErr_SetString(PyExc_MemoryError, "Could not allocate memory for sfzh.");
    return NULL;
  }

  toc("Extracting Python data", setup_start);

  /* With everything set up we can compute the weights for each particle using
   * the requested method. */
  if (strcmp(method, "cic") == 0) {
    weight_loop_cic(grid_props, part_props, grid_props->size, grid_weights,
                    nthreads);
  } else if (strcmp(method, "ngp") == 0) {
    weight_loop_ngp(grid_props, part_props, grid_props->size, grid_weights,
                    nthreads);
  } else {
    PyErr_SetString(PyExc_ValueError, "Unknown grid assignment method.");
    return NULL;
  }

  /* Check we got the output. (Any error messages will already be set) */
  if (grid_weights == NULL) {
    return NULL;
  }

  /* Reconstruct the python array to return. */
  npy_intp np_dims[grid_props->ndim];
  for (int idim = 0; idim < grid_props->ndim; idim++) {
    np_dims[idim] = grid_props->dims[idim];
  }

  PyArrayObject *out_weights = (PyArrayObject *)PyArray_SimpleNewFromData(
      grid_props->ndim, np_dims, NPY_FLOAT64, grid_weights);

  /* Clean up memory! */
  free(part_props);
  free(grid_props);

  toc("Computing SFZH", start_time);

  return Py_BuildValue("N", out_weights);
}

/* Below is all the gubbins needed to make the module importable in Python. */
static PyMethodDef WeightMethods[] = {
    {"compute_grid_weights", (PyCFunction)compute_grid_weights, METH_VARARGS,
     "Method for calculating the weights on a grid."},
    {NULL, NULL, 0, NULL}};

/* Make this importable. */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "compute_weights",                                    /* m_name */
    "A module to calculating particle weigths on a grid", /* m_doc */
    -1,                                                   /* m_size */
    WeightMethods,                                        /* m_methods */
    NULL,                                                 /* m_reload */
    NULL,                                                 /* m_traverse */
    NULL,                                                 /* m_clear */
    NULL,                                                 /* m_free */
};

PyMODINIT_FUNC PyInit_weights(void) {
  PyObject *m = PyModule_Create(&moduledef);
  import_array();
  return m;
}
