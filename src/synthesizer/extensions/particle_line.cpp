/******************************************************************************
 * C extension to calculate SEDs for star particles.
 * Calculates weights on an arbitrary dimensional grid given the mass.
 *****************************************************************************/
/* C includes */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Python includes */
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>

/* Local includes */
#include "macros.h"
#include "property_funcs.h"
#include "timers.h"
#include "weights.h"

/**
 * @brief This calculates particle lines using a cloud in cell approach.
 *
 * This is the serial version of the function.
 *
 * @param grid: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param lines_lum: The output array for the line luminosity.
 * @param lines_cont: The output array for the continuum luminosity.
 */
static void lines_loop_cic_serial(struct grid *grid, struct particles *parts,
                                  double *lines_lum, double *lines_cont) {

  /* Unpack the grid properties. */
  int *dims = grid->dims;
  int ndim = grid->ndim;
  double **grid_props = grid->props;
  const double *grid_lines = grid->lines;
  const double *grid_continuum = grid->continuum;

  /* Unpack the particles properties. */
  double *part_masses = parts->mass;
  double **part_props = parts->props;
  int npart = parts->npart;

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

      /* Nothing to do if fraction is 0. */
      if (frac == 0) {
        continue;
      }

      /* Define the weight. */
      double weight = frac * mass;

      /* We have a contribution, get the flattened index into the grid array. */
      const int grid_ind = get_flat_index(frac_ind, dims, ndim);

      /* Add the contribution to this particle. */
      lines_lum[p] += grid_lines[grid_ind] * weight;
      lines_cont[p] += grid_continuum[grid_ind] * weight;
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
 * @param lines_lum: The output array for the line luminosity.
 * @param lines_cont: The output array for the continuum luminosity.
 * @param nthreads: The number of threads to use.
 */
#ifdef WITH_OPENMP
static void lines_loop_cic_omp(struct grid *grid, struct particles *parts,
                               double *lines_lum, double *lines_cont,
                               int nthreads) {

  /* How many particles should each thread get? */
  int npart_per_thread = (int)(ceil(parts->npart / nthreads));

#pragma omp parallel num_threads(nthreads)
  {

    /* Unpack the grid properties. */
    int *dims = grid->dims;
    int ndim = grid->ndim;
    double **grid_props = grid->props;
    const double *grid_lines = grid->lines;
    const double *grid_continuum = grid->continuum;

    /* Unpack the particles properties. */
    double *part_masses = parts->mass;
    double **part_props = parts->props;
    int npart = parts->npart;

    /* Get the thread id. */
    int tid = omp_get_thread_num();

    /* Calculate start and end indices for each thread */
    int start = tid * npart_per_thread;
    int end = start + npart_per_thread;
    if (end >= npart) {
      end = npart;
    }
#ifdef WITH_DEBUGGING_CHECKS
    else {
#pragma omp critical
      PyErr_SetString(PyExc_RuntimeError,
                      "Not all particles distributed to threads.");
      free(spectra);
      spectra = NULL;
      return;
    }
#endif

    /* Loop over particles. */
    for (int p = start; p < end; p++) {

      /* Get this particle's mass. */
      const double mass = part_masses[p];

      /* Setup the index and mass fraction arrays. */
      int part_indices[ndim];
      double axis_fracs[ndim];

      /* Get the grid indices and cell fractions for the particle. */
      get_part_ind_frac_cic(part_indices, axis_fracs, dims, ndim, grid_props,
                            part_props, p);

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

        if (frac == 0) {
          continue;
        }

        /* Define the weight. */
        double weight = frac * mass;

        /* We have a contribution, get the flattened index into the grid array.
         */
        const int grid_ind = get_flat_index(frac_ind, dims, ndim);

        /* Add the contribution to this particle. */
        lines_lum[p] += grid_lines[grid_ind] * weight;
        lines_cont[p] += grid_continuum[grid_ind] * weight;
      }
    }
  }
}
#endif

/**
 * @brief This calculates particle lines using a cloud in cell approach.
 *
 * This is a wrapper which calls the correct function based on the number of
 * threads requested and whether OpenMP is available.
 *
 * @param grid: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param lines_lum: The output array for the line luminosity.
 * @param lines_cont: The output array for the continuum luminosity.
 * @param nthreads: The number of threads to use.
 */
void lines_loop_cic(struct grid *grid, struct particles *parts,
                    double *lines_lum, double *lines_cont, const int nthreads) {

  double start_time = tic();

  /* Call the correct function for the configuration/number of threads. */

#ifdef WITH_OPENMP

  /* If we have multiple threads and OpenMP we can parallelise. */
  if (nthreads > 1) {
    lines_loop_cic_omp(grid, parts, lines_lum, lines_cont, nthreads);
  }
  /* Otherwise there's no point paying the OpenMP overhead. */
  else {
    lines_loop_cic_serial(grid, parts, lines_lum, lines_cont);
  }

#else

  (void)nthreads;

  /* We don't have OpenMP, just call the serial version. */
  lines_loop_cic_serial(grid, parts, lines_lum, lines_cont);

#endif
  toc("Cloud in Cell particle lines loop", start_time);
}

/**
 * @brief This calculates particle lines using a nearest grid point approach.
 *
 * This is the serial version of the function.
 *
 * @param grid: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param lines_lum: The output array for the line luminosity.
 * @param lines_cont: The output array for the continuum luminosity.
 */
static void lines_loop_ngp_serial(struct grid *grid, struct particles *parts,
                                  double *lines_lum, double *lines_cont) {

  /* Unpack the grid properties. */
  int *dims = grid->dims;
  int ndim = grid->ndim;
  double **grid_props = grid->props;
  const double *grid_lines = grid->lines;
  const double *grid_continuum = grid->continuum;

  /* Unpack the particles properties. */
  double *part_masses = parts->mass;
  double **part_props = parts->props;
  int npart = parts->npart;

  /* Loop over particles. */
  for (int p = 0; p < npart; p++) {

    /* Get this particle's mass. */
    const double mass = part_masses[p];

    /* Setup the index array. */
    int part_indices[ndim];

    /* Get the grid indices for the particle */
    get_part_inds_ngp(part_indices, dims, ndim, grid_props, part_props, p);

    /* Define the weight. */
    double weight = mass;

    /* We have a contribution, get the flattened index into the grid array. */
    const int grid_ind = get_flat_index(part_indices, dims, ndim);

    /* Add the contribution to this particle. */
    lines_lum[p] += grid_lines[grid_ind] * weight;
    lines_cont[p] += grid_continuum[grid_ind] * weight;
  }
}

/**
 * @brief This calculates particle lines using a nearest grid point approach.
 *
 * This is the serial version of the function.
 *
 * @param grid: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param lines_lum: The output array for the line luminosity.
 * @param lines_cont: The output array for the continuum luminosity.
 * @param nthreads: The number of threads to use.
 */
#ifdef WITH_OPENMP
static void lines_loop_ngp_omp(struct grid *grid, struct particles *parts,
                               double *lines_lum, double *lines_cont,
                               int nthreads) {

  /* How many particles should each thread get? */
  int npart_per_thread = (int)(ceil(parts->npart / nthreads));

#pragma omp parallel num_threads(nthreads)
  {

    /* Unpack the grid properties. */
    int *dims = grid->dims;
    int ndim = grid->ndim;
    double **grid_props = grid->props;
    const double *grid_lines = grid->lines;
    const double *grid_continuum = grid->continuum;

    /* Unpack the particles properties. */
    double *part_masses = parts->mass;
    double **part_props = parts->props;
    int npart = parts->npart;

    /* Get the thread id. */
    int tid = omp_get_thread_num();

    /* Calculate start and end indices for each thread */
    int start = tid * npart_per_thread;
    int end = start + npart_per_thread;
    if (end >= npart) {
      end = npart;
    }
#ifdef WITH_DEBUGGING_CHECKS
    else {
#pragma omp critical
      PyErr_SetString(PyExc_RuntimeError,
                      "Not all particles distributed to threads.");
      free(spectra);
      spectra = NULL;
      return;
    }
#endif

    /* Loop over particles. */
    for (int p = start; p < end; p++) {

      /* Get this particle's mass. */
      const double weight = part_masses[p];

      /* Setup the index array. */
      int part_indices[ndim];

      /* Get the grid indices for the particle */
      get_part_inds_ngp(part_indices, dims, ndim, grid_props, part_props, p);

      /* We have a contribution, get the flattened index into the grid array. */
      const int grid_ind = get_flat_index(part_indices, dims, ndim);

      /* Add the contribution to this particle. */
      lines_lum[p] += grid_lines[grid_ind] * weight;
      lines_cont[p] += grid_continuum[grid_ind] * weight;
    }
  }
}
#endif

/**
 * @brief This calculates particle lines using a nearest grid point approach.
 *
 * This is a wrapper which calls the correct function based on the number of
 * threads requested and whether OpenMP is available.
 *
 * @param grid: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param lines_lum: The output array for the line luminosity.
 * @param lines_cont: The output array for the continuum luminosity.
 * @param nthreads: The number of threads to use.
 */
void lines_loop_ngp(struct grid *grid, struct particles *parts,
                    double *lines_lum, double *lines_cont, const int nthreads) {

  double start_time = tic();

  /* Call the correct function for the configuration/number of threads. */

#ifdef WITH_OPENMP

  /* If we have multiple threads and OpenMP we can parallelise. */
  if (nthreads > 1) {
    lines_loop_ngp_omp(grid, parts, lines_lum, lines_cont, nthreads);
  }
  /* Otherwise there's no point paying the OpenMP overhead. */
  else {
    lines_loop_ngp_serial(grid, parts, lines_lum, lines_cont);
  }

#else

  (void)nthreads;

  /* We don't have OpenMP, just call the serial version. */
  lines_loop_ngp_serial(grid, parts, lines_lum, lines_cont);

#endif
  toc("Nearest Grid Point particle lines loop", start_time);
}
/**
 * @brief Computes per particle line emission for a collection of particles.
 *
 * @param np_grid_line: The SPS line emission array.
 * @param np_grid_continuum: The SPS continuum emission array.
 * @param grid_tuple: The tuple containing arrays of grid axis properties.
 * @param part_tuple: The tuple of particle property arrays(in the same order
 *                    as grid_tuple).
 * @param np_part_mass: The particle mass array.
 * @param np_ndims: The size of each grid axis.
 * @param ndim: The number of grid axes.
 * @param npart: The number of particles.
 * @param method: The method to use for assigning weights.
 */
PyObject *compute_particle_line(PyObject *self, PyObject *args) {

  double start = tic();
  double setup_start = tic();

  /* We don't need the self argument but it has to be there. Tell the compiler
   * we don't care. */
  (void)self;

  int ndim, npart, nthreads;
  PyObject *grid_tuple, *part_tuple;
  PyArrayObject *np_grid_lines, *np_grid_continuum;
  PyArrayObject *np_part_mass, *np_ndims;
  char *method;

  if (!PyArg_ParseTuple(args, "OOOOOOiisi", &np_grid_lines, &np_grid_continuum,
                        &grid_tuple, &part_tuple, &np_part_mass, &np_ndims,
                        &ndim, &npart, &method, &nthreads))
    return NULL;

  /* Extract the grid struct. */
  struct grid *grid_props = get_lines_grid_struct(
      grid_tuple, np_ndims, np_grid_lines, np_grid_continuum, ndim, /*nlam*/ 1);
  if (grid_props == NULL) {
    return NULL;
  }

  /* Extract the particle struct. */
  struct particles *part_props = get_part_struct(
      part_tuple, np_part_mass, /*np_velocities*/ NULL, NULL, npart, ndim);
  if (part_props == NULL) {
    return NULL;
  }

  /* Allocate the lines arrays. */
  double *lines_lum = calloc(npart, sizeof(double));
  if (lines_lum == NULL) {
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for lines.");
    return NULL;
  }
  double *lines_cont = calloc(npart, sizeof(double));
  if (lines_cont == NULL) {
    PyErr_SetString(PyExc_MemoryError,
                    "Failed to allocate memory for continuum.");
    return NULL;
  }

  toc("Extracting Python data", setup_start);

  /* With everything set up we can compute the weights for each particle using
   * the requested method. */
  /* NOTE: rather than modify the weights function to make the 2 outputs
   * we'll instead pass an array with line_lum at one end and line_cont at the
   * other and then just extract the result later. */
  if (strcmp(method, "cic") == 0) {
    lines_loop_cic(grid_props, part_props, lines_lum, lines_cont, nthreads);
  } else if (strcmp(method, "ngp") == 0) {
    lines_loop_ngp(grid_props, part_props, lines_lum, lines_cont, nthreads);
  } else {
    PyErr_SetString(PyExc_ValueError, "Unknown grid assignment method (%s).");
    return NULL;
  }

  /* Clean up memory! */
  free(part_props);
  free(grid_props);

  /* Reconstruct the python array to return. */
  npy_intp np_dims[1] = {
      npart,
  };
  PyArrayObject *out_line = (PyArrayObject *)PyArray_SimpleNewFromData(
      1, np_dims, NPY_FLOAT64, lines_lum);
  PyArrayObject *out_cont = (PyArrayObject *)PyArray_SimpleNewFromData(
      1, np_dims, NPY_FLOAT64, lines_cont);

  toc("Computing particles lines", start);

  return Py_BuildValue("(OO)", out_line, out_cont);
}

/* Below is all the gubbins needed to make the module importable in Python. */
static PyMethodDef LineMethods[] = {
    {"compute_particle_line", (PyCFunction)compute_particle_line, METH_VARARGS,
     "Method for calculating particle intrinsic line emission."},
    {NULL, NULL, 0, NULL}};

/* Make this importable. */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "make_particle_line",                           /* m_name */
    "A module to calculate particle line emission", /* m_doc */
    -1,                                             /* m_size */
    LineMethods,                                    /* m_methods */
    NULL,                                           /* m_reload */
    NULL,                                           /* m_traverse */
    NULL,                                           /* m_clear */
    NULL,                                           /* m_free */
};

PyMODINIT_FUNC PyInit_particle_line(void) {
  PyObject *m = PyModule_Create(&moduledef);
  import_array();
  return m;
}
