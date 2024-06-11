/******************************************************************************
 * C extension to calculate integrated SEDs for a galaxy's star particles.
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
#include "weights.h"

/**
 * @brief The callback function to store the weights for each particle.
 *
 * @param weight: The weight for this particle.
 * @param data: The callback data.
 * @param out: The grid weights.
 */
static void store_weight(double weight, struct callback_data *data, void *out) {

  /* Unpack the data. */
  const int *indices = data->indices;
  const int *dims = data->dims;
  const int ndim = data->ndim;

  /* Unravel the indices. */
  int flat_ind = get_flat_index(indices, dims, ndim);

  /* Get the output array. */
  double *out_arr = (double *)out;

  /* Store the weight. */
  out_arr[flat_ind] += weight;
}

/**
 * @brief Compute the integrated line emission from the grid weights.
 *
 * @param grid_props: The grid properties.
 * @param grid_weights: The grid weights computed from the particles.
 *
 * @return The integrated line emission.
 */
static double *get_lines_serial(struct grid *grid_props, double *grid_weights) {

  /* Declare and initialise the vairbales we'll store our result in. */
  double line_lum = 0.0;
  double line_cont = 0.0;

  /* Loop over grid cells populating the lines. */
  for (int grid_ind = 0; grid_ind < grid_props->size; grid_ind++) {

    /* Get the weight. */
    const double weight = grid_weights[grid_ind];

    /* Skip zero weight cells. */
    if (weight <= 0)
      continue;

    /* Add this grid cell's contribution to the lines (fesc is included in
     * the weight).*/
    line_lum += grid_props->lines[grid_ind] * weight;
    line_cont += grid_props->continuum[grid_ind] * weight;
  }

  /* Create an array to return. */
  double result[2] = {line_lum, line_cont};
  return result;
}

/**
 * @brief Compute the integrated line emission from the grid weights.
 *
 * @param grid_props: The grid properties.
 * @param grid_weights: The grid weights computed from the particles.
 * @param nthreads: The number of threads to use.
 */
static double *get_lines_omp(struct grid *grid_props, double *grid_weights,
                             int nthreads) {

#ifdef WITH_OPENMP
  /* Declare and initialise the vairbales we'll store our result in. */
  double line_lum = 0.0;
  double line_cont = 0.0;

#pragma omp parallel num_threads(nthreads)
  {

    /* Loop over grid cells populating the lines. */
#pragma omp for reduction(+ : line_lum, line_cont)
    for (int grid_ind = 0; grid_ind < grid_props->size; grid_ind++) {

      /* Get the weight. */
      const double weight = grid_weights[grid_ind];

      /* Skip zero weight cells. */
      if (weight <= 0)
        continue;

      /* Add this grid cell's contribution to the lines (fesc is included in
       * the weight).*/
      line_lum += grid_props->lines[grid_ind] * weight;
      line_cont += grid_props->continuum[grid_ind] * weight;
    }
  }

  /* Create an array to return. */
  double result[2] = {line_lum, line_cont};
  return result;
#else

  /* We shouldn't be able to get here. */
  PyErr_SetString(PyExc_RuntimeError,
                  "OpenMP not enabled but in an OpenMP function.");
  return NULL;
#endif
}

/**
 * @brief Computes an integrated line emission for a collection of particles.
 *
 * @param np_grid_line: The SPS line emission array.
 * @param np_grid_continuum: The SPS continuum emission array.
 * @param grid_tuple: The tuple containing arrays of grid axis properties.
 * @param part_tuple: The tuple of particle property arrays (in the same order
 *                    as grid_tuple).
 * @param np_part_mass: The particle mass array.
 * @param fesc: The escape fraction.
 * @param np_ndims: The size of each grid axis.
 * @param ndim: The number of grid axes.
 * @param npart: The number of particles.
 * @param method: The method to use for assigning weights.
 */
PyObject *compute_integrated_line(PyObject *self, PyObject *args) {

  /* We don't need the self argument but it has to be there. Tell the compiler
   * we don't care. */
  (void)self;

  int ndim, npart, nthreads;
  PyObject *grid_tuple, *part_tuple;
  PyArrayObject *np_grid_lines, *np_grid_continuum;
  PyArrayObject *np_fesc;
  PyArrayObject *np_part_mass, *np_ndims;
  char *method;

  if (!PyArg_ParseTuple(args, "OOOOOOOiis", &np_grid_lines, &np_grid_continuum,
                        &grid_tuple, &part_tuple, &np_part_mass, &np_fesc,
                        &np_ndims, &ndim, &npart, &method, &nthreads))
    return NULL;

  /* Extract the grid struct. */
  struct grid *grid_props = get_lines_grid_struct(
      grid_tuple, np_ndims, np_grid_lines, np_grid_continuum, ndim, /*nlam*/ 1);
  if (grid_props == NULL) {
    return NULL;
  }

  /* Extract the particle struct. */
  struct particles *part_props =
      get_part_struct(part_tuple, np_part_mass, np_fesc, npart, ndim);
  if (part_props == NULL) {
    return NULL;
  }

  /* With everything set up we can compute the weights for each particle using
   * the requested method. */
  double *grid_weights;
  if (strcmp(method, "cic") == 0) {
    grid_weights = weight_loop_cic(grid_props, part_props, grid_props->size,
                                   store_weight, nthreads);
  } else if (strcmp(method, "ngp") == 0) {
    grid_weights = weight_loop_ngp(grid_props, part_props, grid_props->size,
                                   store_weight, nthreads);
  } else {
    PyErr_SetString(PyExc_ValueError, "Unknown grid assignment method (%s).");
    return NULL;
  }

  /* Check we got grid weights sucessfully. (Any error messages will have
   * already been set) */
  if (grid_weights == NULL) {
    return NULL;
  }

#ifdef WITH_OPENMP
  /* Do we have mutliple threads to do the reduction onto the lines? */
  double *result;
  if (nthreads > 1) {
    result = get_lines_omp(grid_props, grid_weights, nthreads);
  } else {
    result = get_lines_serial(grid_props, grid_weights);
  }
#else
  /* We don't have OpenMP so we can't do the reduction in parallel. */
  double *result = get_lines_serial(grid_props, grid_weights);
#endif

  /* Ensure we got the result ok. */
  if (result == NULL) {
    return NULL;
  }

  /* Unpack the result array (it's 2 elements containing the line and
   * continuum). */
  double line_lum = result[0];
  double line_cont = result[1];

  /* Clean up memory! */
  free(grid_weights);
  free(part_props);
  free(grid_props);

  // Create a Python tuple containing the two doubles
  PyObject *result_tuple = Py_BuildValue("dd", line_lum, line_cont);

  // Return the tuple
  return result_tuple;
}

/* Below is all the gubbins needed to make the module importable in Python. */
static PyMethodDef LineMethods[] = {
    {"compute_integrated_line", (PyCFunction)compute_integrated_line,
     METH_VARARGS, "Method for calculating integrated intrinsic lines."},
    {NULL, NULL, 0, NULL}};

/* Make this importable. */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "get_line",                               /* m_name */
    "A module to calculate integrated lines", /* m_doc */
    -1,                                       /* m_size */
    LineMethods,                              /* m_methods */
    NULL,                                     /* m_reload */
    NULL,                                     /* m_traverse */
    NULL,                                     /* m_clear */
    NULL,                                     /* m_free */
};

PyMODINIT_FUNC PyInit_integrated_line(void) {
  PyObject *m = PyModule_Create(&moduledef);
  import_array();
  return m;
}
