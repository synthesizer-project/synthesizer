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
#include "weights.h"

/**
 * @brief This calculates the line emission/continuum of a particle using a
 * cloud in cell approach.
 *
 * @param grid_props: An array of the properties along each grid axis.
 * @param part_props: An array of the particle properties, in the same property
 *                    order as grid props.
 * @param mass: The mass of the current particle.
 * @param grid_lines: The grid of SPS line emission.
 * @param grid_continuum: The grid of SPS continuum emission.
 * @param dims: The length of each grid dimension.
 * @param ndim: The number of grid dimensions.
 * @param line_lum: The array of particle line luminosities to populate.
 * @param line_cont: The array of particle continuum luminosities to populate.
 * @param fesc: The escape fraction.
 * @param p: The index of the current particle.
 */
void line_loop_cic(const double **grid_props, const double **part_props,
                   const double mass, const double *grid_lines,
                   const double *grid_continuum, const int *dims,
                   const int ndim, double *line_lum, double *line_cont,
                   const double fesc, const int p) {

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

  /* To combine fractions we will need an array of dimensions for the subset.
   * These are always two in size, one for the low and one for high grid
   * point. */
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

    /* We have a contribution, get the flattened index into the grid array. */
    const int grid_ind = get_flat_index(frac_ind, dims, ndim);

    /* Define the weight. */
    double weight = mass * frac;

    /* Add the contribution to this particle. */
    line_lum[p] += grid_lines[grid_ind] * (1 - fesc) * weight;
    line_cont[p] += grid_continuum[grid_ind] * (1 - fesc) * weight;
  }
}

/**
 * @brief This calculates the line emission/continuum of a particle using a
 * nearest grid point approach.
 *
 * @param grid_props: An array of the properties along each grid axis.
 * @param part_props: An array of the particle properties, in the same property
 *                    order as grid props.
 * @param mass: The mass of the current particle.
 * @param grid_lines: The grid of SPS line emission.
 * @param grid_continuum: The grid of SPS continuum emission.
 * @param dims: The length of each grid dimension.
 * @param ndim: The number of grid dimensions.
 * @param line_lum: The array of particle line luminosities to populate.
 * @param line_cont: The array of particle continuum luminosities to populate.
 * @param fesc: The escape fraction.
 * @param p: The index of the current particle.
 */
void line_loop_ngp(const double **grid_props, const double **part_props,
                   const double mass, const double *grid_lines,
                   const double *grid_continuum, const int *dims,
                   const int ndim, double *line_lum, double *line_cont,
                   const double fesc, const int p) {

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
          binary_search(/*low*/ 1, /*high*/ dims[dim] - 1, grid_prop, part_val);
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

  /* Get the weight's index. */
  const int grid_ind = get_flat_index(part_indices, dims, ndim);

  /* Add the contribution to this particle. */
  line_lum[p] += grid_lines[grid_ind] * (1 - fesc) * mass;
  line_cont[p] += grid_continuum[grid_ind] * (1 - fesc) * mass;
}

/**
 * @brief Computes per particle line emission for a collection of particles.
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
PyObject *compute_particle_line(PyObject *self, PyObject *args) {

  const int ndim;
  const int npart, nlam;
  const PyObject *grid_tuple, *part_tuple;
  const PyArrayObject *np_grid_lines, *np_grid_continuum;
  const PyArrayObject *np_fesc;
  const PyArrayObject *np_part_mass, *np_ndims;
  const char *method;

  if (!PyArg_ParseTuple(args, "OOOOOOOiis", &np_grid_lines, &np_grid_continuum,
                        &grid_tuple, &part_tuple, &np_part_mass, &np_fesc,
                        &np_ndims, &ndim, &npart, &method))
    /* Error message is already set here. */
    return NULL;

  /* Quick check to make sure our inputs are valid. */
  if (ndim == 0) {
    PyErr_SetString(
        PyExc_ValueError,
        "Grid appears to be dimensionless! Something awful has happened!");
    return NULL;
  }
  if (npart == 0) {
    PyErr_SetString(PyExc_ValueError, "No particles to process!");
    return NULL;
  }

  /* Extract a pointer to the line grids */
  const double *grid_lines = PyArray_DATA(np_grid_lines);
  if (grid_lines == NULL) {
    PyErr_SetString(PyExc_ValueError, "Failed to extract grid_lines.");
    return NULL;
  }

  /* Extract a pointer to the continuum grid. */
  const double *grid_continuum = PyArray_DATA(np_grid_continuum);
  if (grid_continuum == NULL) {
    PyErr_SetString(PyExc_ValueError, "Failed to extract grid_continuum.");
    return NULL;
  }

  /* Set up arrays to hold the line emission and continuum. */
  double *line_lum = malloc(npart * sizeof(double));
  if (line_lum == NULL) {
    PyErr_SetString(PyExc_MemoryError,
                    "Failed to allocate memory for line_lum.");
    return NULL;
  }
  bzero(line_lum, npart * sizeof(double));
  double *line_cont = malloc(npart * sizeof(double));
  if (line_cont == NULL) {
    PyErr_SetString(PyExc_MemoryError,
                    "Failed to allocate memory for line_cont.");
    return NULL;
  }
  bzero(line_cont, npart * sizeof(double));

  /* Extract a pointer to the grid dims */
  const int *dims = PyArray_DATA(np_ndims);
  if (dims == NULL) {
    PyErr_SetString(PyExc_ValueError, "Failed to extract dims from np_ndims.");
    return NULL;
  }

  /* Extract a pointer to the particle masses. */
  const double *part_mass = PyArray_DATA(np_part_mass);
  if (part_mass == NULL) {
    PyErr_SetString(PyExc_ValueError,
                    "Failed to extract part_mass from np_part_mass.");
    return NULL;
  }

  /* Extract a pointer to the fesc array. */
  const double *fesc = PyArray_DATA(np_fesc);
  if (fesc == NULL) {
    PyErr_SetString(PyExc_ValueError, "Failed to extract fesc from np_fesc.");
    return NULL;
  }

  /* Allocate a single array for grid properties*/
  int nprops = 0;
  for (int dim = 0; dim < ndim; dim++)
    nprops += dims[dim];
  const double **grid_props = malloc(nprops * sizeof(double *));
  if (grid_props == NULL) {
    PyErr_SetString(PyExc_MemoryError,
                    "Failed to allocate memory for grid_props.");
    return NULL;
  }

  /* How many grid elements are there? */
  int grid_size = 1;
  for (int dim = 0; dim < ndim; dim++)
    grid_size *= dims[dim];

  /* Allocate an array to hold the grid weights. */
  double *grid_weights = malloc(grid_size * sizeof(double));
  if (grid_weights == NULL) {
    PyErr_SetString(PyExc_MemoryError,
                    "Failed to allocate memory for grid_weights.");
    return NULL;
  }
  bzero(grid_weights, grid_size * sizeof(double));

  /* Unpack the grid property arrays into a single contiguous array. */
  for (int idim = 0; idim < ndim; idim++) {

    /* Extract the data from the numpy array. */
    const PyArrayObject *np_grid_arr = PyTuple_GetItem(grid_tuple, idim);
    const double *grid_arr = PyArray_DATA(np_grid_arr);

    /* Assign this data to the property array. */
    grid_props[idim] = grid_arr;
  }

  /* Allocate a single array for particle properties. */
  const double **part_props = malloc(npart * ndim * sizeof(double *));
  if (part_props == NULL) {
    PyErr_SetString(PyExc_MemoryError,
                    "Failed to allocate memory for part_props.");
    return NULL;
  }

  /* Unpack the particle property arrays into a single contiguous array. */
  for (int idim = 0; idim < ndim; idim++) {

    /* Extract the data from the numpy array. */
    const PyArrayObject *np_part_arr = PyTuple_GetItem(part_tuple, idim);
    const double *part_arr = PyArray_DATA(np_part_arr);

    /* Assign this data to the property array. */
    part_props[idim] = part_arr;
  }

  /* Loop over particles. */
  for (int p = 0; p < npart; p++) {

    /* Get this particle's mass. */
    const double mass = part_mass[p];

    /* Finally, compute the line for this particle using the
     * requested method. */
    if (strcmp(method, "cic") == 0) {
      line_loop_cic(grid_props, part_props, mass, grid_lines, grid_continuum,
                    dims, ndim, line_lum, line_cont, fesc[p], p);
    } else if (strcmp(method, "ngp") == 0) {
      line_loop_ngp(grid_props, part_props, mass, grid_lines, grid_continuum,
                    dims, ndim, line_lum, line_cont, fesc[p], p);
    } else {
      /* Only print this warning once */
      if (p == 0)
        printf(
            "Unrecognised gird assignment method (%s)! Falling back on CIC\n",
            method);
      line_loop_cic(grid_props, part_props, mass, grid_lines, grid_continuum,
                    dims, ndim, line_lum, line_cont, fesc[p], p);
    }
  }

  /* Clean up memory! */
  free(grid_weights);
  free(part_props);
  free(grid_props);

  /* Reconstruct the python array to return. */
  npy_intp np_dims[1] = {
      npart,
  };
  PyArrayObject *out_line = (PyArrayObject *)PyArray_SimpleNewFromData(
      1, np_dims, NPY_FLOAT64, line_lum);
  PyArrayObject *out_cont = (PyArrayObject *)PyArray_SimpleNewFromData(
      1, np_dims, NPY_FLOAT64, line_cont);

  return Py_BuildValue("(OO)", out_line, out_cont);
}

/* Below is all the gubbins needed to make the module importable in Python. */
static PyMethodDef LineMethods[] = {
    {"compute_particle_line", compute_particle_line, METH_VARARGS,
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
