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
#include "weights.h"

struct lines_output {
  double *line_lum;
  double *line_cont;
};

/**
 * @brief The callback function to store the lines for each particle.
 *
 * @param weight: The weight for this particle.
 * @param out: The particle spectra array.
 */
static void store_lines(double weight, struct callback_data *data, void *out) {

  /* Unpack the data. */
  const int *indices = data->indices;
  const int *dims = data->dims;
  const int ndim = data->ndim;
  const int nlam = data->nlam;
  const double *grid_lines = data->grid_lines;
  const double *grid_continuum = data->grid_continuum;
  const int p = data->particle_index;
  const double fesc = data->fesc;

  /* Get the output arrays. */
  struct lines_output *lines = (struct lines_output *)out;
  double *line_lum = lines->line_lum;
  double *line_cont = lines->line_cont;

  /* We have a contribution, get the flattened index into the grid array. */
  const int grid_ind = get_flat_index(indices, dims, ndim);

  /* Add the contribution to this particle. */
  line_lum[p] += grid_lines[grid_ind] * (1 - fesc) * weight;
  line_cont[p] += grid_continuum[grid_ind] * (1 - fesc) * weight;
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
 * @param fesc: The escape fraction.
 * @param np_ndims: The size of each grid axis.
 * @param ndim: The number of grid axes.
 * @param npart: The number of particles.
 * @param method: The method to use for assigning weights.
 */
PyObject *compute_particle_line(PyObject *self, PyObject *args) {

  /* We don't need the self argument but it has to be there. Tell the compiler
   * we don't care. */
  (void)self;

  int ndim, npart;
  PyObject *grid_tuple, *part_tuple;
  PyArrayObject *np_grid_lines, *np_grid_continuum;
  PyArrayObject *np_fesc;
  PyArrayObject *np_part_mass, *np_ndims;
  char *method;

  if (!PyArg_ParseTuple(args, "OOOOOOOiis", &np_grid_lines, &np_grid_continuum,
                        &grid_tuple, &part_tuple, &np_part_mass, &np_fesc,
                        &np_ndims, &ndim, &npart, &method))
    /* Error message is already set here. */
    return NULL;

  /* Extract the grid struct. */
  struct grid *grid_props = get_lines_grid_struct(
      grid_tuple, np_ndims, np_grid_lines, np_grid_continuum, ndim, /*nlam*/ 0);

  /* Extract the particle struct. */
  struct particles *part_props =
      get_part_struct(part_tuple, np_part_mass, np_fesc, npart, ndim);

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

  /* Create the lines output struct. */
  struct lines_output lines = {.line_lum = line_lum, .line_cont = line_cont};

  /* With everything set up we can compute the weights for each particle using
   * the requested method. */
  if (strcmp(method, "cic") == 0) {
    weight_loop_cic(grid_props, part_props, &lines, store_lines);
  } else if (strcmp(method, "ngp") == 0) {
    weight_loop_ngp(grid_props, part_props, &lines, store_lines);
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
      1, np_dims, NPY_FLOAT64, line_lum);
  PyArrayObject *out_cont = (PyArrayObject *)PyArray_SimpleNewFromData(
      1, np_dims, NPY_FLOAT64, line_cont);

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
