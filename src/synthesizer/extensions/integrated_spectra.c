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
  const double fesc = data->fesc;

  /* Unravel the indices. */
  int flat_ind = get_flat_index(indices, dims, ndim);

  /* Get the output array. */
  double *out_arr = (double *)out;

  /* Store the weight. */
  out_arr[flat_ind] = weight * (1.0 - fesc);
}

/**
 * @brief Computes an integrated SED for a collection of particles.
 *
 * @param np_grid_spectra: The SPS spectra array.
 * o
 * @param grid_tuple: The tuple containing arrays of grid axis properties.
 * @param part_tuple: The tuple of particle property arrays (in the same order
 *                    as grid_tuple).
 * @param np_part_mass: The particle mass array.
 * @param fesc: The escape fraction.
 * @param np_ndims: The size of each grid axis.
 * @param ndim: The number of grid axes.
 * @param npart: The number of particles.
 * @param nlam: The number of wavelength elements.
 */
PyObject *compute_integrated_sed(PyObject *self, PyObject *args) {

  /* We don't need the self argument but it has to be there. Tell the compiler
   * we don't care. */
  (void)self;

  int ndim, npart, nlam;
  PyObject *grid_tuple, *part_tuple;
  PyArrayObject *np_grid_spectra;
  PyArrayObject *np_fesc;
  PyArrayObject *np_part_mass, *np_ndims;
  char *method;

  if (!PyArg_ParseTuple(args, "OOOOOOiiis", &np_grid_spectra, &grid_tuple,
                        &part_tuple, &np_part_mass, &np_fesc, &np_ndims, &ndim,
                        &npart, &nlam, &method))
    return NULL;

  /* Extract the grid struct. */
  struct grid *grid_props = get_spectra_grid_struct(
      grid_tuple, np_ndims, np_grid_spectra, ndim, nlam);
  if (grid_props == NULL) {
    return NULL;
  }

  /* Extract the particle struct. */
  struct particles *part_props =
      get_part_struct(part_tuple, np_part_mass, np_fesc, npart, ndim);
  if (part_props == NULL) {
    return NULL;
  }

  /* Allocate an array to hold the grid weights. */
  double *grid_weights = malloc(grid_props->size * sizeof(double));
  if (grid_weights == NULL) {
    PyErr_SetString(PyExc_MemoryError,
                    "Failed to allocate memory for grid_weights.");
    return NULL;
  }
  bzero(grid_weights, grid_props->size * sizeof(double));

  /* Set up arrays to hold the SEDs themselves. */
  double *spectra = malloc(nlam * sizeof(double));
  if (spectra == NULL) {
    PyErr_SetString(PyExc_ValueError, "Failed to allocate memory for spectra.");
    return NULL;
  }
  bzero(spectra, nlam * sizeof(double));

  /* With everything set up we can compute the weights for each particle using
   * the requested method. */
  if (strcmp(method, "cic") == 0) {
    weight_loop_cic(grid_props, part_props, grid_weights, store_weight);
  } else if (strcmp(method, "ngp") == 0) {
    weight_loop_ngp(grid_props, part_props, grid_weights, store_weight);
  } else {
    PyErr_SetString(PyExc_ValueError, "Unknown grid assignment method (%s).");
    return NULL;
  }

  /* Loop over grid cells. */
  for (int grid_ind = 0; grid_ind < grid_props->size; grid_ind++) {

    /* Get the weight. */
    const double weight = grid_weights[grid_ind];

    /* Skip zero weight cells. */
    if (weight <= 0)
      continue;

    /* Get the spectra ind. */
    int unraveled_ind[ndim + 1];
    get_indices_from_flat(grid_ind, grid_props->ndim, grid_props->dims,
                          unraveled_ind);
    unraveled_ind[ndim] = 0;
    int spectra_ind =
        get_flat_index(unraveled_ind, grid_props->dims, grid_props->ndim + 1);

    /* Add this grid cell's contribution to the spectra */
    for (int ilam = 0; ilam < grid_props->nlam; ilam++) {

      /* Add the contribution to this wavelength. */
      /* fesc is already included in the weight */
      spectra[ilam] += grid_props->spectra[spectra_ind + ilam] * weight;
    }
  }

  /* Clean up memory! */
  free(grid_weights);
  free(part_props);
  free(grid_props);

  /* Reconstruct the python array to return. */
  npy_intp np_dims[1] = {
      nlam,
  };
  PyArrayObject *out_spectra = (PyArrayObject *)PyArray_SimpleNewFromData(
      1, np_dims, NPY_FLOAT64, spectra);

  return Py_BuildValue("N", out_spectra);
}

static PyMethodDef SedMethods[] = {
    {"compute_integrated_sed", (PyCFunction)compute_integrated_sed,
     METH_VARARGS, "Method for calculating integrated intrinsic spectra."},
    {NULL, NULL, 0, NULL}};

/* Make this importable. */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "make_sed",                              /* m_name */
    "A module to calculate integrated seds", /* m_doc */
    -1,                                      /* m_size */
    SedMethods,                              /* m_methods */
    NULL,                                    /* m_reload */
    NULL,                                    /* m_traverse */
    NULL,                                    /* m_clear */
    NULL,                                    /* m_free */
};

PyMODINIT_FUNC PyInit_integrated_spectra(void) {
  PyObject *m = PyModule_Create(&moduledef);
  import_array();
  return m;
}
