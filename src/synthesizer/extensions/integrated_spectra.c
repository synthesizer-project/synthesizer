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
#include "weights.h"

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
PyObject *compute_integrated_sed(PyObject *args) {

  int ndim = 0;
  int npart = 0, nlam = 0;
  PyObject *grid_tuple, *part_tuple;
  PyArrayObject *np_grid_spectra;
  PyArrayObject *np_fesc;
  PyArrayObject *np_part_mass, *np_ndims;
  char *method;

  if (!PyArg_ParseTuple(args, "OOOOOOiiis", &np_grid_spectra, &grid_tuple,
                        &part_tuple, &np_part_mass, &np_fesc, &np_ndims, &ndim,
                        &npart, &nlam, &method))
    return NULL;

  /* Quick check to make sure our inputs are valid. */
  if (ndim == 0) {
    PyErr_SetString(PyExc_ValueError, "ndim must be greater than 0.");
    return NULL;
  }
  if (npart == 0) {
    PyErr_SetString(PyExc_ValueError, "npart must be greater than 0.");
    return NULL;
  }
  if (nlam == 0) {
    PyErr_SetString(PyExc_ValueError, "nlam must be greater than 0.");
    return NULL;
  }

  /* Extract a pointer to the spectra grids */
  const double *grid_spectra = PyArray_DATA(np_grid_spectra);
  if (grid_spectra == NULL) {
    PyErr_SetString(PyExc_ValueError, "Failed to extract grid_spectra.");
    return NULL;
  }

  /* Set up arrays to hold the SEDs themselves. */
  double *spectra = malloc(nlam * sizeof(double));
  if (spectra == NULL) {
    PyErr_SetString(PyExc_ValueError, "Failed to allocate memory for spectra.");
    return NULL;
  }
  bzero(spectra, nlam * sizeof(double));

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

  /* How many grid elements are there? (excluding wavelength axis)*/
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
    PyArrayObject *np_grid_arr =
        (PyArrayObject *)PyTuple_GetItem(grid_tuple, idim);
    if (np_grid_arr == NULL) {
      PyErr_SetString(PyExc_ValueError, "Failed to extract grid_arr.");
      return NULL;
    }
    const double *grid_arr = PyArray_DATA(np_grid_arr);
    if (grid_arr == NULL) {
      PyErr_SetString(PyExc_ValueError, "Failed to extract grid_arr.");
      return NULL;
    }

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
    PyArrayObject *np_part_arr =
        (PyArrayObject *)PyTuple_GetItem(part_tuple, idim);
    if (np_part_arr == NULL) {
      PyErr_SetString(PyExc_ValueError, "Failed to extract part_arr.");
      return NULL;
    }
    const double *part_arr = PyArray_DATA(np_part_arr);
    if (part_arr == NULL) {
      PyErr_SetString(PyExc_ValueError, "Failed to extract part_arr.");
      return NULL;
    }

    /* Assign this data to the property array. */
    part_props[idim] = part_arr;
  }

  /* Loop over particles. */
  for (int p = 0; p < npart; p++) {

    /* Get this particle's mass. */
    const double mass = part_mass[p];

    /* Finally, compute the weights for this particle using the
     * requested method. */
    if (strcmp(method, "cic") == 0) {
      weight_loop_cic(grid_props, part_props, mass, grid_weights, dims, ndim, p,
                      fesc[p]);
    } else if (strcmp(method, "ngp") == 0) {
      weight_loop_ngp(grid_props, part_props, mass, grid_weights, dims, ndim, p,
                      fesc[p]);
    } else {
      /* Only print this warning once! */
      if (p == 0)
        printf(
            "Unrecognised gird assignment method (%s)! Falling back on CIC\n",
            method);
      weight_loop_cic(grid_props, part_props, mass, grid_weights, dims, ndim, p,
                      fesc[p]);
    }

  } /* Loop over particles. */

  /* Loop over grid cells. */
  for (int grid_ind = 0; grid_ind < grid_size; grid_ind++) {

    /* Get the weight. */
    const double weight = grid_weights[grid_ind];

    /* Skip zero weight cells. */
    if (weight <= 0)
      continue;

    /* Get the spectra ind. */
    int unraveled_ind[ndim + 1];
    get_indices_from_flat(grid_ind, ndim, dims, unraveled_ind);
    unraveled_ind[ndim] = 0;
    int spectra_ind = get_flat_index(unraveled_ind, dims, ndim + 1);

    /* Add this grid cell's contribution to the spectra */
    for (int ilam = 0; ilam < nlam; ilam++) {

      /* Add the contribution to this wavelength. */
      /* fesc is already included in the weight */
      spectra[ilam] += grid_spectra[spectra_ind + ilam] * weight;
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
