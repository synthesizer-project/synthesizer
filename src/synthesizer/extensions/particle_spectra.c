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

/**
 * @brief The callback function to store the spectra for each particle.
 *
 * @param weight: The weight for this particle.
 * @param out: The particle spectra array.
 */
static void store_spectra(double weight, struct callback_data *data,
                          void *out) {

  /* Unpack the data. */
  const int *indices = data->indices;
  const int *dims = data->dims;
  const int ndim = data->ndim;
  const int nlam = data->nlam;
  const double *grid_spectra = data->grid_spectra;
  const int p = data->particle_index;
  const double fesc = data->fesc;

  /* Get the output array. */
  double *spectra = (double *)out;

  /* Get the weight's index. */
  const int grid_ind = get_flat_index(indices, dims, ndim);

  /* Get the spectra ind. */
  int unraveled_ind[ndim + 1];
  get_indices_from_flat(grid_ind, ndim, dims, unraveled_ind);
  unraveled_ind[ndim] = 0;
  int spectra_ind = get_flat_index(unraveled_ind, dims, ndim + 1);

  /* Add this grid cell's contribution to the spectra */
  for (int ilam = 0; ilam < nlam; ilam++) {

    /* Add the contribution to this wavelength. */
    spectra[p * nlam + ilam] +=
        grid_spectra[spectra_ind + ilam] * (1 - fesc) * weight;
  }
}

/**
 * @brief Computes an integrated SED for a collection of particles.
 *
 * @param np_grid_spectra: The SPS spectra array.
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
PyObject *compute_particle_seds(PyObject *self, PyObject *args) {

  /* We don't need the self argument but it has to be there. Tell the compiler
   * we don't care. */
  (void)self;

  int ndim, npart, nlam, nthreads;
  PyObject *grid_tuple, *part_tuple;
  PyArrayObject *np_grid_spectra;
  PyArrayObject *np_fesc;
  PyArrayObject *np_part_mass, *np_ndims;
  char *method;

  if (!PyArg_ParseTuple(args, "OOOOOOiiisi", &np_grid_spectra, &grid_tuple,
                        &part_tuple, &np_part_mass, &np_fesc, &np_ndims, &ndim,
                        &npart, &nlam, &method, &nthreads))
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

  /* With everything set up we can compute the weights for each particle using
   * the requested method. */
  double *spectra;
  if (strcmp(method, "cic") == 0) {
    spectra = weight_loop_cic(grid_props, part_props, npart * nlam,
                              store_spectra, nthreads);
  } else if (strcmp(method, "ngp") == 0) {
    spectra = weight_loop_ngp(grid_props, part_props, npart * nlam,
                              store_spectra, nthreads);
  } else {
    PyErr_SetString(PyExc_ValueError, "Unknown grid assignment method (%s).");
    return NULL;
  }

  /* Check we got the spectra sucessfully. (Any error messages will already be
   * set) */
  if (spectra == NULL) {
    return NULL;
  }

  /* Clean up memory! */
  free(part_props);
  free(grid_props);

  /* Reconstruct the python array to return. */
  npy_intp np_dims[2] = {
      npart,
      nlam,
  };
  PyArrayObject *out_spectra = (PyArrayObject *)PyArray_SimpleNewFromData(
      2, np_dims, NPY_FLOAT64, spectra);

  return Py_BuildValue("N", out_spectra);
}

/* Below is all the gubbins needed to make the module importable in Python. */
static PyMethodDef SedMethods[] = {
    {"compute_particle_seds", (PyCFunction)compute_particle_seds, METH_VARARGS,
     "Method for calculating particle intrinsic spectra."},
    {NULL, NULL, 0, NULL}};

/* Make this importable. */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "make_particle_sed",                   /* m_name */
    "A module to calculate particle seds", /* m_doc */
    -1,                                    /* m_size */
    SedMethods,                            /* m_methods */
    NULL,                                  /* m_reload */
    NULL,                                  /* m_traverse */
    NULL,                                  /* m_clear */
    NULL,                                  /* m_free */
};

PyMODINIT_FUNC PyInit_particle_spectra(void) {
  PyObject *m = PyModule_Create(&moduledef);
  import_array();
  return m;
}
