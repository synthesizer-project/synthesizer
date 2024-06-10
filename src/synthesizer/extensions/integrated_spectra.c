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
#include "hashmap.h"
#include "macros.h"
#include "property_funcs.h"
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

  /* Extract a pointer to the grid dims */
  const int *dims = extract_data_int(np_ndims, "dims");
  if (dims == NULL) {
    return NULL;
  }

  /* Extract a pointer to the particle masses. */
  const double *part_mass = extract_data_double(np_part_mass, "part_mass");
  if (part_mass == NULL) {
    return NULL;
  }

  /* Extract a pointer to the fesc array. */
  const double *fesc = extract_data_double(np_fesc, "fesc");
  if (fesc == NULL) {
    return NULL;
  }

  /* Extract the grid properties from the tuple of numpy arrays. */
  const double **grid_props = extract_grid_props(grid_tuple, ndim, dims);
  if (grid_props == NULL) {
    return NULL;
  }

  /* Extract the particle properties from the tuple of numpy arrays. */
  const double **part_props = extract_part_props(part_tuple, ndim, npart);
  if (part_props == NULL) {
    return NULL;
  }

  /* With everything set up we can compute the weights for each particle using
   * the requested method. */
  HashMap *weights;
  if (strcmp(method, "cic") == 0) {
    weights = weight_loop_cic(grid_props, part_props, part_mass, dims, ndim,
                              npart, /*per_part*/ 0);
  } else if (strcmp(method, "ngp") == 0) {
    weights = weight_loop_ngp(grid_props, part_props, part_mass, dims, ndim,
                              npart, /*per_part*/ 0);
  } else {
    PyErr_SetString(PyExc_ValueError, "Unknown grid assignment method (%s).");
    return NULL;
  }

  /* Extract a pointer to the spectra grids */
  const double *grid_spectra =
      extract_data_double(np_grid_spectra, "grid_spectra");
  if (grid_spectra == NULL) {
    return NULL;
  }

  /* Set up arrays to hold the SEDs themselves. */
  double *spectra = malloc(nlam * sizeof(double));
  if (spectra == NULL) {
    PyErr_SetString(PyExc_ValueError, "Failed to allocate memory for spectra.");
    return NULL;
  }
  bzero(spectra, nlam * sizeof(double));

  /* Populate the integrated spectra. */
  for (int i = 0; i < weights->node_pool_used; i++) {
    /* Get the hash map node. */
    Node *node = weights->node_pool[i];

    /* Get the weight and indices. */
    const double weight = node->value;
    const IndexKey key = node->key;
    const int *grid_ind = key.grid_indices;
    const int p = key.particle_index;

    /* Get the spectra ind. */
    int unraveled_ind[ndim + 1];
    memcpy(unraveled_ind, grid_ind, ndim * sizeof(int));
    unraveled_ind[ndim] = 0;
    int spectra_ind = get_flat_index(unraveled_ind, dims, ndim + 1);

    /* Add this grid cell's contribution to the spectra */
    for (int ilam = 0; ilam < nlam; ilam++) {

      /* Add the contribution to this wavelength. */
      /* fesc is already included in the weight */
      spectra[ilam] +=
          grid_spectra[spectra_ind + ilam] * weight * (1.0 - fesc[p]);
    }
  }

  /* Clean up memory! */
  free_hash_map(weights);
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
