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
 * @brief This calculates the spectra of a particle using a cloud in cell
 *        approach.
 *
 * @param grid_props: An array of the properties along each grid axis.
 * @param part_props: An array of the particle properties, in the same property
 *                    order as grid props.
 * @param mass: The mass of the current particle.
 * @param grid_spectra: The grid of SPS spectra.
 * @param dims: The length of each grid dimension.
 * @param ndim: The number of grid dimensions.
 * @param spectra: The array of particle spectra.
 * @param nlam: The number of wavelength elements.
 * @param fesc: The escape fraction.
 * @param p: The index of the current particle.
 */
void spectra_loop_cic(const double **grid_props, const double **part_props,
                      const double mass, const double *grid_spectra,
                      const int *dims, const int ndim, double *spectra,
                      const int nlam, const double fesc, const int p) {

  /* Setup the index and mass fraction arrays. */
  int frac_indices[(int)pow(2, (double)ndim)][ndim];
  double fracs[(int)pow(2, (double)ndim)];

  /* Set up the fractions. */
  for (int icell = 0; icell < (int)pow(2, (double)ndim); icell++) {
    fracs[icell] = 1;
  }

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
      frac = 1;

    } else if (part_val > grid_prop[dims[dim] - 1]) {

      /* Use the grid edge. */
      part_cell = dims[dim] - 1;
      frac = 1;

    } else {

      /* Find the grid index corresponding to this particle property. */
      part_cell =
          binary_search(/*low*/ 1, /*high*/ dims[dim] - 1, grid_prop, part_val);

      /* Calculate the fraction. Note, here we do the cell containing the
       * particle, the cell below is calculated from this fraction. */
      frac = (part_val - grid_prop[part_cell - 1]) /
             (grid_prop[part_cell] - grid_prop[part_cell - 1]);
    }

    /* Set the fractions. */
    fracs[dim * 2 + 1] *= frac;
    fracs[dim * 2] *= 1 - frac;

    /* Set these indices. */
    for (int jdim = 0; jdim < ndim; jdim++) {
      frac_indices[jdim * 2][dim] = part_cell - 1;
      frac_indices[jdim * 2 + 1][dim] = part_cell;
    }
  }

  /* Now loop over this collection of cells collecting and setting their
   * weights. */
  for (int icell = 0; icell < (int)pow(2, (double)ndim); icell++) {

    /* Early skip for cells contributing a 0 fraction. */
    if (fracs[icell] <= 0)
      continue;

    /* We have a contribution, get the flattened index into the grid array. */
    const int grid_ind = get_flat_index(frac_indices[icell], dims, ndim);

    /* Define the weight. */
    double weight = mass * fracs[icell];

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
}

/**
 * @brief This calculates the spectra of a particle using a nearest grid point
 *        approach.
 *
 * @param grid_props: An array of the properties along each grid axis.
 * @param part_props: An array of the particle properties, in the same property
 *                    order as grid props.
 * @param mass: The mass of the current particle.
 * @param grid_spectra: The grid of SPS spectra.
 * @param dims: The length of each grid dimension.
 * @param ndim: The number of grid dimensions.
 * @param spectra: The array of particle spectra.
 * @param nlam: The number of wavelength elements.
 * @param fesc: The escape fraction.
 * @param p: The index of the current particle.
 */
void spectra_loop_ngp(const double **grid_props, const double **part_props,
                      const double mass, const double *grid_spectra,
                      const int *dims, const int ndim, double *spectra,
                      const int nlam, const double fesc, const int p) {

  /* Setup the index array. */
  int part_indices[ndim][ndim];

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

    /* Set these indices. */
    for (int jdim = 0; jdim < ndim; jdim++) {
      part_indices[jdim * 2][dim] = part_cell - 1;
      part_indices[jdim * 2 + 1][dim] = part_cell;
    }
  }

  /* Now loop over this collection of cells collecting and setting their
   * weights. */
  for (int icell = 0; icell < (int)pow(2, (double)ndim); icell++) {

    /* We have a contribution, get the flattened index into the grid array. */
    const int grid_ind = get_flat_index(part_indices[icell], dims, ndim);

    /* Define the weight. */
    double weight = mass;

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

  const int ndim;
  const int npart, nlam;
  const double fesc;
  const PyObject *grid_tuple, *part_tuple;
  const PyArrayObject *np_grid_spectra;
  const PyArrayObject *np_part_mass, *np_ndims;
  const char *method;

  if (!PyArg_ParseTuple(args, "OOOOdOiii", &np_grid_spectra, &grid_tuple,
                        &part_tuple, &np_part_mass, &fesc, &np_ndims, &ndim,
                        &npart, &nlam, &method))
    return NULL;

  /* Quick check to make sure our inputs are valid. */
  if (ndim == 0)
    return NULL;
  if (npart == 0)
    return NULL;
  if (nlam == 0)
    return NULL;

  /* Extract a pointer to the spectra grids */
  const double *grid_spectra = PyArray_DATA(np_grid_spectra);

  /* Set up arrays to hold the SEDs themselves. */
  double *spectra = malloc(npart * nlam * sizeof(double));
  bzero(spectra, npart * nlam * sizeof(double));

  /* Extract a pointer to the grid dims */
  const int *dims = PyArray_DATA(np_ndims);

  /* Extract a pointer to the particle masses. */
  const double *part_mass = PyArray_DATA(np_part_mass);

  /* Allocate a single array for grid properties*/
  int nprops = 0;
  for (int dim = 0; dim < ndim; dim++)
    nprops += dims[dim];
  const double **grid_props = malloc(nprops * sizeof(double *));

  /* How many grid elements are there? (excluding wavelength axis)*/
  int grid_size = 1;
  for (int dim = 0; dim < ndim; dim++)
    grid_size *= dims[dim];

  /* Allocate an array to hold the grid weights. */
  double *grid_weights = malloc(grid_size * sizeof(double));
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

    /* Finally, compute the spectra for this particle using the
     * requested method. */
    if (strcmp(method, "cic") == 0) {
      spectra_loop_cic(grid_props, part_props, mass, grid_spectra, dims, ndim,
                       spectra, nlam, fesc, p);
    } else if (strcmp(method, "ngp") == 0) {
      spectra_loop_ngp(grid_props, part_props, mass, grid_spectra, dims, ndim,
                       spectra, nlam, fesc, p);
    } else {
      printf("Unrecognised gird assignment method (%s)! Falling back on CIC",
             method);
      spectra_loop_cic(grid_props, part_props, mass, grid_spectra, dims, ndim,
                       spectra, nlam, fesc, p);
    }
  }

  /* Clean up memory! */
  free(grid_weights);
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
    {"compute_particle_seds", compute_particle_seds, METH_VARARGS,
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
