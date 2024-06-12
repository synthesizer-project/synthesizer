/******************************************************************************
 * A C module containing helper functions for extracting properties from the
 * numpy objects.
 *****************************************************************************/
#ifndef PROPERTY_FUNCS_H_
#define PROPERTY_FUNCS_H_

/* Python includes */
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>

/* A struct to hold grid properties. */
struct grid {

  /* An array of pointers holding the properties along each axis. */
  double **props;

  /* The number of dimensions. */
  int ndim;

  /* The number of grid cells along each axis. */
  int *dims;

  /* The number of wavelength elements. */
  int nlam;

  /* The number of cells. */
  int size;

  /* The spectra array. */
  double *spectra;

  /* The lines array. */
  double *lines;

  /* The continuum array. */
  double *continuum;
};

/* A struct to hold particle properties. */
struct particles {

  /* An array of pointers holding the properties along each axis. */
  double **props;

  /* The number of particles. */
  int npart;

  /* The particle mass array. */
  double *mass;

  /* Escape fractions. */
  double *fesc;
};

/* Prototypes */
const double *extract_data_double(PyArrayObject *np_arr, char *name);
const int *extract_data_int(PyArrayObject *np_arr, char *name);
const double **extract_grid_props(PyObject *grid_tuple, int ndim, int *dims);
const double **extract_part_props(PyObject *part_tuple, int ndim, int npart);
struct grid *get_spectra_grid_struct(PyObject *grid_tuple,
                                     PyArrayObject *np_ndims,
                                     PyArrayObject *np_grid_spectra,
                                     const int ndim, const int nlam);
struct grid *get_lines_grid_struct(PyObject *grid_tuple,
                                   PyArrayObject *np_ndims,
                                   PyArrayObject *np_grid_lines,
                                   PyArrayObject *np_grid_continuum,
                                   const int ndim, const int nlam);
struct particles *get_part_struct(PyObject *part_tuple,
                                  PyArrayObject *np_part_mass,
                                  PyArrayObject *np_fesc, const int npart,
                                  const int ndim);

#endif // PROPERTY_FUNCS_H_
