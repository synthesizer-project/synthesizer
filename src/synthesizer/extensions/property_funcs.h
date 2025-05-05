/******************************************************************************
 * A C module containing helper functions for extracting properties from the
 * numpy objects.
 *****************************************************************************/
#ifndef PROPERTY_FUNCS_H_
#define PROPERTY_FUNCS_H_

/* Standard includes */
#include <stdlib.h>

/* We need the below because numpy triggers warnings which are errors
 * when we compiled with RUTHLESS. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

/* Python includes */
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>

#pragma GCC diagnostic pop

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

  /* Wavelength */
  double *lam;

  /* The mask array denoting which wavelength elements should be included. */
  npy_bool *lam_mask;
};

/* A struct to hold particle properties. */
struct particles {

  /* An array of pointers holding the properties along each axis. */
  double **props;

  /* The number of particles. */
  int npart;

  /* The particle mass array. */
  double *mass;

  /* Velocities for redshift */
  double *velocities;

  /* The mask array denoting which particles should be included. */
  npy_bool *mask;
};

/* Prototypes */
void *synth_malloc(size_t n, char *msg);
double *extract_data_double(PyArrayObject *np_arr, char *name);
int *extract_data_int(PyArrayObject *np_arr, char *name);
npy_bool *extract_data_bool(PyArrayObject *np_arr, char *name);
double **extract_grid_props(PyObject *grid_tuple, int ndim, int *dims);
double **extract_part_props(PyObject *part_tuple, int ndim, int npart);
struct grid *get_spectra_grid_struct(PyObject *grid_tuple,
                                     PyArrayObject *np_ndims,
                                     PyArrayObject *np_grid_spectra,
                                     PyArrayObject *np_lam,
                                     PyArrayObject *np_lam_mask, const int ndim,
                                     const int nlam);
struct grid *get_lines_grid_struct(PyObject *grid_tuple,
                                   PyArrayObject *np_ndims,
                                   PyArrayObject *np_grid_lines,
                                   PyArrayObject *np_grid_continuum,
                                   const int ndim, const int nlam);
struct particles *get_part_struct(PyObject *part_tuple,
                                  PyArrayObject *np_part_mass,
                                  PyArrayObject *np_velocities,
                                  PyArrayObject *np_mask, const int npart,
                                  const int ndim);

/**
 * @brief Wraps a malloc’ed buffer into a NumPy array, taking ownership.
 *
 * @param ndim     Number of dimensions
 * @param dims     Array of length ndim, giving each dimension size
 * @param typenum  NumPy typenum (e.g. NPY_FLOAT64)
 * @param out      Pointer returned by malloc() (must be at least
 *                 product(dims)*itemsize)
 *
 * @return A new reference to a PyArrayObject which owns 'out', or NULL on
 *         error (buffer is freed on error).
 */
static PyArrayObject *c_array_to_numpy(int ndim, npy_intp *dims, int typenum,
                                       void *out) {
  PyArray_Descr *descr = NULL;
  PyArrayObject *arr = NULL;

  /* Build a dtype descriptor from the typenum */
  descr = PyArray_DescrFromType(typenum);
  if (!descr) {
    free(out);
    return NULL;
  }

  /* Create the numpy array:
   *    - out: the malloc’d memory
   *    - NPY_ARRAY_CARRAY: ensure C-contiguous, aligned, writeable
   *    - NPY_ARRAY_OWNDATA: NumPy will call free(out) when the array dies
   */
  arr = (PyArrayObject *)PyArray_NewFromDescr(
      &PyArray_Type, descr, ndim, dims, NULL, out,
      NPY_ARRAY_CARRAY | NPY_ARRAY_OWNDATA, NULL);
  if (!arr) {
    /* On failure, descriptor was already DECREF’d */
    free(out);
    return NULL;
  }

  return arr;
}

#endif // PROPERTY_FUNCS_H_
