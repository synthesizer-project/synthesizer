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
 * @brief A Python capsule destructor function to free a buffer.
 *
 * This function is called when the capsule is deleted, and it frees
 * the buffer associated with the capsule.
 *
 * @param capsule The capsule object containing the buffer to free.
 */
static void _free_capsule(PyObject *capsule) {
  void *buf = PyCapsule_GetPointer(capsule, NULL);
  free(buf);
}

/**
 * @brief Wraps a malloc’ed buffer into a NumPy array, taking ownership.
 *
 * @param ndim     Number of dimensions
 * @param dims     Array of length ndim, giving each dimension size
 * @param typenum  NumPy typenum (e.g. NPY_FLOAT64)
 * @param buf      Pointer returned by malloc() (must be at least
 * product(dims)*itemsize)
 *
 * @return A new reference to a PyArrayObject which owns 'buf', or NULL on
 * error.
 */
static PyArrayObject *c_array_to_numpy(int ndim, npy_intp *dims, int typenum,
                                       void *out) {

  /* Create the new NumPy array from the buffer. */
  PyArrayObject *arr =
      (PyArrayObject *)PyArray_SimpleNewFromData(ndim, dims, typenum, out);
  if (!arr) {
    free(out);
    return NULL;
  }

  /* Create a capsule to hold the buffer, and set the destructor. */
  PyObject *capsule = PyCapsule_New(out, NULL, _free_capsule);
  if (!capsule) {
    Py_DECREF(arr);
    free(out);
    return NULL;
  }

  /* Tell NumPy to steal a reference to the capsule on array deletion */
  PyArray_SetBaseObject(arr, capsule);

  return arr;
}

#endif // PROPERTY_FUNCS_H_
