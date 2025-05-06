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

/**
 * @brief Allocate an array.
 *
 * Just a wrapper around malloc with a check for NULL.
 *
 * @param n: The number of pointers to allocate.
 */
template <typename T> T *synth_malloc(size_t n, const char *msg) {
  T *ptr = reinterpret_cast<T *>(malloc(n));
  if (ptr == NULL) {
    char error_msg[100];
    snprintf(error_msg, sizeof(error_msg), "Failed to allocate memory for %s.",
             msg);
    PyErr_SetString(PyExc_MemoryError, error_msg);
  }
  bzero(ptr, n);
  return ptr;
}

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

/**
 * @brief A class to hold particle related numpy arrays with getters and
 * setters.
 *
 * This is used to hold the particle properties and mass.
 */
class Particles {
public:
  /* The number of particles. */
  int npart;

  /* Constructor */
  Particles(PyArrayObject *np_weights, PyArrayObject *np_velocities,
            PyArrayObject *np_mask, PyObject *part_tuple, int npart);

  /* Destructor */
  ~Particles();

  /* Prototypes for getters. */
  double *get_weights() const;
  double *get_velocities() const;
  double **get_all_props(int ndim) const;
  double *get_part_props(int idim) const;
  double get_weight_at(int pind) const;
  double get_vel_at(int pind) const;
  npy_bool get_mask_at(int pind) const;
  double get_part_prop_at(int idim, int pind) const;

  /* Is a particle masked? */
  bool part_is_masked(int pind) const;

private:
  /* The numpy array holding the particle weights (e.g. initial mass for
   * SPS grid weighting). */
  PyArrayObject *np_weights_;

  /* The numpy array holding the particle velocities. */
  PyArrayObject *np_velocities_;

  /* The mask (can be Py_None). */
  PyArrayObject *np_mask_;

  /* The particle properties corresponding to the grid axes, this is a tuple
   * of numpy arrays. */
  PyObject *part_tuple_;
};

static inline double get_double_at(PyArrayObject *np_arr, int ind) {
  return *reinterpret_cast<double *>(PyArray_GETPTR1(np_arr, ind));
}

static inline int get_int_at(PyArrayObject *np_arr, int ind) {
  return *reinterpret_cast<int *>(PyArray_GETPTR1(np_arr, ind));
}

static inline npy_bool get_bool_at(PyArrayObject *np_arr, int ind) {
  return *reinterpret_cast<npy_bool *>(PyArray_GETPTR1(np_arr, ind));
}

/* Prototypes */
double *extract_data_double(PyArrayObject *np_arr, const char *name);
int *extract_data_int(PyArrayObject *np_arr, const char *name);
npy_bool *extract_data_bool(PyArrayObject *np_arr, const char *name);
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
