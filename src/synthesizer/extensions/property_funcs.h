/******************************************************************************
 * A C module containing helper functions for extracting properties from the
 * numpy objects.
 *****************************************************************************/
#ifndef PROPERTY_FUNCS_H_
#define PROPERTY_FUNCS_H_

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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
};

/* A struct to hold particle properties. */
struct particles {

  /* An array of pointers holding the properties along each axis. */
  double **props;

  /* Convinience counter for the number of properties. */
  int nprops;

  /* The number of particles. */
  int npart;

  /* The particle mass array. */
  double *mass;

  /* The weight of the particle to be sorted into the grid when computing
   * an Sed from a Grid. */
  double *weight;

  /* Velocities for redshift */
  double *velocities;
};

/**
 * Helper function to extract a NumPy array of doubles from a PyObject.
 *
 * @param param   The name of the attribute to look for.
 * @param obj     The PyObject to search (may be NULL).
 * @param out_size Pointer to a npy_intp that will be set to the number of
 * elements in the array if found.
 *
 * @return A newly allocated C array of doubles (copied from the NumPy array)
 *         or NULL if the attribute isn’t found or isn’t a NumPy array of
 * doubles. The caller is responsible for freeing the returned memory.
 */
static double *get_numpy_array_from_pyobject(const char *param, PyObject *obj,
                                             npy_intp *out_size) {
  if (obj && PyObject_HasAttrString(obj, param)) {
    PyObject *attr = PyObject_GetAttrString(obj, param);
    if (attr && PyArray_Check(attr)) {
      PyArrayObject *array = (PyArrayObject *)attr;
      // Ensure the array is of type double.
      if (PyArray_TYPE(array) == NPY_DOUBLE) {
        // For simplicity we assume a 1-D array.
        if (PyArray_NDIM(array) == 1) {
          *out_size = PyArray_SIZE(array);
          double *data_ptr = (double *)PyArray_DATA(array);
          // Allocate a new C array and copy the data.
          double *result = malloc((*out_size) * sizeof(double));
          if (result != NULL) {
            memcpy(result, data_ptr, (*out_size) * sizeof(double));
          }
          Py_DECREF(attr);
          return result;
        }
      }
    }
    Py_XDECREF(attr);
  }
  return NULL;
}

/**
 * Extract a parameter (a NumPy array of doubles) from the given PyObjects.
 *
 * The search priority is:
 *   1. model
 *   2. emission
 *   3. emitter
 *
 * @param param     A string specifying the parameter name.
 * @param model     A PyObject pointer (or NULL) for the model.
 * @param emission  A PyObject pointer (or NULL) for the emission.
 * @param emitter   A PyObject pointer (or NULL) for the emitter.
 * @param out_size  A pointer to a npy_intp that will receive the number of
 * elements.
 *
 * @return A newly allocated C array of doubles containing the parameter’s data,
 *         or NULL if not found. The caller is responsible for freeing the
 * memory.
 */
double *get_param_array(const char *param, PyObject *model, PyObject *emission,
                        PyObject *emitter, npy_intp *out_size) {
  double *result = NULL;
  *out_size = 0;

  // Priority 1: check the model.
  result = get_numpy_array_from_pyobject(param, model, out_size);
  if (result != NULL)
    return result;

  // Priority 2: check the emission.
  result = get_numpy_array_from_pyobject(param, emission, out_size);
  if (result != NULL)
    return result;

  // Priority 3: check the emitter.
  result = get_numpy_array_from_pyobject(param, emitter, out_size);
  return result;
}

/**
 * Convert any PyObject* pointers that point to Py_None to NULL.
 *
 * This function takes a count of pointer arguments followed by that many
 * pointers to PyObject* (i.e. PyObject **). For each pointer, if it is not
 * NULL and its value is Py_None, it sets the value to NULL.
 *
 * @param count The number of pointer arguments.
 * @param ...   A variable list of PyObject ** arguments.
 */
static inline void convert_none_to_null(int count, ...) {
  va_list args;
  va_start(args, count);
  for (int i = 0; i < count; i++) {
    PyObject **pObj = va_arg(args, PyObject **);
    if (pObj && *pObj == Py_None) {
      *pObj = NULL;
    }
  }
  va_end(args);
}

/* Prototypes */
void *synth_malloc(size_t n, char *msg);
double *extract_data_double(PyArrayObject *np_arr, char *name);
int *extract_data_int(PyArrayObject *np_arr, char *name);
int *extract_data_bool_as_int(PyArrayObject *np_arr, char *name);
double **extract_grid_props(PyObject *grid_tuple, int ndim, int *dims);
double **extract_part_props(PyObject *part_tuple, int ndim, int npart);
struct grid *get_spectra_grid_struct(PyObject *grid_tuple,
                                     PyArrayObject *np_ndims,
                                     PyArrayObject *np_grid_spectra,
                                     PyArrayObject *np_lam, const int ndim,
                                     const int nlam);
struct grid *get_lines_grid_struct(PyObject *grid_tuple,
                                   PyArrayObject *np_ndims,
                                   PyArrayObject *np_grid_lines,
                                   PyArrayObject *np_grid_continuum,
                                   const int ndim, const int nlam);
struct particles *get_part_struct(PyObject *part_tuple,
                                  PyArrayObject *np_part_mass,
                                  PyArrayObject *np_velocities, const int npart,
                                  const int ndim);
struct particles *get_part_struct_from_obj(PyObject *parts, PyObject *grid,
                                           const char *weight_var);
struct grid *get_grid_struct_from_obj(PyObject *py_grid);

#endif // PROPERTY_FUNCS_H_
