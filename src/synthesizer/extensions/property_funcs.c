/******************************************************************************
 * A C module containing helper functions for extracting properties from the
 * numpy objects.
 *****************************************************************************/

/* C headers. */
#include <string.h>

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

/* Header */
#include "property_funcs.h"

/**
 * @brief Allocate an array.
 *
 * Just a wrapper around malloc with a check for NULL.
 *
 * @param n: The number of pointers to allocate.
 */
void *synth_malloc(size_t n, char *msg) {
  void *ptr = malloc(n);
  if (ptr == NULL) {
    char error_msg[100];
    snprintf(error_msg, sizeof(error_msg), "Failed to allocate memory for %s.",
             msg);
    PyErr_SetString(PyExc_MemoryError, error_msg);
  }
  bzero(ptr, n);
  return ptr;
}

/**
 * @brief Extract double data from a numpy array.
 *
 * @param np_arr: The numpy array to extract.
 * @param name: The name of the numpy array. (For error messages)
 */
double *extract_data_double(PyArrayObject *np_arr, char *name) {

  /* Extract a pointer to the spectra grids */
  double *data = PyArray_DATA(np_arr);
  if (data == NULL) {
    char error_msg[100];
    snprintf(error_msg, sizeof(error_msg), "Failed to extract %s.", name);
    PyErr_SetString(PyExc_ValueError, error_msg);
  }
  /* Success. */
  return data;
}

/**
 * @brief Extract int data from a numpy array.
 *
 * @param np_arr: The numpy array to extract.
 * @param name: The name of the numpy array. (For error messages)
 */
int *extract_data_int(PyArrayObject *np_arr, char *name) {

  /* Extract a pointer to the spectra grids */
  int *data = PyArray_DATA(np_arr);
  if (data == NULL) {
    char error_msg[100];
    snprintf(error_msg, sizeof(error_msg), "Failed to extract %s.", name);
    PyErr_SetString(PyExc_ValueError, error_msg);
  }
  /* Success. */
  return data;
}

/**
 * @brief Extract int data from a NumPy array of booleans.
 *
 * This function converts a NumPy array of booleans to a newly allocated C array
 * of integers. Each element in the boolean array is converted to an integer: 1
 * if True, 0 if False.
 *
 * @param np_arr The NumPy array of booleans to convert.
 * @param name   The name of the NumPy array (used in error messages).
 *
 * @return int*  A pointer to the newly allocated integer array, or NULL on
 * error.
 */
int *extract_data_bool_as_int(PyArrayObject *np_arr, char *name) {
  // Check that the input array is not NULL.
  if (!np_arr) {
    PyErr_Format(PyExc_ValueError, "Null numpy array passed to %s", name);
    return NULL;
  }

  // Verify that the array's data type is boolean.
  if (PyArray_TYPE(np_arr) != NPY_BOOL) {
    PyErr_Format(PyExc_TypeError, "%s must be a NumPy array of booleans", name);
    return NULL;
  }

  // Get the total number of elements in the array.
  npy_intp size = PyArray_SIZE(np_arr);

  // Allocate memory for the resulting integer array.
  int *data_int = malloc(size * sizeof(int));
  if (data_int == NULL) {
    char error_msg[100];
    snprintf(error_msg, sizeof(error_msg), "Failed to allocate memory for %s",
             name);
    PyErr_SetString(PyExc_MemoryError, error_msg);
    return NULL;
  }

  // Get a pointer to the boolean data in the NumPy array.
  npy_bool *data_bool = (npy_bool *)PyArray_DATA(np_arr);

  // Convert each boolean value to an integer (1 for true, 0 for false).
  for (npy_intp i = 0; i < size; i++) {
    data_int[i] = data_bool[i] ? 1 : 0;
  }

  // Return the newly allocated integer array.
  return data_int;
}

/**
 * @brief Extract the grid properties from a tuple of numpy arrays.
 *
 * @param grid_tuple: A tuple of numpy arrays containing the grid properties.
 * @param ndim: The number of dimensions in the grid.
 * @param dims: The dimensions of the grid.
 */
double **extract_grid_props(PyObject *grid_tuple, int ndim, int *dims) {

  /* Allocate a single array for grid properties*/
  int nprops = 0;
  for (int dim = 0; dim < ndim; dim++)
    nprops += dims[dim];
  double **grid_props = malloc(nprops * sizeof(double *));
  if (grid_props == NULL) {
    PyErr_SetString(PyExc_MemoryError,
                    "Failed to allocate memory for grid_props.");
    return NULL;
  }

  /* Unpack the grid property arrays into a single contiguous array. */
  for (int idim = 0; idim < ndim; idim++) {

    /* Extract the data from the numpy array. */
    PyArrayObject *np_grid_arr =
        (PyArrayObject *)PyTuple_GetItem(grid_tuple, idim);
    if (np_grid_arr == NULL) {
      PyErr_SetString(PyExc_ValueError, "Failed to extract grid_arr.");
      return NULL;
    }
    double *grid_arr = PyArray_DATA(np_grid_arr);
    if (grid_arr == NULL) {
      PyErr_SetString(PyExc_ValueError, "Failed to extract grid_arr.");
      return NULL;
    }

    /* Assign this data to the property array. */
    grid_props[idim] = grid_arr;
  }

  /* Success. */
  return grid_props;
}

/**
 * @brief Extract the particle properties from a tuple of numpy arrays.
 *
 * @param part_tuple: A tuple of numpy arrays containing the particle
 * properties.
 * @param ndim: The number of dimensions in the grid.
 * @param npart: The number of particles.
 */
double **extract_part_props(PyObject *part_tuple, int ndim, int npart) {

  /* Allocate a single array for particle properties. */
  double **part_props = malloc(npart * ndim * sizeof(double *));
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
    double *part_arr = PyArray_DATA(np_part_arr);
    if (part_arr == NULL) {
      PyErr_SetString(PyExc_ValueError, "Failed to extract part_arr.");
      return NULL;
    }

    /* Assign this data to the property array. */
    for (int ipart = 0; ipart < npart; ipart++) {
      part_props[ipart * ndim + idim] = part_arr + ipart;
    }
  }

  /* Success. */
  return part_props;
}

/**
 * @brief Create the grid struct from the input numpy arrays.
 *
 * This method should be used for spectra grids.
 *
 * @param grid_tuple: A tuple of numpy arrays containing the grid properties.
 * @param np_ndims: The number of grid cells along each axis.
 * @param np_grid_spectra: The grid spectra.
 * @param ndim: The number of dimensions in the grid.
 * @param nlam: The number of wavelength elements.
 *
 * @return struct grid*: A pointer to the grid struct.
 */
struct grid *get_spectra_grid_struct(PyObject *grid_tuple,
                                     PyArrayObject *np_ndims,
                                     PyArrayObject *np_grid_spectra,
                                     PyArrayObject *np_lam, const int ndim,
                                     const int nlam) {

  /* Initialise the grid struct. */
  struct grid *grid = malloc(sizeof(struct grid));
  bzero(grid, sizeof(struct grid));

  /* Quick check to make sure our inputs are valid. */
  if (ndim == 0) {
    PyErr_SetString(PyExc_ValueError, "ndim must be greater than 0.");
    return NULL;
  }
  if (nlam == 0) {
    PyErr_SetString(PyExc_ValueError, "nlam must be greater than 0.");
    return NULL;
  }

  /* Attach the simple integers. */
  grid->ndim = ndim;
  grid->nlam = nlam;

  /* Extract a pointer to the grid dims */
  if (np_ndims != NULL) {

    grid->dims = extract_data_int(np_ndims, "dims");
    if (grid->dims == NULL) {
      return NULL;
    }

    /* Calculate the size of the grid. */
    grid->size = 1;
    for (int dim = 0; dim < ndim; dim++) {
      grid->size *= grid->dims[dim];
    }
  }

  /* Extract the grid properties from the tuple of numpy arrays. */
  if (grid_tuple != NULL) {
    grid->props = extract_grid_props(grid_tuple, ndim, grid->dims);
    if (grid->props == NULL) {
      return NULL;
    }
  }

  /* Extract a pointer to the spectra grids */
  if (np_grid_spectra != NULL) {
    grid->spectra = extract_data_double(np_grid_spectra, "grid_spectra");
    if (grid->spectra == NULL) {
      return NULL;
    }
  }

  /* Extract the wavelength array. */
  if (np_lam != NULL) {
    grid->lam = extract_data_double(np_lam, "lam");
    if (grid->lam == NULL) {
      return NULL;
    }
  }

  return grid;
}

/**
 * @brief Create the grid struct from the input numpy arrays.
 *
 * This method should be used for line grids.
 *
 * @param grid_tuple: A tuple of numpy arrays containing the grid properties.
 * @param np_ndims: The number of grid cells along each axis.
 * @param np_grid_lines: The grid lines.
 * @param np_grid_continuum: The grid continuum.
 * @param ndim: The number of dimensions in the grid.
 * @param nlam: The number of wavelength elements.
 *
 * @return struct grid*: A pointer to the grid struct.
 */
struct grid *get_lines_grid_struct(PyObject *grid_tuple,
                                   PyArrayObject *np_ndims,
                                   PyArrayObject *np_grid_lines,
                                   PyArrayObject *np_grid_continuum,
                                   const int ndim, const int nlam) {

  /* Initialise the grid struct. */
  struct grid *grid = malloc(sizeof(struct grid));
  bzero(grid, sizeof(struct grid));

  /* Quick check to make sure our inputs are valid. */
  if (ndim == 0) {
    PyErr_SetString(PyExc_ValueError, "ndim must be greater than 0.");
    return NULL;
  }
  if (nlam == 0) {
    PyErr_SetString(PyExc_ValueError, "nlam must be greater than 0.");
    return NULL;
  }

  /* Attach the simple integers. */
  grid->ndim = ndim;
  grid->nlam = nlam;

  /* Extract a pointer to the grid dims */
  if (np_ndims != NULL) {
    grid->dims = extract_data_int(np_ndims, "dims");
    if (grid->dims == NULL) {
      return NULL;
    }

    /* Calculate the size of the grid. */
    grid->size = 1;
    for (int dim = 0; dim < ndim; dim++) {
      grid->size *= grid->dims[dim];
    }
  }

  /* Extract the grid properties from the tuple of numpy arrays. */
  if (grid_tuple != NULL) {
    grid->props = extract_grid_props(grid_tuple, ndim, grid->dims);
    if (grid->props == NULL) {
      return NULL;
    }
  }

  /* Extract a pointer to the line grids */
  if (np_grid_lines != NULL) {
    grid->lines = extract_data_double(np_grid_lines, "grid_lines");
    if (grid->lines == NULL) {
      return NULL;
    }
  }

  /* Extract a pointer to the continuum grid. */
  if (np_grid_continuum != NULL) {
    grid->continuum = extract_data_double(np_grid_continuum, "grid_continuum");
    if (grid->continuum == NULL) {
      return NULL;
    }
  }

  return grid;
}
/**
 * @brief Create the particles struct from the input numpy arrays.
 *
 * @param part_tuple: A tuple of numpy arrays containing the particle
 * properties.
 * @param np_part_mass: The particle masses.
 * @param np_fesc: The escape fractions.
 * @param npart: The number of particles.
 *
 * @return struct particles*: A pointer to the particles struct.
 */
struct particles *get_part_struct(PyObject *part_tuple,
                                  PyArrayObject *np_part_mass,
                                  PyArrayObject *np_velocities,
                                  PyArrayObject *np_fesc, const int npart,
                                  const int ndim) {

  /* Initialise the particles struct. */
  struct particles *particles = malloc(sizeof(struct particles));
  bzero(particles, sizeof(struct particles));

  /* Quick check to make sure our inputs are valid. */
  if (npart == 0) {
    PyErr_SetString(PyExc_ValueError, "npart must be greater than 0.");
    return NULL;
  }

  /* Attach the simple integers. */
  particles->npart = npart;

  /* Extract a pointer to the particle masses. */
  if (np_part_mass != NULL) {
    particles->mass = extract_data_double(np_part_mass, "part_mass");
    if (particles->mass == NULL) {
      return NULL;
    }
  }

  /* Extract a pointer to the particle velocities. */
  if (np_velocities != NULL) {
    particles->velocities = extract_data_double(np_velocities, "part_vel");
    if (particles->velocities == NULL) {
      return NULL;
    }
  }

  /* Extract a pointer to the fesc array. */
  if (np_fesc != NULL) {
    particles->fesc = extract_data_double(np_fesc, "fesc");
    if (particles->fesc == NULL) {
      return NULL;
    }
  } else {
    /* If we have no fesc we need an array of zeros. */
    particles->fesc = calloc(npart, sizeof(double));
  }

  /* Extract the particle properties from the tuple of numpy arrays. */
  if (part_tuple != NULL) {
    particles->props = extract_part_props(part_tuple, ndim, npart);
    if (particles->props == NULL) {
      return NULL;
    }
  }

  return particles;
}

/**
 * @brief Retrieve a double pointer from a NumPy array attribute.
 *
 * This function obtains an attribute from the specified Python object and
 * checks whether it is either a NumPy array of doubles or None. If the
 * attribute exists and is a NumPy array of type NPY_DOUBLE, the function
 * returns the underlying data pointer (borrowed). If the attribute exists but
 * is set to None, the function returns NULL. In all error cases (e.g., wrong
 * type), NULL is returned and an appropriate Python exception is set.
 *
 * Note: Since the data pointer is borrowed, ensure that the owner of the NumPy
 * array (the Python object) remains alive for the duration of the pointer's
 * usage.
 *
 * @param obj  The Python object from which to extract the attribute.
 * @param attr The name of the attribute.
 *
 * @return double* The pointer to the underlying array data, or NULL if the
 * attribute is None or an error occurs.
 */
static inline double *get_numpy_attr_double(PyObject *obj, const char *attr) {

  /* This is a horrid hack but we may have to live with it. The
   * operations below demand that numpy has been initialised so that symbols
   * can be resolved. This is fine everywhere else because we are defining
   * modules where we can call import_array() in the module init function.
   * However, here we are defining functions that are called from anywhere
   * and so we need to make sure that numpy is initialised. */
  import_array();

  /* Get the attribute from the Python object */
  PyObject *tmp = PyObject_GetAttrString(obj, attr);

  /* If it did not exist check whether a private version does. */
  if (tmp == NULL) {
    char private_attr[100];
    snprintf(private_attr, sizeof(private_attr), "_%s", attr);
    tmp = PyObject_GetAttrString(obj, private_attr);
  }

  /* If we still don't have it then we have a problem. */
  if (!tmp) {
    /* If PyObject_GetAttrString returned NULL, ensure a clear error message */
    PyErr_Format(PyExc_AttributeError,
                 "The attribute '%s' does not exist on the provided object.",
                 attr);
    return NULL;
  }

  /* If the attribute exists but is explicitly set to None, return NULL */
  if (tmp == Py_None) {
    Py_DECREF(tmp);
    return NULL;
  }

  /* Have we got a numpy array? */
  if (!PyArray_Check(tmp)) {
    Py_DECREF(tmp);
    return NULL;
  }

  /* Check that the attribute is a NumPy array of doubles */
  if (PyArray_TYPE((PyArrayObject *)tmp) != NPY_DOUBLE) {
    PyErr_Format(PyExc_TypeError, "%s must be a NumPy array of doubles or None",
                 attr);
    Py_DECREF(tmp);
    return NULL;
  }

  /* Extract the underlying data pointer using the provided helper function */
  double *data = extract_data_double((PyArrayObject *)tmp, (char *)attr);

  /* Decrement the temporary reference; the actual array remains owned by obj */
  Py_DECREF(tmp);

  return data;
}

/**
 * @brief Retrieve a interger pointer from a NumPy array attribute.
 *
 * This function obtains an attribute from the specified Python object and
 * checks whether it is either a NumPy array of integers or None. If the
 * attribute exists and is a NumPy array of type NPY_INT, the function
 * returns the underlying data pointer (borrowed). If the attribute exists but
 * is set to None, the function returns NULL. In all error cases (e.g., wrong
 * type), NULL is returned and an appropriate Python exception is set.
 *
 * Note: Since the data pointer is borrowed, ensure that the owner of the NumPy
 * array (the Python object) remains alive for the duration of the pointer's
 * usage.
 *
 * @param obj  The Python object from which to extract the attribute.
 * @param attr The name of the attribute.
 *
 * @return int* The pointer to the underlying array data, or NULL if the
 * attribute is None or an error occurs.
 */
static inline int *get_numpy_attr_int(PyObject *obj, const char *attr) {
  /* Get the attribute from the Python object */
  PyObject *tmp = PyObject_GetAttrString(obj, attr);
  if (!tmp) {
    /* If PyObject_GetAttrString returned NULL, ensure a clear error message */
    PyErr_Format(PyExc_AttributeError,
                 "The attribute '%s' does not exist on the provided object.",
                 attr);
    return NULL;
  }

  /* If the attribute exists but is explicitly set to None, return NULL */
  if (tmp == Py_None) {
    Py_DECREF(tmp);
    return NULL;
  }

  /* Check that the attribute is a NumPy array of integers */
  if (!PyArray_Check(tmp) || PyArray_TYPE((PyArrayObject *)tmp) != NPY_INT) {
    PyErr_Format(PyExc_TypeError,
                 "%s must be a NumPy array of integers or None", attr);
    Py_DECREF(tmp);
    return NULL;
  }

  /* Extract the underlying data pointer using the provided helper function */
  int *data = extract_data_int((PyArrayObject *)tmp, (char *)attr);

  /* Decrement the temporary reference; the actual array remains owned by obj */
  Py_DECREF(tmp);

  return data;
}

/**
 * @brief Extract the particle properties from the Python objects.
 *
 * This function borrows pointers to the underlying NumPy array data.
 * It assumes:
 *  - The "parts" object has attributes "nparticles", "masses", "fesc",
 * "velocities", and a weight attribute specified by weight_var. Each attribute
 * must be a NumPy array of doubles.
 *  - The "grid" object has an attribute "axes" which is a list of strings.
 *    For each axis name, the "parts" object has a corresponding attribute (a
 * NumPy array of doubles).
 *
 * Since no data is copied, you must ensure that the Python objects stay alive.
 *
 * @param parts: The Python object containing particle properties.
 * @param grid:  The Python grid object containing the axes attribute.
 * @param weight_var: The name of the weight attribute.
 *
 * @return struct particles*: A pointer to the particles struct, or NULL on
 * error.
 */
struct particles *get_part_struct_from_obj(PyObject *parts, PyObject *grid,
                                           const char *weight_var) {

  /* Allocate the particles struct. */
  struct particles *particles = calloc(1, sizeof(struct particles));
  if (!particles) {
    PyErr_NoMemory();
    return NULL;
  }

  /* Get the number of particles from parts.nparticles */
  PyObject *nparticles_obj = PyObject_GetAttrString(parts, "nparticles");
  if (!nparticles_obj)
    goto error;
  int npart = (int)PyLong_AsLong(nparticles_obj);
  if (PyErr_Occurred())
    goto error;
  particles->npart = npart;
  Py_DECREF(nparticles_obj);

  /* Borrow pointers for masses, weight, fesc, and velocities using the inline
   * function. */
  particles->mass = get_numpy_attr_double(parts, "masses");
  particles->weight = get_numpy_attr_double(parts, weight_var);
  particles->fesc = get_numpy_attr_double(parts, "fesc");
  particles->velocities = get_numpy_attr_double(parts, "velocities");

  /* Some attributes are special cases where they could also be singular
   * floats rather than arrays. */
  if (particles->fesc == NULL) {
    double fesc = PyFloat_AsDouble(PyObject_GetAttrString(parts, "fesc"));
    if (PyErr_Occurred())
      goto error;
    particles->_fesc = fesc;
  } else {
    particles->_fesc = 0.0;
  }

  /* Did an error occur? */
  if (PyErr_Occurred())
    goto error;

  /* Extract grid axes (a list of strings) from grid.axes */
  {
    PyObject *grid_axes = PyObject_GetAttrString(grid, "axes");
    if (!grid_axes || !PyList_Check(grid_axes)) {
      PyErr_SetString(PyExc_TypeError, "grid.axes must be a list");
      Py_XDECREF(grid_axes);
      goto error;
    }
    Py_ssize_t num_axes = PyList_Size(grid_axes);
    particles->nprops = (int)num_axes;

    /* Allocate the array of property pointers. */
    particles->props = calloc(num_axes, sizeof(double *));
    if (!particles->props) {
      Py_DECREF(grid_axes);
      PyErr_NoMemory();
      goto error;
    }

    /* Loop over each axis name, extract the corresponding property, and borrow
     * its data pointer. */
    for (Py_ssize_t i = 0; i < num_axes; i++) {
      PyObject *axis_obj = PyList_GetItem(grid_axes, i);
      if (!PyUnicode_Check(axis_obj)) {
        PyErr_SetString(PyExc_TypeError, "Each grid axis must be a string");
        Py_DECREF(grid_axes);
        goto error;
      }
      const char *axis_name = PyUnicode_AsUTF8(axis_obj);
      if (!axis_name) {
        Py_DECREF(grid_axes);
        goto error;
      }
      double *prop_data = get_numpy_attr_double(parts, axis_name);
      if (!prop_data) {
        Py_DECREF(grid_axes);
        goto error;
      }
      particles->props[i] = prop_data;
    }
    Py_DECREF(grid_axes);
  }

  return particles;

error:
  if (particles) {
    if (particles->props)
      free(particles->props);
    free(particles);
  }
  return NULL;
}

/**
 * @brief Extract the grid properties from the Python objects.
 *
 * This function borrows pointers to the underlying NumPy array data.
 * It assumes:
 * - The "grid" object has attributes "ndim", "nlam", "shape", "ngrid_points",
 *   and "axes". The "shape" attribute must be a tuple of integers, and the
 *   "axes" attribute must be a list of strings.
 * - For each axis name in "axes", the "grid" object has a corresponding
 *   attribute (a NumPy array of doubles).
 *
 * Since no data is copied, you must ensure that the Python objects stay alive.
 *
 * @param py_grid: The Python object containing the grid properties.
 *
 * @return struct grid*: A pointer to the grid struct, or NULL on error.
 */
struct grid *get_grid_struct_from_obj(PyObject *py_grid) {

  /* Allocate the grid struct. */
  struct grid *grid_struct = calloc(1, sizeof(struct grid));
  if (!grid_struct) {
    PyErr_NoMemory();
    return NULL;
  }

  /* Get the number of dimensions from grid.ndim */
  PyObject *ndim_obj = PyObject_GetAttrString(py_grid, "ndim");
  if (!ndim_obj)
    goto error;
  grid_struct->ndim = (int)PyLong_AsLong(ndim_obj);
  if (PyErr_Occurred())
    goto error;
  Py_DECREF(ndim_obj);

  /* Get the number of wavelength elements from grid.nlam */
  PyObject *nlam_obj = PyObject_GetAttrString(py_grid, "nlam");
  if (!nlam_obj)
    goto error;
  grid_struct->nlam = (int)PyLong_AsLong(nlam_obj);
  if (PyErr_Occurred())
    goto error;
  Py_DECREF(nlam_obj);

  /* Extract the grid dimensions from grid.shape, and convert the
   * tuple to a C array */
  PyObject *shape_obj = PyObject_GetAttrString(py_grid, "shape");
  if (!shape_obj)
    goto error;
  if (!PyTuple_Check(shape_obj)) {
    PyErr_SetString(PyExc_TypeError, "grid.shape must be a tuple");
    Py_DECREF(shape_obj);
    goto error;
  }
  Py_ssize_t num_dims = PyTuple_Size(shape_obj);
  grid_struct->dims = calloc(num_dims, sizeof(int));
  if (!grid_struct->dims) {
    Py_DECREF(shape_obj);
    PyErr_NoMemory();
    goto error;
  }
  for (Py_ssize_t i = 0; i < num_dims; i++) {
    PyObject *dim_obj = PyTuple_GetItem(shape_obj, i);
    if (!dim_obj) {
      Py_DECREF(shape_obj);
      goto error;
    }
    grid_struct->dims[i] = (int)PyLong_AsLong(dim_obj);
    if (PyErr_Occurred()) {
      Py_DECREF(shape_obj);
      goto error;
    }
  }

  /* Get the number of grid cells from the ngrid_points attribute */
  PyObject *ngrid_obj = PyObject_GetAttrString(py_grid, "ngrid_points");
  if (!ngrid_obj)
    goto error;
  grid_struct->size = (int)PyLong_AsLong(ngrid_obj);
  if (PyErr_Occurred())
    goto error;

  /* Extract grid axes (a list of strings) from grid.axes */
  {
    PyObject *grid_axes = PyObject_GetAttrString(py_grid, "axes");
    if (!grid_axes || !PyList_Check(grid_axes)) {
      PyErr_SetString(PyExc_TypeError, "grid.axes must be a list");
      Py_XDECREF(grid_axes);
      goto error;
    }

    /* Allocate the array of property pointers. */
    grid_struct->props = calloc(grid_struct->ndim, sizeof(double *));
    if (!grid_struct->props) {
      Py_DECREF(grid_axes);
      PyErr_NoMemory();
      goto error;
    }

    /* Loop over each axis name, extract the corresponding property, and borrow
     * its data pointer. */
    for (Py_ssize_t i = 0; i < grid_struct->ndim; i++) {
      PyObject *axis_obj = PyList_GetItem(grid_axes, i);
      if (!PyUnicode_Check(axis_obj)) {
        PyErr_SetString(PyExc_TypeError, "Each grid axis must be a string");
        Py_DECREF(grid_axes);
        goto error;
      }
      const char *axis_name = PyUnicode_AsUTF8(axis_obj);
      if (!axis_name) {
        Py_DECREF(grid_axes);
        goto error;
      }
      double *prop_data = get_numpy_attr_double(py_grid, axis_name);
      if (!prop_data) {
        Py_DECREF(grid_axes);
        goto error;
      }
      grid_struct->props[i] = prop_data;
    }
    Py_DECREF(grid_axes);
  }

  /* TODO: spectra and lines are missing */
  return grid_struct;

error:
  if (grid_struct) {
    if (grid_struct->props)
      free(grid_struct->props);
    free(grid_struct);
  }
  return NULL;
}
