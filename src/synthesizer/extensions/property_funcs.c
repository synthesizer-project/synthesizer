/******************************************************************************
 * A C module containing helper functions for extracting properties from the
 * numpy objects.
 *****************************************************************************/

/* C headers. */
#include <Python.h>
#include <string.h>

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
 * @brief Extract boolean data from a numpy array.
 *
 * This function returns a pointer to the underlying boolean data stored
 * as npy_bool values (typically unsigned char).
 *
 * @param np_arr: The numpy array to extract.
 * @param name: The name of the numpy array (for error messages).
 * @return Pointer to the npy_bool data, or NULL on error.
 */
npy_bool *extract_data_bool(PyArrayObject *np_arr, char *name) {
  npy_bool *data = (npy_bool *)PyArray_DATA(np_arr);
  if (data == NULL) {
    char error_msg[100];
    snprintf(error_msg, sizeof(error_msg), "Failed to extract %s.", name);
    PyErr_SetString(PyExc_ValueError, error_msg);
    return NULL;
  }
  return data;
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
 * @param np_lam: The wavelength array.
 * @param np_lam_mask: The wavelength mask array.
 * @param ndim: The number of dimensions in the grid.
 * @param nlam: The number of wavelength elements.
 *
 * @return struct grid*: A pointer to the grid struct.
 */
struct grid *get_spectra_grid_struct(PyObject *grid_tuple,
                                     PyArrayObject *np_ndims,
                                     PyArrayObject *np_grid_spectra,
                                     PyArrayObject *np_lam,
                                     PyArrayObject *np_lam_mask, const int ndim,
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

  /* Extract the wavelength mask array, if this is Py_None then we will
   * treat it as NULL. */
  if (np_lam_mask != NULL && np_lam_mask != Py_None) {
    grid->lam_mask = extract_data_bool(np_lam_mask, "lam_mask");
    if (grid->lam_mask == NULL) {
      return NULL;
    }
  } else {
    grid->lam_mask = NULL;
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
 * @param np_velocities: The particle velocities.
 * @param np_mask: The particle mask.
 * @param npart: The number of particles.
 *
 * @return struct particles*: A pointer to the particles struct.
 */
struct particles *get_part_struct(PyObject *part_tuple,
                                  PyArrayObject *np_part_mass,
                                  PyArrayObject *np_velocities,
                                  PyArrayObject *np_mask, const int npart,
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

  /* Extract the particle properties from the tuple of numpy arrays. */
  if (part_tuple != NULL) {
    particles->props = extract_part_props(part_tuple, ndim, npart);
    if (particles->props == NULL) {
      return NULL;
    }
  }

  /* Extract a pointer to the particle mask, if this is Py_None then we will
   * treat it as NULL. */
  if (np_mask != NULL && np_mask != Py_None) {
    particles->mask = extract_data_bool(np_mask, "mask");
    if (particles->mask == NULL) {
      return NULL;
    }
  } else {
    particles->mask = NULL;
  }

  return particles;
}
