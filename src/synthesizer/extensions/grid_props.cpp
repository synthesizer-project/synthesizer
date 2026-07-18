
/* Standard includes */
#include <array>
#include <iostream>
#include <ostream>
#include <stdlib.h>

/* Python includes */
#define PY_ARRAY_UNIQUE_SYMBOL SYNTHESIZER_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy_init.h"

#include <Python.h>

/* Local includes */
#include "cpp_to_python.h"
#include "grid_props.h"
#include "index_utils.h"
#include "property_funcs.h"
#include "python_to_cpp.h"
#include "timers.h"

/**
 * @brief Constructor for the GridProps class.
 *
 * This constructor initializes the GridProps object with the provided
 * spectra, axes, wavelength, wavelength mask, number of wavelengths,
 * and grid weights.
 *
 * @param np_spectra: The numpy array containing the spectra data.
 * @param axes_tuple: A tuple containing numpy arrays for each axis of the
 * grid.
 * @param np_lam: The numpy array containing the wavelength data.
 * @param np_lam_mask: The numpy array containing the wavelength mask.
 * @param nlam: The number of wavelength elements.
 * @param np_grid_weights: The numpy array containing the grid weights,
 * or NULL if not provided.
 *
 *
 */
GridProps::GridProps(PyArrayObject *np_spectra, PyObject *axes_tuple,
                     PyArrayObject *np_lam, PyArrayObject *np_lam_mask,
                     const int nlam, PyArrayObject *np_grid_weights,
                     PyObject *axis_names_tuple)
    : nlam(nlam),
      np_spectra_(np_spectra),
      axes_tuple_(axes_tuple),
      np_lam_(np_lam),
      np_lam_mask_(np_lam_mask),
      np_grid_weights_(np_grid_weights) {

  tic("GridProps.__init__");

  /* The number of dimensions is the length of the axis tuple. */
  ndim = PyTuple_Size(axes_tuple);

  /* If ndim is less than or equal to 0, we have an invalid grid. */
  if (ndim <= 0) {
    PyErr_SetString(PyExc_ValueError,
                    "[GridProps::GridProps]: ndim must be greater than 0.");
    return;
  } else if (ndim > MAX_GRID_NDIM - 1) {

    char error_msg[256];
    snprintf(error_msg, sizeof(error_msg),
             "[GridProps::GridProps]: Invalid ndim: %d < MAX_GRID_NDIM (%d)! "
             "Report this to the "
             "developers, you have exceeded a hardcoded maximum which can be "
             "increased if needed.)",
             ndim, MAX_GRID_NDIM - 1);
    PyErr_SetString(PyExc_ValueError, error_msg);
    return;
  }

  /* Validate that all present floating-point inputs are contiguous and share
   * one supported dtype family before any hot kernels use raw pointers. */
  PyArrayObject *float_arrays[MAX_GRID_NDIM + 3] = {NULL};
  const char *float_names[MAX_GRID_NDIM + 3] = {NULL};
  int float_count = 0;

  if (np_spectra_ != NULL &&
      reinterpret_cast<PyObject *>(np_spectra_) != Py_None) {
    float_arrays[float_count] = np_spectra_;
    float_names[float_count] = "grid_spectra";
    float_count++;
  }

  for (int idim = 0; idim < ndim; idim++) {
    PyArrayObject *np_axis_arr =
        (PyArrayObject *)PyTuple_GetItem(axes_tuple, idim);
    if (np_axis_arr == NULL) {
      PyErr_SetString(PyExc_ValueError,
                      "[GridProps::GridProps]: Failed to extract axis array.");
      return;
    }

    float_arrays[float_count] = np_axis_arr;
    float_names[float_count] = "grid axis";
    float_count++;
  }

  if (np_lam_ != NULL && reinterpret_cast<PyObject *>(np_lam_) != Py_None) {
    float_arrays[float_count] = np_lam_;
    float_names[float_count] = "lam";
    float_count++;
  }

  if (np_grid_weights_ != NULL &&
      reinterpret_cast<PyObject *>(np_grid_weights_) != Py_None) {
    float_arrays[float_count] = np_grid_weights_;
    float_names[float_count] = "grid_weights";
    float_count++;
  }

  if (float_count > 0 &&
      !is_matching_float_dtypes(float_arrays, float_names, float_count,
                                &float_typenum_)) {
    return;
  }

  /* Get the dimensions of the grid from the axis tuple. */
  for (int idim = 0; idim < ndim; idim++) {
    PyArrayObject *np_axis_arr =
        (PyArrayObject *)PyTuple_GetItem(axes_tuple, idim);
    if (np_axis_arr == NULL) {
      PyErr_SetString(PyExc_ValueError,
                      "[GridProps::GridProps]: Failed to extract axis array.");
      return;
    }
    dims[idim] = PyArray_DIM(np_axis_arr, 0);

    axis_names_[idim].clear();
    if (axis_names_tuple != NULL && PySequence_Check(axis_names_tuple) &&
        !PyUnicode_Check(axis_names_tuple)) {
      PyObject *name_obj = PySequence_GetItem(axis_names_tuple, idim);
      if (name_obj != NULL) {
        if (PyUnicode_Check(name_obj)) {
          const char *name = PyUnicode_AsUTF8(name_obj);
          if (name != NULL) {
            axis_names_[idim] = name;
          } else {
            PyErr_Clear();
          }
        }
        Py_DECREF(name_obj);
      } else {
        PyErr_Clear();
      }
    }
  }

  /* Calculate the size of the grid. */
  size = 1;
  for (int dim = 0; dim < ndim; dim++) {
    size *= dims[dim];
  }

  /* Account for the additional wavelength dimension. */
  for (int i = 0; i < ndim; i++) {
    spectra_dims_[i] = dims[i];
  }
  spectra_dims_[ndim] = nlam;

  /* Flag whether we need to populate the grid weights */
  if (has_grid_weights()) {
    need_grid_weights_ = false;
    owns_grid_weights_ = false;
  } else {
    need_grid_weights_ = true;
    owns_grid_weights_ = false;
  }

  toc("GridProps.__init__");
}

/**
 * @brief Convert a multi-dimensional grid index to a flat index.
 *
 * This function handles indices into the Naxis grid space, i.e. it ignores
 * the wavelength axis of the grid.
 *
 * @param multi_index: An array of N-dimensional indices.
 *
 * @return The flat index corresponding to the multi-dimensional index.
 */
int GridProps::ravel_grid_index(
    const std::array<int, MAX_GRID_NDIM> &multi_index) const {
  return get_flat_index(multi_index, dims.data(), ndim);
}

/**
 * @brief Convert a flat index to a multi-dimensional grid index.
 *
 * @param index: The flat index to convert.
 *
 * @return An array of N-dimensional indices corresponding to the flat index.
 */
std::array<int, MAX_GRID_NDIM> GridProps::unravel_grid_index(int index) const {
  std::array<int, MAX_GRID_NDIM> indices = {0};
  get_indices_from_flat(index, ndim, dims, indices);
  return indices;
}

/**
 * @brief Convert a multi-dimensional grid index and wavelength index to a flat
 * index for the spectra array.
 *
 * @param multi_index: An array of N-dimensional indices.
 * @param ilam: The wavelength index.
 *
 * @return The flat index corresponding to the multi-dimensional index and
 * wavelength index.
 */
int GridProps::ravel_spectra_index(
    const std::array<int, MAX_GRID_NDIM> &multi_index, int ilam) const {
  /* Include the wavelength index in the multi-dimensional index. */
  std::array<int, MAX_GRID_NDIM + 1> full_index = {0};
  for (int i = 0; i < ndim; i++) {
    full_index[i] = multi_index[i];
  }
  full_index[ndim] = ilam;  // Set the wavelength index

  return get_flat_index(full_index, spectra_dims_.data(), ndim + 1);
}

/**
 * @brief Convert a flat index to a multi-dimensional grid index and wavelength
 * index.
 *
 * @param index: The flat index to convert.
 *
 * @return An array of N-dimensional indices corresponding to the flat index
 * and the wavelength index.
 */
std::array<int, MAX_GRID_NDIM + 1> GridProps::unravel_spectra_index(
    int index) const {
  std::array<int, MAX_GRID_NDIM + 1> indices = {0};
  get_indices_from_flat(index, ndim + 1, spectra_dims_, indices);
  return indices;
}

/**
 * @brief Check if grid weights are provided.
 *
 * @return True if grid weights are provided, false otherwise.
 */
bool GridProps::has_grid_weights() const {
  return np_grid_weights_ != NULL &&
         reinterpret_cast<PyObject *>(np_grid_weights_) != Py_None;
}

/**
 * @brief Get the numpy array of grid weights.
 *
 * @return The numpy array of grid weights.
 */
PyArrayObject *GridProps::get_np_grid_weights() const {
  if (!has_grid_weights()) {
    PyErr_SetString(PyExc_ValueError,
                    "[GridProps::get_np_grid_weights]: Grid "
                    "weights have not been allocated and populate, or given.");
    return NULL;
  }

  /* Py_BuildValue("N") steals a reference. If the weights were provided by
   * Python we only have a borrowed reference, so we must incref first. If we
   * allocated them ourselves with PyArray_ZEROS, np_grid_weights_ already owns
   * a new reference and must be returned as-is. */
  if (!owns_grid_weights_) {
    Py_INCREF(reinterpret_cast<PyObject *>(np_grid_weights_));
  }
  return np_grid_weights_;
}

/**
 * @brief Get the wavelength mask.
 *
 * @return The wavelength mask array.
 */
bool GridProps::lam_is_masked(int ind) const {
  /* If we don't have a wavelength mask, then the wavelength is not masked. */
  if (np_lam_mask_ == NULL) {
    return false;
  }

  /* If the mask is None, then the wavelength is not masked. */
  if (reinterpret_cast<PyObject *>(np_lam_mask_) == Py_None) {
    return false;
  }

  return !get_bool_at(np_lam_mask_, ind, "wavelength mask");
}

/**
 * @brief Check if grid weights need to be populated.
 *
 * @return True if grid weights need to be populated, false otherwise.
 */
bool GridProps::need_grid_weights() const {
  /* Check we have a grid to populate weights for. */
  if (!has_grid_weights()) {
    PyErr_SetString(PyExc_ValueError,
                    "[GridProps::need_grid_weights]: "
                    "Grid weights have not been allocated.");
    return false;
  }

  return need_grid_weights_;
}

/**
 * @brief Get the shared floating-point dtype used by the grid arrays.
 *
 * @return The resolved NumPy typenum, or -1 if no float arrays were provided.
 */
int GridProps::get_float_typenum() const { return float_typenum_; }
