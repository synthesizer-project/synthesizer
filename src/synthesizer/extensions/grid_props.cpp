
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
#include "timers.h"

/**
 * @brief Constructor for the GridProps class.
 *
 * This constructor initializes the GridProps object with the provided
 * spectra, axes, wavelength, wavelength mask, number of wavelengths,
 * and grid weights.
 *
 * @param np_spectra: The numpy array containing the spectra data.
 * @param axes_tuple: A tuple containing numpy arrays for each axis of the grid.
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
                     const int nlam, PyArrayObject *np_grid_weights)
    : nlam(nlam), np_spectra_(np_spectra), axes_tuple_(axes_tuple),
      np_lam_(np_lam), np_lam_mask_(np_lam_mask),
      np_grid_weights_(np_grid_weights) {

  double start_time = tic();

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
  } else {
    need_grid_weights_ = true;
  }

  toc("Constructing C++ grid properties", start_time);
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
  full_index[ndim] = ilam; // Set the wavelength index

  return get_flat_index(full_index, spectra_dims_.data(), ndim + 1);
}

/**
 * @brief Convert a flat index to a multi-dimensional grid index and wavelength
 * index.
 *
 * @param index: The flat index to convert.
 *
 * @return An array of N-dimensional indices corresponding to the flat index and
 * the wavelength index.
 */
std::array<int, MAX_GRID_NDIM + 1>
GridProps::unravel_spectra_index(int index) const {
  std::array<int, MAX_GRID_NDIM + 1> indices = {0};
  get_indices_from_flat(index, ndim + 1, spectra_dims_, indices);
  return indices;
}

/**
 * @brief Get the spectra array.
 *
 * @return The spectra array.
 */
double *GridProps::get_spectra() const {
  return static_cast<double *>(PyArray_DATA(np_spectra_));
}

/**
 * @brief Get the value of the spectra at a specific grid index and wavelength
 * index.
 *
 * @param grid_ind: The grid index.
 * @param ilam: The wavelength index.
 * @return The value at the specified grid index and wavelength index.
 */
double GridProps::get_spectra_at(int grid_ind, int ilam) const {
  if (grid_ind < 0 || grid_ind >= size) {
    char error_msg[256];
    snprintf(error_msg, sizeof(error_msg),
             "[GridProps::get_spectra_at]: Grid index (%d) out of bounds. "
             "Valid range is [0, %d).",
             grid_ind, size);
    PyErr_SetString(PyExc_IndexError, error_msg);
    return 0.0;
  }
  if (ilam < 0 || ilam >= nlam) {
    char error_msg[256];
    snprintf(
        error_msg, sizeof(error_msg),
        "[GridProps::get_spectra_at]: Wavelength index (%d) out of bounds. "
        "Valid range is [0, %d).",
        ilam, nlam);
    PyErr_SetString(PyExc_IndexError, error_msg);
    return 0.0;
  }

  /* Convert the flat (ravelled) grid index to a multi-dimensional index. */
  std::array<int, MAX_GRID_NDIM> unraveled_ind = unravel_grid_index(grid_ind);

  /* Get the ravelled spectra index for the given multi-dimensional index
   * and wavelength index. */
  int spectra_index = ravel_spectra_index(unraveled_ind, ilam);

  /* Return the value at the spectra index. */
  return get_double_at(np_spectra_, spectra_index);
}

/**
 * @brief Get the wavelength array.
 *
 * @return The wavelength array.
 */
double *GridProps::get_lam() const {
  return static_cast<double *>(PyArray_DATA(np_lam_));
}

/**
 * @brief Get an axis as an array.
 *
 * @return The axis array for the given dimension.
 */
double *GridProps::get_axis(int idim) const {
  if (idim < 0 || idim >= ndim) {
    PyErr_SetString(PyExc_IndexError,
                    "[GridProps::get_axis]: Axis index out of bounds.");
    return NULL;
  }

  PyArrayObject *np_axis_arr =
      (PyArrayObject *)PyTuple_GetItem(axes_tuple_, idim);
  if (np_axis_arr == NULL) {
    PyErr_SetString(PyExc_ValueError,
                    "[GridProps::get_axis]: Failed to extract axis array.");
    return NULL;
  }

  return static_cast<double *>(PyArray_DATA(np_axis_arr));
}

/**
 * @brief Get all axes as an array of pointers.
 *
 * @return An array of pointers to the axes arrays.
 */
std::array<double *, MAX_GRID_NDIM> GridProps::get_all_axes() const {
  std::array<double *, MAX_GRID_NDIM> axes;
  for (int idim = 0; idim < ndim; idim++) {
    double *axis = get_axis(idim);
    if (axis == NULL) {
      PyErr_SetString(PyExc_ValueError,
                      "[GridProps::get_all_axes]: Axis retrieval failed.");
      return {};
    }
    axes[idim] = axis;
  }
  return axes;
}

/**
 * @brief Get the value of an axis at a specific index.
 *
 * @param idim: The dimension index.
 * @param ind: The index in the axis.
 * @return The value at the specified index in the axis.
 */
double GridProps::get_axis_at(int idim, int ind) const {
  if (idim < 0 || idim >= ndim) {
    PyErr_SetString(PyExc_IndexError,
                    "[GridProps::get_axis_at]: Axis index out of bounds.");
    return -1.0;
  }

  PyArrayObject *np_axis_arr =
      (PyArrayObject *)PyTuple_GetItem(axes_tuple_, idim);
  if (np_axis_arr == NULL) {
    PyErr_SetString(PyExc_ValueError,
                    "[GridProps::get_axis_at]: Failed to extract axis array.");
    return -1.0;
  }

  return get_double_at(np_axis_arr, ind);
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
 * @brief Get the grid weights array.
 *
 * If no grid weights have been provided then an empty array is returned ready
 * to be filled.
 *
 * @return The grid weights array.
 */
double *GridProps::get_grid_weights() {
  /* If we already have grid weights, return them. */
  if (has_grid_weights()) {
    grid_weights_ = static_cast<double *>(PyArray_DATA(np_grid_weights_));
    need_grid_weights_ = false; // We don't need to populate them.
    return grid_weights_;
  }

  /* If we don't have grid weights, allocate a new numpy array for filling. */
  npy_intp np_dims_weights[MAX_GRID_NDIM];
  for (int i = 0; i < ndim; i++) {
    np_dims_weights[i] = dims[i];
  }
  np_grid_weights_ =
      (PyArrayObject *)PyArray_ZEROS(ndim, np_dims_weights, NPY_DOUBLE, 0);
  grid_weights_ = static_cast<double *>(PyArray_DATA(np_grid_weights_));
  RETURN_IF_PYERR();

  /* Flag that we need to populate the grid weights. */
  need_grid_weights_ = true;

  return grid_weights_;
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

  return np_grid_weights_;
}

/**
 * @brief Get the grid weight at a specific index.
 *
 * @param ind: The index in the grid weights array.
 * @return The grid weight at the specified index.
 */
double GridProps::get_grid_weight_at(int ind) const {
  if (!has_grid_weights() && grid_weights_ == NULL) {
    PyErr_SetString(
        PyExc_ValueError,
        "[GridProps::get_grid_weight_at]: "
        "Grid weights have not been allocated, populated or provided.");
    return 0.0;
  }

  return grid_weights_[ind];
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

  return !get_bool_at(np_lam_mask_, ind);
}

/**
 * @brief Check if grid weights need to be populated.
 *
 * @return True if grid weights need to be populated, false otherwise.
 */
bool GridProps::need_grid_weights() const {
  /* Check we have a grid to populate weights for. */
  if (!has_grid_weights()) {
    PyErr_SetString(PyExc_ValueError, "[GridProps::need_grid_weights]: "
                                      "Grid weights have not been allocated.");
    return false;
  }

  return need_grid_weights_;
}
