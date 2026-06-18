#ifndef GRID_PROPS_H_
#define GRID_PROPS_H_

/* Standard includes */
#include <array>
#include <stdlib.h>
#include <string>
#include <type_traits>

/* Python includes */
#define PY_ARRAY_UNIQUE_SYMBOL SYNTHESIZER_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy_init.h"

#include <Python.h>

/* Local includes */
#include "cpp_to_python.h"
#include "property_funcs.h"
#include "python_to_cpp.h"

#pragma omp declare target

/* Define the maximum number of dimensions we support right now (this can be
 * increased but we need a constant reasonable value, if this is ever reached
 * we can increase it). */
constexpr int MAX_GRID_NDIM = 10;

class GridProps {

 public:
  /* The number of dimensions. */
  int ndim;

  /* The number of wavelength elements. */
  int nlam;

  /* The number of grid cells along each axis. */
  std::array<int, MAX_GRID_NDIM> dims;

  /* The number of grid cells in total. */
  int size;

  /* Constructor for the GridProps class. */
  GridProps(PyArrayObject *np_spectra, PyObject *axes_tuple,
            PyArrayObject *np_lam, PyArrayObject *np_lam_mask, const int nlam,
            PyArrayObject *np_grid_weights = NULL,
            PyObject *axis_names_tuple = NULL);

  /* Index handlers for indexing the grid properties. */
  int ravel_grid_index(
      const std::array<int, MAX_GRID_NDIM> &multi_index) const;
  std::array<int, MAX_GRID_NDIM> unravel_grid_index(int index) const;
  int ravel_spectra_index(const std::array<int, MAX_GRID_NDIM> &multi_index,
                          int ilam) const;
  std::array<int, MAX_GRID_NDIM + 1> unravel_spectra_index(int index) const;

  /* Prototypes for getters. */
  double *get_spectra() const;
  double get_spectra_at(int grid_ind, int ilam) const;
  double *get_lam() const;
  double *get_axis(int idim) const;
  std::array<double *, MAX_GRID_NDIM> get_all_axes() const;
  double get_axis_at(int idim, int ind) const;
  double *get_grid_weights();
  PyArrayObject *get_np_grid_weights() const;
  double get_grid_weight_at(int ind) const;

  /* Accessors for validated float32/float64 arrays. */
  template <typename Real>
  const Real *get_spectra() const {
    static_assert(std::is_same_v<Real, float> || std::is_same_v<Real, double>,
                  "GridProps supports only float32 and float64 arrays.");
    return data_ptr<const Real>(np_spectra_);
  }

  template <typename Real>
  const Real *get_lam() const {
    static_assert(std::is_same_v<Real, float> || std::is_same_v<Real, double>,
                  "GridProps supports only float32 and float64 arrays.");
    return data_ptr<const Real>(np_lam_);
  }

  template <typename Real>
  const Real *get_axis(int idim) const {
    static_assert(std::is_same_v<Real, float> || std::is_same_v<Real, double>,
                  "GridProps supports only float32 and float64 arrays.");
    if (idim < 0 || idim >= ndim) {
      PyErr_SetString(PyExc_IndexError,
                      "[GridProps::get_axis]: Axis index out of bounds.");
      return NULL;
    }

    PyArrayObject *np_axis_arr =
        (PyArrayObject *)PyTuple_GetItem(axes_tuple_, idim);
    if (np_axis_arr == NULL) {
      PyErr_SetString(PyExc_ValueError,
                      "[GridProps::get_axis]: Failed to extract axis "
                      "array.");
      return NULL;
    }

    return data_ptr<const Real>(np_axis_arr);
  }

  template <typename Real>
  std::array<const Real *, MAX_GRID_NDIM> get_all_axes() const {
    static_assert(std::is_same_v<Real, float> || std::is_same_v<Real, double>,
                  "GridProps supports only float32 and float64 arrays.");
    std::array<const Real *, MAX_GRID_NDIM> axes = {};
    for (int idim = 0; idim < ndim; idim++) {
      const Real *axis = get_axis<Real>(idim);
      if (axis == NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "[GridProps::get_all_axes]: Axis retrieval "
                        "failed.");
        return {};
      }
      axes[idim] = axis;
    }
    return axes;
  }

  template <typename Real>
  Real *get_grid_weights() {
    static_assert(std::is_same_v<Real, float> || std::is_same_v<Real, double>,
                  "GridProps supports only float32 and float64 arrays.");
    constexpr int typenum =
        std::is_same_v<Real, float> ? NPY_FLOAT32 : NPY_FLOAT64;

    if (has_grid_weights()) {
      grid_weights_ = static_cast<void *>(PyArray_DATA(np_grid_weights_));
      need_grid_weights_ = false;
      return static_cast<Real *>(grid_weights_);
    }

    npy_intp np_dims_weights[MAX_GRID_NDIM];
    for (int i = 0; i < ndim; i++) {
      np_dims_weights[i] = dims[i];
    }
    np_grid_weights_ =
        (PyArrayObject *)PyArray_ZEROS(ndim, np_dims_weights, typenum, 0);
    if (np_grid_weights_ == NULL) {
      return NULL;
    }

    grid_weights_ = static_cast<void *>(PyArray_DATA(np_grid_weights_));
    RETURN_IF_PYERR();

    need_grid_weights_ = true;
    owns_grid_weights_ = true;
    return static_cast<Real *>(grid_weights_);
  }

  int get_float_typenum() const;

  /* Is wavelength masked? */
  bool lam_is_masked(int ind) const;

  /* Do we have grid weights already? */
  bool has_grid_weights() const;

  /* Do we need to populate the grid weights? */
  bool need_grid_weights() const;

 private:
  /* The spectra array. */
  PyArrayObject *np_spectra_;

  /* The properties along each axis. */
  PyObject *axes_tuple_;

  /* Names for the axis arrays. */
  std::array<std::string, MAX_GRID_NDIM> axis_names_;

  /* The wavelength array. */
  PyArrayObject *np_lam_;

  /* The wavelength mask array. */
  PyArrayObject *np_lam_mask_;

  /* The grid weights array. */
  PyArrayObject *np_grid_weights_;

  /* A pointer to the grid weights array data. */
  void *grid_weights_ = nullptr;

  /* The shared floating-point dtype used by the grid arrays. */
  int float_typenum_ = -1;

  /* Flag for whether we need to populate the grid weights */
  bool need_grid_weights_ = true;

  /* Did this object allocate the grid weights array itself? */
  bool owns_grid_weights_ = false;

  /* The dimensions of the spectra array (account for the wavelength axis). */
  std::array<int, MAX_GRID_NDIM + 1> spectra_dims_;
};
#pragma omp end declare target

#endif  // GRID_PROPS_H_
