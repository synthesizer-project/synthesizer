#ifndef GRID_PROPS_H_
#define GRID_PROPS_H_

/* Standard includes */
#include <array>
#include <stdlib.h>

/* Python includes */
#define PY_ARRAY_UNIQUE_SYMBOL SYNTHESIZER_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy_init.h"
#include <Python.h>

/* Local includes */
#include "property_funcs.h"

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
            PyArrayObject *np_grid_weights = NULL);

  /* Index handlers for indexing the grid properties. */
  int ravel_grid_index(const std::array<int, MAX_GRID_NDIM> &multi_index) const;
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

  /* The wavelength array. */
  PyArrayObject *np_lam_;

  /* The wavelength mask array. */
  PyArrayObject *np_lam_mask_;

  /* The grid weights array. */
  PyArrayObject *np_grid_weights_;

  /* A pointer to the grid weights array data. */
  double *grid_weights_ = nullptr;

  /* Flag for whether we need to populate the grid weights */
  bool need_grid_weights_ = true;

  /* The dimensions of the spectra array (account for the wavelength axis). */
  std::array<int, MAX_GRID_NDIM + 1> spectra_dims_;
};
#pragma omp end declare target

#endif // GRID_PROPS_H_
