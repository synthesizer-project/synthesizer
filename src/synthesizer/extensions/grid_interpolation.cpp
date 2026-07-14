/******************************************************************************
 * C++ extension module for N-dimensional grid interpolation.
 *
 * This module provides high-performance routines for performing grid
 * interpolation on N-dimensional datasets. It supports both:
 *   1. Nearest Grid Point (NGP) interpolation.
 *   2. Cloud-In-Cell (CIC) / Multilinear interpolation.
 *
 * If openmp support is enabled (via WITH_OPENMP), parallel versions of the
 * loops will run when the `nthreads` parameter is greater than 1.
 *****************************************************************************/
/* C includes */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <array>
#include <vector>

/* Python includes */
#define PY_ARRAY_UNIQUE_SYMBOL SYNTHESIZER_ARRAY_API
#define NO_IMPORT_ARRAY
#include <Python.h>

#include "numpy_init.h"

/* Local includes */
#include "cpp_to_python.h"
#include "grid_props.h"
#include "macros.h"
#include "part_props.h"
#include "property_funcs.h"
#include "timers.h"
#ifdef ATOMIC_TIMING
#include "timers_init.h"
#endif
#include "weights.h"

/* Optional openmp include. */
#ifdef WITH_OPENMP
#include <omp.h>
#endif

#include "grid_interpolation.h"

/**
 * Perform serial Cloud-in-Cell (CIC) multilinear interpolation.
 *
 * @param grid_props Struct containing grid properties and raw data array.
 * @param target_coords Struct containing coordinate values of target points.
 * @param n_extra Dimensionality of additional dimensions (e.g. wavelength).
 * @param out_arr Preallocated flat output array of shape (n_coords * n_extra).
 */
static void interpolate_loop_cic_serial(GridProps *grid_props,
                                        Particles *target_coords, int n_extra,
                                        double *out_arr) {
  const std::array<int, MAX_GRID_NDIM> dims = grid_props->dims;
  const int ndim = grid_props->ndim;
  const int num_sub_cells = 1 << ndim;  // 2^ndim corners of the hypercube

  /* Build sub_dims = [2,2,...,2] once */
  std::array<int, MAX_GRID_NDIM> sub_dims;
  for (int i = 0; i < ndim; ++i) {
    sub_dims[i] = 2;
  }

  /* Precompute sub-cell offsets and linear offsets once for the hypercube
   * corners */
  struct SubCell {
    std::array<int, MAX_GRID_NDIM> offs;
    int linoff;
  };
  std::vector<SubCell> subcells(num_sub_cells);
  {
    std::array<int, MAX_GRID_NDIM> tmp{};
    for (int ic = 0; ic < num_sub_cells; ++ic) {
      get_indices_from_flat(ic, ndim, sub_dims, tmp);
      subcells[ic].offs = tmp;
      subcells[ic].linoff = get_flat_index(tmp, dims.data(), ndim);
    }
  }

  const double *grid_data = grid_props->get_spectra();

  /* Loop over target coordinate points. */
  for (int p = 0; p < target_coords->npart; ++p) {
    std::array<int, MAX_GRID_NDIM> coord_idx;
    std::array<double, MAX_GRID_NDIM> axis_frac;

    // Get the base grid index and fractional distance along each axis
    get_part_ind_frac_cic(coord_idx, axis_frac, grid_props, target_coords, p);

    const int base_lin = get_flat_index(coord_idx, dims.data(), ndim);

    // Sum contribution from all 2^ndim hypercube corners
    for (int ic = 0; ic < num_sub_cells; ++ic) {
      const auto &sc = subcells[ic];

      // Multiplicative combination of 1D linear weights
      double frac = 1.0;
      for (int d = 0; d < ndim; ++d) {
        frac *= sc.offs[d] ? axis_frac[d] : (1.0 - axis_frac[d]);
      }
      if (frac == 0.0) {
        continue;
      }

      const int flat_ind = base_lin + sc.linoff;

      // Accumulate interpolated values across the extra dimensions (e.g.
      // wavelength)
      for (int iextra = 0; iextra < n_extra; ++iextra) {
        out_arr[p * n_extra + iextra] +=
            frac * grid_data[flat_ind * n_extra + iextra];
      }
    }
  }
}

#ifdef WITH_OPENMP
/**
 * Perform OpenMP parallelized Cloud-in-Cell (CIC) multilinear interpolation.
 *
 * @param grid_props Struct containing grid properties and raw data array.
 * @param target_coords Struct containing coordinate values of target points.
 * @param n_extra Dimensionality of additional dimensions (e.g. wavelength).
 * @param out_arr Preallocated flat output array of shape (n_coords * n_extra).
 * @param nthreads Number of threads to use in the openmp parallel region.
 */
static void interpolate_loop_cic_omp(GridProps *grid_props,
                                     Particles *target_coords, int n_extra,
                                     double *out_arr, int nthreads) {
  const std::array<int, MAX_GRID_NDIM> dims = grid_props->dims;
  const int ndim = grid_props->ndim;
  const int num_sub_cells = 1 << ndim;

  /* Build sub_dims = [2,2,...,2] once */
  std::array<int, MAX_GRID_NDIM> sub_dims;
  for (int i = 0; i < ndim; ++i) {
    sub_dims[i] = 2;
  }

  /* Precompute sub-cell offsets and linear offsets once */
  struct SubCell {
    std::array<int, MAX_GRID_NDIM> offs;
    int linoff;
  };
  std::vector<SubCell> subcells(num_sub_cells);
  {
    std::array<int, MAX_GRID_NDIM> tmp{};
    for (int ic = 0; ic < num_sub_cells; ++ic) {
      get_indices_from_flat(ic, ndim, sub_dims, tmp);
      subcells[ic].offs = tmp;
      subcells[ic].linoff = get_flat_index(tmp, dims.data(), ndim);
    }
  }

  const double *grid_data = grid_props->get_spectra();

  // Distribute loops over target points using OpenMP static scheduling
#pragma omp parallel for num_threads(nthreads) schedule(static)
  for (int p = 0; p < target_coords->npart; ++p) {
    std::array<int, MAX_GRID_NDIM> coord_idx;
    std::array<double, MAX_GRID_NDIM> axis_frac;
    get_part_ind_frac_cic(coord_idx, axis_frac, grid_props, target_coords, p);

    const int base_lin = get_flat_index(coord_idx, dims.data(), ndim);

    for (int ic = 0; ic < num_sub_cells; ++ic) {
      const auto &sc = subcells[ic];

      double frac = 1.0;
      for (int d = 0; d < ndim; ++d) {
        frac *= sc.offs[d] ? axis_frac[d] : (1.0 - axis_frac[d]);
      }
      if (frac == 0.0) {
        continue;
      }

      const int flat_ind = base_lin + sc.linoff;

      for (int iextra = 0; iextra < n_extra; ++iextra) {
        out_arr[p * n_extra + iextra] +=
            frac * grid_data[flat_ind * n_extra + iextra];
      }
    }
  }
}
#endif

/**
 * Perform serial Nearest Grid Point (NGP) interpolation.
 *
 * @param grid_props Struct containing grid properties and raw data array.
 * @param target_coords Struct containing coordinate values of target points.
 * @param n_extra Dimensionality of additional dimensions (e.g. wavelength).
 * @param out_arr Preallocated flat output array of shape (n_coords * n_extra).
 */
static void interpolate_loop_ngp_serial(GridProps *grid_props,
                                        Particles *target_coords, int n_extra,
                                        double *out_arr) {
  const std::array<int, MAX_GRID_NDIM> dims = grid_props->dims;
  const int ndim = grid_props->ndim;
  const double *grid_data = grid_props->get_spectra();

  for (int p = 0; p < target_coords->npart; ++p) {
    std::array<int, MAX_GRID_NDIM> coord_idx;
    // Determine the single closest grid coordinate index
    get_part_inds_ngp(coord_idx, grid_props, target_coords, p);

    const int flat_ind = get_flat_index(coord_idx, dims.data(), ndim);

    // Copy the values directly from closest grid cell
    for (int iextra = 0; iextra < n_extra; ++iextra) {
      out_arr[p * n_extra + iextra] = grid_data[flat_ind * n_extra + iextra];
    }
  }
}

#ifdef WITH_OPENMP
/**
 * Perform OpenMP parallelized Nearest Grid Point (NGP) interpolation.
 *
 * @param grid_props Struct containing grid properties and raw data array.
 * @param target_coords Struct containing coordinate values of target points.
 * @param n_extra Dimensionality of additional dimensions (e.g. wavelength).
 * @param out_arr Preallocated flat output array of shape (n_coords * n_extra).
 * @param nthreads Number of threads to use in the openmp parallel region.
 */
static void interpolate_loop_ngp_omp(GridProps *grid_props,
                                     Particles *target_coords, int n_extra,
                                     double *out_arr, int nthreads) {
  const std::array<int, MAX_GRID_NDIM> dims = grid_props->dims;
  const int ndim = grid_props->ndim;
  const double *grid_data = grid_props->get_spectra();

#pragma omp parallel for num_threads(nthreads) schedule(static)
  for (int p = 0; p < target_coords->npart; ++p) {
    std::array<int, MAX_GRID_NDIM> coord_idx;
    get_part_inds_ngp(coord_idx, grid_props, target_coords, p);

    const int flat_ind = get_flat_index(coord_idx, dims.data(), ndim);

    for (int iextra = 0; iextra < n_extra; ++iextra) {
      out_arr[p * n_extra + iextra] = grid_data[flat_ind * n_extra + iextra];
    }
  }
}
#endif

/**
 * CPython wrapper/entry point for grid interpolation.
 *
 * Parsed Python arguments:
 *   - np_grid_data: Numpy array of the raw grid data values.
 *   - grid_tuple: Tuple of grid axes coordinates.
 *   - coords_tuple: Tuple of target coordinate arrays.
 *   - ndim: Dimensionality of grid.
 *   - n_coords: Number of target coordinate values.
 *   - n_extra: Dimensionality of extra array parameters (e.g.
 * wavelength/lines).
 *   - method: "cic" (linear) or "ngp" (nearest).
 *   - nthreads: Number of openmp threads.
 *   - prop_names: Optional property axis names.
 */
PyObject *interpolate_grid_array(PyObject *self, PyObject *args) {
  tic("interpolate_grid_array");

  (void)self;

  int ndim, n_coords, n_extra, nthreads;
  PyObject *grid_tuple, *coords_tuple;
  PyObject *prop_names = NULL;
  PyArrayObject *np_grid_data;
  char *method;

  // Parse tuple inputs
  if (!PyArg_ParseTuple(args, "OOOiiisi|O", &np_grid_data, &grid_tuple,
                        &coords_tuple, &ndim, &n_coords, &n_extra, &method,
                        &nthreads, &prop_names))
    return nullptr;

  /* Extract the grid struct. */
  GridProps *grid_props =
      new GridProps(np_grid_data, grid_tuple,
                    /*np_lam*/ nullptr, /*np_lam_mask*/ nullptr, n_extra,
                    /*np_grid_weights*/ NULL, prop_names);

  RETURN_IF_PYERR();

  /* Create the object that holds target coordinate properties. */
  Particles *target_coords =
      new Particles(/*np_weights*/ nullptr, /*np_velocities*/ nullptr,
                    /*np_mask*/ nullptr, coords_tuple, prop_names, n_coords);

  RETURN_IF_PYERR();

  /* Allocate the output numpy array. */
  npy_intp np_dims[2] = {n_coords, n_extra};
  PyArrayObject *out_arr_obj =
      (PyArrayObject *)PyArray_ZEROS(2, np_dims, NPY_DOUBLE, 0);
  if (out_arr_obj == nullptr) {
    delete target_coords;
    delete grid_props;
    return nullptr;
  }
  double *out_arr = static_cast<double *>(PyArray_DATA(out_arr_obj));

  /* Run the appropriate interpolation loop. */
  if (strcmp(method, "cic") == 0) {
#ifdef WITH_OPENMP
    if (nthreads > 1) {
      interpolate_loop_cic_omp(grid_props, target_coords, n_extra, out_arr,
                               nthreads);
    } else {
      interpolate_loop_cic_serial(grid_props, target_coords, n_extra, out_arr);
    }
#else
    (void)nthreads;
    interpolate_loop_cic_serial(grid_props, target_coords, n_extra, out_arr);
#endif
  } else if (strcmp(method, "ngp") == 0) {
#ifdef WITH_OPENMP
    if (nthreads > 1) {
      interpolate_loop_ngp_omp(grid_props, target_coords, n_extra, out_arr,
                               nthreads);
    } else {
      interpolate_loop_ngp_serial(grid_props, target_coords, n_extra, out_arr);
    }
#else
    (void)nthreads;
    interpolate_loop_ngp_serial(grid_props, target_coords, n_extra, out_arr);
#endif
  } else {
    PyErr_SetString(PyExc_ValueError, "Unknown interpolation method.");
    delete target_coords;
    delete grid_props;
    Py_XDECREF(out_arr_obj);
    return nullptr;
  }

  // Deallocate internal structs
  delete target_coords;
  delete grid_props;

  toc("interpolate_grid_array");

  return Py_BuildValue("N", out_arr_obj);
}

static PyMethodDef GridInterpolationMethods[] = {
    {"interpolate_grid_array", (PyCFunction)interpolate_grid_array,
     METH_VARARGS, "Method for interpolating a grid array at coordinates."},
    {nullptr, nullptr, 0, nullptr}};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "grid_interpolation",
    "A module for performing grid interpolation",
    -1,
    GridInterpolationMethods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

PyMODINIT_FUNC PyInit_grid_interpolation(void) {
  PyObject *m = PyModule_Create(&moduledef);
  if (m == NULL) return NULL;
  if (numpy_import() < 0) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to import numpy.");
    Py_DECREF(m);
    return NULL;
  }
#ifdef ATOMIC_TIMING
  if (import_toc_capsule() < 0) {
    Py_DECREF(m);
    return NULL;
  }
#endif
  return m;
}
