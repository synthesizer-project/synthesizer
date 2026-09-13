/******************************************************************************
 * C++ extension module for N-dimensional grid interpolation.
 *
 * This module provides high-performance routines for performing grid
 * interpolation on N-dimensional datasets. It supports both:
 *   1. Nearest Grid Point (NGP) interpolation.
 *   2. Cloud-In-Cell (CIC) / Multilinear interpolation.
 *
 * The grid data, the target coordinates and the output array can each be
 * float32 or float64. The kernels below are templated on all three dtypes and
 * dispatched at the Python boundary, so no input is ever cast or copied
 * behind the scenes.
 *
 * If openmp support is enabled (via WITH_OPENMP), parallel versions of the
 * loops will run when the `nthreads` parameter is greater than 1.
 *****************************************************************************/
/* C includes */
#include <array>
#include <cmath>
#include <math.h>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

/* Python includes */
#define PY_ARRAY_UNIQUE_SYMBOL SYNTHESIZER_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy_init.h"

#include <Python.h>

/* Local includes */
#include "cpp_to_python.h"
#include "grid_props.h"
#include "macros.h"
#include "part_props.h"
#include "property_funcs.h"
#include "python_to_cpp.h"
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
 * @brief Perform Cloud-in-Cell (CIC) multilinear interpolation.
 *
 * Each target point writes to its own row of the output array, so the loop
 * over points is trivially parallel.
 *
 * @tparam CoordReal The floating-point type of the target coordinate arrays.
 * @tparam GridReal The floating-point type of the grid data and axis arrays.
 * @tparam OutT The floating-point type stored in the output buffer.
 *
 * @param grid_props: Struct containing grid properties and raw data array.
 * @param target_coords: Struct containing the target point coordinates.
 * @param n_extra: Dimensionality of additional dimensions (e.g. wavelength).
 * @param out_arr: Preallocated zeroed output array of shape
 *                 (n_coords * n_extra).
 * @param nthreads: Number of threads to use in the openmp parallel region.
 */
template <typename CoordReal, typename GridReal, typename OutT>
static void interpolate_loop_cic(GridProps *grid_props,
                                 Particles *target_coords, int n_extra,
                                 OutT *out_arr, int nthreads) {
  tic("interpolate_loop_cic");

  const int ndim = grid_props->ndim;
  const int num_sub_cells = 1 << ndim; /* 2^ndim corners of the hypercube */
  const int npart = target_coords->npart;

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
      subcells[ic].linoff = grid_props->ravel_grid_index(tmp);
    }
  }

  const GridReal *__restrict grid_data = grid_props->get_spectra<GridReal>();

  /* Loop over target coordinate points. */
#ifdef WITH_OPENMP
#pragma omp parallel for num_threads(nthreads) \
    schedule(static) if (nthreads > 1)
#else
  (void)nthreads;
#endif
  for (int p = 0; p < npart; ++p) {

    /* Find the base cell and the fractional distance along each axis. */
    std::array<int, MAX_GRID_NDIM> coord_indices;
    std::array<GridReal, MAX_GRID_NDIM> axis_fracs;
    get_part_ind_frac_cic<CoordReal, GridReal>(coord_indices, axis_fracs,
                                               grid_props, target_coords, p);
    const int base_lin = grid_props->ravel_grid_index(coord_indices);

    /* Get this point's output row. */
    OutT *__restrict out_row =
        out_arr + static_cast<size_t>(p) * static_cast<size_t>(n_extra);

    /* Sum contribution from all 2^ndim hypercube corners. */
    for (int ic = 0; ic < num_sub_cells; ++ic) {
      const auto &sc = subcells[ic];

      /* Multiplicative combination of 1D linear weights. */
      double frac = 1.0;
      for (int d = 0; d < ndim; ++d) {
        frac *= sc.offs[d] ? axis_fracs[d]
                           : (static_cast<GridReal>(1) - axis_fracs[d]);
      }
      if (frac == 0.0) {
        continue;
      }
      const OutT weight = static_cast<OutT>(frac);

      /* Accumulate interpolated values across the extra dimensions (e.g.
       * wavelength). */
      const GridReal *__restrict cell =
          grid_data +
          (static_cast<size_t>(base_lin) + static_cast<size_t>(sc.linoff)) *
              static_cast<size_t>(n_extra);
      for (int iextra = 0; iextra < n_extra; ++iextra) {
        out_row[iextra] =
            std::fma(static_cast<OutT>(cell[iextra]), weight, out_row[iextra]);
      }
    }
  }

  toc("interpolate_loop_cic");
}

/**
 * @brief Perform Nearest Grid Point (NGP) interpolation.
 *
 * @tparam CoordReal The floating-point type of the target coordinate arrays.
 * @tparam GridReal The floating-point type of the grid data and axis arrays.
 * @tparam OutT The floating-point type stored in the output buffer.
 *
 * @param grid_props: Struct containing grid properties and raw data array.
 * @param target_coords: Struct containing the target point coordinates.
 * @param n_extra: Dimensionality of additional dimensions (e.g. wavelength).
 * @param out_arr: Preallocated output array of shape (n_coords * n_extra).
 * @param nthreads: Number of threads to use in the openmp parallel region.
 */
template <typename CoordReal, typename GridReal, typename OutT>
static void interpolate_loop_ngp(GridProps *grid_props,
                                 Particles *target_coords, int n_extra,
                                 OutT *out_arr, int nthreads) {
  tic("interpolate_loop_ngp");

  const int npart = target_coords->npart;
  const GridReal *__restrict grid_data = grid_props->get_spectra<GridReal>();

#ifdef WITH_OPENMP
#pragma omp parallel for num_threads(nthreads) \
    schedule(static) if (nthreads > 1)
#else
  (void)nthreads;
#endif
  for (int p = 0; p < npart; ++p) {

    /* Find the closest grid point to this coordinate. */
    std::array<int, MAX_GRID_NDIM> coord_indices;
    get_part_inds_ngp<CoordReal, GridReal>(coord_indices, grid_props,
                                           target_coords, p);
    const int flat_ind = grid_props->ravel_grid_index(coord_indices);

    /* Copy the values directly from the closest grid cell. */
    OutT *__restrict out_row =
        out_arr + static_cast<size_t>(p) * static_cast<size_t>(n_extra);
    const GridReal *__restrict cell =
        grid_data +
        static_cast<size_t>(flat_ind) * static_cast<size_t>(n_extra);
    for (int iextra = 0; iextra < n_extra; ++iextra) {
      out_row[iextra] = static_cast<OutT>(cell[iextra]);
    }
  }

  toc("interpolate_loop_ngp");
}

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
 *   - out_dtype: Requested floating-point dtype of the returned array.
 *   - prop_names: Optional property axis names.
 */
PyObject *interpolate_grid_array(PyObject *self, PyObject *args) {
  tic("interpolate_grid_array");

  (void)self;

  int ndim, n_coords, n_extra, nthreads;
  PyObject *grid_tuple, *coords_tuple;
  PyObject *out_dtype;
  PyObject *prop_names = NULL;
  PyArrayObject *np_grid_data;
  char *method;

  // Parse tuple inputs
  if (!PyArg_ParseTuple(args, "OOOiiisiO|O", &np_grid_data, &grid_tuple,
                        &coords_tuple, &ndim, &n_coords, &n_extra, &method,
                        &nthreads, &out_dtype, &prop_names))
    return nullptr;

  /* Extract the grid struct. */
  auto grid_props = std::make_unique<GridProps>(
      np_grid_data, grid_tuple,
      /*np_lam*/ nullptr, /*np_lam_mask*/ nullptr, n_extra,
      /*np_grid_weights*/ nullptr, prop_names);

  RETURN_IF_PYERR();

  /* Create the object that holds target coordinate properties. */
  auto target_coords = std::make_unique<Particles>(
      /*np_weights*/ nullptr, /*np_velocities*/ nullptr,
      /*np_mask*/ nullptr, coords_tuple, prop_names, n_coords);

  RETURN_IF_PYERR();

  /* Resolve the dtypes we will dispatch the kernels on. */
  const int grid_typenum = grid_props->get_float_typenum();
  const int coord_typenum = target_coords->get_float_typenum();
  const int output_typenum = resolve_output_typenum(out_dtype, "out_dtype");
  if (output_typenum < 0) {
    return nullptr;
  }

  /* Allocate the output numpy array in the requested precision. */
  npy_intp np_dims[2] = {n_coords, n_extra};
  PyArrayObject *out_arr_obj =
      (PyArrayObject *)PyArray_ZEROS(2, np_dims, output_typenum, 0);
  if (out_arr_obj == nullptr) {
    return nullptr;
  }

  /* Run the appropriate interpolation loop. */
  if (strcmp(method, "cic") == 0) {
    dispatch_float(coord_typenum, [&](auto c) {
      dispatch_float(grid_typenum, [&](auto g) {
        dispatch_float(output_typenum, [&](auto o) {
          using CoordReal = decltype(c);
          using GridReal = decltype(g);
          using OutT = decltype(o);
          interpolate_loop_cic<CoordReal, GridReal, OutT>(
              grid_props.get(), target_coords.get(), n_extra,
              static_cast<OutT *>(PyArray_DATA(out_arr_obj)), nthreads);
        });
      });
    });
  } else if (strcmp(method, "ngp") == 0) {
    dispatch_float(coord_typenum, [&](auto c) {
      dispatch_float(grid_typenum, [&](auto g) {
        dispatch_float(output_typenum, [&](auto o) {
          using CoordReal = decltype(c);
          using GridReal = decltype(g);
          using OutT = decltype(o);
          interpolate_loop_ngp<CoordReal, GridReal, OutT>(
              grid_props.get(), target_coords.get(), n_extra,
              static_cast<OutT *>(PyArray_DATA(out_arr_obj)), nthreads);
        });
      });
    });
  } else {
    PyErr_SetString(PyExc_ValueError, "Unknown interpolation method.");
    Py_XDECREF(out_arr_obj);
    return nullptr;
  }

  if (PyErr_Occurred() != NULL) {
    Py_DECREF(out_arr_obj);
    return nullptr;
  }

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
