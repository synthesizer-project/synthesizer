/******************************************************************************
 * C extension for combined Draine and Li attenuation extraction.
 *****************************************************************************/

/* C includes */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* Python includes */
#define PY_ARRAY_UNIQUE_SYMBOL SYNTHESIZER_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy_init.h"
#include <Python.h>

/* Local includes */
#include "property_funcs.h"
#include "timers.h"
#ifdef ATOMIC_TIMING
#include "timers_init.h"
#endif
#include "weights.h"

/**
 * @brief Combined Draine-Li attenuation loop.
 *
 * This computes the total attenuation from multiple grain-component spectra
 * grids in a single pass over the particles.
 */
static void draine_li_loop_serial(
    double **spectra_grids, int ngrains, const double *grid_axis,
    int ngrid, const double *dtg_values, const double *dust_columns,
    const npy_bool *valid_mask, const double tau_scale, int npart, int nlam,
    double *tau_out) {
  for (int p = 0; p < npart; ++p) {
    for (int g = 0; g < ngrains; ++g) {
      const npy_intp idx = static_cast<npy_intp>(p) * ngrains + g;
      if (!valid_mask[idx]) {
        continue;
      }

      const double dtg = dtg_values[idx];
      const int upper = binary_search(0, ngrid - 1, grid_axis, dtg);
      const int lower = upper - 1;
      const double low = grid_axis[lower];
      const double high = grid_axis[upper];
      const double frac = (high == low) ? 0.0 : (dtg - low) / (high - low);
      const double dust_col = dust_columns[idx];
      const double *grid = spectra_grids[g];
      const int lower_base = (lower * nlam);
      const int upper_base = (upper * nlam);
      const double dust_tau_scale = tau_scale * dust_col;

      for (int ilam = 0; ilam < nlam; ++ilam) {
        const double curve =
            (1.0 - frac) * grid[lower_base + ilam] + frac * grid[upper_base + ilam];
        tau_out[p * nlam + ilam] += curve * dust_tau_scale;
      }
    }
  }
}

/**
 * @brief Python wrapper for combined Draine-Li attenuation extraction.
 */
static PyObject *compute_draine_li_attenuation(PyObject *self, PyObject *args) {
  (void)self;

  PyObject *spectra_list_obj;
  PyArrayObject *np_grid_axis;
  PyArrayObject *np_dtg_values;
  PyArrayObject *np_dust_columns;
  PyArrayObject *np_valid_mask;
  double tau_scale;

  if (!PyArg_ParseTuple(args, "OO!O!O!O!d", &spectra_list_obj, &PyArray_Type,
                        &np_grid_axis, &PyArray_Type, &np_dtg_values,
                        &PyArray_Type, &np_dust_columns, &PyArray_Type,
                        &np_valid_mask, &tau_scale)) {
    return NULL;
  }

  if (!PySequence_Check(spectra_list_obj) || PyUnicode_Check(spectra_list_obj)) {
    PyErr_SetString(PyExc_TypeError, "spectra_list must be a sequence of arrays.");
    return NULL;
  }

  const Py_ssize_t ngrains_py = PySequence_Size(spectra_list_obj);
  if (ngrains_py <= 0) {
    PyErr_SetString(PyExc_ValueError, "spectra_list must contain at least one grid.");
    return NULL;
  }
  const int ngrains = static_cast<int>(ngrains_py);

  const double *grid_axis = extract_data_double(np_grid_axis, "grid_axis");
  const double *dtg_values = extract_data_double(np_dtg_values, "dtg_values");
  const double *dust_columns = extract_data_double(np_dust_columns, "dust_columns");
  const npy_bool *valid_mask =
      static_cast<npy_bool *>(PyArray_DATA(np_valid_mask));
  if (grid_axis == NULL || dtg_values == NULL || dust_columns == NULL) {
    return NULL;
  }

  if (PyArray_TYPE(np_valid_mask) != NPY_BOOL ||
      !PyArray_IS_C_CONTIGUOUS(np_valid_mask)) {
    PyErr_SetString(PyExc_TypeError, "valid_mask must be a C-contiguous bool array.");
    return NULL;
  }

  const int ngrid = static_cast<int>(PyArray_DIM(np_grid_axis, 0));
  if (PyArray_NDIM(np_dtg_values) != 2 || PyArray_NDIM(np_dust_columns) != 2 ||
      PyArray_NDIM(np_valid_mask) != 2) {
    PyErr_SetString(PyExc_ValueError,
                    "dtg_values, dust_columns, and valid_mask must be 2D arrays.");
    return NULL;
  }
  const int npart = static_cast<int>(PyArray_DIM(np_dtg_values, 0));
  if (PyArray_DIM(np_dtg_values, 1) != ngrains ||
      PyArray_DIM(np_dust_columns, 0) != npart ||
      PyArray_DIM(np_dust_columns, 1) != ngrains ||
      PyArray_DIM(np_valid_mask, 0) != npart ||
      PyArray_DIM(np_valid_mask, 1) != ngrains) {
    PyErr_SetString(PyExc_ValueError,
                    "dtg_values, dust_columns, and valid_mask must have shape (npart, ngrains).");
    return NULL;
  }

  double **spectra_grids = static_cast<double **>(malloc(ngrains * sizeof(double *)));
  if (spectra_grids == NULL) {
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate spectra grid pointer array.");
    return NULL;
  }

  int nlam = -1;
  for (int g = 0; g < ngrains; ++g) {
    PyObject *item = PySequence_GetItem(spectra_list_obj, g);
    if (item == NULL) {
      free(spectra_grids);
      return NULL;
    }
    if (!PyArray_Check(item)) {
      Py_DECREF(item);
      free(spectra_grids);
      PyErr_SetString(PyExc_TypeError, "Each spectra grid must be a numpy array.");
      return NULL;
    }

    PyArrayObject *np_grid = reinterpret_cast<PyArrayObject *>(item);
    if (PyArray_NDIM(np_grid) != 2 || PyArray_DIM(np_grid, 0) != ngrid) {
      Py_DECREF(item);
      free(spectra_grids);
      PyErr_SetString(PyExc_ValueError,
                      "Each spectra grid must have shape (ngrid, nlam).");
      return NULL;
    }
    if (nlam < 0) {
      nlam = static_cast<int>(PyArray_DIM(np_grid, 1));
    } else if (PyArray_DIM(np_grid, 1) != nlam) {
      Py_DECREF(item);
      free(spectra_grids);
      PyErr_SetString(PyExc_ValueError,
                      "All spectra grids must have the same nlam.");
      return NULL;
    }

    spectra_grids[g] = extract_data_double(np_grid, "spectra_grid");
    Py_DECREF(item);
    if (spectra_grids[g] == NULL) {
      free(spectra_grids);
      return NULL;
    }
  }

  npy_intp out_dims[2] = {npart, nlam};
  PyArrayObject *np_tau =
      (PyArrayObject *)PyArray_ZEROS(2, out_dims, NPY_DOUBLE, 0);
  if (np_tau == NULL) {
    free(spectra_grids);
    return NULL;
  }
  double *tau_out = static_cast<double *>(PyArray_DATA(np_tau));

  tic("Combined DraineLi attenuation loop");
  draine_li_loop_serial(spectra_grids, ngrains, grid_axis, ngrid, dtg_values,
                        dust_columns, valid_mask, tau_scale, npart, nlam,
                        tau_out);
  toc("Combined DraineLi attenuation loop");

  free(spectra_grids);
  return reinterpret_cast<PyObject *>(np_tau);
}

static PyMethodDef DraineLiMethods[] = {
    {"compute_draine_li_attenuation",
     (PyCFunction)compute_draine_li_attenuation, METH_VARARGS,
     "Compute combined Draine-Li attenuation for multiple grain grids."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "draine_li_attenuation",
    "A module to calculate combined Draine-Li attenuation.",
    -1,
    DraineLiMethods,
    NULL,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC PyInit_draine_li_attenuation(void) {
  PyObject *m = PyModule_Create(&moduledef);
  if (m == NULL)
    return NULL;
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
