/******************************************************************************
 * C extension to calculate integrated SEDs for a galaxy's star particles.
 * Calculates weights on an arbitrary dimensional grid given the mass.
 *****************************************************************************/
/* C includes */
#include <array>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

/**
 * @brief Computes an integrated SED for a collection of particles.
 *
 * @param np_grid_spectra: The SPS spectra array.
 * o
 * @param grid_tuple: The tuple containing arrays of grid axis properties.
 * @param part_tuple: The tuple of particle property arrays (in the same order
 *                    as grid_tuple).
 * @param np_part_mass: The particle mass array.
 * @param np_ndims: The size of each grid axis.
 * @param ndim: The number of grid axes.
 * @param npart: The number of particles.
 * @param nlam: The number of wavelength elements.
 * @param out_dtype: Requested floating-point dtype for the returned SFZH.
 */
PyObject *compute_sfzh(PyObject *self, PyObject *args) {

  tic("compute_sfzh");

  /* We don't need the self argument but it has to be there. Tell the compiler
   * we don't care. */
  (void)self;

  int ndim, npart, nthreads;
  PyObject *grid_tuple, *part_tuple;
  PyObject *prop_names = NULL;
  PyObject *out_dtype = Py_None;
  PyArrayObject *np_part_mass, *np_ndims;
  PyArrayObject *np_mask;
  char *method;

  /* Parse the Python-level inputs. */
  if (!PyArg_ParseTuple(args, "OOOOiisiO|OO", &grid_tuple, &part_tuple,
                        &np_part_mass, &np_ndims, &ndim, &npart, &method,
                        &nthreads, &np_mask, &prop_names, &out_dtype))
    return NULL;

  /* Extract the grid struct. */
  GridProps *grid_props =
      new GridProps(/*np_grid_spectra*/ nullptr, grid_tuple,
                    /*np_lam*/ nullptr, /*np_lam_mask*/ nullptr, 1,
                    /*np_grid_weights*/ NULL, prop_names);
  RETURN_IF_PYERR();

  /* Extract the particle struct. */
  Particles *parts = new Particles(np_part_mass, /*np_velocities*/ NULL,
                                   np_mask, part_tuple, prop_names, npart);
  RETURN_IF_PYERR();

  /* Resolve the shared input precision family before allocating the output. */
  const int grid_typenum = grid_props->get_float_typenum();
  const int part_typenum = parts->get_float_typenum();
  if (grid_typenum != -1 && part_typenum != -1 &&
      grid_typenum != part_typenum) {
    PyErr_SetString(
        PyExc_TypeError,
        "Grid and particle arrays must share the same floating-point dtype.");
    delete parts;
    delete grid_props;
    return NULL;
  }

  /* Default to the shared input precision family, or float64 if neither has any
   * float arrays. */
  const int input_typenum = grid_typenum != -1 ? grid_typenum : part_typenum;
  int output_typenum = input_typenum;
  if (out_dtype != Py_None) {
    output_typenum = resolve_output_typenum(out_dtype, "out_dtype");
    if (output_typenum < 0) {
      delete parts;
      delete grid_props;
      return NULL;
    }
  }

  /* Allocate the SFZH array in the requested output precision. */
  void *sfzh = NULL;
  {
    int dispatch_key = (output_typenum == NPY_FLOAT64);

    /* Dispatch: call the matching typed kernel based on the dispatch key. */
    switch (dispatch_key) {
    case 0:
      sfzh = static_cast<void *>(grid_props->get_grid_weights<float>());
      break;
    default:
      sfzh = static_cast<void *>(grid_props->get_grid_weights<double>());
      break;
    }
  }
  RETURN_IF_PYERR();

  /* With everything set up we can compute the weights for each particle using
   * the requested method. */
  if (strcmp(method, "cic") == 0) {
    {
      int dispatch_key = ((input_typenum == NPY_FLOAT64) << 1) |
                         (output_typenum == NPY_FLOAT64);

      /* Dispatch: call the matching typed kernel based on the dispatch key. */
      switch (dispatch_key) {
      case 0:
        weight_loop_cic<float, float>(grid_props, parts, grid_props->size,
                                      static_cast<float *>(sfzh), nthreads);
        break;
      case 1:
        weight_loop_cic<float, double>(grid_props, parts, grid_props->size,
                                       static_cast<double *>(sfzh), nthreads);
        break;
      case 2:
        weight_loop_cic<double, float>(grid_props, parts, grid_props->size,
                                       static_cast<float *>(sfzh), nthreads);
        break;
      default:
        weight_loop_cic<double, double>(grid_props, parts, grid_props->size,
                                        static_cast<double *>(sfzh), nthreads);
        break;
      }
    }
  } else if (strcmp(method, "ngp") == 0) {
    {
      int dispatch_key = ((input_typenum == NPY_FLOAT64) << 1) |
                         (output_typenum == NPY_FLOAT64);

      /* Dispatch: call the matching typed kernel based on the dispatch key. */
      switch (dispatch_key) {
      case 0:
        weight_loop_ngp<float, float>(grid_props, parts, grid_props->size,
                                      static_cast<float *>(sfzh), nthreads);
        break;
      case 1:
        weight_loop_ngp<float, double>(grid_props, parts, grid_props->size,
                                       static_cast<double *>(sfzh), nthreads);
        break;
      case 2:
        weight_loop_ngp<double, float>(grid_props, parts, grid_props->size,
                                       static_cast<float *>(sfzh), nthreads);
        break;
      default:
        weight_loop_ngp<double, double>(grid_props, parts, grid_props->size,
                                        static_cast<double *>(sfzh), nthreads);
        break;
      }
    }
  } else {
    PyErr_Format(PyExc_ValueError, "Unknown grid assignment method (%s).",
                 method);
    delete parts;
    delete grid_props;
    return NULL;
  }
  RETURN_IF_PYERR();

  /* Extract the grid weights we'll write out. */
  PyArrayObject *np_sfzh = grid_props->get_np_grid_weights();

  /* Clean up memory! */
  delete parts;
  delete grid_props;

  toc("compute_sfzh");

  return Py_BuildValue("N", np_sfzh);
}

/* Below is all the gubbins needed to make the module importable in Python. */
static PyMethodDef SFZHMethods[] = {{"compute_sfzh", (PyCFunction)compute_sfzh,
                                     METH_VARARGS,
                                     "Method for calculating the SFZH."},
                                    {NULL, NULL, 0, NULL}};

/* Make this importable. */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "make_sfzh",                             /* m_name */
    "A module to calculating particle SFZH", /* m_doc */
    -1,                                      /* m_size */
    SFZHMethods,                             /* m_methods */
    NULL,                                    /* m_reload */
    NULL,                                    /* m_traverse */
    NULL,                                    /* m_clear */
    NULL,                                    /* m_free */
};

PyMODINIT_FUNC PyInit_sfzh(void) {
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
