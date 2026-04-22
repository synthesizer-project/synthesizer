/******************************************************************************
 * C extension helpers for LOS kernel-table construction.
 *****************************************************************************/

/* C headers. */
#include <math.h>
#include <string.h>

#ifdef WITH_OPENMP
#include <omp.h>
#endif

/* Python headers. */
#define PY_ARRAY_UNIQUE_SYMBOL SYNTHESIZER_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy_init.h"
#include <Python.h>

/* Local includes. */
#include "property_funcs.h"

/**
 * @brief Evaluate the uniform kernel.
 */
static inline double uniform(const double r) {
  if (r < 1.0) {
    return 1.0 / ((4.0 / 3.0) * M_PI);
  }
  return 0.0;
}

/**
 * @brief Evaluate the SPH Anarchy kernel.
 */
static inline double sph_anarchy(const double r) {
  if (r <= 1.0) {
    const double one_minus_r = 1.0 - r;
    return (21.0 / (2.0 * M_PI)) * (one_minus_r * one_minus_r *
                                     one_minus_r * one_minus_r *
                                     (1.0 + 4.0 * r));
  }
  return 0.0;
}

/**
 * @brief Evaluate the Gadget-2 kernel.
 */
static inline double gadget_2(const double r) {
  if (r < 0.5) {
    return (8.0 / M_PI) * (1.0 - 6.0 * (r * r) + 6.0 * (r * r * r));
  }
  if (r < 1.0) {
    const double one_minus_r = 1.0 - r;
    return (16.0 / M_PI) *
           (one_minus_r * one_minus_r * one_minus_r);
  }
  return 0.0;
}

/**
 * @brief Evaluate the cubic kernel.
 */
static inline double cubic(const double r) {
  if (r < 0.5) {
    return 2.546479089470 + 15.278874536822 * (r - 1.0) * r * r;
  }
  if (r < 1.0) {
    const double one_minus_r = 1.0 - r;
    return 5.092958178941 *
           (one_minus_r * one_minus_r * one_minus_r);
  }
  return 0.0;
}

/**
 * @brief Evaluate the quintic kernel.
 */
static inline double quintic(const double r) {
  if (r < 0.333333333) {
    return 27.0 * (6.4457752 * r * r * r * r * (1.0 - r) -
                   1.4323945 * r * r + 0.17507044);
  }
  if (r < 0.666666667) {
    return 27.0 *
           (3.2228876 * r * r * r * r * (r - 3.0) +
            10.7429587 * r * r * r - 5.01338071 * r * r +
            0.5968310366 * r + 0.1352817016);
  }
  if (r < 1.0) {
    return 27.0 * 0.64457752 *
           (-r * r * r * r * r + 5.0 * r * r * r * r -
            10.0 * r * r * r + 10.0 * r * r - 5.0 * r + 1.0);
  }
  return 0.0;
}

typedef double (*kernel_func)(double);

/**
 * @brief Map a kernel name onto the corresponding analytic function.
 */
static kernel_func get_kernel_function(const char *name) {
  if (strcmp(name, "uniform") == 0) {
    return uniform;
  }
  if (strcmp(name, "sph_anarchy") == 0) {
    return sph_anarchy;
  }
  if (strcmp(name, "gadget_2") == 0) {
    return gadget_2;
  }
  if (strcmp(name, "cubic") == 0) {
    return cubic;
  }
  if (strcmp(name, "quintic") == 0) {
    return quintic;
  }
  return NULL;
}

/**
 * @brief Build the truncated LOS kernel lookup table.
 *
 * The output table is stored with projected-separation index first and
 * LOS-coordinate index second.
 */
static void build_truncated_los_kernel(double *kernel, const double *q_grid,
                                       const double *z_grid,
                                       const int qdim, const int zdim,
                                       kernel_func func) {
#ifdef WITH_OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int iq = 0; iq < qdim; iq++) {
    const double q = q_grid[iq];
    double cumulative = 0.0;
    double prev_value = 0.0;
    double prev_z = z_grid[0];

    {
      const double radius = sqrt(prev_z * prev_z + q * q);
      if (radius < 1.0) {
        prev_value = func(radius);
      }
      kernel[iq * zdim] = 0.0;
    }

    for (int iz = 1; iz < zdim; iz++) {
      const double z = z_grid[iz];
      const double radius = sqrt(z * z + q * q);
      double value = 0.0;
      if (radius < 1.0) {
        value = func(radius);
      }

      cumulative += 0.5 * (prev_value + value) * (z - prev_z);
      kernel[iq * zdim + iz] = cumulative;

      prev_value = value;
      prev_z = z;
    }
  }
}

/**
 * @brief Python wrapper for truncated LOS kernel-table construction.
 *
 * Args:
 *   q_grid: 1D float64 C-contiguous projected-separation grid.
 *   z_grid: 1D float64 C-contiguous LOS-coordinate grid.
 *   kernel_name: Name of the analytic kernel to evaluate.
 *
 * Returns:
 *   A 2D float64 NumPy array with shape ``(q_grid.size, z_grid.size)``.
 */
PyObject *compute_truncated_los_kernel(PyObject *self, PyObject *args) {

  (void)self;

  PyArrayObject *np_q_grid, *np_z_grid;
  const char *kernel_name;

  if (!PyArg_ParseTuple(args, "OOs", &np_q_grid, &np_z_grid, &kernel_name)) {
    return NULL;
  }

  if (PyArray_NDIM(np_q_grid) != 1 || PyArray_NDIM(np_z_grid) != 1) {
    PyErr_SetString(PyExc_ValueError,
                    "q_grid and z_grid must both be 1D arrays.");
    return NULL;
  }

  const double *q_grid = extract_data_double(np_q_grid, "q_grid");
  const double *z_grid = extract_data_double(np_z_grid, "z_grid");
  if (q_grid == NULL || z_grid == NULL) {
    return NULL;
  }

  kernel_func func = get_kernel_function(kernel_name);
  if (func == NULL) {
    PyErr_SetString(PyExc_ValueError, "Kernel name not defined");
    return NULL;
  }

  const int qdim = static_cast<int>(PyArray_DIM(np_q_grid, 0));
  const int zdim = static_cast<int>(PyArray_DIM(np_z_grid, 0));

  npy_intp dims[2] = {qdim, zdim};
  PyArrayObject *np_kernel =
      (PyArrayObject *)PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
  double *kernel = static_cast<double *>(PyArray_DATA(np_kernel));

  build_truncated_los_kernel(kernel, q_grid, z_grid, qdim, zdim, func);

  return Py_BuildValue("N", np_kernel);
}

static PyMethodDef KernelMethods[] = {
    {"compute_truncated_los_kernel", (PyCFunction)compute_truncated_los_kernel,
     METH_VARARGS, "Build the truncated LOS kernel table."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "kernel",
    "A module to build LOS kernel tables",
    -1,
    KernelMethods,
    NULL,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC PyInit_kernel(void) {
  PyObject *m = PyModule_Create(&moduledef);
  if (m == NULL) {
    return NULL;
  }
  if (numpy_import() < 0) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to import numpy.");
    Py_DECREF(m);
    return NULL;
  }
  return m;
}
