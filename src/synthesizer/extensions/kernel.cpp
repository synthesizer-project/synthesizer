/******************************************************************************
 * C extension helpers for LOS kernel evaluation and table construction.
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
#include "integration.h"
#include "property_funcs.h"

/**
 * @brief Evaluate the uniform kernel at a dimensionless radius.
 *
 * All kernel functions in this file assume the public ``Kernel`` convention
 * used on the Python side: the support radius is normalised to unity and the
 * returned value is the 3D kernel density at the requested radius.
 *
 * @param r The dimensionless radius.
 *
 * @return The kernel value.
 */
static inline double uniform(const double r) {
  if (r < 1.0) {
    return 1.0 / ((4.0 / 3.0) * M_PI);
  }
  return 0.0;
}

/**
 * @brief Evaluate the SPH Anarchy kernel at a dimensionless radius.
 *
 * @param r The dimensionless radius.
 *
 * @return The kernel value.
 */
static inline double sph_anarchy(const double r) {
  if (r <= 1.0) {
    const double one_minus_r = 1.0 - r;
    return (21.0 / (2.0 * M_PI)) * (one_minus_r * one_minus_r * one_minus_r *
                                    one_minus_r * (1.0 + 4.0 * r));
  }
  return 0.0;
}

/**
 * @brief Evaluate the Gadget-2 kernel at a dimensionless radius.
 *
 * @param r The dimensionless radius.
 *
 * @return The kernel value.
 */
static inline double gadget_2(const double r) {
  if (r < 0.5) {
    return (8.0 / M_PI) * (1.0 - 6.0 * (r * r) + 6.0 * (r * r * r));
  }
  if (r < 1.0) {
    const double one_minus_r = 1.0 - r;
    return (16.0 / M_PI) * (one_minus_r * one_minus_r * one_minus_r);
  }
  return 0.0;
}

/**
 * @brief Evaluate the cubic kernel at a dimensionless radius.
 *
 * @param r The dimensionless radius.
 *
 * @return The kernel value.
 */
static inline double cubic(const double r) {
  if (r < 0.5) {
    return 2.546479089470 + 15.278874536822 * (r - 1.0) * r * r;
  }
  if (r < 1.0) {
    const double one_minus_r = 1.0 - r;
    return 5.092958178941 * (one_minus_r * one_minus_r * one_minus_r);
  }
  return 0.0;
}

/**
 * @brief Evaluate the quintic kernel at a dimensionless radius.
 *
 * @param r The dimensionless radius.
 *
 * @return The kernel value.
 */
static inline double quintic(const double r) {
  if (r < 0.333333333) {
    return 27.0 * (6.4457752 * r * r * r * r * (1.0 - r) - 1.4323945 * r * r +
                   0.17507044);
  }
  if (r < 0.666666667) {
    return 27.0 *
           (3.2228876 * r * r * r * r * (r - 3.0) + 10.7429587 * r * r * r -
            5.01338071 * r * r + 0.5968310366 * r + 0.1352817016);
  }
  if (r < 1.0) {
    return 27.0 * 0.64457752 *
           (-r * r * r * r * r + 5.0 * r * r * r * r - 10.0 * r * r * r +
            10.0 * r * r - 5.0 * r + 1.0);
  }
  return 0.0;
}

typedef double (*kernel_func)(double);

/* Use a fixed even number of sub-intervals for Simpson integration so the
 * projected LOS kernel builder is deterministic and easy to compare against
 * the Python quad-based reference in tests. */
static const int PROJECTED_KERNEL_SIMPSON_STEPS = 256;

/**
 * @brief Map a public kernel name onto the corresponding analytic function.
 *
 * Keeping the name dispatch in one place ensures the Python wrappers and the
 * truncated LOS table builder both use exactly the same implementation.
 *
 * @param name The public kernel name.
 *
 * @return Pointer to the matching kernel function, or ``NULL`` if unknown.
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
 * @brief Integrate the projected LOS kernel at one impact parameter.
 *
 * This evaluates
 *
 *     2 * integral_0^sqrt(1 - q^2) W(sqrt(z^2 + q^2)) dz
 *
 * using the shared composite Simpson helper from ``integration.cpp`` on a
 * fixed tabulated LOS grid. The fixed rule keeps the extension simple and fast
 * while still producing a stable value that can be checked against the
 * existing ``scipy.integrate.quad`` reference in the unit tests.
 *
 * @param q The dimensionless impact parameter.
 * @param func The analytic kernel function to evaluate.
 *
 * @return The projected LOS kernel value at ``q``.
 */
static inline double integrate_projected_kernel(const double q,
                                                kernel_func func) {
  if (q < 0.0 || q >= 1.0) {
    return 0.0;
  }

  const double zmax = sqrt(1.0 - q * q);
  if (zmax == 0.0) {
    return 0.0;
  }

  double z_values[PROJECTED_KERNEL_SIMPSON_STEPS + 1];
  double integrand[PROJECTED_KERNEL_SIMPSON_STEPS + 1];
  const double dz = zmax / PROJECTED_KERNEL_SIMPSON_STEPS;

  for (int iz = 0; iz <= PROJECTED_KERNEL_SIMPSON_STEPS; iz++) {
    const double z = dz * iz;
    z_values[iz] = z;
    double radius = sqrt(z * z + q * q);

    /* The projected LOS integral is unchanged by the value at the single
     * endpoint z=zmax, but fixed-grid Simpson integration samples that point
     * explicitly. Nudge the final sample infinitesimally inside the compact
     * support so kernels with a hard edge, such as the uniform kernel, do not
     * pick up an artificial zero-valued endpoint. */
    if (iz == PROJECTED_KERNEL_SIMPSON_STEPS && radius >= 1.0) {
      radius = nextafter(1.0, 0.0);
    }

    integrand[iz] = func(radius);
  }

  return 2.0 * simps_1d(z_values, integrand,
                        PROJECTED_KERNEL_SIMPSON_STEPS + 1);
}

/**
 * @brief Build the projected LOS kernel lookup table.
 *
 * The output is a 1D table indexed by dimensionless projected separation. Each
 * entry stores the full LOS integral through the source kernel at that impact
 * parameter.
 *
 * @param kernel The output projected kernel table.
 * @param q_grid The projected-separation lookup grid.
 * @param qdim The number of projected-separation bins.
 * @param func The analytic kernel function to evaluate.
 */
static void build_projected_kernel(double *kernel, const double *q_grid,
                                   const int qdim, kernel_func func) {
#ifdef WITH_OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int iq = 0; iq < qdim; iq++) {
    kernel[iq] = integrate_projected_kernel(q_grid[iq], func);
  }
}

/**
 * @brief Build the truncated LOS kernel lookup table.
 *
 * The output table is stored with projected-separation index first and
 * LOS-coordinate index second. For each projected separation we walk once
 * along the LOS grid and accumulate the trapezoidal integral in place. This
 * avoids the large temporary Python arrays that dominated the old build path.
 *
 * @param kernel The output kernel table in row-major order.
 * @param q_grid The projected-separation grid.
 * @param z_grid The LOS truncation grid.
 * @param qdim The number of projected-separation bins.
 * @param zdim The number of LOS bins.
 * @param func The analytic kernel function to evaluate.
 */
static void build_truncated_los_kernel(double *kernel, const double *q_grid,
                                       const double *z_grid, const int qdim,
                                       const int zdim, kernel_func func) {
#ifdef WITH_OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int iq = 0; iq < qdim; iq++) {
    const double q = q_grid[iq];
    double cumulative = 0.0;
    double prev_value = 0.0;
    double prev_z = z_grid[0];

    /* The cumulative integral is defined to be zero at the first tabulated
     * LOS coordinate. We still evaluate the kernel there so the first
     * trapezoid uses the correct left-hand endpoint value. */
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
 * @brief Evaluate a named kernel on a 1D radius array.
 *
 * This exposes the analytic kernel implementations to Python so the public
 * ``uniform`` / ``cubic`` / etc. helpers can delegate to the same code used by
 * the C++ table builders instead of duplicating the formulas.
 *
 * @param self The module instance (unused).
 * @param args Python arguments containing a 1D radius array and kernel name.
 *
 * @return A 1D float64 NumPy array of kernel values.
 */
PyObject *evaluate_kernel(PyObject *self, PyObject *args) {

  (void)self;

  /* Parse the Python inputs: a contiguous radius array and the public kernel
   * name to evaluate. */
  PyArrayObject *np_radii;
  const char *kernel_name;

  if (!PyArg_ParseTuple(args, "O!s", &PyArray_Type, &np_radii, &kernel_name)) {
    return NULL;
  }

  /* The evaluator is intentionally simple: one radius axis in, one values
   * axis out. Reject higher-dimensional inputs here rather than silently
   * flattening them inside the extension. */
  if (PyArray_NDIM(np_radii) != 1) {
    PyErr_SetString(PyExc_ValueError, "radii must be a 1D array.");
    return NULL;
  }

  /* Extract a raw pointer to the float64 radius data. The shared property
   * helper also checks the dtype and contiguity requirements for us. */
  const double *radii = extract_data_double(np_radii, "radii");
  if (radii == NULL) {
    return NULL;
  }

  /* Resolve the analytic kernel once so the loop body only pays a direct
   * function-pointer call for each tabulated radius. */
  kernel_func func = get_kernel_function(kernel_name);
  if (func == NULL) {
    PyErr_SetString(PyExc_ValueError, "Kernel name not defined");
    return NULL;
  }

  /* Allocate the NumPy output array that will hold the evaluated kernel
   * values one-for-one with the input radii. */
  const int ndim = static_cast<int>(PyArray_DIM(np_radii, 0));
  npy_intp dims[1] = {ndim};
  PyArrayObject *np_values =
      (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
  double *values = static_cast<double *>(PyArray_DATA(np_values));

  /* The evaluation is embarrassingly parallel across radii, so when OpenMP is
   * available we can distribute the loop without any synchronisation. */
#ifdef WITH_OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < ndim; i++) {
    values[i] = func(radii[i]);
  }

  /* Transfer ownership of the newly created array back to Python. */
  return Py_BuildValue("N", np_values);
}

/**
 * @brief Python wrapper for projected LOS kernel-table construction.
 *
 * Python prepares the projected-separation lookup grid and this wrapper fills
 * the corresponding full LOS kernel table using the shared C++ kernel
 * evaluator.
 *
 * @param self The module instance (unused).
 * @param args Python arguments containing the q-grid and kernel name.
 *
 * @return A 1D float64 NumPy array with the projected LOS kernel values.
 */
PyObject *compute_projected_kernel(PyObject *self, PyObject *args) {

  (void)self;

  /* Parse the precomputed projected-separation grid and the public kernel
   * name used to select the analytic kernel shape. */
  PyArrayObject *np_q_grid;
  const char *kernel_name;

  if (!PyArg_ParseTuple(args, "O!s", &PyArray_Type, &np_q_grid,
                        &kernel_name)) {
    return NULL;
  }

  /* The projected kernel is defined on a single 1D q-axis. */
  if (PyArray_NDIM(np_q_grid) != 1) {
    PyErr_SetString(PyExc_ValueError, "q_grid must be a 1D array.");
    return NULL;
  }

  /* Extract the raw float64 axis data after validating dtype and contiguity. */
  const double *q_grid = extract_data_double(np_q_grid, "q_grid");
  if (q_grid == NULL) {
    return NULL;
  }

  /* Resolve the analytic kernel once before entering the numeric builder. */
  kernel_func func = get_kernel_function(kernel_name);
  if (func == NULL) {
    PyErr_SetString(PyExc_ValueError, "Kernel name not defined");
    return NULL;
  }

  /* Allocate the dense output table using the supplied q-grid size. */
  const int qdim = static_cast<int>(PyArray_DIM(np_q_grid, 0));
  npy_intp dims[1] = {qdim};
  PyArrayObject *np_kernel =
      (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
  double *kernel = static_cast<double *>(PyArray_DATA(np_kernel));

  /* Fill the output table in place using the fixed-rule projected integrator. */
  build_projected_kernel(kernel, q_grid, qdim, func);

  /* Transfer ownership of the completed NumPy array back to Python. */
  return Py_BuildValue("N", np_kernel);
}

/**
 * @brief Python wrapper for truncated LOS kernel-table construction.
 *
 * Python prepares the q and z lookup grids, while this wrapper selects the
 * analytic kernel and returns the cumulative LOS table as a dense NumPy array.
 *
 * @param self The module instance (unused).
 * @param args Python arguments containing the q-grid, z-grid, and kernel name.
 *
 * @return A 2D float64 NumPy array with shape ``(q_grid.size, z_grid.size)``.
 */
PyObject *compute_truncated_los_kernel(PyObject *self, PyObject *args) {

  (void)self;

  /* Parse the precomputed q-grid and z-grid supplied by Python along with the
   * public kernel name that selects the analytic kernel shape. */
  PyArrayObject *np_q_grid, *np_z_grid;
  const char *kernel_name;

  if (!PyArg_ParseTuple(args, "O!O!s", &PyArray_Type, &np_q_grid, &PyArray_Type,
                        &np_z_grid, &kernel_name)) {
    return NULL;
  }

  /* The truncated LOS table is defined on a rectilinear q-z grid, so reject
   * anything other than two 1D lookup axes here. */
  if (PyArray_NDIM(np_q_grid) != 1 || PyArray_NDIM(np_z_grid) != 1) {
    PyErr_SetString(PyExc_ValueError,
                    "q_grid and z_grid must both be 1D arrays.");
    return NULL;
  }

  /* Extract the raw float64 axis data after validating dtype and contiguity. */
  const double *q_grid = extract_data_double(np_q_grid, "q_grid");
  const double *z_grid = extract_data_double(np_z_grid, "z_grid");
  if (q_grid == NULL || z_grid == NULL) {
    return NULL;
  }

  /* Resolve the analytic kernel once before entering the numeric builder. */
  kernel_func func = get_kernel_function(kernel_name);
  if (func == NULL) {
    PyErr_SetString(PyExc_ValueError, "Kernel name not defined");
    return NULL;
  }

  /* Allocate the dense output table using the supplied q and z grid sizes. */
  const int qdim = static_cast<int>(PyArray_DIM(np_q_grid, 0));
  const int zdim = static_cast<int>(PyArray_DIM(np_z_grid, 0));

  npy_intp dims[2] = {qdim, zdim};
  PyArrayObject *np_kernel =
      (PyArrayObject *)PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
  double *kernel = static_cast<double *>(PyArray_DATA(np_kernel));

  /* Fill the output table in place using the shared C++ kernel evaluator. */
  build_truncated_los_kernel(kernel, q_grid, z_grid, qdim, zdim, func);

  /* Transfer ownership of the completed NumPy array back to Python. */
  return Py_BuildValue("N", np_kernel);
}

/* Expose the Python-callable entry points for this extension module. */
static PyMethodDef KernelMethods[] = {
    {"evaluate_kernel", (PyCFunction)evaluate_kernel, METH_VARARGS,
     "Evaluate a named kernel on a 1D array of radii."},
    {"compute_projected_kernel", (PyCFunction)compute_projected_kernel,
     METH_VARARGS, "Build the projected LOS kernel table."},
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

/* Create the module and initialise the NumPy C API before any of the wrapped
 * entry points are called from Python. */
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
