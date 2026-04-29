/******************************************************************************
 * Truncated LOS kernel table builder.
 *
 * This module builds the 2D truncated LOS kernel lookup table used when the
 * input particle lies inside the source kernel. The table stores the cumulative
 * foreground LOS integral as a function of projected separation and truncation
 * coordinate.
 *****************************************************************************/

#include "kernel_functions.h"
#include "kernels.h"

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
    const double radius = sqrt(prev_z * prev_z + q * q);
    if (radius < 1.0) {
      prev_value = func(radius);
    }

    for (int iz = 1; iz < zdim; iz++) {
      const double z = z_grid[iz];
      const double radius = sqrt(z * z + q * q);
      double value = 0.0;
      if (radius < 1.0) {
        value = func(radius);
      }

      /* Walk along the LOS grid once and accumulate the cumulative foreground
       * contribution directly into the output table. */
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

  PyArrayObject *np_q_grid, *np_z_grid;
  const char *kernel_name;

  if (!PyArg_ParseTuple(args, "O!O!s", &PyArray_Type, &np_q_grid, &PyArray_Type,
                        &np_z_grid, &kernel_name)) {
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

  /* Reuse the same analytic kernel dispatch as the projected table so the 1D
   * and truncated 2D builders stay numerically consistent. */
  kernel_func func = get_kernel_function(kernel_name);
  if (func == NULL) {
    PyErr_SetString(PyExc_ValueError, "Kernel name not defined");
    return NULL;
  }

const int qdim = static_cast<int>(PyArray_DIM(np_q_grid, 0));
  const int zdim = static_cast<int>(PyArray_DIM(np_z_grid, 0));
  if (qdim == 0) {
    PyErr_SetString(PyExc_ValueError,
                  "q_grid must contain at least one element.");
    return NULL;
  }
  if (zdim == 0) {
    PyErr_SetString(PyExc_ValueError,
                  "z_grid must contain at least one element.");
    return NULL;
  }

  npy_intp dims[2] = {qdim, zdim};
  PyArrayObject *np_kernel =
      (PyArrayObject *)PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
  if (np_kernel == NULL) {
    PyErr_NoMemory();
    return NULL;
  }
  double *kernel = static_cast<double *>(PyArray_DATA(np_kernel));

  build_truncated_los_kernel(kernel, q_grid, z_grid, qdim, zdim, func);

  return Py_BuildValue("N", np_kernel);
}
