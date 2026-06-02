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
 * @tparam Real The floating-point type (float or double).
 * @param kernel The output kernel table in row-major order.
 * @param q_grid The projected-separation grid.
 * @param z_grid The LOS truncation grid.
 * @param qdim The number of projected-separation bins.
 * @param zdim The number of LOS bins.
 * @param func The analytic kernel function to evaluate.
 */
template <typename Real>
static void build_truncated_los_kernel(Real *kernel, const Real *q_grid,
                                       const Real *z_grid, const int qdim,
                                       const int zdim, kernel_func<Real> func) {
#ifdef WITH_OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int iq = 0; iq < qdim; iq++) {
    const Real q = q_grid[iq];
    Real cumulative = static_cast<Real>(0.0);
    Real prev_value = static_cast<Real>(0.0);
    Real prev_z = z_grid[0];

    /* The cumulative integral is defined to be zero at the first tabulated
     * LOS coordinate. We still evaluate the kernel there so the first
     * trapezoid uses the correct left-hand endpoint value. */
    const Real radius = sqrt(prev_z * prev_z + q * q);
    if (radius < static_cast<Real>(1.0)) {
      prev_value = func(radius);
    }

    for (int iz = 1; iz < zdim; iz++) {
      const Real z = z_grid[iz];
      const Real radius = sqrt(z * z + q * q);
      Real value = static_cast<Real>(0.0);
      if (radius < static_cast<Real>(1.0)) {
        value = func(radius);
      }

      /* Walk along the LOS grid once and accumulate the cumulative foreground
       * contribution directly into the output table. */
      cumulative += static_cast<Real>(0.5) * (prev_value + value) * (z - prev_z);
      kernel[iq * zdim + iz] = cumulative;

      prev_value = value;
      prev_z = z;
    }
  }
}

/**
 * @brief Templated implementation of truncated LOS kernel table construction.
 *
 * @tparam Real The floating-point type (float or double).
 */
template <typename Real>
static PyObject *compute_truncated_los_kernel_impl(PyObject *self,
                                                   PyArrayObject *np_q_grid,
                                                   PyArrayObject *np_z_grid,
                                                   const char *kernel_name) {
  (void)self;

  const Real *q_grid = extract_data<Real>(np_q_grid, "q_grid");
  const Real *z_grid = extract_data<Real>(np_z_grid, "z_grid");
  if (q_grid == NULL || z_grid == NULL) {
    return NULL;
  }

  /* Reuse the same analytic kernel dispatch as the projected table so the 1D
   * and truncated 2D builders stay numerically consistent. */
  kernel_func<Real> func = get_kernel_function<Real>(kernel_name);
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

  const int typenum =
      std::is_same_v<Real, float> ? NPY_FLOAT32 : NPY_FLOAT64;
  npy_intp dims[2] = {qdim, zdim};
  PyArrayObject *np_kernel =
      (PyArrayObject *)PyArray_ZEROS(2, dims, typenum, 0);
  if (np_kernel == NULL) {
    PyErr_NoMemory();
    return NULL;
  }
  Real *kernel = static_cast<Real *>(PyArray_DATA(np_kernel));

  build_truncated_los_kernel<Real>(kernel, q_grid, z_grid, qdim, zdim, func);

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
 * @return A 2D float64 or float32 NumPy array with shape
 *         ``(q_grid.size, z_grid.size)``, matching the input precision.
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

  /* Validate the dtype and dispatch to the correct instantiation. */
  const int input_typenum = PyArray_TYPE(np_q_grid);
  if (input_typenum != PyArray_TYPE(np_z_grid)) {
    PyErr_SetString(PyExc_TypeError,
                    "q_grid and z_grid must have the same dtype.");
    return NULL;
  }

  if (input_typenum == NPY_FLOAT32) {
    return compute_truncated_los_kernel_impl<float>(self, np_q_grid, np_z_grid,
                                                    kernel_name);
  }
  if (input_typenum == NPY_FLOAT64) {
    return compute_truncated_los_kernel_impl<double>(self, np_q_grid, np_z_grid,
                                                     kernel_name);
  }

  PyErr_SetString(PyExc_TypeError,
                  "q_grid and z_grid must be float32 or float64.");
  return NULL;
}
