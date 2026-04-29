/******************************************************************************
 * Projected LOS kernel table builder.
 *
 * This module builds the 1D projected LOS kernel lookup table used by the
 * point-particle LOS path. Each entry stores the full LOS integral through
 * the source kernel at a given dimensionless impact parameter.
 *****************************************************************************/

#include "kernels.h"
#include "kernel_functions.h"

/**
 * @brief Integrate the projected LOS kernel at one impact parameter.
 *
 * This evaluates
 *
 *     2 * integral_0^sqrt(1 - q^2) W(sqrt(z^2 + q^2)) dz
 *
 * using the composite trapezoidal rule on a fixed tabulated LOS grid. This
 * matches the integration method used by the truncated LOS table builder and
 * produces a stable value that can be checked against the Python reference in
 * the unit tests.
 *
 * @param q The dimensionless impact parameter.
 * @param func The analytic kernel function to evaluate.
 * @param nsteps The number of trapezoidal integration steps.
 *
 * @return The projected LOS kernel value at ``q``.
 */
static inline double integrate_projected_kernel(const double q,
                                                kernel_func func,
                                                const int nsteps) {
  if (q < 0.0 || q >= 1.0) {
    return 0.0;
  }

  const double zmax = sqrt(1.0 - q * q);
  if (zmax == 0.0) {
    return 0.0;
  }

  const double dz = zmax / nsteps;

  double z_values[nsteps + 1];
  double integrand[nsteps + 1];

  for (int iz = 0; iz <= nsteps; iz++) {
    const double z = dz * iz;
    z_values[iz] = z;
    double radius = sqrt(z * z + q * q);

    /* Nudge the final sample infinitesimally inside the compact support so
     * kernels with a hard edge do not pick up an artificial zero-valued
     * endpoint. */
    if (iz == nsteps && radius >= 1.0) {
      radius = nextafter(1.0, 0.0);
    }

    integrand[iz] = func(radius);
  }

  return 2.0 * trapz_1d(z_values, integrand, nsteps + 1);
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
 * @param nsteps The number of trapezoidal integration steps per q bin.
 */
static void build_projected_kernel(double *kernel, const double *q_grid,
                                   const int qdim, kernel_func func,
                                   const int nsteps) {
#ifdef WITH_OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int iq = 0; iq < qdim; iq++) {
    /* Each q bin is independent, so we can tabulate the projected kernel with
     * a simple embarrassingly-parallel loop. */
    kernel[iq] = integrate_projected_kernel(q_grid[iq], func, nsteps);
  }
}

/**
 * @brief Python wrapper for projected LOS kernel-table construction.
 *
 * Python prepares the projected-separation lookup grid and integration
 * resolution, while this wrapper selects the analytic kernel and returns the
 * filled projected table as a dense NumPy array.
 *
 * @param self The module instance (unused).
 * @param args Python arguments containing the q-grid, kernel name, and the
 *        number of integration steps.
 *
 * @return A 1D float64 NumPy array with the projected LOS kernel values.
 */
PyObject *compute_projected_kernel(PyObject *self, PyObject *args) {

  (void)self;

  PyArrayObject *np_q_grid;
  const char *kernel_name;
  int nsteps;

  if (!PyArg_ParseTuple(args, "O!si", &PyArray_Type, &np_q_grid,
                        &kernel_name, &nsteps)) {
    return NULL;
  }

  if (PyArray_NDIM(np_q_grid) != 1) {
    PyErr_SetString(PyExc_ValueError, "q_grid must be a 1D array.");
    return NULL;
  }

  /* The q-grid is prepared in Python so the public Kernel class controls the
   * tabulation resolution, while this wrapper only handles numeric filling. */
  const double *q_grid = extract_data_double(np_q_grid, "q_grid");
  if (q_grid == NULL) {
    return NULL;
  }

  kernel_func func = get_kernel_function(kernel_name);
  if (func == NULL) {
    PyErr_SetString(PyExc_ValueError, "Kernel name not defined");
    return NULL;
  }

  const int qdim = static_cast<int>(PyArray_DIM(np_q_grid, 0));
  npy_intp dims[1] = {qdim};
  PyArrayObject *np_kernel =
      (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
  if (np_kernel == NULL) {
    PyErr_NoMemory();
    return NULL;
  }
  double *kernel = static_cast<double *>(PyArray_DATA(np_kernel));

  build_projected_kernel(kernel, q_grid, qdim, func, nsteps);

  return Py_BuildValue("N", np_kernel);
}
