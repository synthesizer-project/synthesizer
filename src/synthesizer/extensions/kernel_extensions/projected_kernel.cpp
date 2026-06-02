/******************************************************************************
 * Projected LOS kernel table builder.
 *
 * This module builds the 1D projected LOS kernel lookup table used by the
 * point-particle LOS path. Each entry stores the full LOS integral through
 * the source kernel at a given dimensionless impact parameter.
 *****************************************************************************/

#include "kernels.h"
#include "kernel_functions.h"

#include <vector>

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
 * @tparam Real The floating-point type (float or double).
 * @param q The dimensionless impact parameter.
 * @param func The analytic kernel function to evaluate.
 * @param nsteps The number of trapezoidal integration steps.
 *
 * @return The projected LOS kernel value at ``q``.
 */
template <typename Real>
static inline Real integrate_projected_kernel(const Real q,
                                              kernel_func<Real> func,
                                              const int nsteps) {
  if (q < static_cast<Real>(0.0) || q >= static_cast<Real>(1.0)) {
    return static_cast<Real>(0.0);
  }

  const Real zmax = sqrt(static_cast<Real>(1.0) - q * q);
  if (zmax == static_cast<Real>(0.0)) {
    return static_cast<Real>(0.0);
  }

  const Real dz = zmax / static_cast<Real>(nsteps);

  std::vector<Real> z_values(nsteps + 1);
  std::vector<Real> integrand(nsteps + 1);

  for (int iz = 0; iz <= nsteps; iz++) {
    const Real z = dz * static_cast<Real>(iz);
    z_values[iz] = z;
    Real radius = sqrt(z * z + q * q);

    /* Nudge the final sample infinitesimally inside the compact support so
     * kernels with a hard edge do not pick up an artificial zero-valued
     * endpoint. */
    if (iz == nsteps && radius >= static_cast<Real>(1.0)) {
      radius = nextafter(static_cast<Real>(1.0), static_cast<Real>(0.0));
    }

    integrand[iz] = func(radius);
  }

  return static_cast<Real>(2.0) *
         trapz_1d<Real>(z_values.data(), integrand.data(), nsteps + 1);
}

/**
 * @brief Build the projected LOS kernel lookup table.
 *
 * The output is a 1D table indexed by dimensionless projected separation. Each
 * entry stores the full LOS integral through the source kernel at that impact
 * parameter.
 *
 * @tparam Real The floating-point type (float or double).
 * @param kernel The output projected kernel table.
 * @param q_grid The projected-separation lookup grid.
 * @param qdim The number of projected-separation bins.
 * @param func The analytic kernel function to evaluate.
 * @param nsteps The number of trapezoidal integration steps per q bin.
 */
template <typename Real>
static void build_projected_kernel(Real *kernel, const Real *q_grid,
                                   const int qdim, kernel_func<Real> func,
                                   const int nsteps) {
#ifdef WITH_OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int iq = 0; iq < qdim; iq++) {
    /* Each q bin is independent, so we can tabulate the projected kernel with
     * a simple embarrassingly-parallel loop. */
    kernel[iq] = integrate_projected_kernel<Real>(q_grid[iq], func, nsteps);
  }
}

/**
 * @brief Templated implementation of projected LOS kernel table construction.
 *
 * @tparam Real The floating-point type (float or double).
 */
template <typename Real>
static PyObject *compute_projected_kernel_impl(PyObject *self,
                                               PyArrayObject *np_q_grid,
                                               const char *kernel_name,
                                               int nsteps) {
  (void)self;

  const Real *q_grid = extract_data<Real>(np_q_grid, "q_grid");
  if (q_grid == NULL) {
    return NULL;
  }

  kernel_func<Real> func = get_kernel_function<Real>(kernel_name);
  if (func == NULL) {
    PyErr_SetString(PyExc_ValueError, "Kernel name not defined");
    return NULL;
  }

  const int qdim = static_cast<int>(PyArray_DIM(np_q_grid, 0));
  const int typenum =
      std::is_same_v<Real, float> ? NPY_FLOAT32 : NPY_FLOAT64;
  npy_intp dims[1] = {qdim};
  PyArrayObject *np_kernel =
      (PyArrayObject *)PyArray_ZEROS(1, dims, typenum, 0);
  if (np_kernel == NULL) {
    PyErr_NoMemory();
    return NULL;
  }
  Real *kernel = static_cast<Real *>(PyArray_DATA(np_kernel));

  build_projected_kernel<Real>(kernel, q_grid, qdim, func, nsteps);

  return Py_BuildValue("N", np_kernel);
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
 * @return A 1D float64 or float32 NumPy array with the projected LOS kernel
 *         values, matching the input precision.
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

  if (nsteps <= 0) {
    PyErr_SetString(PyExc_ValueError,
                  "nsteps must be a positive integer.");
    return NULL;
  }

  if (PyArray_NDIM(np_q_grid) != 1) {
    PyErr_SetString(PyExc_ValueError, "q_grid must be a 1D array.");
    return NULL;
  }

  /* Validate the dtype and dispatch to the correct instantiation. */
  const int input_typenum = PyArray_TYPE(np_q_grid);
  if (input_typenum == NPY_FLOAT32) {
    return compute_projected_kernel_impl<float>(self, np_q_grid, kernel_name,
                                                nsteps);
  }
  if (input_typenum == NPY_FLOAT64) {
    return compute_projected_kernel_impl<double>(self, np_q_grid, kernel_name,
                                                 nsteps);
  }

  PyErr_SetString(PyExc_TypeError,
                  "q_grid must be float32 or float64.");
  return NULL;
}
