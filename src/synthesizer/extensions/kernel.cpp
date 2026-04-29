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
#include "kernel_utils.h"
#include "property_funcs.h"
#include "timers.h"
#ifdef ATOMIC_TIMING
#include "timers_init.h"
#endif

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
    return 27.0 *
           (6.4457752 * r * r * r * r * (1.0 - r) - 1.4323945 * r * r +
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
 * @brief Evaluate the overlap kernel table for smoothed LOS calculations.
 *
 * This helper builds the `G(q, u, eta)` table used by the smoothed-input LOS
 * path. The surrounding Python code prepares the coordinate grids, sampled
 * input-kernel points, and truncated LOS kernel table. The dense numeric loop
 * over `(q, u, eta)` is then executed here in C++.
 *
 * The work naturally factorises over `eta` slices, so the OpenMP path
 * parallelises that outer loop when multiple threads are requested. This is a
 * good fit for the table layout because each `(q, u)` plane at fixed `eta`
 * writes to a disjoint contiguous slab of the output array and therefore needs
 * no synchronisation between threads.
 *
 * For each tabulated `eta = h_i / h_j` we place the sampled input kernel
 * support at coordinates `(eta * qx, eta * qy, eta * qz)`, place the source
 * particle centre at projected offset `q * (1 + eta)`, and shift the upper LOS
 * integration boundary by `u * (1 + eta)`. The truncated LOS table is then
 * queried once per sampled point and the result is averaged with the sampled
 * input-kernel weights.
 *
 * The sampled points are expressed in units of the input smoothing length,
 * whereas the truncated LOS table is expressed in units of the source
 * smoothing length. Scaling the sampled coordinates by `eta = h_i / h_j`
 * performs that change of units before the truncated lookup is evaluated.
 *
 * @param overlap_kernel The output overlap table stored as `(q, u, eta)` in
 *        row-major order.
 * @param q_grid The overlap-table q-axis.
 * @param u_grid The overlap-table u-axis.
 * @param eta_grid The overlap-table eta-axis.
 * @param sample_x The x coordinates of the sampled input-kernel points.
 * @param sample_y The y coordinates of the sampled input-kernel points.
 * @param sample_z The z coordinates of the sampled input-kernel points.
 * @param sample_weights The radial-kernel weights for the sampled points.
 * @param weight_sum The total sampled input-kernel weight.
 * @param truncated_kernel The truncated LOS kernel table.
 * @param qdim The number of overlap-table q-grid entries.
 * @param udim The number of overlap-table u-grid entries.
 * @param etadim The number of overlap-table eta-grid entries.
 * @param nsample The number of sampled input-kernel points.
 * @param trunc_qdim The number of truncated-table q-grid entries.
 * @param trunc_zdim The number of truncated-table z-grid entries.
 * @param nthreads The number of OpenMP threads requested.
 */
static void build_overlap_kernel_table(
    double *overlap_kernel, const double *q_grid, const double *u_grid,
    const double *eta_grid, const double *sample_x, const double *sample_y,
    const double *sample_z, const double *sample_weights,
    const double weight_sum, const double *truncated_kernel, const int qdim,
    const int udim, const int etadim, const int nsample, const int trunc_qdim,
    const int trunc_zdim, const int nthreads) {

#ifdef WITH_OPENMP
#pragma omp parallel for num_threads(nthreads) schedule(static) if (nthreads > 1)
#endif
  for (int ieta = 0; ieta < etadim; ieta++) {
    const double eta = eta_grid[ieta];

    /* In source-kernel units the input support radius is `eta` and the source
     * support radius is `1`, so the sum of support radii is `1 + eta`. */
    const double support_sum = 1.0 + eta;

    for (int iq = 0; iq < qdim; iq++) {
      /* By construction the source centre lies on the x axis, so the q-grid
       * only needs to move the source in x. */
      const double source_x = q_grid[iq] * support_sum;

      for (int iu = 0; iu < udim; iu++) {
        /* The u-grid shifts the input sample points relative to the source
         * centre along the LOS, again in units of the summed support radii. */
        const double z_shift = u_grid[iu] * support_sum;
        double weighted_sum = 0.0;

        for (int is = 0; is < nsample; is++) {
          /* Rescale the sampled input-kernel point from input-kernel units to
           * source-kernel units before comparing it to the source geometry. */
          const double input_x = eta * sample_x[is];
          const double input_y = eta * sample_y[is];
          const double input_z = eta * sample_z[is];

          /* The source centre is placed on the x axis, so the y separation is
           * just the sampled input y coordinate in this frame. */
          const double dx = input_x - source_x;
          const double dy = input_y;
          const double projected_q = sqrt(dx * dx + dy * dy);

          /* Shift the sampled input point along the LOS and use that as the
           * truncation coordinate into the source-kernel cumulative table. */
          const double z_trunc = input_z + z_shift;

          const double value = get_truncated_kernel_value(
              truncated_kernel, trunc_qdim, trunc_zdim, projected_q, z_trunc);

          weighted_sum += sample_weights[is] * value;
        }

        /* Normalise by the sampled input-kernel weight so the overlap table is
         * a kernel-weighted average rather than an unnormalised sum. */
        overlap_kernel[(iq * udim + iu) * etadim + ieta] =
            weighted_sum / weight_sum;
      }
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

  PyArrayObject *np_radii;
  const char *kernel_name;

  if (!PyArg_ParseTuple(args, "O!s", &PyArray_Type, &np_radii, &kernel_name)) {
    return NULL;
  }

  if (PyArray_NDIM(np_radii) != 1) {
    PyErr_SetString(PyExc_ValueError, "radii must be a 1D array.");
    return NULL;
  }
  if (PyArray_TYPE(np_radii) != NPY_DOUBLE) {
    PyErr_SetString(PyExc_TypeError, "radii must be float64.");
    return NULL;
  }

  /* The helper expects a contiguous float64 buffer and returns a borrowed data
   * pointer owned by the NumPy array object. */
  const double *radii = extract_data_double(np_radii, "radii");
  if (radii == NULL) {
    return NULL;
  }

  kernel_func func = get_kernel_function(kernel_name);
  if (func == NULL) {
    PyErr_SetString(PyExc_ValueError, "Kernel name not defined");
    return NULL;
  }

  /* Allocate the result array on the Python side and fill it in place from
   * the shared analytic kernel implementation. */
  const int ndim = static_cast<int>(PyArray_DIM(np_radii, 0));
  npy_intp dims[1] = {ndim};
  PyArrayObject *np_values =
      (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
  if (np_values == NULL) {
    PyErr_NoMemory();
    return NULL;
  }
  double *values = static_cast<double *>(PyArray_DATA(np_values));

#ifdef WITH_OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < ndim; i++) {
    values[i] = func(radii[i]);
  }

  return Py_BuildValue("N", np_values);
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

/**
 * @brief Build the smoothed-input overlap kernel table.
 *
 * Python prepares the table axes, sampled input-kernel points, and the
 * truncated LOS kernel table. This extension evaluates the expensive overlap
 * table in C++ and returns a NumPy array with shape `(qdim, udim, etadim)`.
 *
 * @param self The module instance (unused).
 * @param args Python arguments containing the overlap grids, sampled points,
 *        truncated LOS table, and explicit dimensions.
 *
 * @return A NumPy array containing the overlap kernel table.
 */
PyObject *compute_overlap_kernel(PyObject *self, PyObject *args) {

  (void)self;

  int qdim, udim, etadim, nsample, trunc_qdim, trunc_zdim, nthreads;
  PyArrayObject *np_q_grid, *np_u_grid, *np_eta_grid, *np_sample_x,
      *np_sample_y, *np_sample_z, *np_sample_weights, *np_truncated_kernel,
      *np_trunc_q, *np_trunc_z;

  if (!PyArg_ParseTuple(args, "OOOOOOOOOOiiiiiii", &np_q_grid, &np_u_grid,
                        &np_eta_grid, &np_sample_x, &np_sample_y, &np_sample_z,
                        &np_sample_weights, &np_truncated_kernel, &np_trunc_q,
                        &np_trunc_z, &qdim, &udim, &etadim, &nsample,
                        &trunc_qdim, &trunc_zdim, &nthreads)) {
    return NULL;
  }

  const double *q_grid = extract_data_double(np_q_grid, "q_grid");
  const double *u_grid = extract_data_double(np_u_grid, "u_grid");
  const double *eta_grid = extract_data_double(np_eta_grid, "eta_grid");
  const double *sample_x = extract_data_double(np_sample_x, "sample_x");
  const double *sample_y = extract_data_double(np_sample_y, "sample_y");
  const double *sample_z = extract_data_double(np_sample_z, "sample_z");
  const double *sample_weights =
      extract_data_double(np_sample_weights, "sample_weights");
  const double *truncated_kernel =
      extract_data_double(np_truncated_kernel, "truncated_kernel");
  const double *trunc_q = extract_data_double(np_trunc_q, "trunc_q");
  const double *trunc_z = extract_data_double(np_trunc_z, "trunc_z");

  if (q_grid == NULL || u_grid == NULL || eta_grid == NULL ||
      sample_x == NULL || sample_y == NULL || sample_z == NULL ||
      sample_weights == NULL || truncated_kernel == NULL || trunc_q == NULL ||
      trunc_z == NULL) {
    return NULL;
  }

  /* The truncated q/z coordinate arrays are currently passed through from
   * Python for interface symmetry and possible future validation, but the
   * overlap builder only needs their dimensions because interpolation into the
   * truncated table uses the known normalised support range directly. */
  (void)trunc_q;
  (void)trunc_z;

  npy_intp dims[3] = {qdim, udim, etadim};
  PyArrayObject *np_overlap_kernel =
      (PyArrayObject *)PyArray_ZEROS(3, dims, NPY_DOUBLE, 0);
  if (np_overlap_kernel == NULL) {
    PyErr_NoMemory();
    return NULL;
  }
  double *overlap_kernel =
      static_cast<double *>(PyArray_DATA(np_overlap_kernel));

  /* Precompute the total sampled input-kernel weight once so each table cell
   * can convert its weighted sum into a proper kernel-weighted average. */
  double weight_sum = 0.0;
  for (int is = 0; is < nsample; is++) {
    weight_sum += sample_weights[is];
  }

  tic("Evaluating overlap kernel table");
  build_overlap_kernel_table(overlap_kernel, q_grid, u_grid, eta_grid, sample_x,
                             sample_y, sample_z, sample_weights, weight_sum,
                             truncated_kernel, qdim, udim, etadim, nsample,
                             trunc_qdim, trunc_zdim, nthreads);
  toc("Evaluating overlap kernel table");

  return Py_BuildValue("N", np_overlap_kernel);
}

/* Expose the Python-callable entry points for this extension module. */
static PyMethodDef KernelMethods[] = {
    {"evaluate_kernel", (PyCFunction)evaluate_kernel, METH_VARARGS,
     "Evaluate a named kernel on a 1D array of radii."},
    {"compute_projected_kernel", (PyCFunction)compute_projected_kernel,
     METH_VARARGS, "Build the projected LOS kernel table."},
    {"compute_truncated_los_kernel", (PyCFunction)compute_truncated_los_kernel,
     METH_VARARGS, "Build the truncated LOS kernel table."},
    {"compute_overlap_kernel", (PyCFunction)compute_overlap_kernel,
     METH_VARARGS,
     "Method for building the smoothed-input overlap kernel table."},
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
#ifdef ATOMIC_TIMING
  if (import_toc_capsule() < 0) {
    Py_DECREF(m);
    return NULL;
  }
#endif
  return m;
}
