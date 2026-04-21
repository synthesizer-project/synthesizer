/******************************************************************************
 * C extension helpers for LOS kernel-table construction.
 *****************************************************************************/

/* C headers. */
#include <math.h>

#ifdef WITH_OPENMP
#include <omp.h>
#endif

/* Python headers. */
#define PY_ARRAY_UNIQUE_SYMBOL SYNTHESIZER_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy_init.h"
#include <Python.h>

/* Local includes. */
#include "kernel_utils.h"
#include "property_funcs.h"
#include "timers.h"
#ifdef ATOMIC_TIMING
#include "timers_init.h"
#endif

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
 * For each tabulated `eta = h_i / h_j` we place the sampled input
 * kernel support at coordinates `(eta * qx, eta * qy, eta * qz)`, place the
 * source-particle centre at projected offset `q * (1 + eta)`, and shift the
 * upper LOS integration boundary by `u * (1 + eta)`. The truncated LOS table is
 * then queried once per sampled point and the result is averaged with the
 * sampled input-kernel weights.
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
 * @param trunc_q The projected-separation axis of the truncated table.
 * @param trunc_z The LOS-coordinate axis of the truncated table.
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
    const double weight_sum, const double *truncated_kernel,
    const double *trunc_q, const double *trunc_z, const int qdim,
    const int udim, const int etadim, const int nsample, const int trunc_qdim,
    const int trunc_zdim, const int nthreads) {

#ifdef WITH_OPENMP
#pragma omp parallel for num_threads(nthreads)                                 \
    schedule(static) if (nthreads > 1)
#endif
  for (int ieta = 0; ieta < etadim; ieta++) {

    /* Each eta slice is independent, making this the natural parallel axis. */
    const double eta = eta_grid[ieta];
    const double rsum = 1.0 + eta;

    for (int iq = 0; iq < qdim; iq++) {
      const double source_x = q_grid[iq] * rsum;

      for (int iu = 0; iu < udim; iu++) {
        const double z_shift = u_grid[iu] * rsum;
        double weighted_sum = 0.0;

        /* Average the truncated LOS source contribution across the support of
         * the input kernel using the precomputed sample points and weights. The
         * sample coordinates are dimensionless locations inside the unit input
         * support sphere, so multiplying them by eta maps them onto the input
         * kernel support appropriate for this eta slice. */
        for (int is = 0; is < nsample; is++) {
          const double input_x = eta * sample_x[is];
          const double input_y = eta * sample_y[is];
          const double input_z = eta * sample_z[is];

          const double dx = input_x - source_x;
          const double dy = input_y;
          const double projected_q = sqrt(dx * dx + dy * dy);
          const double z_upper = input_z + z_shift;

          /* The truncated LOS table already encodes the front / straddling /
           * behind behaviour for a source kernel at fixed projected radius, so
           * the overlap builder only needs to evaluate it at this sampled LOS
           * upper boundary and fold the result into the weighted average. */
          const double value = get_truncated_kernel_value(
              truncated_kernel, trunc_qdim, trunc_zdim, projected_q, z_upper);

          weighted_sum += sample_weights[is] * value;
        }

        overlap_kernel[(iq * udim + iu) * etadim + ieta] =
            weighted_sum / weight_sum;
      }
    }
  }
}

/**
 * @brief Build the smoothed-input overlap kernel table.
 *
 * Python prepares the table axes, sampled input-kernel points, and the
 * truncated LOS kernel table. This extension evaluates the expensive overlap
 * table in C++ and returns a NumPy array with shape `(qdim, udim, etadim)`.
 *
 * `Kernel` remains responsible for constructing the tabulated coordinate arrays
 * and helper tables, while this extension evaluates the dense numeric triple
 * loop.
 *
 * @param np_q_grid The overlap-table q-axis.
 * @param np_u_grid The overlap-table u-axis.
 * @param np_eta_grid The overlap-table eta-axis.
 * @param np_sample_x The sampled input-kernel x coordinates.
 * @param np_sample_y The sampled input-kernel y coordinates.
 * @param np_sample_z The sampled input-kernel z coordinates.
 * @param np_sample_weights The sampled input-kernel weights.
 * @param np_truncated_kernel The truncated LOS kernel table.
 * @param np_trunc_q The truncated-table q-axis.
 * @param np_trunc_z The truncated-table z-axis.
 * @param qdim The number of overlap-table q-grid entries.
 * @param udim The number of overlap-table u-grid entries.
 * @param etadim The number of overlap-table eta-grid entries.
 * @param nsample The number of sampled input-kernel points.
 * @param trunc_qdim The number of truncated-table q-grid entries.
 * @param trunc_zdim The number of truncated-table z-grid entries.
 * @param nthreads The number of OpenMP threads requested.
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

  npy_intp dims[3] = {qdim, udim, etadim};
  PyArrayObject *np_overlap_kernel =
      (PyArrayObject *)PyArray_ZEROS(3, dims, NPY_DOUBLE, 0);
  double *overlap_kernel =
      static_cast<double *>(PyArray_DATA(np_overlap_kernel));

  /* The sampled input-kernel weights are normalised after accumulation so the
   * returned table stores a true kernel average rather than an unnormalised
   * weighted sum. */
  double weight_sum = 0.0;
  for (int is = 0; is < nsample; is++) {
    weight_sum += sample_weights[is];
  }

  tic("Evaluating overlap kernel table");
  build_overlap_kernel_table(overlap_kernel, q_grid, u_grid, eta_grid, sample_x,
                             sample_y, sample_z, sample_weights, weight_sum,
                             truncated_kernel, trunc_q, trunc_z, qdim, udim,
                             etadim, nsample, trunc_qdim, trunc_zdim, nthreads);
  toc("Evaluating overlap kernel table");

  return Py_BuildValue("N", np_overlap_kernel);
}

/* Below is all the gubbins needed to make the module importable in Python. */
static PyMethodDef KernelMethods[] = {
    {"compute_overlap_kernel", (PyCFunction)compute_overlap_kernel,
     METH_VARARGS,
     "Method for building the smoothed-input overlap kernel table."},
    {NULL, NULL, 0, NULL}};

/* Make this importable. */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "kernel",                              /* m_name */
    "A module to build LOS kernel tables", /* m_doc */
    -1,                                    /* m_size */
    KernelMethods,                         /* m_methods */
    NULL,                                  /* m_reload */
    NULL,                                  /* m_traverse */
    NULL,                                  /* m_clear */
    NULL,                                  /* m_free */
};

PyMODINIT_FUNC PyInit_kernel(void) {
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
