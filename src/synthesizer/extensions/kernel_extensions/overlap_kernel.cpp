/******************************************************************************
 * Smoothed-input overlap kernel table builder.
 *
 * This module builds the 3D overlap kernel table G(q, u, eta) used by the
 * smoothed-input LOS path. The table encodes the line-of-sight overlap between
 * an input (smoothing-length h_i) and source (smoothing-length h_j) kernel
 * as a function of projected separation, LOS offset, and smoothing-length ratio.
 *****************************************************************************/

#include "kernels.h"
#include "kernel_functions.h"

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
 * @brief Python wrapper for smoothed-input overlap kernel table construction.
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