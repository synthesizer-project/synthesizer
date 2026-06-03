/******************************************************************************
 * Smoothed-input overlap kernel table builder.
 *
 * This module builds the 3D overlap kernel table G(q, u, eta) used by the
 * smoothed-input LOS path. The table encodes the line-of-sight overlap between
 * an input (smoothing-length h_i) and source (smoothing-length h_j) kernel
 * as a function of projected separation, LOS offset, and smoothing-length
 * ratio.
 *****************************************************************************/

#include "kernel_functions.h"
#include "kernels.h"

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
 * @tparam Real The floating-point type.
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
template <typename Real>
static void build_overlap_kernel_table(
    Real *overlap_kernel, const Real *q_grid, const Real *u_grid,
    const Real *eta_grid, const Real *sample_x, const Real *sample_y,
    const Real *sample_z, const Real *sample_weights, const Real weight_sum,
    const Real *truncated_kernel, const int qdim, const int udim,
    const int etadim, const int nsample, const int trunc_qdim,
    const int trunc_zdim, const int nthreads) {

#ifdef WITH_OPENMP
#pragma omp parallel for num_threads(nthreads) \
    schedule(static) if (nthreads > 1)
#endif
  for (int ieta = 0; ieta < etadim; ieta++) {
    const Real eta = eta_grid[ieta];

    /* In source-kernel units the input support radius is `eta` and the source
     * support radius is `1`, so the sum of support radii is `1 + eta`. */
    const Real support_sum = static_cast<Real>(1.0) + eta;

    for (int iq = 0; iq < qdim; iq++) {
      /* By construction the source centre lies on the x axis, so the q-grid
       * only needs to move the source in x. */
      const Real source_x = q_grid[iq] * support_sum;

      for (int iu = 0; iu < udim; iu++) {
        /* The u-grid shifts the input sample points relative to the source
         * centre along the LOS, again in units of the summed support radii. */
        const Real z_shift = u_grid[iu] * support_sum;
        Real weighted_sum = static_cast<Real>(0.0);

        for (int is = 0; is < nsample; is++) {
          /* Rescale the sampled input-kernel point from input-kernel units to
           * source-kernel units before comparing it to the source geometry. */
          const Real input_x = eta * sample_x[is];
          const Real input_y = eta * sample_y[is];
          const Real input_z = eta * sample_z[is];

          /* The source centre is placed on the x axis, so the y separation is
           * just the sampled input y coordinate in this frame. */
          const Real dx = input_x - source_x;
          const Real dy = input_y;
          const Real projected_q = sqrt(dx * dx + dy * dy);

          /* Shift the sampled input point along the LOS and use that as the
           * truncation coordinate into the source-kernel cumulative table. */
          const Real z_trunc = input_z + z_shift;

          const Real value = get_truncated_kernel_value<Real>(
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
 * @brief Templated implementation of overlap kernel table construction.
 *
 * @tparam Real The floating-point type (float or double).
 */
template <typename Real>
static PyObject *compute_overlap_kernel_impl(PyObject *self, PyObject *args) {
  (void)self;

  int qdim, udim, etadim, nsample, trunc_qdim, trunc_zdim, nthreads;
  PyArrayObject *np_q_grid, *np_u_grid, *np_eta_grid, *np_sample_x,
      *np_sample_y, *np_sample_z, *np_sample_weights, *np_truncated_kernel,
      *np_trunc_q, *np_trunc_z;

  if (!PyArg_ParseTuple(
          args, "O!O!O!O!O!O!O!O!O!O!iiiiiii", &PyArray_Type, &np_q_grid,
          &PyArray_Type, &np_u_grid, &PyArray_Type, &np_eta_grid,
          &PyArray_Type, &np_sample_x, &PyArray_Type, &np_sample_y,
          &PyArray_Type, &np_sample_z, &PyArray_Type, &np_sample_weights,
          &PyArray_Type, &np_truncated_kernel, &PyArray_Type, &np_trunc_q,
          &PyArray_Type, &np_trunc_z, &qdim, &udim, &etadim, &nsample,
          &trunc_qdim, &trunc_zdim, &nthreads)) {
    return NULL;
  }

  if (qdim <= 0 || udim <= 0 || etadim <= 0 || nsample <= 0 ||
      trunc_qdim <= 0 || trunc_zdim <= 0) {
    PyErr_SetString(PyExc_ValueError,
                    "All overlap-kernel dimensions must be positive.");
    return NULL;
  }

  if (PyArray_NDIM(np_q_grid) != 1 || PyArray_NDIM(np_u_grid) != 1 ||
      PyArray_NDIM(np_eta_grid) != 1 || PyArray_NDIM(np_sample_x) != 1 ||
      PyArray_NDIM(np_sample_y) != 1 || PyArray_NDIM(np_sample_z) != 1 ||
      PyArray_NDIM(np_sample_weights) != 1 || PyArray_NDIM(np_trunc_q) != 1 ||
      PyArray_NDIM(np_trunc_z) != 1) {
    PyErr_SetString(
        PyExc_ValueError,
        "All overlap-kernel coordinate and sample arrays must be 1D.");
    return NULL;
  }
  if (PyArray_NDIM(np_truncated_kernel) != 2) {
    PyErr_SetString(PyExc_ValueError, "truncated_kernel must be a 2D array.");
    return NULL;
  }

  const Real *q_grid = extract_data<Real>(np_q_grid, "q_grid");
  const Real *u_grid = extract_data<Real>(np_u_grid, "u_grid");
  const Real *eta_grid = extract_data<Real>(np_eta_grid, "eta_grid");
  const Real *sample_x = extract_data<Real>(np_sample_x, "sample_x");
  const Real *sample_y = extract_data<Real>(np_sample_y, "sample_y");
  const Real *sample_z = extract_data<Real>(np_sample_z, "sample_z");
  const Real *sample_weights =
      extract_data<Real>(np_sample_weights, "sample_weights");
  const Real *truncated_kernel =
      extract_data<Real>(np_truncated_kernel, "truncated_kernel");
  const Real *trunc_q = extract_data<Real>(np_trunc_q, "trunc_q");
  const Real *trunc_z = extract_data<Real>(np_trunc_z, "trunc_z");

  if (q_grid == NULL || u_grid == NULL || eta_grid == NULL ||
      sample_x == NULL || sample_y == NULL || sample_z == NULL ||
      sample_weights == NULL || truncated_kernel == NULL || trunc_q == NULL ||
      trunc_z == NULL) {
    return NULL;
  }

  /* Validate dimensions match the declared sizes. */
  if (static_cast<int>(PyArray_DIM(np_q_grid, 0)) != qdim) {
    PyErr_SetString(PyExc_ValueError, "q_grid dimension does not match qdim");
    return NULL;
  }
  if (static_cast<int>(PyArray_DIM(np_u_grid, 0)) != udim) {
    PyErr_SetString(PyExc_ValueError, "u_grid dimension does not match udim");
    return NULL;
  }
  if (static_cast<int>(PyArray_DIM(np_eta_grid, 0)) != etadim) {
    PyErr_SetString(PyExc_ValueError,
                    "eta_grid dimension does not match etadim");
    return NULL;
  }
  if (static_cast<int>(PyArray_DIM(np_sample_x, 0)) != nsample) {
    PyErr_SetString(PyExc_ValueError,
                    "sample_x dimension does not match nsample");
    return NULL;
  }
  if (static_cast<int>(PyArray_DIM(np_sample_y, 0)) != nsample) {
    PyErr_SetString(PyExc_ValueError,
                    "sample_y dimension does not match nsample");
    return NULL;
  }
  if (static_cast<int>(PyArray_DIM(np_sample_z, 0)) != nsample) {
    PyErr_SetString(PyExc_ValueError,
                    "sample_z dimension does not match nsample");
    return NULL;
  }
  if (static_cast<int>(PyArray_DIM(np_sample_weights, 0)) != nsample) {
    PyErr_SetString(PyExc_ValueError,
                    "sample_weights dimension does not match nsample");
    return NULL;
  }
  if (static_cast<int>(PyArray_DIM(np_truncated_kernel, 0)) != trunc_qdim ||
      static_cast<int>(PyArray_DIM(np_truncated_kernel, 1)) != trunc_zdim) {
    PyErr_SetString(
        PyExc_ValueError,
        "truncated_kernel dimensions do not match trunc_qdim x trunc_zdim");
    return NULL;
  }
  if (static_cast<int>(PyArray_DIM(np_trunc_q, 0)) != trunc_qdim) {
    PyErr_SetString(PyExc_ValueError,
                    "trunc_q dimension does not match trunc_qdim");
    return NULL;
  }
  if (static_cast<int>(PyArray_DIM(np_trunc_z, 0)) != trunc_zdim) {
    PyErr_SetString(PyExc_ValueError,
                    "trunc_z dimension does not match trunc_zdim");
    return NULL;
  }

  /* The truncated q/z coordinate arrays are currently passed through from
   * Python for interface symmetry and possible future validation, but the
   * overlap builder only needs their dimensions because interpolation into the
   * truncated table uses the known normalised support range directly. */
  (void)trunc_q;
  (void)trunc_z;

  const int typenum = std::is_same_v<Real, float> ? NPY_FLOAT32 : NPY_FLOAT64;
  npy_intp dims[3] = {qdim, udim, etadim};
  PyArrayObject *np_overlap_kernel =
      (PyArrayObject *)PyArray_ZEROS(3, dims, typenum, 0);
  if (np_overlap_kernel == NULL) {
    PyErr_NoMemory();
    return NULL;
  }
  Real *overlap_kernel = static_cast<Real *>(PyArray_DATA(np_overlap_kernel));

  /* Precompute the total sampled input-kernel weight once so each table cell
   * can convert its weighted sum into a proper kernel-weighted average. */
  Real weight_sum = static_cast<Real>(0.0);
  for (int is = 0; is < nsample; is++) {
    weight_sum += sample_weights[is];
  }

  tic("Evaluating overlap kernel table");
  build_overlap_kernel_table<Real>(
      overlap_kernel, q_grid, u_grid, eta_grid, sample_x, sample_y, sample_z,
      sample_weights, weight_sum, truncated_kernel, qdim, udim, etadim,
      nsample, trunc_qdim, trunc_zdim, nthreads);
  toc("Evaluating overlap kernel table");

  return Py_BuildValue("N", np_overlap_kernel);
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

  /* Check the dtype of the first array to determine precision without fully
   * parsing the tuple (the impl will do that). */
  PyObject *first_arg = PyTuple_GetItem(args, 0);
  if (first_arg == NULL) {
    return NULL;
  }
  if (!PyArray_Check(first_arg)) {
    PyErr_SetString(PyExc_TypeError, "First argument must be a numpy array.");
    return NULL;
  }
  const int input_typenum = PyArray_TYPE((PyArrayObject *)first_arg);
  /* Dispatch: encode input precision into a 1-bit key. */
  int dispatch_key = (input_typenum == NPY_FLOAT64);

  /* Dispatch: call the matching typed kernel based on the dispatch key. */
  switch (dispatch_key) {
    case 0:
      return compute_overlap_kernel_impl<float>(self, args);
    case 1:
      return compute_overlap_kernel_impl<double>(self, args);
    default:
      PyErr_SetString(PyExc_TypeError,
                      "Overlap kernel arrays must be float32 or float64.");
      return NULL;
  }
}
