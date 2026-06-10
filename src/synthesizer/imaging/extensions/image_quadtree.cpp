/******************************************************************************
 * C functions for pixel-centric SPH image rendering using a 2D quadtree.
 *
 * Each pixel queries the quadtree for candidate particles, then computes
 * the exact pixel-kernel overlap integral using the same area-integration
 * algorithm validated in the octree backend (3x3 grid for interior pixels,
 * adaptive grid for boundary pixels).
 *****************************************************************************/

/* C includes. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* C++ includes. */
#include <algorithm>
#include <vector>

/* Python includes. */
#define PY_ARRAY_UNIQUE_SYMBOL SYNTHESIZER_ARRAY_API
#define NO_IMPORT_ARRAY
#include <Python.h>

#include "../../extensions/numpy_init.h"

/* Local includes. */
#include "../../extensions/cpp_to_python.h"
#include "../../extensions/property_funcs.h"
#include "../../extensions/quadtree.h"
#include "../../extensions/timers.h"
#ifdef ATOMIC_TIMING
#include "../../extensions/timers_init.h"
#endif
#include "kernel_smoothing_funcs.h"

#ifdef WITH_OPENMP
#include <omp.h>
#endif

/**
 * @brief Build a q^2-indexed kernel lookup table from a q-indexed table.
 *
 * The original kernel table is uniformly spaced in q (impact parameter
 * normalised by smoothing length).  We convert it to uniform spacing in
 * q^2 = (dx^2 + dy^2) / h^2, eliminating sqrt from the inner loops (D1).
 * The conversion costs kdim sqrts, done once at render start.
 *
 * @param kernel  The q-indexed kernel table.
 * @param kdim    Number of entries in the kernel table.
 * @param threshold  Kernel support threshold.
 * @return q^2-indexed kernel table (vector of kdim doubles).
 */
static std::vector<double> build_kernel_q2_table(const double *kernel,
                                                 int kdim, double threshold) {

  double threshold_squ = threshold * threshold;
  std::vector<double> kq2(kdim);

  for (int i = 0; i < kdim; i++) {
    double q2 = i * threshold_squ / (kdim - 1);
    double q = sqrt(q2);
    kq2[i] = interpolate_kernel(q, kernel, kdim, threshold);
  }

  return kq2;
}

/**
 * @brief Interpolate the kernel value from a q^2-indexed lookup table.
 *
 * Uses linear interpolation in q^2 space, avoiding sqrt in the hot path
 * (D1).  ``q2`` should be (dx^2 + dy^2) / h^2.
 *
 * @param q2  Squared normalised distance.
 * @param kernel_q2  Kernel table uniformly spaced in q^2.
 * @param kdim  Number of entries.
 * @param threshold_squ  threshold * threshold.
 * @return Interpolated kernel value.
 */
static inline double interpolate_kernel_q2(double q2, const double *kernel_q2,
                                           int kdim, double threshold_squ) {

  if (q2 >= threshold_squ) {
    return 0.0;
  }

  double q2_scaled = q2 * (kdim - 1) / threshold_squ;

  if (q2_scaled >= kdim - 1) {
    return kernel_q2[kdim - 1];
  }

  int idx = (int)q2_scaled;
  double frac = q2_scaled - idx;
  return kernel_q2[idx] * (1.0 - frac) + kernel_q2[idx + 1] * frac;
}

/**
 * @brief Check whether a particle's entire kernel support lies inside a
 * single pixel (fast path).
 *
 * @param part_x, part_y  Particle position.
 * @param pix_x_min ... pix_y_max  Pixel world-coordinate bounds.
 * @param kernel_radius  h * threshold.
 * @return True if the kernel is wholly inside the pixel.
 */
static inline bool kernel_fully_inside_pixel_raw(
    double part_x, double part_y, double pix_x_min, double pix_x_max,
    double pix_y_min, double pix_y_max, double kernel_radius) {

  double dx_left = part_x - pix_x_min;
  double dx_right = pix_x_max - part_x;
  double dy_bottom = part_y - pix_y_min;
  double dy_top = pix_y_max - part_y;

  return (dx_left >= 0.0 && dx_right >= 0.0 && dy_bottom >= 0.0 &&
          dy_top >= 0.0 && dx_left >= kernel_radius &&
          dx_right >= kernel_radius && dy_bottom >= kernel_radius &&
          dy_top >= kernel_radius);
}

/**
 * @brief Compute kernel contribution when the entire pixel is inside the
 * kernel support.
 *
 * Uses a fixed 3x3 grid of sample points over the pixel area and averages
 * the kernel values, applying proper SPH normalisation.
 *
 * @param pix_x_min, pix_y_min  Pixel corner.
 * @param res  Pixel resolution.
 * @param part_x, part_y  Particle position.
 * @param h_squ  Smoothing length squared.
 * @param kernel_q2  q^2-indexed kernel table.
 * @param kdim  Table dimension.
 * @param threshold_squ  threshold^2.
 * @return Normalised kernel contribution (dimensionless fraction of total
 *         kernel mass).
 */
static inline double pixel_inside_kernel_contribution_q2(
    double pix_x_min, double pix_y_min, double res, double part_x,
    double part_y, double h_squ, const double *kernel_q2, int kdim,
    double threshold_squ) {

  const int grid = 3;
  double sum = 0.0;

  for (int si = 0; si < grid; si++) {
    for (int sj = 0; sj < grid; sj++) {
      double sample_x = pix_x_min + (si + 0.5) * res / grid;
      double sample_y = pix_y_min + (sj + 0.5) * res / grid;
      double dx = sample_x - part_x;
      double dy = sample_y - part_y;
      double q2 = (dx * dx + dy * dy) / h_squ;
      sum += interpolate_kernel_q2(q2, kernel_q2, kdim, threshold_squ);
    }
  }

  double avg = sum / (double)(grid * grid);
  /* SPH normalisation: kernel value / h^2 * pixel_area */
  return avg / h_squ * res * res;
}

/**
 * @brief Compute kernel contribution when the pixel straddles the kernel
 * boundary (partial overlap).
 *
 * Uses an adaptive grid whose density depends on the ratio of pixel size
 * to smoothing length (same logic as the octree backend).
 *
 * @param part_x, part_y  Particle position.
 * @param h  Smoothing length.
 * @param h_squ  Smoothing length squared.
 * @param pix_x_min, pix_y_min  Pixel corner.
 * @param res  Pixel resolution.
 * @param kernel_q2  q^2-indexed kernel table.
 * @param kdim  Table dimension.
 * @param threshold_squ  threshold^2.
 * @return Normalised kernel contribution.
 */
static inline double pixel_kernel_partial_overlap_q2(
    double part_x, double part_y, double h, double h_squ, double pix_x_min,
    double pix_y_min, double res, const double *kernel_q2, int kdim,
    double threshold_squ) {

  /* Base resolution: at least 4 samples per axis, increase when the
   * smoothing length is small compared to the pixel. */
  int n_sub = std::max(4, static_cast<int>(ceil(2.0 * res / h)));

  double kvalue_sum = 0.0;
  int n_samples = 0;

  for (int si = 0; si < n_sub; si++) {
    for (int sj = 0; sj < n_sub; sj++) {
      double sample_x = pix_x_min + (si + 0.5) * res / n_sub;
      double sample_y = pix_y_min + (sj + 0.5) * res / n_sub;
      double dx = sample_x - part_x;
      double dy = sample_y - part_y;
      double q2 = (dx * dx + dy * dy) / h_squ;
      double kval = interpolate_kernel_q2(q2, kernel_q2, kdim, threshold_squ);
      kvalue_sum += kval;
      n_samples++;
    }
  }

  double avg = kvalue_sum / (double)n_samples;
  return avg / h_squ * res * res;
}

/**
 * @brief Render a range of pixel rows using the quadtree.
 *
 * This is the core rendering function.  It queries the quadtree once per
 * pixel (A1), then for each candidate particle computes the exact
 * pixel-kernel overlap integral using the same area-integration algorithm
 * as the octree backend (3x3 grid for interior pixels, adaptive grid for
 * boundary pixels).  Each pixel writes exclusively to its own output slot
 * — no atomics needed.
 *
 * @param px_start  First pixel row (inclusive).
 * @param px_end    Last pixel row (exclusive).
 * @param pix_values  Flat array [npart * nimgs].
 * @param kernel_q2   q^2-indexed kernel table (D1).
 * @param res         Pixel resolution.
 * @param npix_x, npix_y  Image dimensions.
 * @param nimgs       Number of filters.
 * @param threshold   Kernel support threshold.
 * @param threshold_squ  threshold^2.
 * @param kdim        Kernel table dimension.
 * @param res_squ     res * res.
 * @param norm_factor Flux-conservation normalisation.
 * @param img         Output array [npix_x * npix_y * nimgs].
 * @param root        Quadtree root.
 */
static void render_pixel_rows(int px_start, int px_end,
                              const double *pix_values,
                              const double *kernel_q2, double res, int npix_x,
                              int npix_y, int nimgs, double threshold,
                              double threshold_squ, int kdim, double res_squ,
                              double norm_factor, double *img,
                              const struct quadcell *root) {

  /* Thread-local vectors for quadtree query results. */
  std::vector<const struct quadcell *> leaves;
  std::vector<int> starts;
  std::vector<int> counts;

  /* Thread-local accumulation buffer. */
  std::vector<double> accum(nimgs);

  for (int px = px_start; px < px_end; px++) {
    for (int py = 0; py < npix_y; py++) {

      /* Clear thread-local storage. */
      leaves.clear();
      starts.clear();
      counts.clear();
      for (int f = 0; f < nimgs; f++) {
        accum[f] = 0.0;
      }

      /* Pixel world-coordinate bounds. */
      double pix_x_min = px * res;
      double pix_x_max = (px + 1) * res;
      double pix_y_min = py * res;
      double pix_y_max = (py + 1) * res;

      /* A1: Single quadtree query for this pixel (B1 rect-to-rect prune). */
      query_quadtree_for_pixel(root, pix_x_min, pix_x_max, pix_y_min,
                               pix_y_max, threshold, leaves, starts, counts);

      if (leaves.empty()) {
        continue;
      }

      /* Evaluate every candidate particle with area integration. */
      for (size_t li = 0; li < leaves.size(); li++) {
        const struct quadcell *leaf = leaves[li];
        int start = starts[li];
        int end = start + counts[li];

        for (int p = start; p < end; p++) {

          double part_x = leaf->pos_x[p];
          double part_y = leaf->pos_y[p];
          double h = leaf->sml_arr[p];
          double h_squ = leaf->sml_squ_arr[p];
          double kernel_radius = h * threshold;

          /* Fast path: entire kernel support lies inside this pixel. */
          if (kernel_fully_inside_pixel_raw(part_x, part_y, pix_x_min,
                                            pix_x_max, pix_y_min, pix_y_max,
                                            kernel_radius)) {
            double contrib = 1.0 * norm_factor;
            int idx = leaf->indices[p];
            for (int f = 0; f < nimgs; f++) {
              accum[f] += contrib * pix_values[idx * nimgs + f];
            }
            continue;
          }

          /* Determine how this pixel overlaps the kernel. */
          double q_min, q_max, q_center;
          calculate_pixel_kernel_overlap(part_x, part_y, pix_x_min, pix_x_max,
                                         pix_y_min, pix_y_max, h, q_min, q_max,
                                         q_center);

          /* No overlap at all. */
          if (q_min >= threshold) {
            continue;
          }

          double kvalue;
          if (q_max < threshold) {
            /* Pixel entirely within kernel support. */
            kvalue = pixel_inside_kernel_contribution_q2(
                pix_x_min, pix_y_min, res, part_x, part_y, h_squ, kernel_q2,
                kdim, threshold_squ);
          } else {
            /* Partial overlap — adaptive grid integration. */
            kvalue = pixel_kernel_partial_overlap_q2(
                part_x, part_y, h, h_squ, pix_x_min, pix_y_min, res, kernel_q2,
                kdim, threshold_squ);
          }

          kvalue *= norm_factor;

          int idx = leaf->indices[p];
          for (int f = 0; f < nimgs; f++) {
            accum[f] += kvalue * pix_values[idx * nimgs + f];
          }
        }
      }

      /* Write to global image — no atomics needed. */
      for (int f = 0; f < nimgs; f++) {
        img[px * npix_y * nimgs + py * nimgs + f] = accum[f];
      }
    }
  }
}

/**
 * @brief Populate the image using the quadtree-based pixel-centric renderer.
 *
 * Dispatches to serial or OpenMP-parallel rendering.
 */
static void render_image_quadtree(const double *pix_values,
                                  const double *kernel_q2, double res,
                                  int npix_x, int npix_y, int nimgs,
                                  double threshold, int kdim,
                                  double norm_factor, double *img,
                                  const struct quadcell *root, int nthreads) {

  tic("render_image_quadtree");

  double threshold_squ = threshold * threshold;
  double res_squ = res * res;

#ifdef WITH_OPENMP
  if (nthreads > 1) {

#pragma omp parallel num_threads(nthreads)
    {
      int tid = omp_get_thread_num();
      int nthreads_actual = omp_get_num_threads();

      int rows_per_thread = npix_x / nthreads_actual;
      int remainder = npix_x % nthreads_actual;
      int px_start, px_end;

      if (tid < remainder) {
        px_start = tid * (rows_per_thread + 1);
        px_end = px_start + rows_per_thread + 1;
      } else {
        px_start = tid * rows_per_thread + remainder;
        px_end = px_start + rows_per_thread;
      }

      render_pixel_rows(px_start, px_end, pix_values, kernel_q2, res, npix_x,
                        npix_y, nimgs, threshold, threshold_squ, kdim, res_squ,
                        norm_factor, img, root);
    }

  } else {

    render_pixel_rows(0, npix_x, pix_values, kernel_q2, res, npix_x, npix_y,
                      nimgs, threshold, threshold_squ, kdim, res_squ,
                      norm_factor, img, root);
  }
#else
  (void)nthreads;

  render_pixel_rows(0, npix_x, pix_values, kernel_q2, res, npix_x, npix_y,
                    nimgs, threshold, threshold_squ, kdim, res_squ,
                    norm_factor, img, root);
#endif

  toc("render_image_quadtree");
}

/**
 * @brief Compute an image from particle data using the quadtree-based
 * pixel-centric renderer with area integration.
 *
 * @param np_pix_values  Particle values [npart * nimgs].
 * @param np_smoothing_lengths  Smoothing lengths.
 * @param np_pos  Particle (x, y) positions [npart, 2].
 * @param np_kernel  q-indexed projected kernel table.
 * @param res  Pixel resolution in world units.
 * @param npix_x, npix_y  Image dimensions.
 * @param npart  Number of particles.
 * @param threshold  Kernel support threshold.
 * @param kdim  Kernel table dimension.
 * @param nimgs  Number of filters.
 * @param nthreads  OpenMP thread count.
 */
PyObject *make_img_quadtree(PyObject *self, PyObject *args) {

  tic("make_img_quadtree");

  tic("make_img_quadtree.extract_python_data");

  (void)self;

  double res, threshold;
  int npix_x, npix_y, npart, kdim, nthreads, nimgs;
  PyArrayObject *np_pix_values, *np_kernel;
  PyArrayObject *np_smoothing_lengths, *np_pos;

  if (!PyArg_ParseTuple(args, "OOOOdiiidiii", &np_pix_values,
                        &np_smoothing_lengths, &np_pos, &np_kernel, &res,
                        &npix_x, &npix_y, &npart, &threshold, &kdim, &nimgs,
                        &nthreads))
    return NULL;

  const double *pix_values = extract_data_double(np_pix_values, "pix_values");
  const double *smoothing_lengths =
      extract_data_double(np_smoothing_lengths, "smoothing_lengths");
  const double *pos = extract_data_double(np_pos, "pos");
  const double *kernel = extract_data_double(np_kernel, "kernel");

  if (pix_values == NULL || smoothing_lengths == NULL || pos == NULL ||
      kernel == NULL) {
    return NULL;
  }

  /* Extract contiguous x, y arrays from (npart, 2) position data. */
  std::vector<double> pos_x_vec(npart);
  std::vector<double> pos_y_vec(npart);
  for (int i = 0; i < npart; i++) {
    pos_x_vec[i] = pos[i * 2];
    pos_y_vec[i] = pos[i * 2 + 1];
  }

  toc("make_img_quadtree.extract_python_data");

  /* D1: Build the q^2-indexed kernel table (kdim sqrts, once). */
  tic("make_img_quadtree.build_kernel_q2");
  std::vector<double> kernel_q2 =
      build_kernel_q2_table(kernel, kdim, threshold);
  toc("make_img_quadtree.build_kernel_q2");

  /* Construct the quadtree (G1, C1, C2). */
  tic("make_img_quadtree.construct_quadtree");
  int ncells = 0;
  struct quadcell *root = new struct quadcell;
  construct_quadtree(pos_x_vec.data(), pos_y_vec.data(), smoothing_lengths,
                     npart, root, &ncells, QUADTREE_MAX_DEPTH,
                     100, /* min_count */
                     res /* G1: pixel-resolution early stop */);
  toc("make_img_quadtree.construct_quadtree");

  /* Zeroed output array [npix_x, npix_y, nimgs]. */
  tic("make_img_quadtree.create_output_array");
  npy_intp np_img_dims[3] = {npix_x, npix_y, nimgs};
  PyArrayObject *np_img =
      (PyArrayObject *)PyArray_ZEROS(3, np_img_dims, NPY_DOUBLE, 0);
  double *img = (double *)PyArray_DATA(np_img);
  toc("make_img_quadtree.create_output_array");

  /* Flux-conservation normalisation for truncated kernels. */
  double norm_factor = compute_kernel_norm(kernel, kdim, threshold);

  /* Render. */
  render_image_quadtree(pix_values, kernel_q2.data(), res, npix_x, npix_y,
                        nimgs, threshold, kdim, norm_factor, img, root,
                        nthreads);

  cleanup_quadtree(root);

  toc("make_img_quadtree");

  return Py_BuildValue("N", np_img);
}

static PyMethodDef ImageQuadTreeMethods[] = {
    {"make_img_quadtree", (PyCFunction)make_img_quadtree, METH_VARARGS,
     "Pixel-centric SPH imaging via 2D quadtree with area integration."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "image_quadtree",
    "Pixel-centric SPH imaging with a 2D quadtree.",
    -1,
    ImageQuadTreeMethods,
    NULL,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC PyInit_image_quadtree(void) {
  PyObject *m = PyModule_Create(&moduledef);
  if (m == NULL) return NULL;
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
