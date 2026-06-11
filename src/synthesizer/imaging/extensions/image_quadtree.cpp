/******************************************************************************
 * C functions for pixel-centric SPH image rendering using a 2D quadtree.
 *
 * The image is divided into tiles.  Each tile queries the quadtree once to
 * collect candidate particles (amortising tree-traversal cost across many
 * pixels).  Within a tile, every pixel iterates the candidate set and
 * computes exact pixel-kernel overlap integrals.  Tiles are processed in
 * parallel with no atomic writes — each tile's pixel writes are exclusive.
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

/* ------------------------------------------------------------------ */
/*  D1: q^2 kernel table                                              */
/* ------------------------------------------------------------------ */

static std::vector<double> build_kernel_q2_table(const double *kernel,
                                                 int kdim, double threshold) {
  double threshold_squ = threshold * threshold;
  std::vector<double> kq2(kdim);
  for (int i = 0; i < kdim; i++) {
    double q2 = i * threshold_squ / (kdim - 1);
    kq2[i] = interpolate_kernel(sqrt(q2), kernel, kdim, threshold);
  }
  return kq2;
}

static inline double interpolate_kernel_q2(double q2, const double *kernel_q2,
                                           int kdim, double threshold_squ) {
  if (q2 >= threshold_squ) return 0.0;
  double q2_scaled = q2 * (kdim - 1) / threshold_squ;
  if (q2_scaled >= kdim - 1) return kernel_q2[kdim - 1];
  int idx = (int)q2_scaled;
  double frac = q2_scaled - idx;
  return kernel_q2[idx] * (1.0 - frac) + kernel_q2[idx + 1] * frac;
}

/* ------------------------------------------------------------------ */
/*  Area-integration helpers (same algorithm as the octree backend)   */
/* ------------------------------------------------------------------ */

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

static inline double pixel_inside_kernel_contribution_q2(
    double pix_x_min, double pix_y_min, double res, double part_x,
    double part_y, double h_squ, const double *kernel_q2, int kdim,
    double threshold_squ) {
  const int grid = 3;
  double sum = 0.0;
  for (int si = 0; si < grid; si++) {
    for (int sj = 0; sj < grid; sj++) {
      double sx = pix_x_min + (si + 0.5) * res / grid;
      double sy = pix_y_min + (sj + 0.5) * res / grid;
      double dx = sx - part_x, dy = sy - part_y;
      sum += interpolate_kernel_q2((dx * dx + dy * dy) / h_squ, kernel_q2,
                                   kdim, threshold_squ);
    }
  }
  return sum / (double)(grid * grid) / h_squ * res * res;
}

static inline double pixel_kernel_partial_overlap_q2(
    double part_x, double part_y, double h, double h_squ, double pix_x_min,
    double pix_y_min, double res, const double *kernel_q2, int kdim,
    double threshold_squ) {
  int n_sub = std::max(4, static_cast<int>(ceil(2.0 * res / h)));
  double sum = 0.0;
  for (int si = 0; si < n_sub; si++) {
    for (int sj = 0; sj < n_sub; sj++) {
      double sx = pix_x_min + (si + 0.5) * res / n_sub;
      double sy = pix_y_min + (sj + 0.5) * res / n_sub;
      double dx = sx - part_x, dy = sy - part_y;
      sum += interpolate_kernel_q2((dx * dx + dy * dy) / h_squ, kernel_q2,
                                   kdim, threshold_squ);
    }
  }
  return sum / (double)(n_sub * n_sub) / h_squ * res * res;
}

/* ------------------------------------------------------------------ */
/*  Core renderer: tile-based, pixel-centric                          */
/* ------------------------------------------------------------------ */

/**
 * @brief Render a rectangular tile of pixels.
 *
 * Queries the quadtree once for the tile's expanded bounding box (A1, B1),
 * then evaluates every pixel in the tile against the candidate set.  No
 * atomics — the caller must ensure tiles are processed by disjoint threads.
 */
static void render_tile(int px0, int px1, int py0, int py1,
                        const double *pix_values, const double *kernel_q2,
                        double res, int npix_x, int npix_y, int nimgs,
                        double threshold, double threshold_squ, int kdim,
                        double norm_factor, double *img,
                        const struct quadtree *tree) {

  /* Work out the world-coordinate bounds of the tile.  The quadtree
   * query prunes per-cell using each cell's own max_sml_squ (B1), so
   * we pass the UNEXPANDED tile rect — cells whose largest kernel can
   * reach the tile will be kept automatically. */
  double tile_xmin = px0 * res;
  double tile_xmax = (px1 - 1) * res + res;
  double tile_ymin = py0 * res;
  double tile_ymax = (py1 - 1) * res + res;

  /* A1: single quadtree query for the whole tile. */
  tic("render_image_quadtree.tile_query");
  std::vector<const struct quadcell *> leaves;
  std::vector<int> starts, counts;
  query_quadtree_for_pixel(tree, tile_xmin, tile_xmax, tile_ymin, tile_ymax,
                           threshold, leaves, starts, counts);
  toc("render_image_quadtree.tile_query");

  if (leaves.empty()) return;

  /* Per-pixel accumulation buffer for this tile (no atomics needed). */
  int tw = px1 - px0, th = py1 - py0;
  std::vector<double> tile_img(tw * th * nimgs, 0.0);

  tic("render_image_quadtree.tile_inner");
  for (int px = px0; px < px1; px++) {
    for (int py = py0; py < py1; py++) {

      double pix_x_min = px * res;
      double pix_x_max = (px + 1) * res;
      double pix_y_min = py * res;
      double pix_y_max = (py + 1) * res;

      double *accum = &tile_img[((px - px0) * th + (py - py0)) * nimgs];

      for (size_t li = 0; li < leaves.size(); li++) {
        const struct quadcell *leaf = leaves[li];
        int start = starts[li];
        int end = start + counts[li];

        for (int p = start; p < end; p++) {
          double part_x = leaf->pos_x[p];
          double part_y = leaf->pos_y[p];
          double h = leaf->sml_arr[p];
          double h_squ = leaf->sml_squ_arr[p];
          double kern_r = h * threshold;

          double kvalue;
          if (kernel_fully_inside_pixel_raw(part_x, part_y, pix_x_min,
                                            pix_x_max, pix_y_min, pix_y_max,
                                            kern_r)) {
            kvalue = 1.0;
          } else {
            double q_min, q_max, q_center;
            calculate_pixel_kernel_overlap(part_x, part_y, pix_x_min,
                                           pix_x_max, pix_y_min, pix_y_max, h,
                                           q_min, q_max, q_center);
            if (q_min >= threshold) continue;
            if (q_max < threshold) {
              kvalue = pixel_inside_kernel_contribution_q2(
                  pix_x_min, pix_y_min, res, part_x, part_y, h_squ, kernel_q2,
                  kdim, threshold_squ);
            } else {
              kvalue = pixel_kernel_partial_overlap_q2(
                  part_x, part_y, h, h_squ, pix_x_min, pix_y_min, res,
                  kernel_q2, kdim, threshold_squ);
            }
          }
          kvalue *= norm_factor;

          int idx = leaf->indices[p];
          for (int f = 0; f < nimgs; f++) {
            accum[f] += kvalue * pix_values[idx * nimgs + f];
          }
        }
      }
    }
  }
  toc("render_image_quadtree.tile_inner");

  /* Copy tile image to global image — no atomics needed. */
  tic("render_image_quadtree.tile_copy");
  for (int lx = 0; lx < tw; lx++) {
    for (int ly = 0; ly < th; ly++) {
      int gx = px0 + lx, gy = py0 + ly;
      if (gx >= npix_x || gy >= npix_y) continue;
      int ti = (lx * th + ly) * nimgs;
      int gi = gx * npix_y * nimgs + gy * nimgs;
      for (int f = 0; f < nimgs; f++) img[gi + f] = tile_img[ti + f];
    }
  }
  toc("render_image_quadtree.tile_copy");
}

/**
 * @brief Populate the image using tile-based pixel-centric rendering.
 */
static void render_image_quadtree(const double *pix_values,
                                  const double *kernel_q2, double res,
                                  int npix_x, int npix_y, int nimgs,
                                  double threshold, int kdim,
                                  double norm_factor, double *img,
                                  const struct quadtree *tree, int nthreads) {

  tic("render_image_quadtree");

  double threshold_squ = threshold * threshold;

  /* Tile size: large enough to amortise tree traversals, small enough
   * for good load balance and cache footprint. */
  const int TILE = 64;

#ifdef WITH_OPENMP
  if (nthreads > 1) {

#pragma omp parallel for num_threads(nthreads) schedule(dynamic) collapse(2)
    for (int tx = 0; tx < npix_x; tx += TILE) {
      for (int ty = 0; ty < npix_y; ty += TILE) {
        int px0 = tx;
        int px1 = std::min(tx + TILE, npix_x);
        int py0 = ty;
        int py1 = std::min(ty + TILE, npix_y);
        render_tile(px0, px1, py0, py1, pix_values, kernel_q2, res, npix_x,
                    npix_y, nimgs, threshold, threshold_squ, kdim, norm_factor,
                    img, tree);
      }
    }

  } else {
    for (int tx = 0; tx < npix_x; tx += TILE) {
      for (int ty = 0; ty < npix_y; ty += TILE) {
        int px0 = tx, px1 = std::min(tx + TILE, npix_x);
        int py0 = ty, py1 = std::min(ty + TILE, npix_y);
        render_tile(px0, px1, py0, py1, pix_values, kernel_q2, res, npix_x,
                    npix_y, nimgs, threshold, threshold_squ, kdim, norm_factor,
                    img, tree);
      }
    }
  }
#else
  (void)nthreads;
  for (int tx = 0; tx < npix_x; tx += TILE) {
    for (int ty = 0; ty < npix_y; ty += TILE) {
      int px0 = tx, px1 = std::min(tx + TILE, npix_x);
      int py0 = ty, py1 = std::min(ty + TILE, npix_y);
      render_tile(px0, px1, py0, py1, pix_values, kernel_q2, res, npix_x,
                  npix_y, nimgs, threshold, threshold_squ, kdim, norm_factor,
                  img, root);
    }
  }
#endif

  toc("render_image_quadtree");
}

/* ------------------------------------------------------------------ */
/*  Python entry point                                                */
/* ------------------------------------------------------------------ */

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
      kernel == NULL)
    return NULL;

  std::vector<double> pos_x_vec(npart);
  std::vector<double> pos_y_vec(npart);
  for (int i = 0; i < npart; i++) {
    pos_x_vec[i] = pos[i * 2];
    pos_y_vec[i] = pos[i * 2 + 1];
  }
  toc("make_img_quadtree.extract_python_data");

  tic("make_img_quadtree.build_kernel_q2");
  std::vector<double> kernel_q2 =
      build_kernel_q2_table(kernel, kdim, threshold);
  toc("make_img_quadtree.build_kernel_q2");

  tic("make_img_quadtree.construct_quadtree");
  struct quadtree *tree =
      construct_quadtree(pos_x_vec.data(), pos_y_vec.data(), smoothing_lengths,
                         npart, QUADTREE_MAX_DEPTH, 100, res, 16);
  toc("make_img_quadtree.construct_quadtree");

  tic("make_img_quadtree.create_output_array");
  npy_intp np_img_dims[3] = {npix_x, npix_y, nimgs};
  PyArrayObject *np_img =
      (PyArrayObject *)PyArray_ZEROS(3, np_img_dims, NPY_DOUBLE, 0);
  double *img = (double *)PyArray_DATA(np_img);
  toc("make_img_quadtree.create_output_array");

  double norm_factor = compute_kernel_norm(kernel, kdim, threshold);

  render_image_quadtree(pix_values, kernel_q2.data(), res, npix_x, npix_y,
                        nimgs, threshold, kdim, norm_factor, img, tree,
                        nthreads);

  cleanup_quadtree(tree);
  toc("make_img_quadtree");
  return Py_BuildValue("N", np_img);
}

static PyMethodDef ImageQuadTreeMethods[] = {
    {"make_img_quadtree", (PyCFunction)make_img_quadtree, METH_VARARGS,
     "Tile-based pixel-centric SPH imaging via 2D quadtree."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "image_quadtree",
    "Tile-based pixel-centric SPH imaging with a 2D quadtree.",
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
