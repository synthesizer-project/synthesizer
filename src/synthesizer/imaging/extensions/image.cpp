/******************************************************************************
 * C functions for calculating the value of a stellar particles SPH kernel
 *****************************************************************************/

/* C includes. */
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <math.h>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

/* Python includes. */
#define PY_ARRAY_UNIQUE_SYMBOL SYNTHESIZER_ARRAY_API
#define NO_IMPORT_ARRAY
#include "../../extensions/numpy_init.h"
#include <Python.h>

/* Local includes. */
#include "../../extensions/cpp_to_python.h"
#include "../../extensions/octree.h"
#include "../../extensions/property_funcs.h"
#include "../../extensions/timers.h"

#ifdef WITH_OPENMP
#include <omp.h>
#endif

/**
 * @brief Inline function for kernel interpolation
 */
inline double interpolate_kernel(double q, const double *kernel, int kdim,
                                 double threshold) {
  if (q >= threshold) {
    return 0.0;
  }

  double q_scaled = q * kdim;

  if (q_scaled >= kdim - 1) {
    return kernel[kdim - 1];
  }

  int kindex_low = static_cast<int>(q_scaled);
  double frac = q_scaled - kindex_low;

  return kernel[kindex_low] * (1.0 - frac) + kernel[kindex_low + 1] * frac;
}

/**
 * @brief Structure to hold cell with its computational cost
 */
struct weighted_cell {
  struct cell *cell_ptr;
  double cost;
  bool can_subdivide;

  weighted_cell(struct cell *c) : cell_ptr(c) {
    // Cost = max_smoothing_length * particle_count
    // This represents kernel area × work per kernel
    double max_sml = sqrt(c->max_sml_squ);
    cost = max_sml * c->part_count;
    can_subdivide = c->split;
  }
};

/**
 * @brief Comparison function for finding most expensive cell
 */
bool compare_by_cost(const weighted_cell &a, const weighted_cell &b) {
  return a.cost > b.cost; // Descending order
}

/**
 * @brief Build balanced work list using adaptive subdivision
 */
std::vector<weighted_cell>
build_balanced_work_list(struct cell *root, int nthreads,
                         double balance_tolerance = 2.0) {

  double start_time = tic();

  std::vector<weighted_cell> work_list;
  work_list.emplace_back(root);

  int target_cells =
      std::max(2 * nthreads, 8); // At least 2x threads, minimum 8

  while (true) {
    // Calculate statistics
    double total_cost = 0.0;
    double max_cost = 0.0;
    int subdividable_cells = 0;

    for (const auto &wc : work_list) {
      total_cost += wc.cost;
      max_cost = std::max(max_cost, wc.cost);
      if (wc.can_subdivide)
        subdividable_cells++;
    }

    double avg_cost = total_cost / work_list.size();

    // Check termination conditions
    bool enough_cells = work_list.size() >= target_cells;
    bool balanced = (max_cost / avg_cost) <= balance_tolerance;
    bool can_continue = subdividable_cells > 0;

    // Condition 1: Need more cells than threads (with reasonable minimum)
    // Condition 2: Cells should be roughly balanced in cost
    if ((enough_cells && balanced) || !can_continue) {
      break;
    }

    // Find the most expensive cell that can be subdivided
    auto most_expensive =
        std::max_element(work_list.begin(), work_list.end(),
                         [](const weighted_cell &a, const weighted_cell &b) {
                           if (!a.can_subdivide && b.can_subdivide)
                             return true;
                           if (a.can_subdivide && !b.can_subdivide)
                             return false;
                           return a.cost < b.cost;
                         });

    if (!most_expensive->can_subdivide) {
      printf("  No more cells can be subdivided\n");
      break;
    }

    // Subdivide the most expensive cell
    struct cell *expensive_cell = most_expensive->cell_ptr;

    // Remove the expensive cell from the list
    work_list.erase(most_expensive);

    // Add its children
    for (int ip = 0; ip < 8; ip++) {
      struct cell *child = &expensive_cell->progeny[ip];
      if (child->part_count > 0) {
        work_list.emplace_back(child);
      }
    }
  }

  // Final statistics
  double final_total_cost = 0.0;
  double final_max_cost = 0.0;
  double final_min_cost = 1e99;

  for (const auto &wc : work_list) {
    final_total_cost += wc.cost;
    final_max_cost = std::max(final_max_cost, wc.cost);
    final_min_cost = std::min(final_min_cost, wc.cost);
  }

  double final_avg_cost = final_total_cost / work_list.size();

  toc("Splitting cell tree over threads", start_time);

  return work_list;
}
/**
 * @brief Recursively populate a pixel.
 *
 * This will recurse to the leaves of the cell tree, any cells further than the
 * maximum smoothing length from the position will be skipped. Once in the
 * leaves the particles themselves will be checked to see if their SPH kernel
 * overlaps with the line of sight of the star particle.
 *
 * @param c The cell to calculate the surface densities for.
 * @param pix_x The x position of the pixel.
 * @param pix_y The y position of the pixel.
 * @param threshold The threshold for the kernel.
 * @param kdim The dimension of the kernel.
 * @param kernel The kernel to use for the calculation.
 * @param out The output array to populate with the pixel values.
 * @param nimgs The number of images to populate.
 * @param pix_values The pixel values to use for each image.
 */
static void populate_pixel_recursive(const struct cell *c, double threshold,
                                     int kdim, const double *kernel, int npart,
                                     double *out, int nimgs,
                                     const double *pix_values, const double res,
                                     const int npix_x, const int npix_y) {

  /* Is the cell split? */
  if (c->split) {

    /* Ok, so we recurse... */
    for (int ip = 0; ip < 8; ip++) {
      struct cell *cp = &c->progeny[ip];

      /* Skip empty progeny. */
      if (cp->part_count == 0) {
        continue;
      }

      /* Recurse... */
      populate_pixel_recursive(cp, threshold, kdim, kernel, npart, out, nimgs,
                               pix_values, res, npix_x, npix_y);
    }

  } else {

    /* We're in a leaf if we get here, unpack the particles. */
    int npart = c->part_count;
    struct particle *parts = c->particles;

    /* Loop over the particles adding their contribution to the pixel value. */
    for (int p = 0; p < npart; p++) {

      /* Get the particle. */
      struct particle *part = &parts[p];

      /* Get the particle position in terms of pixels. */
      int i = (int)floor(part->pos[0] / res);
      int j = (int)floor(part->pos[1] / res);

      /* If the smoothing length is less than half the resolution just add it
       * to the nearest pixel. */
      if (part->sml < res / 2.0) {
        int img_index = 0;
        for (int nimg = 0; nimg < nimgs; nimg++) {
          img_index = i * npix_y * nimgs + j * nimgs + nimg;
          out[img_index] += pix_values[part->index + nimg * npart];
        }
        continue;
      }

      /* How many pixels do we need to walk out in the kernel?  (with a
       * buffer of 1 pixel to ensure we cover the kernel). */
      int delta_pix = (int)ceil(part->sml / res) + 1;

      /* Loop over the pixels in the kernel. */
      for (int ii = -delta_pix; ii <= delta_pix; ii++) {
        for (int jj = -delta_pix; jj <= delta_pix; jj++) {

          /* Get this pixels indices. */
          int iii = i + ii;
          int jjj = j + jj;

          /* Skip pixels that are out of bounds. */
          if (iii < 0 || iii >= npix_x || jjj < 0 || jjj >= npix_y) {
            continue;
          }

          /* Calculate the pixel position. */
          double pix_x = iii * res;
          double pix_y = jjj * res;

          /* Calculate the x and y separations. */
          double dx = pix_x - part->pos[0];
          double dy = pix_y - part->pos[1];

          /* Calculate the impact parameter. */
          double b = sqrt(dx * dx + dy * dy);

          /* Compute the impact parameter in terms of the smoothing length. */
          double q = b / part->sml;

          /* Early skip if the pixel is outside the kernel threshold. */
          if (q > threshold) {
            continue;
          }

          /* Get the kernel value at this pixel position. */
          float kvalue_interpolated =
              interpolate_kernel(q, kernel, kdim, threshold);
          float kvalue =
              kvalue_interpolated / (part->sml * part->sml) * res * res;

          /* Loop over images and add the contribution to each pixel. */
          for (int nimg = 0; nimg < nimgs; nimg++) {
            int img_index = iii * npix_y * nimgs + jjj * nimgs + nimg;
            out[img_index] += kvalue * pix_values[part->index + nimg * npart];
          }
        }
      }
    }
  }
}

/**
 * @brief Populate the image across all threads.
 *
 * This function will first divide the cell tree to optimally balance work
 * across threads, and then it will populate the image by recursively
 * processing each cell in the partitioned work list.
 *
 * @param pix_values: The particle data to be sorted into pixels
 *                   (luminosity/flux/mass etc.).
 * @param kernel: The kernel data (integrated along the z axis and softed by
 *                impact parameter).
 * @param res: The pixel resolution.
 * @param npix_x: The number of pixels along the x axis.
 * @param npix_y: The number of pixels along the y axis.
 * @param npart: The number of particles.
 * @param threshold: The threshold of the SPH kernel.
 * @param kdim: The number of elements in the kernel.
 * @param img: The image to be populated.
 * @param nimgs: The number of images to populate.
 * @param root: The root of the tree.
 * @param nthreads: The number of threads to use.
 * @return void
 */
#ifdef WITH_OPENMP
void populate_smoothed_image_parallel(const double *pix_values,
                                      const double *kernel, const double res,
                                      const int npix_x, const int npix_y,
                                      const int npart, const double threshold,
                                      const int kdim, double *img,
                                      const int nimgs, struct cell *root,
                                      const int nthreads) {

  /* Build a balanced work list. */
  std::vector<weighted_cell> work_list =
      build_balanced_work_list(root, nthreads);

  /* Parallel loop over the work list. */
#pragma omp parallel for num_threads(nthreads) schedule(dynamic, 1)
  for (int i = 0; i < work_list.size(); i++) {
    const weighted_cell &wc = work_list[i];
    struct cell *c = wc.cell_ptr;

    /* Populate the pixel recursively. */
    populate_pixel_recursive(c, threshold, kdim, kernel, npart, img, nimgs,
                             pix_values, res, npix_x, npix_y);
  }
}
#endif

/**
 * @brief Populate an image in serial using a tree structure.
 *
 * The SPH kernel of a particle (integrated along the z axis) is used to
 * calculate the pixel weight for all pixels within a stellar particles
 * kernel. Once the kernel value is found at a pixels position the pixel value
 * is added multiplied by the kernels weight.
 *
 * This function will recurse through the cell tree and populate the
 * image by calculating the contribution of each particle to the pixels
 * in the image.
 *
 * @param pix_values: The particle data to be sorted into pixels
 *                    (luminosity/flux/mass etc.).
 * @param smoothing_lengths: The stellar particle smoothing lengths.
 * @param pos: The coordinates of the particles.
 * @param kernel: The kernel data (integrated along the z axis and softed by
 *               impact parameter).
 * @param res: The pixel resolution.
 * @param npix_x: The number of pixels along the x axis.
 * @param npix_y: The number of pixels along the y axis.
 * @param npart: The number of particles.
 * @param threshold: The threshold of the SPH kernel.
 * @param kdim: The number of elements in the kernel.
 * @param img: The image to be populated.
 * @param nimgs: The number of images to populate.
 * @param root: The root of the tree.
 */
void populate_smoothed_image_serial(const double *pix_values,
                                    const double *kernel, const double res,
                                    const int npix_x, const int npix_y,
                                    const int npart, const double threshold,
                                    const int kdim, double *img,
                                    const int nimgs, struct cell *root) {

  /* Populate the pixel recursively. */
  populate_pixel_recursive(root, threshold, kdim, kernel, npart, img, nimgs,
                           pix_values, res, npix_x, npix_y);
}

/**
 * @brief Populate the image.
 *
 * This is a wrapper which will call the correct version of the function to
 * populate the image based on whether or not OpenMP is available and the
 * number of threads to use.
 *
 * @param pix_values: The particle data to be sorted into pixels
 *                    (luminosity/flux/mass etc.).
 * @param kernel: The kernel data (integrated along the z axis and softed by
 *                impact parameter).
 * @param res: The pixel resolution.
 * @param npix_x: The number of pixels along the x axis.
 * @param npix_y: The number of pixels along the y axis.
 * @param npart: The number of particles.
 * @param threshold: The threshold of the SPH kernel.
 * @param kdim: The number of elements in the kernel.
 * @param img: The image to be populated.
 * @param nimgs: The number of images to populate.
 * @param root: The root of the tree.
 * @param nthreads: The number of threads to use.
 */
void populate_smoothed_image(const double *pix_values, const double *kernel,
                             const double res, const int npix_x,
                             const int npix_y, const int npart,
                             const double threshold, const int kdim,
                             double *img, const int nimgs, struct cell *root,
                             const int nthreads) {
  double start = tic();

  /* If we have multiple threads and OpenMP we can parallelise. */
#ifdef WITH_OPENMP
  if (nthreads > 1) {

    /* Populate the image in parallel. */
    populate_smoothed_image_parallel(pix_values, kernel, res, npix_x, npix_y,
                                     npart, threshold, kdim, img, nimgs, root,
                                     nthreads);

  } else {

    /* If we don't have OpenMP call the serial version. */
    populate_smoothed_image_serial(pix_values, kernel, res, npix_x, npix_y,
                                   npart, threshold, kdim, img, nimgs, root);
  }
#else
  (void)nthreads;

  /* If we don't have OpenMP call the serial version. */
  populate_smoothed_image_serial(pix_values, kernel, res, npix_x, npix_y, npart,
                                 threshold, kdim, img, nimgs, root);
#endif

  toc("Populating smoothed image", start);
}

/**
 * @brief Function to compute an image from particle data and a kernel.
 *
 * The SPH kernel of a particle (integrated along the z axis) is used to
 * calculate the pixel weight for all pixels within a stellar particles
 * kernel. Once the kernel value is found at a pixels position the pixel
 * value is added multiplied by the kernels weight.
 *
 * @param np_pix_values: The particle data to be sorted into pixels
 *                       (luminosity/flux/mass etc.).
 * @param np_smoothing_lengths: The stellar particle smoothing lengths.
 * @param np_pos: The coordinates of the particles.
 * @param np_kernel: The kernel data (integrated along the z axis and
 * softed by impact parameter).
 * @param res: The pixel resolution.
 * @param npix: The number of pixels along an axis.
 * @param npart: The number of particles.
 * @param threshold: The threshold of the SPH kernel.
 * @param kdim: The number of elements in the kernel.
 * @param nthreads: The number of threads to use.
 */
PyObject *make_img(PyObject *self, PyObject *args) {

  double start_time = tic();

  /* We don't need the self argument but it has to be there. Tell the
   * compiler we don't care. */
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

  /* Get pointers to the actual data. */
  const double *pix_values = extract_data_double(np_pix_values, "pix_values");
  const double *smoothing_lengths =
      extract_data_double(np_smoothing_lengths, "smoothing_lengths");
  const double *pos = extract_data_double(np_pos, "pos");
  const double *kernel = extract_data_double(np_kernel, "kernel");

  toc("Extracting Python data", start_time);

  double tree_start = tic();

  /* Allocate cells array. The first cell will be the root and then we
   * will dynamically nibble off cells for the progeny. */
  int ncells = 1;
  struct cell *root = new struct cell;

  /* Consturct the cell tree. */
  construct_cell_tree(pos, smoothing_lengths, smoothing_lengths, npart, root,
                      ncells, MAX_DEPTH, 100);

  toc("Constructing cell tree", tree_start);

  double out_start = tic();

  /* Create the zeroed image numpy array. */
  npy_intp np_img_dims[3] = {npix_x, npix_y, nimgs};
  PyArrayObject *np_img =
      (PyArrayObject *)PyArray_ZEROS(3, np_img_dims, NPY_DOUBLE, 0);
  double *img = (double *)PyArray_DATA(np_img);

  toc("Creating output image", out_start);

  /* Populate the image. */
  populate_smoothed_image(pix_values, kernel, res, npix_x, npix_y, npart,
                          threshold, kdim, img, nimgs, root, nthreads);

  /* Cleanup the cell tree. */
  cleanup_cell_tree(root);

  toc("Computing smoothed image", start_time);

  return Py_BuildValue("N", np_img);
}

static PyMethodDef ImageMethods[] = {
    {"make_img", (PyCFunction)make_img, METH_VARARGS,
     "Method for smoothing particles into an image."},
    {NULL, NULL, 0, NULL},
};

/* Make this importable. */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "image",                                  /* m_name */
    "A module to make images from particles", /* m_doc */
    -1,                                       /* m_size */
    ImageMethods,                             /* m_methods */
    NULL,                                     /* m_reload */
    NULL,                                     /* m_traverse */
    NULL,                                     /* m_clear */
    NULL,                                     /* m_free */
};

PyMODINIT_FUNC PyInit_image(void) {
  PyObject *m = PyModule_Create(&moduledef);
  if (numpy_import() < 0) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to import numpy.");
    return NULL;
  }
  return m;
}
