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
 * @brief Inline function for kernel interpolation.
 *
 * This funciton interpolates the value of the kernel at a given distance
 * from the center, `q`, it starts from the precomputed kernel look up table
 * and refines this value using linear interpolation.
 *
 * @param q The distance from the center, normalized by the smoothing length.
 * @param kernel The precomputed kernel look up table.
 * @param kdim The dimension of the kernel (number of entries in the kernel
 * array).
 * @param threshold The threshold value for the kernel, above which the kernel
 * value is zero.
 *
 * @return The interpolated kernel value at distance `q`.
 */
inline double interpolate_kernel(double q, const double *kernel, int kdim,
                                 double threshold) {

  /* Early exit if q is above the threshold */
  if (q >= threshold) {
    return 0.0;
  }

  /* Convert q to a fraction of the kernel dimension */
  double q_scaled = q * kdim;

  /* Return the maximum kernel value if q is beyond the last entry */
  if (q_scaled >= kdim - 1) {
    return kernel[kdim - 1];
  }

  /* Linear interpolation between the two nearest kernel values */
  int kindex_low = static_cast<int>(q_scaled);
  double frac = q_scaled - kindex_low;
  return kernel[kindex_low] * (1.0 - frac) + kernel[kindex_low + 1] * frac;
}

/**
 * @brief Calculate pixel-kernel overlap distances.
 *
 * This function calculates the minimum and maximum distances from a particle
 * to a pixel, which are used to determine if/how the pixel overlaps with the
 * particle's kernel.
 *
 * @param part_x Particle x position.
 * @param part_y Particle y position.
 * @param pix_x_min Pixel minimum x coordinate.
 * @param pix_x_max Pixel maximum x coordinate.
 * @param pix_y_min Pixel minimum y coordinate.
 * @param pix_y_max Pixel maximum y coordinate.
 * @param sml Particle smoothing length.
 * @param q_min Output: minimum distance (to closest point in pixel) / sml.
 * @param q_max Output: maximum distance (to farthest corner) / sml.
 * @param q_center Output: distance to pixel center / sml.
 */
inline void calculate_pixel_kernel_overlap(double part_x, double part_y,
                                           double pix_x_min, double pix_x_max,
                                           double pix_y_min, double pix_y_max,
                                           double sml, double &q_min,
                                           double &q_max, double &q_center) {

  /* Find the closest point in the pixel to the particle. */
  double closest_x = std::max(pix_x_min, std::min(part_x, pix_x_max));
  double closest_y = std::max(pix_y_min, std::min(part_y, pix_y_max));
  double dx_min = closest_x - part_x;
  double dy_min = closest_y - part_y;
  double b_min = sqrt(dx_min * dx_min + dy_min * dy_min);
  q_min = b_min / sml;

  /* Find the farthest corner from the particle. */
  q_max = 0.0;
  for (int ci = 0; ci <= 1; ci++) {
    for (int cj = 0; cj <= 1; cj++) {
      double corner_x = (ci == 0) ? pix_x_min : pix_x_max;
      double corner_y = (cj == 0) ? pix_y_min : pix_y_max;
      double dx_c = corner_x - part_x;
      double dy_c = corner_y - part_y;
      double b_corner = sqrt(dx_c * dx_c + dy_c * dy_c);
      double q_corner = b_corner / sml;
      if (q_corner > q_max) {
        q_max = q_corner;
      }
    }
  }

  /* Calculate pixel center distance. */
  double pix_x_center = (pix_x_min + pix_x_max) / 2.0;
  double pix_y_center = (pix_y_min + pix_y_max) / 2.0;
  double dx_center = pix_x_center - part_x;
  double dy_center = pix_y_center - part_y;
  double b_center = sqrt(dx_center * dx_center + dy_center * dy_center);
  q_center = b_center / sml;
}

/**
 * @brief Calculate contribution when entire pixel is inside the kernel.
 *
 * When the pixel is wholly contained within the kernel support, we evaluate
 * the kernel at the pixel center and apply proper SPH normalization.
 *
 * @param q_center Normalized distance from particle to pixel center.
 * @param kernel Pointer to kernel interpolation table.
 * @param kdim Dimension of kernel table.
 * @param threshold Kernel threshold value.
 * @param sml Smoothing length.
 * @param res Pixel resolution.
 * @return The kernel contribution value (normalized).
 */
inline double pixel_inside_kernel_contribution(double pix_x_min,
                                               double pix_y_min, double res,
                                               const struct particle *part,
                                               const double *kernel, int kdim,
                                               double threshold) {

  /* Use a small fixed grid to sample the pixel area (fast, deterministic). */
  const int grid = 3; // 3x3 samples
  double sum = 0.0;

  for (int si = 0; si < grid; si++) {
    for (int sj = 0; sj < grid; sj++) {
      double sample_x = pix_x_min + (si + 0.5) * res / grid;
      double sample_y = pix_y_min + (sj + 0.5) * res / grid;

      double dx = sample_x - part->pos[0];
      double dy = sample_y - part->pos[1];
      double q = sqrt(dx * dx + dy * dy) / part->sml;

      sum += interpolate_kernel(q, kernel, kdim, threshold);
    }
  }

  double avg = sum / static_cast<double>(grid * grid);

  /* Apply SPH normalization: kernel value / h^2 * pixel_area */
  return avg / (part->sml * part->sml) * res * res;
}

/**
 * @brief Compute normalization factor so truncated kernels conserve flux.
 */
inline double compute_kernel_norm(const double *kernel, int kdim,
                                  double threshold) {
  double dq = 1.0 / (kdim - 1);
  double full_integral = 0.0;
  double partial_integral = 0.0;

  int max_idx = kdim - 1;
  double integral = 0.0;

  for (int i = 0; i < max_idx; i++) {
    double q0 = i * dq;
    double q1 = (i + 1) * dq;
    double w0 = kernel[i];
    double w1 = kernel[i + 1];
    double avg = 0.5 * (w0 + w1);
    /* Integrate 2π q K(q) dq via trapezoid; ∫q dq over [q0,q1] = 0.5(q1^2-q0^2) */
    double seg = 2.0 * M_PI * avg * 0.5 * (q1 * q1 - q0 * q0);
    integral += seg;
    if (q0 < threshold) {
      double q1c = q1;
      double w1c = w1;
      if (q1c > threshold) {
        q1c = threshold;
        double frac = (q1c - q0) / dq;
        w1c = w0 * (1.0 - frac) + kernel[i + 1] * frac;
      }
      double avgt = 0.5 * (w0 + w1c);
      double segt = 2.0 * M_PI * avgt * 0.5 * (q1c * q1c - q0 * q0);
      partial_integral += segt;
    }
  }

  full_integral = integral;

  if (partial_integral <= 0.0) {
    return 1.0;
  }
  /* Rescale so total integral over [0,1] is unity, and partial region matches. */
  return full_integral / partial_integral;
}

/**
 * @brief Calculate contribution when pixel is partially inside the kernel.
 *
 * When the pixel straddles the kernel boundary, we use adaptive Monte Carlo
 * sampling to accurately integrate the kernel contribution over the pixel area.
 *
 * @param part Pointer to the particle.
 * @param pix_x_min Pixel minimum x coordinate.
 * @param pix_x_max Pixel maximum x coordinate.
 * @param pix_y_min Pixel minimum y coordinate.
 * @param pix_y_max Pixel maximum y coordinate.
 * @param kernel Pointer to kernel interpolation table.
 * @param kdim Dimension of kernel table.
 * @param threshold Kernel threshold value.
 * @param res Pixel resolution.
 * @return The kernel contribution value (normalized).
 */
inline double pixel_kernel_partial_overlap_contribution(
    const struct particle *part, double pix_x_min, double pix_x_max,
    double pix_y_min, double pix_y_max, const double *kernel, int kdim,
    double threshold, double res) {

  /* Adaptive sampling based on smoothing length.
   * Minimum 3 samples per axis, increase for small smoothing lengths. */
  int n_sub =
      std::max(4, static_cast<int>(ceil(2.0 * res / part->sml))); // capped grid

  double kvalue_sum = 0.0;
  int n_samples = 0;

  /* Deterministic grid integration over pixel area */
  for (int si = 0; si < n_sub; si++) {
    for (int sj = 0; sj < n_sub; sj++) {

      /* Sample at sub-pixel positions */
      double sample_x = pix_x_min + (si + 0.5) * res / n_sub;
      double sample_y = pix_y_min + (sj + 0.5) * res / n_sub;

      /* Calculate distance to particle */
      double dx_s = sample_x - part->pos[0];
      double dy_s = sample_y - part->pos[1];
      double b_s = sqrt(dx_s * dx_s + dy_s * dy_s);
      double q_s = b_s / part->sml;

      /* Add kernel contribution from this sample point */
      double kval = interpolate_kernel(q_s, kernel, kdim, threshold);
      kvalue_sum += kval;
      n_samples++;
    }
  }

  /* Average kernel value over sample points */
  double kvalue_interpolated = kvalue_sum / static_cast<double>(n_samples);

  /* Apply SPH normalization: kernel value / h^2 * pixel_area */
  return kvalue_interpolated / (part->sml * part->sml) * res * res;
}

/**
 * @brief Check if the entire kernel support lies within the pixel.
 */
inline bool kernel_fully_inside_pixel(const struct particle *part,
                                      double pix_x_min, double pix_x_max,
                                      double pix_y_min, double pix_y_max,
                                      double kernel_radius) {
  /* Particle must be inside the pixel, and the kernel radius must not cross
   * any pixel edge. */
  double dx_left = part->pos[0] - pix_x_min;
  double dx_right = pix_x_max - part->pos[0];
  double dy_bottom = part->pos[1] - pix_y_min;
  double dy_top = pix_y_max - part->pos[1];

  return (dx_left >= 0.0 && dx_right >= 0.0 && dy_bottom >= 0.0 &&
          dy_top >= 0.0 && dx_left >= kernel_radius &&
          dx_right >= kernel_radius && dy_bottom >= kernel_radius &&
          dy_top >= kernel_radius);
}

/**
 * @brief Structure to hold cell with its computational cost
 */
struct weighted_cell {

  /*! Pointer to the cell */
  struct cell *cell_ptr;

  /*! Cost of the cell, calculated as max_smoothing_length * particle_count */
  double cost;

  /*! Whether the cell can be subdivided, i.e. if the cell is split */
  bool can_subdivide;

  /**
   * @brief Constructor to initialize a weighted cell
   *
   * @param c Pointer to the cell to be weighted
   */
  weighted_cell(struct cell *c) : cell_ptr(c) {
    // Cost = max_smoothing_length * particle_count
    // This represents kernel area × work per kernel
    double max_sml = sqrt(c->max_sml_squ);
    cost = max_sml * c->part_count;
    can_subdivide = c->split;
  }
};

/**
 * @brief Comparison function for finding most expensive cell.
 *
 * This function compares two weighted cells based on their cost.
 *
 * It is used to sort the cells in descending order of cost,
 * so that the most expensive cells are processed first.
 *
 * @param a First weighted cell to compare
 * @param b Second weighted cell to compare
 *
 * @return True if the cost of cell `a` is greater than that of cell `b`,
 *         false otherwise.
 */
inline bool compare_by_cost(const weighted_cell &a, const weighted_cell &b) {
  return a.cost > b.cost; // Descending order
}

/**
 * @brief Build balanced work list using adaptive subdivision
 */
static std::vector<weighted_cell>
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
                                     int kdim, const double *kernel,
                                     double norm_factor, int npart, double *out,
                                     int nimgs, const double *pix_values,
                                     const double res, const int npix_x,
                                     const int npix_y) {

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
      populate_pixel_recursive(cp, threshold, kdim, kernel, norm_factor, npart,
                               out, nimgs, pix_values, res, npix_x, npix_y);
    }

  } else {

    /* We're in a leaf if we get here. To avoid memory contention between
     * threads, we'll allocate a local image buffer for just the region
     * this leaf cell affects, process all particles into that buffer,
     * then copy the results back to the global image at the end. */

    /* Calculate the maximum kernel radius for particles in this cell. */
    double max_sml = sqrt(c->max_sml_squ);
    double max_kernel_radius = max_sml * threshold;

    /* Add a buffer around the cell to account for kernel overlap. We need
     * to include the maximum kernel radius plus an extra pixel buffer to
     * ensure we capture all affected pixels. */
    double buffer = max_kernel_radius + res;

    /* Calculate the world coordinate bounds of the region this cell affects. */
    double x_min_world = c->loc[0] - buffer;
    double x_max_world = c->loc[0] + c->width + buffer;
    double y_min_world = c->loc[1] - buffer;
    double y_max_world = c->loc[1] + c->width + buffer;

    /* Convert world coordinates to pixel indices, clamping to image bounds. */
    int i_min = std::max(0, (int)floor(x_min_world / res));
    int i_max = std::min(npix_x, (int)ceil(x_max_world / res));
    int j_min = std::max(0, (int)floor(y_min_world / res));
    int j_max = std::min(npix_y, (int)ceil(y_max_world / res));

    /* Calculate the dimensions of our local image buffer. */
    int local_width = i_max - i_min;
    int local_height = j_max - j_min;

    /* Safety check: ensure we have valid dimensions. */
    if (local_width <= 0 || local_height <= 0) {
      PyErr_SetString(PyExc_ValueError,
                      "Invalid local image dimensions calculated.");
      return;
    }

    /* Allocate and zero-initialize the local image buffer. This will store
     * contributions from all particles in this cell before we copy them
     * back to the global image. */
    std::vector<double> local_img(local_width * local_height * nimgs, 0.0);

    /* Unpack the particles from this leaf cell. */
    struct particle *parts = c->particles;

    /* Loop over the particles adding their contribution to the local buffer. */
    for (int p = 0; p < c->part_count; p++) {

      /* Get the particle. */
      struct particle *part = &parts[p];

      /* Get the particle position in terms of pixels. */
      int i = (int)floor(part->pos[0] / res);
      int j = (int)floor(part->pos[1] / res);

      /* Note: We removed the early escape for small smoothing lengths
       * (sml < res/2) because it doesn't properly handle the kernel
       * distribution. Instead, we always use the proper kernel integration
       * below, which correctly handles all smoothing length regimes. */

      /* How many pixels do we need to walk out in the kernel? (with a
       * buffer of 1 pixel to ensure we cover the kernel). */
      int delta_pix = (int)ceil(part->sml * threshold / res) + 1;

      /* Loop over the pixels in the kernel. */
      for (int ii = -delta_pix; ii <= delta_pix; ii++) {
        for (int jj = -delta_pix; jj <= delta_pix; jj++) {

          /* Get this pixel's global indices. */
          int iii = i + ii;
          int jjj = j + jj;

          /* Skip pixels that are outside our local region. This includes
           * pixels that are out of bounds of the global image as well as
           * pixels that fall outside the region this cell affects. */
          if (iii < i_min || iii >= i_max || jjj < j_min || jjj >= j_max) {
            continue;
          }

          /* Calculate the pixel bounds. */
          double pix_x_min = iii * res;
          double pix_x_max = (iii + 1) * res;
          double pix_y_min = jjj * res;
          double pix_y_max = (jjj + 1) * res;

          /* Calculate kernel support radius */
          double kernel_radius = part->sml * threshold;

          /* Fast path: kernel wholly inside this pixel. */
          double kvalue;
          if (kernel_fully_inside_pixel(part, pix_x_min, pix_x_max, pix_y_min,
                                        pix_y_max, kernel_radius)) {
            kvalue = 1.0; /* Entire kernel mass inside this pixel */
          } else {
            /* Determine overlap of kernel and pixel. */
            double q_min, q_max, q_center;
            calculate_pixel_kernel_overlap(
                part->pos[0], part->pos[1], pix_x_min, pix_x_max, pix_y_min,
                pix_y_max, part->sml, q_min, q_max, q_center);

            /* Skip pixels that don't overlap with the kernel at all. */
            if (q_min >= threshold) {
              continue;
            }

            if (q_max < threshold) {
              /* Pixel wholly inside kernel: center-sample with SPH
               * normalization (fast path). */
              kvalue = pixel_inside_kernel_contribution(
                  pix_x_min, pix_y_min, res, part, kernel, kdim, threshold);
            } else {
              /* Partial overlap: integrate over pixel area. */
              kvalue = pixel_kernel_partial_overlap_contribution(
                  part, pix_x_min, pix_x_max, pix_y_min, pix_y_max, kernel,
                  kdim, threshold, res);
            }
          }

          /* Apply kernel normalization to conserve flux when truncated. */
          kvalue *= norm_factor;

          /* Convert global pixel coordinates to local buffer coordinates. */
          int local_i = iii - i_min;
          int local_j = jjj - j_min;

          /* Safety check bounds. */
          if (local_i >= 0 && local_i < local_width && local_j >= 0 &&
              local_j < local_height) {

            /* Loop over images and add the contribution to the local buffer. */
            for (int nimg = 0; nimg < nimgs; nimg++) {
              int local_idx =
                  local_i * local_height * nimgs + local_j * nimgs + nimg;
              if (local_idx >= 0 && local_idx < local_img.size()) {
                local_img[local_idx] +=
                    kvalue * pix_values[part->index * nimgs + nimg];
              }
            }
          }
        }
      }
    }

    /* Now copy the local buffer contents back to the global image. We need
     * to use atomic operations here because multiple threads may be writing
     * to overlapping regions of the global image due to kernel overlap
     * between neighboring cells. */
    for (int local_i = 0; local_i < local_width; local_i++) {
      for (int local_j = 0; local_j < local_height; local_j++) {

        /* Convert local coordinates back to global pixel indices. */
        int global_i = i_min + local_i;
        int global_j = j_min + local_j;

        /* Safety check global bounds. */
        if (global_i >= 0 && global_i < npix_x && global_j >= 0 &&
            global_j < npix_y) {

          /* Copy contributions for all images. */
          for (int nimg = 0; nimg < nimgs; nimg++) {
            int local_idx =
                local_i * local_height * nimgs + local_j * nimgs + nimg;
            int global_idx =
                global_i * npix_y * nimgs + global_j * nimgs + nimg;

            /* Atomically add the local contribution to the global image to
             * handle potential race conditions from overlapping kernels. */
#ifdef WITH_OPENMP
#pragma omp atomic
#endif
            out[global_idx] += local_img[local_idx];
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
                                      const int kdim, double norm_factor,
                                      double *img,
                                      const int nimgs, struct cell *root,
                                      const int nthreads) {

  /* Build a balanced work list. */
  std::vector<weighted_cell> work_list =
      build_balanced_work_list(root, nthreads);

  /* Parallel loop over the work list. */
#pragma omp parallel for num_threads(nthreads) schedule(dynamic)
  for (int i = 0; i < work_list.size(); i++) {
    const weighted_cell &wc = work_list[i];
    struct cell *c = wc.cell_ptr;

    /* Populate the pixel recursively. */
    populate_pixel_recursive(c, threshold, kdim, kernel, norm_factor, npart,
                             img, nimgs, pix_values, res, npix_x, npix_y);
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
                                    const int kdim, double norm_factor,
                                    double *img,
                                    const int nimgs, struct cell *root) {

  /* Build a balanced work list (this isn't really necessary in serial,
   * but it keeps the code consistent with the parallel version and has
   * negligible cost and does give cache benefits). */
  std::vector<weighted_cell> work_list = build_balanced_work_list(root, 1);

  /* Loop over the work list. */
  for (int i = 0; i < work_list.size(); i++) {
    const weighted_cell &wc = work_list[i];
    struct cell *c = wc.cell_ptr;

    /* Populate the pixel recursively. */
    populate_pixel_recursive(c, threshold, kdim, kernel, norm_factor, npart,
                             img, nimgs, pix_values, res, npix_x, npix_y);
  }
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

  /* Compute normalization to conserve flux when truncating kernel. */
  double norm_factor = compute_kernel_norm(kernel, kdim, threshold);

  /* If we have multiple threads and OpenMP we can parallelise. */
#ifdef WITH_OPENMP
  if (nthreads > 1) {

    /* Populate the image in parallel. */
    populate_smoothed_image_parallel(pix_values, kernel, res, npix_x, npix_y,
                                     npart, threshold, kdim, norm_factor, img,
                                     nimgs, root, nthreads);

  } else {

    /* If we don't have OpenMP call the serial version. */
    populate_smoothed_image_serial(pix_values, kernel, res, npix_x, npix_y,
                                   npart, threshold, kdim, norm_factor, img,
                                   nimgs, root);
  }
#else
  (void)nthreads;

  /* If we don't have OpenMP call the serial version. */
  populate_smoothed_image_serial(pix_values, kernel, res, npix_x, npix_y, npart,
                                 threshold, kdim, norm_factor, img, nimgs,
                                 root);
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

  /* Construct the cell tree. */
  construct_cell_tree(pos, smoothing_lengths, smoothing_lengths, npart, root,
                      ncells, MAX_DEPTH, 100);

  toc("Constructing cell tree", tree_start);

  double out_start = tic();

  /* Create the zeroed image numpy array. */
  npy_intp np_img_dims[3] = {npix_x, npix_y, nimgs};
  PyArrayObject *np_img =
      (PyArrayObject *)PyArray_ZEROS(3, np_img_dims, NPY_DOUBLE, 0);
  double *img = (double *)PyArray_DATA(np_img);

  toc("Creating output image array", out_start);

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
