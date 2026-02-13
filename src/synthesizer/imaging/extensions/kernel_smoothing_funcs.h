/******************************************************************************
 * Shared kernel functions for SPH imaging
 *
 * This header contains reusable functions for SPH kernel interpolation and
 * pixel-kernel overlap calculations that are used by both 2D imaging and
 * spectral cube generation.
 *****************************************************************************/

#ifndef KERNEL_SMOOTHING_FUNCS_H
#define KERNEL_SMOOTHING_FUNCS_H

#include <algorithm>
#include <cmath>

/* Include octree header for particle struct definition. */
#include "../../extensions/octree.h"

/**
 * @brief Inline function for kernel interpolation.
 *
 * This function interpolates the value of the kernel at a given distance
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
 * @brief Compute normalization factor so truncated kernels conserve flux.
 *
 * This function calculates the ratio between the full kernel integral and
 * the truncated kernel integral (up to threshold), allowing us to rescale
 * kernel values to ensure flux conservation when using a finite support.
 *
 * @param kernel The precomputed kernel look up table.
 * @param kdim The dimension of the kernel array.
 * @param threshold The threshold value for kernel truncation.
 *
 * @return Normalization factor to multiply kernel values by.
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
    /* Integrate 2π q K(q) dq via trapezoid; ∫q dq over [q0,q1] =
     * 0.5(q1^2-q0^2) */
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
  /* Rescale so total integral over [0,1] is unity, and partial region
   * matches. */
  return full_integral / partial_integral;
}

/**
 * @brief Check if the entire kernel support lies within the pixel.
 *
 * @param part Pointer to the particle.
 * @param pix_x_min Pixel minimum x coordinate.
 * @param pix_x_max Pixel maximum x coordinate.
 * @param pix_y_min Pixel minimum y coordinate.
 * @param pix_y_max Pixel maximum y coordinate.
 * @param kernel_radius Kernel support radius (sml * threshold).
 *
 * @return True if kernel is fully inside pixel, false otherwise.
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
 * @brief Calculate contribution when entire pixel is inside the kernel.
 *
 * When the pixel is wholly contained within the kernel support, we evaluate
 * the kernel using a 3x3 grid and apply proper SPH normalization.
 *
 * @param pix_x_min Pixel minimum x coordinate.
 * @param pix_y_min Pixel minimum y coordinate.
 * @param res Pixel resolution.
 * @param part Pointer to the particle.
 * @param kernel Pointer to kernel interpolation table.
 * @param kdim Dimension of kernel table.
 * @param threshold Kernel threshold value.
 *
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
 * @brief Calculate contribution when pixel is partially inside the kernel.
 *
 * When the pixel straddles the kernel boundary, we use adaptive grid
 * sampling to accurately integrate the kernel contribution over the pixel
 * area.
 *
 * @param part Pointer to the particle.
 * @param pix_x_min Pixel minimum x coordinate.
 * @param pix_y_min Pixel minimum y coordinate.
 * @param kernel Pointer to kernel interpolation table.
 * @param kdim Dimension of kernel table.
 * @param threshold Kernel threshold value.
 * @param res Pixel resolution.
 *
 * @return The kernel contribution value (normalized).
 */
inline double pixel_kernel_partial_overlap_contribution(
    const struct particle *part, double pix_x_min, double pix_y_min,
    const double *kernel, int kdim, double threshold, double res) {

  /* Adaptive sampling based on smoothing length.
   * Minimum 4 samples per axis, increase for small smoothing lengths. */
  int n_sub = std::max(4, static_cast<int>(ceil(2.0 * res / part->sml)));

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

#endif /* KERNEL_SMOOTHING_FUNCS_H */
