/******************************************************************************
 * A header containing helper functions for interpolating kernel lookup tables
 * used by the LOS surface density extensions.
 *****************************************************************************/
#ifndef KERNEL_UTILS_H_
#define KERNEL_UTILS_H_

/* C++ includes. */
#include <algorithm>

/* Define the number of sample points per axis used to average the LOS column
 * density across the support of a smoothed input particle. */
#define LOS_SMOOTHED_INPUT_NDIM 16

/**
 * @brief Get the value of a 1D kernel lookup table at a dimensionless radius.
 *
 * The LOS machinery tabulates kernels on a uniform grid in the dimensionless
 * radius q = r / h. This helper performs a simple linear interpolation on that
 * table.
 *
 * Values outside the tabulated support are taken to be zero.
 *
 * @param kernel The 1D kernel lookup table.
 * @param kdim The number of entries in the kernel lookup table.
 * @param q The dimensionless radius at which to evaluate the kernel.
 *
 * @return The interpolated kernel value.
 */
static inline double get_kernel_value(const double *kernel, const int kdim,
                                      const double q) {

  /* Early exit for outside the kernel. */
  if (q < 0.0 || q >= 1.0) {
    return 0.0;
  }

  /* Map the dimensionless radius to the kernel.*/
  const double scaled = q * (kdim - 1);
  const int index = static_cast<int>(scaled);
  const int next_index = std::min(kdim - 1, index + 1);

  /* Calculate the fractional distance between the two kernel samples. */
  const double frac = scaled - index;

  /* Linearly interpolate and return the kernel value. */
  return kernel[index] + frac * (kernel[next_index] - kernel[index]);
}

/**
 * @brief Get a value from the truncated LOS kernel lookup table.
 *
 * This function is used when only a fraction of the source particle kernel
 * contributes to the foreground column density at the sampled point.
 *
 * The truncated LOS kernel is a 2D table tabulated as a function of the
 * dimensionless projected separation q and the dimensionless LOS coordinate z.
 *
 * The LOS coordinate is clamped to the tabulated range [-1, 1], which matches
 * the support-normalised kernel extent used when constructing the table.
 * Values outside the projected support remain zero.
 *
 * @param kernel The 2D truncated LOS kernel lookup table stored in row-major
 *        order with projected-separation index first.
 * @param kdim The number of projected-separation entries.
 * @param zdim The number of LOS-coordinate entries.
 * @param q The dimensionless projected impact parameter.
 * @param z The dimensionless LOS coordinate.
 *
 * @return The interpolated cumulative LOS kernel value.
 */
static inline double get_truncated_kernel_value(const double *kernel,
                                                const int kdim, const int zdim,
                                                const double q,
                                                const double z) {

  /* Early exit for outside the projected kernel support. */
  if (q < 0.0 || q >= 1.0) {
    return 0.0;
  }

  /* Clamp the LOS coordinate to the tabulated range. */
  const double clamped_z = std::max(-1.0, std::min(1.0, z));

  /* Map the dimensionless radius to the kernel.*/
  const double scaled_q = q * (kdim - 1);
  const int q_index = static_cast<int>(scaled_q);
  const int q_next = std::min(kdim - 1, q_index + 1);

  /* Calculate the fractional distance between the two kernel samples. */
  const double q_frac = scaled_q - q_index;

  /* Map the clamped LOS coordinate to the kernel. */
  const double scaled_z = 0.5 * (clamped_z + 1.0) * (zdim - 1);
  const int z_index = static_cast<int>(scaled_z);
  const int z_next = std::min(zdim - 1, z_index + 1);

  /* Calculate the fractional distance between the two kernel samples. */
  const double z_frac = scaled_z - z_index;

  /* Perform bilinear interpolation on the 2D kernel table. */
  const double v00 = kernel[q_index * zdim + z_index];
  const double v01 = kernel[q_index * zdim + z_next];
  const double v10 = kernel[q_next * zdim + z_index];
  const double v11 = kernel[q_next * zdim + z_next];

  /* Interpolate in the LOS coordinate direction first. */
  const double vz0 = v00 + z_frac * (v01 - v00);
  const double vz1 = v10 + z_frac * (v11 - v10);

  /* Then interpolate in the projected-separation direction and return. */
  return vz0 + q_frac * (vz1 - vz0);
}

#endif // KERNEL_UTILS_H_
