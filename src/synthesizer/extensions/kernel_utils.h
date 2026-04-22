#ifndef KERNEL_UTILS_H_
#define KERNEL_UTILS_H_

/* C++ includes. */
#include <algorithm>

/**
 * @brief Get the value of a 1D kernel lookup table at a dimensionless radius.
 *
 * @param kernel The 1D kernel lookup table.
 * @param kdim The number of entries in the kernel table.
 * @param q The dimensionless radius at which to evaluate the table.
 *
 * @return The interpolated kernel value.
 */
static inline double get_kernel_value(const double *kernel, const int kdim,
                                      const double q) {

  /* Outside the support radius the kernel vanishes. */
  if (q < 0.0 || q >= 1.0) {
    return 0.0;
  }

  /* Map the dimensionless radius directly onto the tabulated grid. */
  const double scaled_q = q * (kdim - 1);
  const int q_index = static_cast<int>(scaled_q);
  const int q_next = std::min(kdim - 1, q_index + 1);
  const double q_frac = scaled_q - q_index;

  const double v0 = kernel[q_index];
  const double v1 = kernel[q_next];
  return v0 + q_frac * (v1 - v0);
}

/**
 * @brief Get a value from the truncated LOS kernel lookup table.
 *
 * The truncated LOS kernel is a 2D table tabulated as a function of the
 * dimensionless projected separation q and the dimensionless LOS coordinate z.
 * It stores the cumulative source contribution up to a given LOS truncation
 * coordinate and is used when an input particle lies inside the source kernel.
 *
 * @param kernel The 2D truncated LOS kernel lookup table stored in row-major
 *        order with projected-separation index first.
 * @param kdim The number of projected-separation entries.
 * @param zdim The number of LOS-coordinate entries.
 * @param q The dimensionless projected impact parameter.
 * @param z The dimensionless LOS truncation coordinate.
 *
 * @return The interpolated cumulative LOS kernel value.
 */
static inline double get_truncated_kernel_value(const double *kernel,
                                                const int kdim, const int zdim,
                                                const double q,
                                                const double z) {

  /* Outside the projected kernel support the contribution vanishes. */
  if (q < 0.0 || q >= 1.0) {
    return 0.0;
  }

  /* Clamp the LOS truncation coordinate to the tabulated support range. */
  const double clamped_z = std::max(-1.0, std::min(1.0, z));

  /* Locate the projected-separation interpolation cell. */
  const double scaled_q = q * (kdim - 1);
  const int q_index = static_cast<int>(scaled_q);
  const int q_next = std::min(kdim - 1, q_index + 1);
  const double q_frac = scaled_q - q_index;

  /* Locate the LOS truncation interpolation cell. */
  const double scaled_z = 0.5 * (clamped_z + 1.0) * (zdim - 1);
  const int z_index = static_cast<int>(scaled_z);
  const int z_next = std::min(zdim - 1, z_index + 1);
  const double z_frac = scaled_z - z_index;

  /* Bilinearly interpolate the table. */
  const double v00 = kernel[q_index * zdim + z_index];
  const double v01 = kernel[q_index * zdim + z_next];
  const double v10 = kernel[q_next * zdim + z_index];
  const double v11 = kernel[q_next * zdim + z_next];

  const double vz0 = v00 + z_frac * (v01 - v00);
  const double vz1 = v10 + z_frac * (v11 - v10);
  return vz0 + q_frac * (vz1 - vz0);
}

#endif // KERNEL_UTILS_H_
