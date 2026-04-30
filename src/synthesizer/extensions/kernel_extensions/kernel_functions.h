/******************************************************************************
 * Kernel function definitions for LOS kernel table construction.
 *
 * All kernel functions assume the public ``Kernel`` convention used on the
 * Python side: the support radius is normalised to unity and the returned
 * value is the 3D kernel density at the requested radius.
 *
 * The dispatch helper ``get_kernel_function`` maps a public kernel name onto
 * the corresponding analytic function so the table builders and Python
 * wrappers all use exactly the same implementation.
 *****************************************************************************/

#ifndef KERNEL_FUNCTIONS_H
#define KERNEL_FUNCTIONS_H

#include "kernels.h"

/**
 * @brief Evaluate the uniform kernel at a dimensionless radius.
 *
 * @param r The dimensionless radius.
 *
 * @return The kernel value.
 */
static inline double uniform(const double r) {
  if (r < 1.0) {
    return 1.0 / ((4.0 / 3.0) * M_PI);
  }
  return 0.0;
}

/**
 * @brief Evaluate the SPH Anarchy kernel at a dimensionless radius.
 *
 * @param r The dimensionless radius.
 *
 * @return The kernel value.
 */
static inline double sph_anarchy(const double r) {
  if (r <= 1.0) {
    const double one_minus_r = 1.0 - r;
    return (21.0 / (2.0 * M_PI)) * (one_minus_r * one_minus_r * one_minus_r *
                                     one_minus_r * (1.0 + 4.0 * r));
  }
  return 0.0;
}

/**
 * @brief Evaluate the Gadget-2 kernel at a dimensionless radius.
 *
 * @param r The dimensionless radius.
 *
 * @return The kernel value.
 */
static inline double gadget_2(const double r) {
  if (r < 0.5) {
    return (8.0 / M_PI) * (1.0 - 6.0 * (r * r) + 6.0 * (r * r * r));
  }
  if (r < 1.0) {
    const double one_minus_r = 1.0 - r;
    return (16.0 / M_PI) * (one_minus_r * one_minus_r * one_minus_r);
  }
  return 0.0;
}

/**
 * @brief Evaluate the cubic kernel at a dimensionless radius.
 *
 * @param r The dimensionless radius.
 *
 * @return The kernel value.
 */
static inline double cubic(const double r) {
  if (r < 0.5) {
    return 2.546479089470 + 15.278874536822 * (r - 1.0) * r * r;
  }
  if (r < 1.0) {
    const double one_minus_r = 1.0 - r;
    return 5.092958178941 * (one_minus_r * one_minus_r * one_minus_r);
  }
  return 0.0;
}

/**
 * @brief Evaluate the quintic kernel at a dimensionless radius.
 *
 * @param r The dimensionless radius.
 *
 * @return The kernel value.
 */
static inline double quintic(const double r) {
  if (r < 0.333333333) {
    return 27.0 *
           (6.4457752 * r * r * r * r * (1.0 - r) - 1.4323945 * r * r +
            0.17507044);
  }
  if (r < 0.666666667) {
    return 27.0 *
           (3.2228876 * r * r * r * r * (r - 3.0) + 10.7429587 * r * r * r -
            5.01338071 * r * r + 0.5968310366 * r + 0.1352817016);
  }
  if (r < 1.0) {
    return 27.0 * 0.64457752 *
           (-r * r * r * r * r + 5.0 * r * r * r * r - 10.0 * r * r * r +
            10.0 * r * r - 5.0 * r + 1.0);
  }
  return 0.0;
}

/**
 * @brief Map a public kernel name onto the corresponding analytic function.
 *
 * Keeping the name dispatch in one place ensures the Python wrappers and the
 * truncated LOS table builder both use exactly the same implementation.
 *
 * @param name The public kernel name.
 *
 * @return Pointer to the matching kernel function, or ``NULL`` if unknown.
 */
static kernel_func get_kernel_function(const char *name) {
  if (strcmp(name, "uniform") == 0) {
    return uniform;
  }
  if (strcmp(name, "sph_anarchy") == 0) {
    return sph_anarchy;
  }
  if (strcmp(name, "gadget_2") == 0) {
    return gadget_2;
  }
  if (strcmp(name, "cubic") == 0) {
    return cubic;
  }
  if (strcmp(name, "quintic") == 0) {
    return quintic;
  }
  return NULL;
}

#endif /* KERNEL_FUNCTIONS_H */
