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
template <typename Real>
static inline Real uniform(const Real r) {
  if (r < static_cast<Real>(1.0)) {
    return static_cast<Real>(1.0) / (static_cast<Real>(4.0 / 3.0) * static_cast<Real>(M_PI));
  }
  return static_cast<Real>(0.0);
}

/**
 * @brief Evaluate the SPH Anarchy kernel at a dimensionless radius.
 *
 * @param r The dimensionless radius.
 *
 * @return The kernel value.
 */
template <typename Real>
static inline Real sph_anarchy(const Real r) {
  if (r <= static_cast<Real>(1.0)) {
    const Real one_minus_r = static_cast<Real>(1.0) - r;
    return static_cast<Real>(21.0 / (2.0 * M_PI)) *
           (one_minus_r * one_minus_r * one_minus_r *
            one_minus_r * (static_cast<Real>(1.0) + static_cast<Real>(4.0) * r));
  }
  return static_cast<Real>(0.0);
}

/**
 * @brief Evaluate the Gadget-2 kernel at a dimensionless radius.
 *
 * @param r The dimensionless radius.
 *
 * @return The kernel value.
 */
template <typename Real>
static inline Real gadget_2(const Real r) {
  if (r < static_cast<Real>(0.5)) {
    return static_cast<Real>(8.0 / M_PI) *
           (static_cast<Real>(1.0) - static_cast<Real>(6.0) * (r * r) +
            static_cast<Real>(6.0) * (r * r * r));
  }
  if (r < static_cast<Real>(1.0)) {
    const Real one_minus_r = static_cast<Real>(1.0) - r;
    return static_cast<Real>(16.0 / M_PI) *
           (one_minus_r * one_minus_r * one_minus_r);
  }
  return static_cast<Real>(0.0);
}

/**
 * @brief Evaluate the cubic kernel at a dimensionless radius.
 *
 * @param r The dimensionless radius.
 *
 * @return The kernel value.
 */
template <typename Real>
static inline Real cubic(const Real r) {
  if (r < static_cast<Real>(0.5)) {
    return static_cast<Real>(2.546479089470) +
           static_cast<Real>(15.278874536822) * (r - static_cast<Real>(1.0)) * r * r;
  }
  if (r < static_cast<Real>(1.0)) {
    const Real one_minus_r = static_cast<Real>(1.0) - r;
    return static_cast<Real>(5.092958178941) *
           (one_minus_r * one_minus_r * one_minus_r);
  }
  return static_cast<Real>(0.0);
}

/**
 * @brief Evaluate the quartic spline (M5) kernel at a dimensionless radius.
 *
 * This is the SPHENIX quartic spline from Borrow et al. (2022), rescaled from
 * support radius 5h/2 to the public convention where the support radius is 1.
 *
 * @param r The dimensionless radius.
 *
 * @return The kernel value.
 */
template <typename Real>
static inline Real quartic(const Real r) {
  const Real q = static_cast<Real>(2.5) * r;
  const Real norm = static_cast<Real>(25.0 / (32.0 * M_PI));

  if (q < static_cast<Real>(0.5)) {
    const Real a = static_cast<Real>(2.5) - q;
    const Real b = static_cast<Real>(1.5) - q;
    const Real c = static_cast<Real>(0.5) - q;
    return norm * (a * a * a * a - static_cast<Real>(5.0) * b * b * b * b +
                   static_cast<Real>(10.0) * c * c * c * c);
  }
  if (q < static_cast<Real>(1.5)) {
    const Real a = static_cast<Real>(2.5) - q;
    const Real b = static_cast<Real>(1.5) - q;
    return norm * (a * a * a * a - static_cast<Real>(5.0) * b * b * b * b);
  }
  if (q < static_cast<Real>(2.5)) {
    const Real a = static_cast<Real>(2.5) - q;
    return norm * (a * a * a * a);
  }
  return static_cast<Real>(0.0);
}

/**
 * @brief Evaluate the quintic kernel at a dimensionless radius.
 *
 * @param r The dimensionless radius.
 *
 * @return The kernel value.
 */
template <typename Real>
static inline Real quintic(const Real r) {
  if (r < static_cast<Real>(0.333333333)) {
    return static_cast<Real>(27.0) *
           (static_cast<Real>(6.4457752) * r * r * r * r *
                (static_cast<Real>(1.0) - r) -
            static_cast<Real>(1.4323945) * r * r +
            static_cast<Real>(0.17507044));
  }
  if (r < static_cast<Real>(0.666666667)) {
    return static_cast<Real>(27.0) *
           (static_cast<Real>(3.2228876) * r * r * r * r *
                (r - static_cast<Real>(3.0)) +
            static_cast<Real>(10.7429587) * r * r * r -
            static_cast<Real>(5.01338071) * r * r +
            static_cast<Real>(0.5968310366) * r +
            static_cast<Real>(0.1352817016));
  }
  if (r < static_cast<Real>(1.0)) {
    return static_cast<Real>(27.0) * static_cast<Real>(0.64457752) *
           (-r * r * r * r * r + static_cast<Real>(5.0) * r * r * r * r -
            static_cast<Real>(10.0) * r * r * r +
            static_cast<Real>(10.0) * r * r - static_cast<Real>(5.0) * r +
            static_cast<Real>(1.0));
  }
  return static_cast<Real>(0.0);
}

/**
 * @brief Map a public kernel name onto the corresponding analytic function.
 *
 * Keeping the name dispatch in one place ensures the Python wrappers and the
 * truncated LOS table builder both use exactly the same implementation.
 *
 * @tparam Real The floating-point type (float or double).
 * @param name The public kernel name.
 *
 * @return Pointer to the matching kernel function, or ``NULL`` if unknown.
 */
template <typename Real>
static kernel_func<Real> get_kernel_function(const char *name) {
  if (strcmp(name, "uniform") == 0) {
    return uniform<Real>;
  }
  if (strcmp(name, "sph_anarchy") == 0) {
    return sph_anarchy<Real>;
  }
  if (strcmp(name, "gadget_2") == 0) {
    return gadget_2<Real>;
  }
  if (strcmp(name, "cubic") == 0) {
    return cubic<Real>;
  }
  if (strcmp(name, "quartic") == 0) {
    return quartic<Real>;
  }
  if (strcmp(name, "quintic") == 0) {
    return quintic<Real>;
  }
  return NULL;
}

#endif /* KERNEL_FUNCTIONS_H */
