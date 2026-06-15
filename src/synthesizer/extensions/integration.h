#ifndef INTEGRATION_H_
#define INTEGRATION_H_

#include <cstddef>

/**
 * @brief Integrate one 1D function using composite Simpson's rule.
 *
 * The abscissae are supplied explicitly so this helper can be reused by both
 * generic array integrations and specialised kernel builders. The spacing must
 * be uniform.
 *
 * @tparam XReal The floating-point type of the x array.
 * @tparam YReal The floating-point type of the y array.
 * @tparam OutT The floating-point type of the returned integral.
 *
 * @param x 1D array of x values.
 * @param y 1D array of y values.
 * @param n Number of samples.
 *
 * @return The integrated value.
 */
template <typename XReal, typename YReal, typename OutT = YReal>
inline OutT simps_1d(const XReal *x, const YReal *y, size_t n) {
  OutT integral = static_cast<OutT>(0.0);

  if (n < 2) {
    return integral;
  }

  /* Apply Simpson's rule over pairs of intervals. */
  for (size_t j = 0; j < (n - 1) / 2; ++j) {
    const size_t k = 2 * j;
    const OutT h0 = static_cast<OutT>(x[k + 1] - x[k]);
    const OutT h1 = static_cast<OutT>(x[k + 2] - x[k + 1]);

    /* Skip degenerate intervals rather than dividing by zero. */
    if (h0 == static_cast<OutT>(0.0) || h1 == static_cast<OutT>(0.0)) {
      continue;
    }

    integral +=
        (h0 + h1) / static_cast<OutT>(6.0) *
        ((static_cast<OutT>(2.0) - h1 / h0) * static_cast<OutT>(y[k]) +
         (((h0 + h1) * (h0 + h1)) / (h0 * h1)) * static_cast<OutT>(y[k + 1]) +
         (static_cast<OutT>(2.0) - h0 / h1) * static_cast<OutT>(y[k + 2]));
  }

  /* Finish with a trapezoidal tail when the sample count leaves one interval
   * uncovered by the Simpson pairs. */
  if ((n - 1) % 2 != 0) {
    integral += static_cast<OutT>(0.5) *
                static_cast<OutT>(x[n - 1] - x[n - 2]) *
                (static_cast<OutT>(y[n - 1]) + static_cast<OutT>(y[n - 2]));
  }

  return integral;
}

/**
 * @brief Integrate one 1D function using the trapezoidal rule.
 *
 * @tparam XReal The floating-point type of the x array.
 * @tparam YReal The floating-point type of the y array.
 * @tparam OutT The floating-point type of the returned integral.
 *
 * @param x 1D array of x values.
 * @param y 1D array of y values.
 * @param n Number of samples.
 *
 * @return The integrated value.
 */
template <typename XReal, typename YReal, typename OutT = YReal>
inline OutT trapz_1d(const XReal *x, const YReal *y, size_t n) {
  OutT integral = static_cast<OutT>(0.0);

  if (n < 2) {
    return integral;
  }

  /* Sum the trapezoidal contribution from every interval. */
  for (size_t j = 0; j < n - 1; ++j) {
    integral += static_cast<OutT>(0.5) * static_cast<OutT>(x[j + 1] - x[j]) *
                (static_cast<OutT>(y[j + 1]) + static_cast<OutT>(y[j]));
  }

  return integral;
}

#endif  // INTEGRATION_H_
