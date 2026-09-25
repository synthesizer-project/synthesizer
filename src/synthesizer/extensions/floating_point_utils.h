#ifndef FLOATING_POINT_UTILS_H_
#define FLOATING_POINT_UTILS_H_

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

/**
 * @brief Test for NaN without relying on floating-point compiler semantics.
 *
 * Synthesizer builds with finite-math optimisations, which may fold
 * ``std::isnan`` to false. Inspecting the IEEE-754 representation preserves
 * NaN handling under those flags.
 */
template <typename Real>
static inline bool is_nan_bits(Real value) {

  /* Check that the type is either 32-bit or 64-bit float, we don't support
   * other sizes here. */
  static_assert(
      sizeof(Real) == sizeof(uint32_t) || sizeof(Real) == sizeof(uint64_t),
      "is_nan_bits only supports 32-bit and 64-bit floats");

  /* Check the IEEE-754 representation of the float to determine if
   * it is NaN. */
  if constexpr (sizeof(Real) == sizeof(uint32_t)) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT32_C(0x7fffffff)) > UINT32_C(0x7f800000);
  } else {
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT64_C(0x7fffffffffffffff)) >
           UINT64_C(0x7ff0000000000000);
  }
}

/* Set by the kernels when a particle weight could not be represented in the
 * output precision (and was applied at float64 instead). The Python wrappers
 * clear it before a kernel runs and warn if it was set afterwards. */
inline std::atomic<bool> weight_rescaled{false};

/**
 * @brief Set whether any particle weight has been rescaled.
 *
 * @param value: The new value of the flag.
 */
static inline void set_weight_rescaled(bool value) {
  weight_rescaled.store(value, std::memory_order_relaxed);
}

/**
 * @brief Get whether any particle weight has been rescaled.
 *
 * @return True if a weight was rescaled since the flag was last cleared.
 */
static inline bool get_weight_rescaled() {
  return weight_rescaled.load(std::memory_order_relaxed);
}

/**
 * @brief Get a particle weight that can be applied in the output precision.
 *
 * Kernels multiply the grid spectra by the particle weight in the output
 * precision. A weight too large for that precision (e.g. a bolometric
 * luminosity of ~1e45 erg/s at float32) would become inf even when the
 * resulting spectra fit. Such weights are split into a mantissa, used in the
 * kernel as normal, and a power-of-two scale, applied to the finished row at
 * float64 by set_scaled_row. Weights that fit are returned unchanged, so the
 * common case costs a single comparison.
 *
 * @param weight: The particle weight.
 * @param scale: Set to the scale to apply afterwards (1 if none is needed).
 *
 * @return The weight to use in the kernel.
 */
template <typename OutT>
static inline OutT get_split_weight(double weight, double &scale) {
  if (std::fabs(weight) <=
      static_cast<double>(std::numeric_limits<OutT>::max())) {
    scale = 1.0;
    return static_cast<OutT>(weight);
  }
  int exponent;
  const double mantissa = std::frexp(weight, &exponent);
  scale = std::ldexp(1.0, exponent);
  set_weight_rescaled(true);
  return static_cast<OutT>(mantissa);
}

/**
 * @brief Set a finished output row to its values times a weight scale.
 *
 * Applies the scale from get_split_weight. The multiplication is done at
 * float64 and only the result is rounded to the output precision.
 *
 * @param row: The output row.
 * @param n: The number of elements in the row.
 * @param scale: The scale returned by get_split_weight.
 */
template <typename OutT>
static inline void set_scaled_row(OutT *row, size_t n, double scale) {
  if (scale == 1.0) {
    return;
  }
  for (size_t i = 0; i < n; i++) {
    row[i] = static_cast<OutT>(static_cast<double>(row[i]) * scale);
  }
}

/**
 * @brief Set an output row to itself plus float64 values times a scale.
 *
 * Used by kernels which accumulate each particle's contribution at float64
 * before storing it (e.g. the Doppler-shifted spectra), applying the scale
 * from get_split_weight at float64.
 *
 * @param row: The output row to add to.
 * @param values: The float64 accumulated values.
 * @param n: The number of elements in the row.
 * @param scale: The scale returned by get_split_weight.
 */
template <typename OutT>
static inline void set_accumulated_row(OutT *row, const double *values,
                                       size_t n, double scale) {
  for (size_t i = 0; i < n; i++) {
    row[i] += static_cast<OutT>(values[i] * scale);
  }
}

#endif  // FLOATING_POINT_UTILS_H_
