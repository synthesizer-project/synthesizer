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
 * output precision (and was applied at float64 instead), and when an output
 * value overflowed the output precision anyway. The Python wrappers reset
 * these before a kernel runs and turn them into warnings afterwards. */
inline std::atomic<bool> weight_rescaled{false};
inline std::atomic<bool> output_overflowed{false};

/**
 * @brief Split a particle weight so it can be applied in the output precision.
 *
 * Kernels multiply the grid spectra by the particle weight in the output
 * precision. A weight too large for that precision (e.g. a bolometric
 * luminosity of ~1e45 erg/s at float32) would become inf even when the
 * resulting spectra fit. Such weights are split into a mantissa, used in the
 * kernel as normal, and a power-of-two scale, applied to the finished row at
 * float64 by apply_weight_scale. Weights that fit are returned unchanged, so
 * the common case costs a single comparison.
 *
 * @param weight: The particle weight.
 * @param scale: Set to the scale to apply afterwards (1 if none is needed).
 *
 * @return The weight to use in the kernel.
 */
template <typename OutT>
static inline OutT split_weight(double weight, double &scale) {
  if (std::fabs(weight) <=
      static_cast<double>(std::numeric_limits<OutT>::max())) {
    scale = 1.0;
    return static_cast<OutT>(weight);
  }
  int exponent;
  const double mantissa = std::frexp(weight, &exponent);
  scale = std::ldexp(1.0, exponent);
  weight_rescaled.store(true, std::memory_order_relaxed);
  return static_cast<OutT>(mantissa);
}

/**
 * @brief Apply the scale from split_weight to a finished output row.
 *
 * The multiplication is done at float64 and only the result is rounded to
 * the output precision. Any value that still cannot be represented is flagged.
 *
 * @param row: The output row.
 * @param n: The number of elements in the row.
 * @param scale: The scale returned by split_weight.
 */
template <typename OutT>
static inline void apply_weight_scale(OutT *row, size_t n, double scale) {
  if (scale == 1.0) {
    return;
  }
  const double max_value =
      static_cast<double>(std::numeric_limits<OutT>::max());
  bool overflowed = false;
  for (size_t i = 0; i < n; i++) {
    const double value = static_cast<double>(row[i]) * scale;
    overflowed |= std::fabs(value) > max_value;
    row[i] = static_cast<OutT>(value);
  }
  if (overflowed) {
    output_overflowed.store(true, std::memory_order_relaxed);
  }
}

/**
 * @brief Add a float64 accumulation, times a split_weight scale, to a row.
 *
 * Used by kernels which accumulate each particle's contribution at float64
 * before storing it (e.g. the Doppler-shifted spectra). Any value too large
 * for the output precision is flagged.
 *
 * @param row: The output row to add to.
 * @param values: The float64 accumulated values.
 * @param n: The number of elements in the row.
 * @param scale: The scale returned by split_weight.
 */
template <typename OutT>
static inline void add_scaled_row(OutT *row, const double *values, size_t n,
                                  double scale) {
  const double max_value =
      static_cast<double>(std::numeric_limits<OutT>::max());
  bool overflowed = false;
  for (size_t i = 0; i < n; i++) {
    const double value = values[i] * scale;
    overflowed |= std::fabs(value) > max_value;
    row[i] += static_cast<OutT>(value);
  }
  if (overflowed) {
    output_overflowed.store(true, std::memory_order_relaxed);
  }
}

#endif  // FLOATING_POINT_UTILS_H_
