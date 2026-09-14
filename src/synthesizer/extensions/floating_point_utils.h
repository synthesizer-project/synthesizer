#ifndef FLOATING_POINT_UTILS_H_
#define FLOATING_POINT_UTILS_H_

#include <cstdint>
#include <cstring>

/**
 * @brief Test for NaN without relying on floating-point compiler semantics.
 *
 * Synthesizer builds with finite-math optimisations, which may fold
 * ``std::isnan`` to false. Inspecting the IEEE-754 representation preserves
 * NaN handling under those flags.
 */
template <typename Real>
static inline bool is_nan_bits(Real value) {
  static_assert(
      sizeof(Real) == sizeof(uint32_t) || sizeof(Real) == sizeof(uint64_t),
      "is_nan_bits only supports 32-bit and 64-bit floats");

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

#endif  // FLOATING_POINT_UTILS_H_
