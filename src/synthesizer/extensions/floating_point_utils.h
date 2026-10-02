#ifndef FLOATING_POINT_UTILS_H_
#define FLOATING_POINT_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

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

/**
 * @brief Check whether any element of an array is +/-inf.
 *
 * The raw bits are read straight from memory as an integer. The extensions
 * build with -ffast-math, so the compiler may assume any float value is
 * finite and fold a test on it (std::isinf, or even a bit test on a loaded
 * float) to false.
 *
 * @tparam Real The floating-point type of the array.
 *
 * @param values: The array values.
 * @param size: The number of elements.
 * @param nthreads: The number of threads to use.
 *
 * @return True if any element is +/-inf.
 */
template <typename Real>
static inline bool contains_inf(const Real *values, size_t size,
                                int nthreads) {
  /* The unsigned integer matching the float's width, and the bit patterns of
   * an infinity once the sign bit is masked off. */
  using Bits =
      typename std::conditional<sizeof(Real) == 4, uint32_t, uint64_t>::type;
  const Bits abs_mask = static_cast<Bits>(~Bits(0) >> 1);
  const Bits inf_bits = sizeof(Real) == 4 ? Bits(UINT32_C(0x7f800000))
                                          : Bits(UINT64_C(0x7ff0000000000000));

  const unsigned char *bytes = reinterpret_cast<const unsigned char *>(values);
  bool found = false;

#ifdef WITH_OPENMP
#pragma omp parallel for num_threads(nthreads) schedule(static) \
    reduction(|| : found)
#else
  (void)nthreads;
#endif
  for (ptrdiff_t index = 0; index < (ptrdiff_t)size; ++index) {
    Bits bits;
    std::memcpy(&bits, bytes + index * sizeof(Real), sizeof(Bits));
    found = found || ((bits & abs_mask) == inf_bits);
  }

  return found;
}

#endif  // FLOATING_POINT_UTILS_H_
