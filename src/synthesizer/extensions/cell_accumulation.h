/******************************************************************************
 * Shared inner accumulation for grid-cell weighted spectra extraction.
 *
 * Both the plain and the Doppler-shifted particle spectra extensions combine
 * a handful of grid spectra rows into one per-particle row. The number of
 * contributing cells is only known at runtime (it is 2^ndim minus any cells
 * with a zero CIC fraction), which stops the compiler unrolling the cell loop
 * and therefore stops it vectorising over wavelength. Specialising the counts
 * a CIC patch can actually produce restores vectorisation.
 *****************************************************************************/
#ifndef CELL_ACCUMULATION_H_
#define CELL_ACCUMULATION_H_

#include <cmath>
#include <cstddef>

/**
 * @brief Accumulate the weighted grid rows of one particle.
 *
 * Computes out[ilam] = sum_icell cells[icell][ilam] * weights[icell].
 *
 * @tparam SpecReal The floating-point type of the grid spectra.
 * @tparam OutT The floating-point type of the output buffer.
 *
 * @param cells: The grid spectra rows contributing to this particle.
 * @param weights: The weight applied to each contributing row.
 * @param ncells: The number of contributing rows (1 to 2^ndim).
 * @param out: The destination row.
 * @param nlam: The number of wavelength bins.
 */
template <typename SpecReal, typename OutT>
static inline void accumulate_cell_spectra(const SpecReal *const *cells,
                                           const OutT *weights, int ncells,
                                           OutT *__restrict out, size_t nlam) {
#define SYNTH_ACCUM_CELLS(N)                                         \
  case N: {                                                          \
    const SpecReal *c[N];                                            \
    OutT w[N];                                                       \
    for (int i = 0; i < N; i++) {                                    \
      c[i] = cells[i];                                               \
      w[i] = weights[i];                                             \
    }                                                                \
    _Pragma("omp simd") for (size_t ilam = 0; ilam < nlam; ilam++) { \
      OutT spec_val = static_cast<OutT>(0);                          \
      for (int i = 0; i < N; i++) {                                  \
        spec_val += static_cast<OutT>(c[i][ilam]) * w[i];            \
      }                                                              \
      out[ilam] = spec_val;                                          \
    }                                                                \
    return;                                                          \
  }

  switch (ncells) {
    SYNTH_ACCUM_CELLS(1)
    SYNTH_ACCUM_CELLS(2)
    SYNTH_ACCUM_CELLS(3)
    SYNTH_ACCUM_CELLS(4)
    SYNTH_ACCUM_CELLS(5)
    SYNTH_ACCUM_CELLS(6)
    SYNTH_ACCUM_CELLS(7)
    SYNTH_ACCUM_CELLS(8)
    default:
      break;
  }
#undef SYNTH_ACCUM_CELLS

  /* More cells than we specialise for (grids with ndim > 3). */
  for (size_t ilam = 0; ilam < nlam; ilam++) {
    OutT spec_val = static_cast<OutT>(0);
    for (int icell = 0; icell < ncells; icell++) {
      spec_val = std::fma(static_cast<OutT>(cells[icell][ilam]),
                          weights[icell], spec_val);
    }
    out[ilam] = spec_val;
  }
}

#endif  // CELL_ACCUMULATION_H_
