/******************************************************************************
 * A header containing helper functions for interpolating kernel lookup tables
 * used by the LOS surface density extensions.
 *****************************************************************************/
#ifndef KERNEL_UTILS_H_
#define KERNEL_UTILS_H_

/* C++ includes. */
#include <algorithm>
#include <cmath>

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
 * @brief Locate the interpolation cell on a uniform tabulated axis.
 *
 * The overlap kernel uses uniformly sampled q and u axes. For those grids we
 * can replace the generic monotonic-grid search with a direct affine mapping
 * from coordinate to tabulated index, removing the binary search cost from the
 * hot pair-evaluation path.
 *
 * @param value The coordinate to locate.
 * @param min_value The lower coordinate bound of the grid.
 * @param max_value The upper coordinate bound of the grid.
 * @param dim The number of grid entries.
 * @param index The returned lower interpolation index.
 * @param frac The returned fractional position within that cell.
 */
static inline void get_uniform_interp_index(const double value,
                                            const double min_value,
                                            const double max_value,
                                            const int dim, int &index,
                                            double &frac) {

  if (dim <= 1) {
    index = 0;
    frac = 0.0;
    return;
  }

  if (value <= min_value) {
    index = 0;
    frac = 0.0;
    return;
  }
  if (value >= max_value) {
    index = dim - 2;
    frac = 1.0;
    return;
  }

  const double scaled =
      (value - min_value) * (static_cast<double>(dim - 1) / (max_value - min_value));
  index = static_cast<int>(scaled);
  frac = scaled - index;
}

/**
 * @brief Locate the interpolation cell on a logarithmically uniform axis.
 *
 * The overlap kernel's eta axis is built with `geomspace`, so the tabulated
 * entries are uniform in log(eta). This helper converts eta to log-space and
 * performs the same direct affine mapping used for the uniform q and u axes.
 *
 * @param value The coordinate to locate.
 * @param min_value The lower coordinate bound of the grid.
 * @param max_value The upper coordinate bound of the grid.
 * @param dim The number of grid entries.
 * @param index The returned lower interpolation index.
 * @param frac The returned fractional position within that cell.
 */
static inline void get_geometric_interp_index(const double value,
                                              const double min_value,
                                              const double max_value,
                                              const int dim, int &index,
                                              double &frac) {

  if (dim <= 1) {
    index = 0;
    frac = 0.0;
    return;
  }

  if (value <= min_value) {
    index = 0;
    frac = 0.0;
    return;
  }
  if (value >= max_value) {
    index = dim - 2;
    frac = 1.0;
    return;
  }

  const double log_min = std::log(min_value);
  const double log_max = std::log(max_value);
  const double scaled = (std::log(value) - log_min) *
                        (static_cast<double>(dim - 1) / (log_max - log_min));
  index = static_cast<int>(scaled);
  frac = scaled - index;
}

/**
 * @brief Get a value from the truncated LOS kernel lookup table.
 *
 * The truncated LOS kernel is a 2D table tabulated as a function of the
 * dimensionless projected separation q and the dimensionless LOS coordinate z.
 * It stores the cumulative foreground contribution of a source kernel up to a
 * given LOS coordinate and is used when building the smoothed-input overlap
 * table.
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

  /* Clamp the LOS coordinate to the tabulated support range. */
  const double clamped_z = std::max(-1.0, std::min(1.0, z));

  /* Locate the interpolation cell in projected separation. */
  const double scaled_q = q * (kdim - 1);
  const int q_index = static_cast<int>(scaled_q);
  const int q_next = std::min(kdim - 1, q_index + 1);
  const double q_frac = scaled_q - q_index;

  /* Locate the interpolation cell in LOS coordinate. */
  const double scaled_z = 0.5 * (clamped_z + 1.0) * (zdim - 1);
  const int z_index = static_cast<int>(scaled_z);
  const int z_next = std::min(zdim - 1, z_index + 1);
  const double z_frac = scaled_z - z_index;

  /* Bilinearly interpolate the cumulative kernel table. */
  const double v00 = kernel[q_index * zdim + z_index];
  const double v01 = kernel[q_index * zdim + z_next];
  const double v10 = kernel[q_next * zdim + z_index];
  const double v11 = kernel[q_next * zdim + z_next];

  const double vz0 = v00 + z_frac * (v01 - v00);
  const double vz1 = v10 + z_frac * (v11 - v10);
  return vz0 + q_frac * (vz1 - vz0);
}

/**
 * @brief Get a value from the smoothed-input overlap kernel lookup table.
 *
 * The overlap table is stored in row-major order with the axes arranged as
 * `(q, u, eta)`, matching the NumPy layout of a C-contiguous array with shape
 * `(qdim, udim, etadim)`.
 *
 * The coordinate definitions are:
 *
 *   q   = b / (R_i + R_j)
 *   u   = (z_i - z_j) / (R_i + R_j)
 *   eta = h_i / h_j
 *
 * where `b` is the projected particle-centre separation, `R_i` and `R_j` are
 * the support radii of the input and source particles, and `h_i` and `h_j` are
 * their smoothing lengths.
 *
 * Return semantics follow the physical support of the overlap problem:
 *
 * - `q >= 1` implies no projected overlap, so the function returns zero;
 * - `u <= -1` implies the source kernel is entirely behind the input kernel,
 *   so the function returns zero;
 * - `u >= 1` implies the source kernel is entirely in front, so the lookup is
 *   clamped to the saturated `u = 1` face of the table;
 * - `eta` values outside the tabulated range are clamped to the nearest table
 *   boundary.
 *
 * Interpolation inside the table uses standard CIC / trilinear weights.
 * Because the tabulated q and u axes are uniform and the eta axis is uniform in
 * log-space, the interpolation cell is located with direct index arithmetic
 * rather than generic binary searches.
 *
 * @param kernel The overlap kernel table.
 * @param q_grid The projected-overlap coordinate grid.
 * @param u_grid The LOS-offset coordinate grid.
 * @param eta_grid The smoothing-length-ratio grid.
 * @param qdim The number of q-grid entries.
 * @param udim The number of u-grid entries.
 * @param etadim The number of eta-grid entries.
 * @param q The dimensionless projected separation.
 * @param u The dimensionless LOS offset.
 * @param eta The smoothing-length ratio.
 *
 * @return The interpolated overlap-kernel value.
 */
static inline double get_overlap_kernel_value(
    const double *kernel, const double *q_grid, const double *u_grid,
    const double *eta_grid, const int qdim, const int udim, const int etadim,
    const double q, const double u, const double eta) {

  /* Outside the projected or LOS support there is no contribution. */
  if (q < 0.0 || q >= 1.0 || u <= -1.0) {
    return 0.0;
  }

  /* Clamp the LOS offset to the saturated fully-front face of the table. */
  const double clamped_u = std::min(u, 1.0);

  /* Locate the interpolation cell on each axis. */
  int q_index = 0;
  int u_index = 0;
  int eta_index = 0;
  double q_frac = 0.0;
  double u_frac = 0.0;
  double eta_frac = 0.0;
  get_uniform_interp_index(q, q_grid[0], q_grid[qdim - 1], qdim, q_index,
                           q_frac);
  get_uniform_interp_index(clamped_u, u_grid[0], u_grid[udim - 1], udim,
                           u_index, u_frac);
  get_geometric_interp_index(eta, eta_grid[0], eta_grid[etadim - 1], etadim,
                             eta_index, eta_frac);

  /* Get the upper interpolation indices, handling edge-clamped lookups. */
  const int q_next = std::min(qdim - 1, q_index + 1);
  const int u_next = std::min(udim - 1, u_index + 1);
  const int eta_next = std::min(etadim - 1, eta_index + 1);

  /* Helper lambda to flatten the row-major (q, u, eta) indexing. */
  const auto value_at = [&](const int iq, const int iu, const int ieta) {
    return kernel[(iq * udim + iu) * etadim + ieta];
  };

  /* Interpolate along the eta direction first on each q/u corner. */
  const double c000 = value_at(q_index, u_index, eta_index);
  const double c001 = value_at(q_index, u_index, eta_next);
  const double c010 = value_at(q_index, u_next, eta_index);
  const double c011 = value_at(q_index, u_next, eta_next);
  const double c100 = value_at(q_next, u_index, eta_index);
  const double c101 = value_at(q_next, u_index, eta_next);
  const double c110 = value_at(q_next, u_next, eta_index);
  const double c111 = value_at(q_next, u_next, eta_next);

  const double c00 = c000 + eta_frac * (c001 - c000);
  const double c01 = c010 + eta_frac * (c011 - c010);
  const double c10 = c100 + eta_frac * (c101 - c100);
  const double c11 = c110 + eta_frac * (c111 - c110);

  /* Then interpolate along the LOS-offset direction. */
  const double c0 = c00 + u_frac * (c01 - c00);
  const double c1 = c10 + u_frac * (c11 - c10);

  /* Finally interpolate along the projected-separation direction. */
  return c0 + q_frac * (c1 - c0);
}

#endif // KERNEL_UTILS_H_
