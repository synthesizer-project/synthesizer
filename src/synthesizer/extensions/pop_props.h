/******************************************************************************
 * A header defining the Populations struct used by the parametric extensions.
 *
 * This is the parametric sibling of the Particles class (see part_props.h).
 * Where a particle is a point in the grid's N dimensional space, a
 * population is a set of N dimensional bins (boxes), each holding a mass
 * that is uniformly distributed within it. All populations share the same
 * bin edges along each axis and differ only in their masses.
 *
 * The inputs (edges, masses and mask) are ordered by the populations' own
 * bin axes, which needn't match the grid's axis order or units. Everything
 * per axis in the struct is ordered like the grid, with bin_axis giving the
 * input axis matching each grid axis and unit_scale the factor converting
 * that axis' edges into the grid axis' units.
 *
 * The struct is a plain holder of validated pointers, all validation of the
 * Python inputs happens where it is filled (see parametric_spectra.cpp).
 *****************************************************************************/
#ifndef POP_PROPS_H_
#define POP_PROPS_H_

/* Standard includes */
#include <array>
#include <limits>

/* Python includes */
#define PY_ARRAY_UNIQUE_SYMBOL SYNTHESIZER_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy_init.h"

#include <Python.h>

/* Local includes */
#include "grid_props.h"
#include "index_utils.h"

struct Populations {
  /* The number of populations. */
  int npop = 0;

  /* The number of axes (always equal to the grid's ndim). */
  int ndim = 0;

  /* The input (bin) axis matching each grid axis. */
  std::array<int, MAX_GRID_NDIM> bin_axis = {};

  /* The number of bins along each grid axis (one less than the number of
   * edges). */
  std::array<int, MAX_GRID_NDIM> nbins = {};

  /* The number of bins along each input axis, i.e. the masses' layout. */
  std::array<int, MAX_GRID_NDIM> input_nbins = {};

  /* The factor converting each grid axis' edges (and mask bounds) into the
   * grid axis' units. */
  std::array<double, MAX_GRID_NDIM> unit_scale = {};

  /* The number of bins in a single population (product of nbins). */
  int ncells = 0;

  /* The bin edges along each grid axis, in linear (physical) units before
   * scaling by unit_scale. These are read at the population dtype. */
  std::array<const void *, MAX_GRID_NDIM> edges = {};

  /* Is the grid axis matching each bin axis in log10 space? */
  std::array<bool, MAX_GRID_NDIM> log_axis = {};

  /* The masses with shape (npop, input_nbins[0], ...,
   * input_nbins[ndim - 1]). */
  const void *masses = nullptr;

  /* The range [mask_lo, mask_hi] (linear units before scaling by
   * unit_scale, inclusive) along each grid axis that bins are masked to. Only
   * the part of a bin inside the range contributes, and with no mask the range
   * covers every finite value.
   *
   * NOTE: The extensions are compiled with -ffast-math, which makes
   * infinities undefined behaviour, so the bounds must always be finite. */
  std::array<double, MAX_GRID_NDIM> mask_lo;
  std::array<double, MAX_GRID_NDIM> mask_hi;

  /* The shared floating-point dtype of the edges and masses. */
  int float_typenum = NPY_FLOAT64;

  /**
   * @brief Construct an empty set of populations with no mask.
   */
  Populations() {
    /* Start with a range covering every finite value along every axis. */
    mask_lo.fill(-std::numeric_limits<double>::max());
    mask_hi.fill(std::numeric_limits<double>::max());
  }

  /**
   * @brief Get the bin edges along an axis at the population dtype.
   *
   * @tparam Real The floating-point type of the population arrays.
   * @param idim: The axis index.
   *
   * @return A pointer to the (nbins[idim] + 1) bin edges along the axis.
   */
  template <typename Real>
  const Real *get_edges(int idim) const {
    return static_cast<const Real *>(edges[idim]);
  }

  /**
   * @brief Get the masses at the population dtype.
   *
   * @tparam Real The floating-point type of the population arrays.
   *
   * @return A pointer to the (npop * ncells) masses.
   */
  template <typename Real>
  const Real *get_masses() const {
    return static_cast<const Real *>(masses);
  }

  /**
   * @brief Get a bin's index along each grid axis.
   *
   * @param icell: The bin's flat index within its population (in the
   *               masses' layout).
   * @param bin_inds: The output index along each grid axis.
   */
  void get_bin_inds(int icell,
                    std::array<int, MAX_GRID_NDIM> &bin_inds) const {
    /* Unravel the index in the masses' (input axis) layout. */
    std::array<int, MAX_GRID_NDIM> input_inds;
    get_indices_from_flat(icell, ndim, input_nbins, input_inds);

    /* Reorder it to match the grid's axes. */
    for (int idim = 0; idim < ndim; idim++) {
      bin_inds[idim] = input_inds[bin_axis[idim]];
    }
  }
};

#endif  // POP_PROPS_H_
