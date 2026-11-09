/******************************************************************************
 * A header defining the Populations class used by the parametric extensions.
 *
 * This is the parametric sibling of the Particles class (see part_props.h).
 * Where a particle is a point in the grid's N dimensional space, a
 * population is a set of N dimensional bins (boxes), each holding a mass
 * that is uniformly distributed within it. All populations share the same
 * bin edges along each axis and differ only in their masses.
 *
 * The inputs (edges, masses and mask) are ordered by the populations' own
 * bin axes, which needn't match the grid's axis order or units. Everything
 * per axis in the class is ordered like the grid, with bin_axis giving the
 * input axis matching each grid axis and unit_scale the factor converting
 * that axis' edges into the grid axis' units.
 *****************************************************************************/
#ifndef POP_PROPS_H_
#define POP_PROPS_H_

/* Standard includes */
#include <array>
#include <string>

/* Python includes */
#define PY_ARRAY_UNIQUE_SYMBOL SYNTHESIZER_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy_init.h"

#include <Python.h>

/* Local includes */
#include "grid_props.h"
#include "index_utils.h"

/**
 * @brief A class to hold the bins of a set of parametric populations.
 *
 * This is used to hold the bin edges, masses and mask of the populations.
 */
class Populations {
 public:
  /* The number of populations. */
  int npop;

  /* The number of axes (always equal to the grid's ndim). */
  int ndim;

  /* The number of bins along each grid axis (one less than the number of
   * edges). */
  std::array<int, MAX_GRID_NDIM> nbins;

  /* The number of bins in a single population (product of nbins). */
  int ncells;

  /* Constructor */
  Populations(PyObject *edges_tuple, PyArrayObject *np_masses,
              PyObject *log_flags, PyObject *bin_axes, PyObject *unit_scales,
              PyObject *mask_obj, PyObject *prop_names, int ndim);

  /**
   * @brief Get the resolved floating-point dtype of the edges and masses.
   *
   * @return The numpy type number.
   */
  int get_float_typenum() const { return float_typenum_; }

  /**
   * @brief Get the bin edges along a grid axis at the population dtype.
   *
   * @tparam Real The floating-point type of the population arrays.
   * @param idim: The grid axis index.
   *
   * @return A pointer to the (nbins[idim] + 1) bin edges along the axis.
   */
  template <typename Real>
  const Real *get_edges(int idim) const {
    return static_cast<const Real *>(edges_[idim]);
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
    return static_cast<const Real *>(masses_);
  }

  /**
   * @brief Are the grid coordinates along a grid axis log10 values?
   *
   * @param idim: The grid axis index.
   *
   * @return True if the grid axis is in log10 space.
   */
  bool is_log_axis(int idim) const { return log_axis_[idim]; }

  /**
   * @brief Get the factor converting a grid axis' edges into its units.
   *
   * @param idim: The grid axis index.
   *
   * @return The unit conversion factor.
   */
  double get_unit_scale(int idim) const { return unit_scale_[idim]; }

  /**
   * @brief Get the lower bound of the mask's range along a grid axis.
   *
   * @param idim: The grid axis index.
   *
   * @return The lower bound (in the edges' units).
   */
  double get_mask_lo(int idim) const { return mask_lo_[idim]; }

  /**
   * @brief Get the upper bound of the mask's range along a grid axis.
   *
   * @param idim: The grid axis index.
   *
   * @return The upper bound (in the edges' units).
   */
  double get_mask_hi(int idim) const { return mask_hi_[idim]; }

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
    get_indices_from_flat(icell, ndim, input_nbins_, input_inds);

    /* Reorder it to match the grid's axes. */
    for (int idim = 0; idim < ndim; idim++) {
      bin_inds[idim] = input_inds[bin_axis_[idim]];
    }
  }

 private:
  /* The input (bin) axis matching each grid axis. */
  std::array<int, MAX_GRID_NDIM> bin_axis_;

  /* The number of bins along each input axis, i.e. the masses' layout. */
  std::array<int, MAX_GRID_NDIM> input_nbins_;

  /* The bin edges along each grid axis, in linear (physical) units before
   * scaling by unit_scale_. These are read at the population dtype. */
  std::array<const void *, MAX_GRID_NDIM> edges_;

  /* Is the grid axis matching each bin axis in log10 space? */
  std::array<bool, MAX_GRID_NDIM> log_axis_;

  /* The factor converting each grid axis' edges (and mask bounds) into the
   * grid axis' units. */
  std::array<double, MAX_GRID_NDIM> unit_scale_;

  /* The masses with shape (npop, input_nbins_[0], ...,
   * input_nbins_[ndim - 1]). */
  const void *masses_;

  /* The range [mask_lo_, mask_hi_] (linear units before scaling by
   * unit_scale_, inclusive) along each grid axis that bins are masked to.
   * Only the part of a bin inside the range contributes, and with no mask
   * the range covers every finite value.
   *
   * NOTE: The extensions are compiled with -ffast-math, which makes
   * infinities undefined behaviour, so the bounds must always be finite. */
  std::array<double, MAX_GRID_NDIM> mask_lo_;
  std::array<double, MAX_GRID_NDIM> mask_hi_;

  /* The name of each grid axis (for error messages). */
  std::array<std::string, MAX_GRID_NDIM> axis_names_;

  /* The shared floating-point dtype of the edges and masses. */
  int float_typenum_ = -1;
};

#endif  // POP_PROPS_H_
