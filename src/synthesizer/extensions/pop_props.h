/******************************************************************************
 * A header defining the Populations struct used by the parametric extensions.
 *
 * This is the parametric sibling of the Particles class (see part_props.h).
 * Where a particle is a point in the grid's N dimensional space, a
 * population is a set of N dimensional bins (boxes), each holding a mass
 * that is uniformly distributed within it. All populations share the same
 * bin edges along each axis and differ only in their masses.
 *
 * The struct is a plain holder of validated pointers, all validation of the
 * Python inputs happens where it is filled (see parametric_spectra.cpp).
 *****************************************************************************/
#ifndef POP_PROPS_H_
#define POP_PROPS_H_

/* Standard includes */
#include <array>

/* Python includes */
#define PY_ARRAY_UNIQUE_SYMBOL SYNTHESIZER_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy_init.h"

#include <Python.h>

/* Local includes */
#include "grid_props.h"

struct Populations {
  /* The number of populations. */
  int npop = 0;

  /* The number of axes (always equal to the grid's ndim). */
  int ndim = 0;

  /* The number of bins along each axis (one less than the number of edges).
   */
  std::array<int, MAX_GRID_NDIM> nbins = {};

  /* The number of bins in a single population (product of nbins). */
  int ncells = 0;

  /* The bin edges along each axis, in linear (physical) units. These are
   * read at the population dtype. */
  std::array<const void *, MAX_GRID_NDIM> edges = {};

  /* Is the grid axis matching each bin axis in log10 space? */
  std::array<bool, MAX_GRID_NDIM> log_axis = {};

  /* The masses with shape (npop, nbins[0], ..., nbins[ndim - 1]). */
  const void *masses = nullptr;

  /* The optional mask with the same shape as the masses (true means the bin
   * is included), or nullptr for no mask. */
  const npy_bool *mask = nullptr;

  /* The shared floating-point dtype of the edges and masses. */
  int float_typenum = NPY_FLOAT64;

  /* Get the bin edges along an axis at the population dtype. */
  template <typename Real>
  const Real *get_edges(int idim) const {
    return static_cast<const Real *>(edges[idim]);
  }

  /* Get the masses at the population dtype. */
  template <typename Real>
  const Real *get_masses() const {
    return static_cast<const Real *>(masses);
  }

  /* Is a bin (flat index across all populations) masked out? */
  bool bin_is_masked(size_t ind) const {
    return mask != nullptr && !mask[ind];
  }
};

#endif  // POP_PROPS_H_
