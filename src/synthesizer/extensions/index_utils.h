#ifndef INDEX_UTILS_H_
#define INDEX_UTILS_H_

#include "grid_props.h"
#include <array>

/**
 * @brief Compute an ndimensional index from a flat index.
 *
 * @param flat_ind: The flattened index to unravel.
 * @param ndim: The number of dimensions for the unraveled index.
 * @param dims: The size of each dimension.
 * @param indices: The output N-dimensional indices.
 */
static inline void
get_indices_from_flat(int flat_ind, int ndim,
                      std::array<int, MAX_GRID_NDIM> dims,
                      std::array<int, MAX_GRID_NDIM> &indices) {

  /* Loop over indices calculating each one. */
  for (int i = ndim - 1; i > -1; i--) {
    indices[i] = flat_ind % dims[i];
    flat_ind /= dims[i];
  }
}

/**
 * @brief Compute an ndimensional index from a flat index.
 *
 * This version accounts for the additional wavelength dimension in the grid
 * properties.
 *
 * @param flat_ind: The flattened index to unravel.
 * @param ndim: The number of dimensions for the unraveled index.
 * @param dims: The size of each dimension.
 * @param indices: The output N-dimensional indices.
 */
static inline void
get_indices_from_flat(int flat_ind, int ndim,
                      std::array<int, MAX_GRID_NDIM + 1> dims,
                      std::array<int, MAX_GRID_NDIM + 1> &indices) {

  /* Loop over indices calculating each one. */
  for (int i = ndim - 1; i > -1; i--) {
    indices[i] = flat_ind % dims[i];
    flat_ind /= dims[i];
  }
}

/**
 * @brief Compute a flat grid index based on the grid dimensions.
 *
 * @param multi_index: An array of N-dimensional indices.
 * @param dims: The length of each dimension.
 * @param ndim: The number of dimensions.
 */
static inline int
get_flat_index(const std::array<int, MAX_GRID_NDIM> multi_index,
               const int *dims, const int ndim) {

  int index = 0, stride = 1;
  for (int i = ndim - 1; i >= 0; i--) {
    index += stride * multi_index[i];
    stride *= dims[i];
  }

  return index;
}

/**
 * @brief Compute a flat grid index based on the grid dimensions.
 *
 * This version accounts for the additional wavelength dimension in the grid
 * properties.
 *
 * @param multi_index: An array of N-dimensional indices.
 * @param dims: The length of each dimension.
 * @param ndim: The number of dimensions.
 */
static inline int
get_flat_index(const std::array<int, MAX_GRID_NDIM + 1> multi_index,
               const int *dims, const int ndim) {
  int index = 0, stride = 1;
  for (int i = ndim - 1; i >= 0; i--) {
    index += stride * multi_index[i];
    stride *= dims[i];
  }

  return index;
}

#endif // INDEX_UTILS_H_
