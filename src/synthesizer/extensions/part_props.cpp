// Standard includes
#include <array>
#include <limits>
#include <vector>

#ifdef WITH_OPENMP
#include <omp.h>
#endif

// Local includes
#include "part_props.h"
#include "timers.h"

// Declare the GridProps class to avoid circular dependency.
class GridProps;

/**
 * @brief Constructor for the particles class.
 *
 * @param np_weights: The numpy array holding the particle weights.
 * @param np_velocities: The numpy array holding the particle velocities.
 * @param np_mask: The numpy array holding the particle mask.
 * @param part_tuple: The tuple of numpy arrays holding the particle properties.
 */
Particles::Particles(PyArrayObject *np_weights, PyArrayObject *np_velocities,
                     PyArrayObject *np_mask, PyObject *part_tuple, int npart_)
    : np_weights_(np_weights), np_velocities_(np_velocities), np_mask_(np_mask),
      part_tuple_(part_tuple) {

  double start_time = tic();

  /* Assign the number of particles. */
  npart = npart_;

  toc("Constructing C++ Particles object", start_time);
}

/**
 * @brief Destructor for the particles class.
 */
Particles::~Particles() {
  /* We don't own the numpy arrays, so we don't need to free them. */
  np_weights_ = NULL;
  np_velocities_ = NULL;
  np_mask_ = NULL;

  /* The part_tuple is a tuple of numpy arrays, we don't own it either. */
  part_tuple_ = NULL;

  /* We don't need to do anything else here, the numpy arrays will be freed
   * automatically when the Python objects are destroyed. */
  /* Note: If we had allocated any memory in this class, we would free it here,
   * but we don't own the numpy arrays, so we don't need to do anything. */
}

/**
 * @brief Get the weights of the particles.
 *
 * @return The weights of the particles.
 */
double *Particles::get_weights() const {
  return (double *)PyArray_DATA(np_weights_);
}

/**
 * @brief Get the velocities of the particles.
 *
 * @return The velocities of the particles.
 */
double *Particles::get_velocities() const {
  return (double *)PyArray_DATA(np_velocities_);
}

/**
 * @brief Get the properties of the particles.
 *
 * @return The properties of the particles.
 */
double **Particles::get_all_props(int ndim) const {
  /* Allocate a single array for particle properties. */
  double **part_props = new double *[ndim];
  if (part_props == NULL) {
    PyErr_SetString(PyExc_MemoryError,
                    "Failed to allocate memory for part_props.");
    return NULL;
  }

  /* Unpack the particle property arrays into a single contiguous array. */
  for (int idim = 0; idim < ndim; idim++) {

    /* Extract the data from the numpy array. */
    PyArrayObject *np_part_arr =
        (PyArrayObject *)PyTuple_GetItem(part_tuple_, idim);
    if (np_part_arr == NULL) {
      PyErr_SetString(PyExc_ValueError, "Failed to extract part_arr.");
      return NULL;
    }
    part_props[idim] = (double *)PyArray_DATA(np_part_arr);
  }

  /* Success. */
  return part_props;
}

/**
 * @brief Get the properties of the particles.
 *
 * @return The properties of the particles.
 */
double *Particles::get_part_props(int idim) const {
  /* Get the array stored at idim. */
  PyArrayObject *np_part_arr =
      (PyArrayObject *)PyTuple_GetItem(part_tuple_, idim);
  if (np_part_arr == NULL) {
    PyErr_SetString(PyExc_ValueError, "Failed to extract part_arr.");
    return NULL;
  }

  /* Extract the data from the numpy array. */
  double *part_arr = (double *)PyArray_DATA(np_part_arr);
  if (part_arr == NULL) {
    PyErr_SetString(PyExc_ValueError, "Failed to extract part_arr.");
    return NULL;
  }
  return part_arr;
}

/**
 * @brief Get the weight of a particle at a given index.
 *
 * @param pind: The index of the particle.
 * @return The weight of the particle at the given index.
 */
double Particles::get_weight_at(int pind) const {
  return get_double_at(np_weights_, pind);
}

/**
 * @brief Get the velocity of a particle at a given index.
 *
 * @param pind: The index of the particle.
 * @return The velocity of the particle at the given index.
 */
double Particles::get_vel_at(int pind) const {
  return get_double_at(np_velocities_, pind);
}

/**
 * @brief Get the mask of a particle at a given index.
 *
 * @param pind: The index of the particle.
 * @return The mask of the particle at the given index.
 */
npy_bool Particles::get_mask_at(int pind) const {
  /* If the mask is NULL, return true (i.e. not masked). */
  if (np_mask_ == NULL) {
    return true;
  }

  /* If the mask is Py_None, return true (i.e. not masked). */
  if (reinterpret_cast<PyObject *>(np_mask_) == Py_None) {
    return true;
  }

  /* Otherwise, is this element masked? */
  return get_bool_at(np_mask_, pind);
}

/**
 * @brief Get the property of a particle at a given index.
 *
 * @param idim: The index of the property.
 * @param pind: The index of the particle.
 * @return The property of the particle at the given index.
 */
double Particles::get_part_prop_at(int idim, int pind) const {
  /* Get the array stored at idim. */
  PyArrayObject *np_part_arr =
      (PyArrayObject *)PyTuple_GetItem(part_tuple_, idim);
  if (np_part_arr == NULL) {
    PyErr_SetString(PyExc_ValueError, "Failed to extract part_arr.");
    return std::numeric_limits<double>::quiet_NaN();
  }

  /* If we have a size 1 array then we have a fixed scalar value. In this case
   * we return the first element. */
  if (PyArray_SIZE(np_part_arr) == 1) {
    return get_double_at(np_part_arr, 0);
  }

  return get_double_at(np_part_arr, pind);
}

/**
 * @brief Check if a particle is masked.
 *
 * @param pind: The index of the particle.
 * @return True if the particle is masked, false otherwise.
 */
bool Particles::part_is_masked(int pind) const {
  /* If the mask is NULL, return false (i.e. not masked). */
  if (np_mask_ == NULL) {
    return false;
  }

  /* If the mask is Py_None, return false (i.e. not masked). */
  if (reinterpret_cast<PyObject *>(np_mask_) == Py_None) {
    return false;
  }

  /* Otherwise, is this element masked? */
  return !get_bool_at(np_mask_, pind);
}

/**
 * @brief Get the grid indices and fractions for a particle using CIC.
 *
 * This function computes the indices of the grid cells that a particle
 * occupies, along with the fractions of the particle's mass in each cell.
 *
 * This is the serial version of the function.
 *
 * @param GridProps: The properties of the grid.
 * @param parts: The particle properties.
 */
static void get_particle_indices_and_fracs_serial(GridProps *grid_props,
                                                  Particles *parts) {

  /* Unpack the grid properties. */
  const int ndim = grid_props->ndim;
  int nlam = grid_props->nlam;

  // Pre-allocate exactly npart slots to avoid resizing
  parts->grid_indices.resize(parts->npart);
  parts->grid_fracs.resize(parts->npart * ndim);

  /* Loop over particles. */
  for (int p = 0; p < parts->npart; p++) {

    /* Skip masked particles. */
    if (parts->part_is_masked(p)) {
      parts->grid_indices[p] = -1;
      for (int idim = 0; idim < ndim; idim++) {
        parts->grid_fracs[p * ndim + idim] = 0.0;
      }
      continue;
    }

    /* Get the grid indices and cell fractions for the particle. */
    std::array<int, MAX_GRID_NDIM> part_indices;
    std::array<double, MAX_GRID_NDIM> axis_fracs;
    get_part_ind_frac_cic(part_indices, axis_fracs, grid_props, parts, p);

    /* Compute base linear index for this particle */
    const int grid_ind = grid_props->ravel_grid_index(part_indices);

    /* Store the grid indices and fractions in the particle class. */
    parts->grid_indices[p] = grid_ind;

    /* Store the per‐dimension fraction in the particle class. */
    for (int idim = 0; idim < ndim; idim++) {
      parts->grid_fracs[p * ndim + idim] = axis_fracs[idim];
    }
  }
}

/**
 * @brief Get the grid indices and fractions for a particle using CIC.
 *
 * This function computes the indices of the grid cells that a particle
 * occupies, along with the fractions of the particle's mass in each cell.
 *
 * This is the parallel version of the function.
 *
 * @param grid_props: The properties of the grid.
 * @param parts: The particle properties.
 * @param nthreads: The number of threads to use for parallel processing.
 */
static void get_particle_indices_and_fracs_parallel(GridProps *grid_props,
                                                    Particles *parts,
                                                    int nthreads) {

  // Unpack the grid properties.
  const int ndim = grid_props->ndim;
  const int npart = parts->npart;

  // Pre-allocate exactly npart slots so we can write into slices on
  // each thread without resizing.
  parts->grid_indices.resize(npart);
  parts->grid_fracs.resize(npart * ndim);

#pragma omp parallel for num_threads(nthreads) schedule(static)
  // Loop over particles in parallel
  for (int p = 0; p < npart; p++) {

    // Skip masked particles
    if (parts->part_is_masked(p)) {
      parts->grid_indices[p] = -1;
      for (int d = 0; d < ndim; d++) {
        parts->grid_fracs[p * ndim + d] = 0.0;
      }
      continue;
    }

    // Get the grid indices and cell fractions for the particle.
    std::array<int, MAX_GRID_NDIM> part_indices;
    std::array<double, MAX_GRID_NDIM> axis_fracs;
    get_part_ind_frac_cic(part_indices, axis_fracs, grid_props, parts, p);

    // Compute base linear index for this particle
    int grid_ind = grid_props->ravel_grid_index(part_indices);
    parts->grid_indices[p] = grid_ind;

    // Store the per‐dimension fractions
    for (int d = 0; d < ndim; d++) {
      parts->grid_fracs[p * ndim + d] = axis_fracs[d];
    }
  }
}

/**
 * @brief Calculate the grid indices and fractions for all particles.
 *
 * This is a wrapper function that calls the correct version based on
 * the number of threads requested or whether OpenMP is available.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param nthreads: The number of threads to use.
 */
void get_particle_indices_and_fracs(GridProps *grid_props, Particles *parts,
                                    int nthreads) {

  double start = tic();

#ifdef WITH_OPENMP
  if (nthreads > 1) {
    get_particle_indices_and_fracs_parallel(grid_props, parts, nthreads);
  } else {
    get_particle_indices_and_fracs_serial(grid_props, parts);
  }
#else
  get_particle_indices_and_fracs_serial(grid_props, parts);
#endif /* WITH_OPENMP */

  toc("Finding particle grid indices and fractions", start);
}

/**
 * @brief Get the grid indices for a particle using Nearest Grid Point (NGP).
 *
 * This is the serial version of the function.
 *
 * @param GridProps: The properties of the grid.
 * @param parts: The particle properties.
 */
static void get_particle_indices_serial(GridProps *grid_props,
                                        Particles *parts) {

  /* Unpack the grid properties. */
  const int ndim = grid_props->ndim;
  int nlam = grid_props->nlam;

  // Pre-allocate exactly npart slots to avoid resizing
  parts->grid_indices.resize(parts->npart);

  /* Loop over particles. */
  for (int p = 0; p < parts->npart; p++) {

    /* Skip masked particles. */
    if (parts->part_is_masked(p)) {
      parts->grid_indices[p] = -1;
      continue;
    }

    /* Get the grid indices and cell fractions for the particle. */
    std::array<int, MAX_GRID_NDIM> part_indices;
    get_part_inds_ngp(part_indices, grid_props, parts, p);

    /* Compute the flattened grid index for this particle and store it. */
    const int grid_ind = grid_props->ravel_grid_index(part_indices);
    parts->grid_indices[p] = grid_ind;
  }
}

/**
 * @brief Get the grid indices for a particle using Nearest Grid Point (NGP).
 *
 * This is the parallel version of the function.
 *
 * @param grid_props: The properties of the grid.
 * @param parts: The particle properties.
 * @param nthreads: The number of threads to use for parallel processing.
 */
static void get_particle_indices_parallel(GridProps *grid_props,
                                          Particles *parts, int nthreads) {
  const int ndim = grid_props->ndim;
  const int npart = parts->npart;

  // Pre-allocate exactly npart slots so we can write into slices on
  // each thread without resizing.
  parts->grid_indices.resize(npart);

#pragma omp parallel for num_threads(nthreads) schedule(static)
  // Loop over particles in parallel
  for (int p = 0; p < npart; p++) {

    // Skip masked particles
    if (parts->part_is_masked(p)) {
      parts->grid_indices[p] = -1;
      continue;
    }

    // Get the grid indices for the particle.
    std::array<int, MAX_GRID_NDIM> part_indices;
    get_part_inds_ngp(part_indices, grid_props, parts, p);

    // Compute the flattened grid index for this particle and store it.
    int grid_ind = grid_props->ravel_grid_index(part_indices);
    parts->grid_indices[p] = grid_ind;
  }
}

/**
 * @brief Calculate the grid indices for all particles.
 *
 * This is a wrapper function that calls the correct version based on
 * the number of threads requested or whether OpenMP is available.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param nthreads: The number of threads to use.
 */
void get_particle_indices(GridProps *grid_props, Particles *parts,
                          int nthreads) {

  double start = tic();

#ifdef WITH_OPENMP
  if (nthreads > 1) {
    get_particle_indices_parallel(grid_props, parts, nthreads);
  } else {
    get_particle_indices_serial(grid_props, parts);
  }
#else
  get_particle_indices_serial(grid_props, parts);
#endif /* WITH_OPENMP */

  toc("Finding particle grid indices", start);
}
