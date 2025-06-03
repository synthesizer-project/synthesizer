
#include "part_props.h"
#include "timers.h"

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
  /* Deallocate the numpy arrays. */
  // Py_XDECREF(np_weights_);
  // Py_XDECREF(np_velocities_);
  // Py_XDECREF(np_mask_);
  // Py_XDECREF(part_tuple_);
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
  double **part_props =
      reinterpret_cast<double **>(malloc(ndim * sizeof(double *)));
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
    return -1.0;
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
