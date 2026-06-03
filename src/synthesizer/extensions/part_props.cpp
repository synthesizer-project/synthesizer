// Standard includes
#include <array>
#include <limits>
#include <vector>

#ifdef WITH_OPENMP
#include <omp.h>
#endif

// Local includes
#include "grid_props.h"
#include "part_props.h"
#include "timers.h"
#include "weights.h"

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
                     PyArrayObject *np_mask, PyObject *part_tuple,
                     PyObject *part_names_tuple, int npart_)
    : np_weights_(np_weights), np_velocities_(np_velocities),
      np_mask_(np_mask), part_tuple_(part_tuple) {

  tic("Particles.__init__");

  /* Assign the number of particles. */
  npart = npart_;

  /* Validate that all floating-point particle inputs are contiguous and share
   * one supported dtype family before any typed kernels use raw pointers. */
  PyArrayObject *float_arrays[MAX_GRID_NDIM + 2] = {NULL};
  const char *float_names[MAX_GRID_NDIM + 2] = {NULL};
  int float_count = 0;

  if (np_weights_ != NULL && reinterpret_cast<PyObject *>(np_weights_) != Py_None) {
    float_arrays[float_count] = np_weights_;
    float_names[float_count] = "weights";
    float_count++;
  }

  if (np_velocities_ != NULL &&
      reinterpret_cast<PyObject *>(np_velocities_) != Py_None) {
    float_arrays[float_count] = np_velocities_;
    float_names[float_count] = "velocities";
    float_count++;
  }

  if (part_tuple_ != NULL && PyTuple_Check(part_tuple_)) {
    const Py_ssize_t n_props = PyTuple_Size(part_tuple_);
    for (Py_ssize_t i = 0; i < n_props; i++) {
      PyObject *item = PyTuple_GetItem(part_tuple_, i);
      if (item == NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "[Particles::Particles]: Failed to extract particle property array.");
        return;
      }
      if (!PyArray_Check(item)) {
        PyErr_SetString(PyExc_TypeError,
                        "[Particles::Particles]: Particle properties must be numpy arrays.");
        return;
      }
      PyArrayObject *np_part_arr = reinterpret_cast<PyArrayObject *>(item);
      float_arrays[float_count] = np_part_arr;
      float_names[float_count] = "particle property";
      float_count++;
    }
  }

  if (float_count > 0 &&
      !is_matching_float_dtypes(float_arrays, float_names, float_count,
                                &float_typenum_)) {
    return;
  }

  if (part_names_tuple != NULL && PySequence_Check(part_names_tuple) &&
      !PyUnicode_Check(part_names_tuple)) {
    Py_ssize_t n_names = PySequence_Size(part_names_tuple);
    if (n_names < 0) {
      PyErr_Clear();
    } else {
      part_names_.reserve(n_names);
      for (Py_ssize_t i = 0; i < n_names; ++i) {
        PyObject *name_obj = PySequence_GetItem(part_names_tuple, i);
        if (name_obj == NULL) {
          PyErr_Clear();
          part_names_.emplace_back("");
          continue;
        }

        if (PyUnicode_Check(name_obj)) {
          const char *name = PyUnicode_AsUTF8(name_obj);
          if (name != NULL) {
            part_names_.emplace_back(name);
          } else {
            PyErr_Clear();
            part_names_.emplace_back("");
          }
        } else {
          part_names_.emplace_back("");
        }

        Py_DECREF(name_obj);
      }
    }
  }

  toc("Particles.__init__");
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
  part_names_.clear();

  /* We don't need to do anything else here, the numpy arrays will be freed
   * automatically when the Python objects are destroyed. */
  /* Note: If we had allocated any memory in this class, we would free it here,
   * but we don't own the numpy arrays, so we don't need to do anything. */
}

/**
 * @brief Get the resolved floating-point dtype used by particle arrays.
 *
 * @return The resolved NumPy typenum, or -1 if no float arrays were provided.
 */
int Particles::get_float_typenum() const { return float_typenum_; }

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
  return get_at<double>(np_weights_, pind, "weights");
}

/**
 * @brief Get the velocity of a particle at a given index.
 *
 * @param pind: The index of the particle.
 * @return The velocity of the particle at the given index.
 */
double Particles::get_vel_at(int pind) const {
  return get_at<double>(np_velocities_, pind, "velocities");
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
  return get_bool_at(np_mask_, pind, "mask");
}

/**
 * @brief Get the property of a particle at a given index.
 *
 * @param idim: The index of the property.
 * @param pind: The index of the particle.
 * @return The property of the particle at the given index.
 */
double Particles::get_part_prop_at(int idim, int pind) const {
  char fallback_name[64];
  fallback_name[0] = '\0';

  if (idim >= 0 && idim < static_cast<int>(part_names_.size()) &&
      !part_names_[idim].empty()) {
    snprintf(fallback_name, sizeof(fallback_name), "%s",
             part_names_[idim].c_str());
  }

  if (fallback_name[0] == '\0') {
    snprintf(fallback_name, sizeof(fallback_name), "particle property %d",
             idim);
  }

  const char *array_name = fallback_name;

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
    return get_at<double>(np_part_arr, 0, array_name);
  }

  return get_at<double>(np_part_arr, pind, array_name);
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
  return !get_bool_at(np_mask_, pind, "mask");
}
