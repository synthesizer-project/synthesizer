#ifndef PART_PROPS_H_
#define PART_PROPS_H_

/* Standard includes */
#include <stdlib.h>
#include <string>
#include <vector>

/* Python includes */
#define PY_ARRAY_UNIQUE_SYMBOL SYNTHESIZER_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy_init.h"
#include <Python.h>

/* Local includes */
#include "property_funcs.h"
#include "python_to_cpp.h"

/**
 * @brief A class to hold particle related numpy arrays with getters and
 * setters.
 *
 * This is used to hold the particle properties and mass.
 */
class Particles {
public:
  /* The number of particles. */
  int npart;

  /* Constructor */
  Particles(PyArrayObject *np_weights, PyArrayObject *np_velocities,
            PyArrayObject *np_mask, PyObject *part_tuple,
            PyObject *part_names_tuple, int npart);

  /* Destructor */
  ~Particles();

  /* Prototypes for getters. */
  double *get_weights() const;
  double *get_velocities() const;
  double **get_all_props(int ndim) const;
  double *get_part_props(int idim) const;
  double get_weight_at(int pind) const;
  double get_vel_at(int pind) const;
  npy_bool get_mask_at(int pind) const;
  double get_part_prop_at(int idim, int pind) const;

  /* The resolved floating-point dtype for particle arrays. */
  int get_float_typenum() const;

  /* Accessors for validated float32/float64 arrays. */
  template <typename Real> const Real *get_weights() const {
    return data_ptr<const Real>(np_weights_);
  }

  template <typename Real> const Real *get_velocities() const {
    if (np_velocities_ == NULL || reinterpret_cast<PyObject *>(np_velocities_) == Py_None) {
      return NULL;
    }
    return data_ptr<const Real>(np_velocities_);
  }

  template <typename Real> const Real *get_part_props(int idim) const {
    PyArrayObject *np_part_arr =
        (PyArrayObject *)PyTuple_GetItem(part_tuple_, idim);
    if (np_part_arr == NULL) {
      PyErr_SetString(PyExc_ValueError,
                      "[Particles::get_part_props]: Failed to extract part_arr.");
      return NULL;
    }
    return data_ptr<const Real>(np_part_arr);
  }

  template <typename Real> Real get_weight_at(int pind) const {
    if (pind < 0 || pind >= npart) {
      PyErr_Format(PyExc_IndexError,
                   "[Particles::get_weight_at]: Index (%d) out of bounds for weights. "
                   "Valid range is [0, %d).",
                   pind, npart);
      return static_cast<Real>(0);
    }
    return get_weights<Real>()[pind];
  }

  template <typename Real> Real get_vel_at(int pind) const {
    if (np_velocities_ == NULL || reinterpret_cast<PyObject *>(np_velocities_) == Py_None) {
      PyErr_SetString(PyExc_ValueError,
                      "[Particles::get_vel_at]: Velocities were not provided.");
      return static_cast<Real>(0);
    }

    if (pind < 0 || pind >= npart) {
      PyErr_Format(PyExc_IndexError,
                   "[Particles::get_vel_at]: Index (%d) out of bounds for velocities. "
                   "Valid range is [0, %d).",
                   pind, npart);
      return static_cast<Real>(0);
    }

    return get_velocities<Real>()[pind];
  }

  template <typename Real> Real get_part_prop_at(int idim, int pind) const {
    PyArrayObject *np_part_arr =
        (PyArrayObject *)PyTuple_GetItem(part_tuple_, idim);
    if (np_part_arr == NULL) {
      PyErr_SetString(PyExc_ValueError,
                      "[Particles::get_part_prop_at]: Failed to extract part_arr.");
      return static_cast<Real>(0);
    }

    /* If we have a size 1 array then we have a fixed scalar value. */
    if (PyArray_SIZE(np_part_arr) == 1) {
      return data_ptr<const Real>(np_part_arr)[0];
    }

    if (pind < 0 || pind >= npart) {
      PyErr_Format(PyExc_IndexError,
                   "[Particles::get_part_prop_at]: Index (%d) out of bounds for particle properties. "
                   "Valid range is [0, %d).",
                   pind, npart);
      return static_cast<Real>(0);
    }

    return data_ptr<const Real>(np_part_arr)[pind];
  }

  /* Is a particle masked? */
  bool part_is_masked(int pind) const;

 private:
  /* The numpy array holding the particle weights (e.g. initial mass for
   * SPS grid weighting). */
  PyArrayObject *np_weights_;

  /* The numpy array holding the particle velocities. */
  PyArrayObject *np_velocities_;

  /* The mask (can be Py_None). */
  PyArrayObject *np_mask_;

  /* The particle properties corresponding to the grid axes, this is a tuple
   * of numpy arrays. */
  PyObject *part_tuple_;

  /* Names for the particle property arrays. */
  std::vector<std::string> part_names_;

  /* The shared floating-point dtype used by the particle arrays. */
  int float_typenum_ = -1;
};

#endif // PART_PROPS_H_
