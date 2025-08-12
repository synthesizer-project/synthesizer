#ifndef PART_PROPS_H_
#define PART_PROPS_H_

/* Standard includes */
#include <stdlib.h>

/* Python includes */
#define PY_ARRAY_UNIQUE_SYMBOL SYNTHESIZER_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy_init.h"
#include <Python.h>

/* Local includes */
#include "property_funcs.h"

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
            PyArrayObject *np_mask, PyObject *part_tuple, int npart);

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
};

#endif // PART_PROPS_H_
