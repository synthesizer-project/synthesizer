/******************************************************************************
 * C extension to calculate line of sight metal surface densities for star
particles.
/*****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/ndarraytypes.h>
#include <numpy/ndarrayobject.h>

/* Define a macro to handle that bzero is non-standard. */
#define bzero(b,len) (memset((b), '\0', (len)), (void) 0)

/**
 * @brief Computes the line of sight metal surface densities for each of the
 *        stars passed to this function.
 *
 * @param 
 */
PyObject *compute_dust_surface_dens(PyObject *self, PyObject *args) {

  const int nstar, ngas, kdim;
  const double threshold;
  const PyArrayObject *np_kernel, *np_star_pos, *np_gas_pos, *np_gas_sml;
  const PyArrayObject *np_gas_met, *np_gas_mass, *np_gas_dtm;

  if(!PyArg_ParseTuple(args, "OOOOOOOiiid", &np_kernel, &np_star_pos,
                       np_gas_pos, &np_gas_sml, &np_gas_met, &np_gas_mass,
                       &np_gas_dtm, &nstar, &ngas, &kdim, &threshold))
    return NULL;

  /* Quick check to make sure our inputs are valid. */
  if (nstar == 0) return NULL;
  if (ngas == 0) return NULL;
  if (kdim == 0) return NULL;

  /* Set up arrays to hold the surface densities themselves. */
  double *los_dustsds = malloc(nstar * sizeof(double));
  bzero(los_dustsds, nstar * sizeof(double));

  /* Extract a pointers to the actual data in the numpy arrays. */
  const double *kernel = PyArray_DATA(np_kernel);
  const double *star_pos = PyArray_DATA(np_star_pos);
  const double *gas_pos = PyArray_DATA(np_gas_pos);
  const double *gas_sml = PyArray_DATA(np_gas_sml);
  const double *gas_met = PyArray_DATA(np_gas_met);
  const double *gas_mass = PyArray_DATA(np_gas_mass);
  const double *gas_dtm = PyArray_DATA(np_gas_dtm);

  /* Loop over stars */
  for (int istar = 0; istar < nstar; istar++) {
    for (int igas = 0; igas < ngas; igas++) {

      /* Skip straight away if the gas particle is behind the star. */
      if (gas_pos[igas * 3 + 2] > star_pos[istar * 3 + 2])
        continue;

      /* Calculate the x and y separations. */
      double x = gas_pos[igas * 3 + 0] - star_pos[istar * 3 + 0];
      double y = gas_pos[igas * 3 + 1] - star_pos[istar * 3 + 1];

      /* Early skip if the star doesn't fall in the gas particles kernel. */
      if (abs(x) > threshold * gas_sml[igas] ||
          abs(y) > threshold * gas_sml[igas])
        continue;

      /* Convert separation to distance. */
      double rsqu = x * x + y * y;

      /* Calculate the impact parameter. */
      double sml_squ = gas_sml[igas] * gas_sml[igas];
      double q = rsqu / sml_squ;

      /* Skip gas particles outside the kernel. */
      if (q > threshold) continue;

      /* Get the value of the kernel at q. */
      int index = kdim * q;
      double kvalue = kernel[index];

      /* Finally, compute the metal surface density itself. */
      los_dustsds[istar] += gas_dtm[igas] * gas_mass[igas] *
        gas_met[igas] / sml_squ * kvalue;
      
    }
  }

  /* Reconstruct the python array to return. */
  npy_intp np_dims[1] = {nstar,};
  PyArrayObject *out_los_dustsds =
    (PyArrayObject *) PyArray_SimpleNewFromData(1, np_dims, NPY_FLOAT64,
                                                los_dustsds);

  return Py_BuildValue("N", out_los_metsds);
}

/* Below is all the gubbins needed to make the module importable in Python. */
static PyMethodDef LosMethods[] = {
  {"compute_dust_surface_dens", compute_metal_surface_dens, METH_VARARGS,
   "Method for calculating line of sight metal surface densities."},
  {NULL, NULL, 0, NULL} 
};

/* Make this importable. */
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "los_dust_surface_dens",                                /* m_name */
        "A module to calculate los metal surface densities",   /* m_doc */
        -1,                                                    /* m_size */
        LosMethods,                                            /* m_methods */
        NULL,                                                  /* m_reload */
        NULL,                                                  /* m_traverse */
        NULL,                                                  /* m_clear */
        NULL,                                                  /* m_free */
    };

PyMODINIT_FUNC PyInit_los(void) {
    PyObject *m = PyModule_Create(&moduledef);
    import_array();
    return m;
}
