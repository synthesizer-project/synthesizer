/******************************************************************************
 * Cython extension to calculate SED weights for star particles. 
 * Calculates weights on an age / metallicity grid given the mass.
 * Args: 
 *   z - 1 dimensional array of metallicity values (ascending)
 *   a - 1 dimensional array of age values (ascending) 
 *   particle - 2 dimensional array of particle properties (N, 3) 
 *   first column metallicity, second column age, third column mass 
 * Returns: 
 *   w - 2d array, dimensions (z, a), of weights to apply to SED array 
/*****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <Python.h>

//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/ndarraytypes.h>
#include <numpy/ndarrayobject.h>

/* Define a macro to handle that bzero is non-standard. */
#define bzero(b,len) (memset((b), '\0', (len)), (void) 0)

/**
 * @brief Performs a binary search to find the closest value in a float array
 *        to comparison float value.
 *
 * NOTE: by design this function always returns the lower index of the 2
 *       bounding the value.
 *
 * @param arr: The array to find the index of the closest value in.
 * @param n: The number of entries in arr.
 * @param x: The value you compare against.
 */
int closestIndex(double *arr, int n, double x) {

  /* Define the starting indices. */
  int l = 0, r = n-1;

  /* While we don't have a pair of adjacent indices. */
  while (r - l > 1) {

    /* Define the midpoint. */
    int mid = l + (r - l) / 2;

    /* Where is the midpoint relative to the value? */
    if (arr[mid] <= x) {
      l = mid;
    } else {
      r = mid;
      
    }
  }
  return l + 1;
}

/**
 * @brief Cython extension to calculate SED weights for star particles. 
 *        Calculates weights on an age / metallicity grid given the mass.
 *
 * @param z: The metallicity values (ascending).
 * @param a: The age values (ascending).
 * @param particle_z: Particle metallicity.
 * @param particle_a: Particle age.
 * @param particle_m: Particle initial mass.
 * @param lena: The number of age values in a.
 * @param lenz: The number of metallicity values in z.
 * @param npart: The number of stellar particles.
 */
PyObject *calculate_weights(PyObject *self, PyObject *args) {

  const double *z, *a, *particle_z, *particle_a, *particle_m;
  const int lena, lenz, npart;
  PyArrayObject *np_z, *np_a;
  PyArrayObject *np_particle_z, *np_particle_a, *np_particle_m;

  if(!PyArg_ParseTuple(args, "OOOOOiii", &np_z, &np_a, &np_particle_z,
                       &np_particle_a, &np_particle_m, &lena, &lenz, &npart))
    return NULL;

  int ihigh, ilow, jhigh, jlow;
  double ifrac, jfrac, metal, age, mass, mfrac;

  /* Get pointers to the actual data within the numpy arrays. */
  z = np_z->data;
  a = np_a->data;
  particle_z = np_particle_z->data;
  particle_a = np_particle_a->data;
  particle_m = np_particle_m->data;

  /* Extract a pointer to the data from the numpy array. */
  double *weights = malloc(lena * lenz * sizeof(double));
  bzero(weights, lena * lenz * sizeof(double));

  /* Loop over particles. */
  for (int p = 0; p < npart; p++) {

    /* Get this particles properties. */
    metal = particle_z[p];
    age = particle_a[p];
    mass = particle_m[p];

    /* Get the cloest value in metallicity. The lower bound is always
     * returned. */
    ilow = closestIndex(a, lena, age);

    /* Are outside the array? */
    /* TODO: For values outside the range should we return any weight? */
    if (ilow == 0) {
      ihigh = ilow;  /* set upper index to lower. */
      ifrac = 0;  /* set fraction to zero. */
    } else if (ilow == lena) {
      ilow -= 1;  /* set upper index to the array length. */
      ihigh = ilow;
      ifrac = 0; /* set fraction to zero. */
    } else {

      /* Get the indices bounding the particle value. */
      ihigh = ilow;
      ilow -= 1;

      /* Calculate the fraction. */
      ifrac = (age - a[ilow]) / (a[ihigh] - a[ilow]);
    }
    
    /* Get the cloest value in age. The lower bound is always returned. */
    jlow = closestIndex(z, lenz, metal);

    /* Are outside the array? */
    /* TODO: For values outside the range should we return any weight? */
    if (jlow == 0) {
      jhigh = jlow;
      jfrac = 0;
    }else if (jlow == lenz) {
      jlow -= 1;
      jhigh = jlow;
      jfrac = 0;
    } else{

      /* Get the indices bounding the particle value. */
      jhigh = jlow;
      jlow -= 1;

      /* Calculate the fraction. */
      jfrac = (metal - z[jlow]) / (z[jhigh] - z[jlow]);
    }

    /* Compute the mass fraction. */
    mfrac = mass * (1.-jfrac);

    /* Compute the weight. */
    weights[ilow * lenz + jlow] += mfrac * (1.-ifrac);

    /* Ensure we're not adding weights more than once when outside range */
    if (jlow != jhigh) {
      weights[ilow * lenz + jhigh] += mfrac * ifrac;
    }
    if (ilow != ihigh) {
      mfrac = mass * jfrac;
      weights[ihigh* lenz + jlow] += mfrac * (1.-ifrac);
      if (jlow != jhigh) {
        weights[ihigh * lenz + jhigh] += mfrac * ifrac;
      }
    }
  }

  /* Reconstruct the python array to return. */
  npy_intp dims[2] = {lena, lenz};
  PyArrayObject *out_weights =
    (PyArrayObject *) PyArray_SimpleNewFromData(2, dims, NPY_FLOAT64, weights);
  out_weights->flags = NPY_OWNDATA;
  
  return Py_BuildValue("N", out_weights);
}


/* Below is all the gubbins needed to make the module importable in Python. */
static PyMethodDef WeightMethods[] = {
  {"calculate_weights", calculate_weights, METH_VARARGS,
   "Method for calculating particle weights."},
};

/* Make this importable. */
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "weights",                                 /* m_name */
        "A module to calculate particle weights",  /* m_doc */
        -1,                                        /* m_size */
        WeightMethods,                             /* m_methods */
        NULL,                                      /* m_reload */
        NULL,                                      /* m_traverse */
        NULL,                                      /* m_clear */
        NULL,                                      /* m_free */
    };

PyMODINIT_FUNC PyInit_weights(void) {
    PyObject *m = PyModule_Create(&moduledef);
    import_array();
    return m;
}
