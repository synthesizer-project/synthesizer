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
 * @brief Compute a flat grid index based on the grid dimensions.
 *
 * @param indices: An array of N-dimensional indices.
 * @param dims: The length of each dimension.
 * @param ndim: The number of dimensions.
 */
int get_flat_index(const int *indices, const int *dims, const int ndim) {

  /* Define the index. */
  int index = indices[0];

  /* If we haven't been given dims then they are always 2. */
  if (dims == NULL) {
    
    /* Loop over dimensions accumalting the index along each dimension. */
    for (int i = 1; i < ndim; i++) {
      index = index * 2 + indices[i];
    }
    
  } else {
    
    /* Loop over dimensions accumalting the index along each dimension. */
    for (int i = 1; i < ndim; i++) {
      index = index * dims[i - 1] + indices[i];
    }
    
  }
  
  return index;
  
}

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
int closest_index(const double *arr, const int n, const double x) {

  /* Define the starting indices. */
  int l = 0, r = n-1;

  /* Here we need to handle if we are outside the range of values. If so
   * there's no point in searching and we return the edge nearest to the
   * value. */
  if (x < arr[0]) {
    return 0;
  } else if (x > arr[n - 1]) {
    return n;
  }

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
 * @brief This calculates the mass fractions in each right most grid cell.
 *
 * To do this for an N-dimensional array this is done recursively.
 *
 * @param grid_tuple: The tuple contain each array of grid properties.
 * @param part_tuple: The tuple containing each array of particle properties.
 * @param p: Index of the current particle.
 * @param dim: The current dimension in the recursion.
 * @param ndim: The number of grid dimensions.
 * @param dims: The length of each grid dimension.
 * @param indices: The array for storing N-dimensional grid indicies.
 * @param fracs: The array for storing the mass fractions. NOTE: The left most
 *               grid cell's mass fraction is simply (1 - frac[dim])
 */
void recursive_frac_loop(const PyObject *grid_tuple, const PyObject *part_tuple,
                         int p, int dim, const int ndim, const int *dims,
                         int *indices, double *fracs) {

  /* Are we done yet? */
  if (dim >= ndim) {
    return;
  }

  int low, high;

  /* Get a pointer to the grid property data itself. */
  const PyArrayObject *np_grid_arr = PyTuple_GetItem(grid_tuple, dim);
  const double *grid_arr = np_grid_arr->data;
    
  /* Get a pointer to the particle property data itself. */
  const PyArrayObject *np_part_arr = PyTuple_GetItem(part_tuple, dim);
  const double *part_arr = np_part_arr->data;
  const double part_val = part_arr[p];
  
  /* Get the cloest value to val in arr. The upper bound is always
   * returned unless an edge is hit. */
  low = closest_index(grid_arr, dims[dim], part_val);

  /* Are we outside the array? */
  if (low == 0) {
    fracs[dim] = 0;  /* set fraction to zero. */
  } else if (low == dims[dim]) {
    low -= 1;  /* set upper index to the array length. */
    fracs[dim] = 0; /* set fraction to zero. */
  } else {
    
    /* Get the indices bounding the particle value. */
    high = low;
    low -= 1;
    
    /* Calculate the fraction. Note, this represents the mass fraction in
     * the high cell. */
    fracs[dim] = (part_val - grid_arr[low]) / (grid_arr[high] - grid_arr[low]);
  }

  /* Set these indices. */
  indices[dim] = low;

  /* Recurse... */
  recursive_frac_loop(grid_tuple, part_tuple, p, dim + 1, ndim, dims, indices,
                      fracs);
}

/**
 * @brief This calculates the grid weights in each grid cell.
 *
 * To do this for an N-dimensional array this is done recursively.
 *
 * @param mass: The mass of the current particle.
 * @param part_tuple: The tuple containing each array of particle properties.
 * @param p: Index of the current particle.
 * @param dim: The current dimension in the recursion.
 * @param ndim: The number of grid dimensions.
 * @param dims: The length of each grid dimension.
 * @param indices: The array for storing N-dimensional grid indicies.
 * @param fracs: The array for storing the mass fractions. NOTE: The left most
 *               grid cell's mass fraction is simply (1 - frac[dim])
 */
void recursive_weight_loop(const double mass, int*sub_indices,
                           int *frac_indices, int *weight_indices,
                           double *weights, double *fracs,
                           int dim, const int *dims, const int ndim) {

  /* Are we done yet? */
  if (dim >= ndim) {

    /* Get the index for this particle in the weights array. */
    const int weight_ind = get_flat_index(sub_indices, /*dims*/NULL, ndim);

    /* Get the flattened index into the grid array. */
    weight_indices[weight_ind] = get_flat_index(frac_indices, dims, ndim);

    /* Compute the weight. */
    weights[weight_ind] = mass;
    for (int i = 0; i < ndim; i++) {

      if (sub_indices[i]) {
        weights[weight_ind] = fracs[i];
      } else {
        weights[weight_ind] = (1 - fracs[i]);
      }
    }

    /* We're done! */
    return;
  }

  /* Loop over this dimension */
  for (int i = 0; i < 2; i++) {

    /* Where are we in the sub_array? */
    sub_indices[dim] = i;

    /* Where are we in the grid array? */
    frac_indices[dim] += i;
    
    /* Recurse... */
    recursive_weight_loop(mass, sub_indices, frac_indices,
                          weight_indices, weights, fracs, dim + 1, dims, ndim);
  }
}

/**
 * @brief Calculates the weights for each grid cell occupied by a particle.
 *
 * @param grid_tuple:
 * @param part_tuple:
 * @param len_tuple:
 * @param part_mass:
 * @param particle_m: Particle initial mass.
 * @param lena: The number of age values in a.
 * @param lenz: The number of metallicity values in z.
 * @param npart: The number of stellar particles.
 */
void calculate_weights(const PyObject *grid_tuple, const PyObject *part_tuple,
                       const int *dims, const double *part_mass, int p,
                       const int grid_dim, int *weight_indices,
                       double *weights, const int nlam, double *fracs,
                       int *frac_indices, int *sub_indices) {
  
  /* Get this particle's mass. */
  const double mass = part_mass[p];

  /* Compute grid indices and the mass faction in each grid cell. */
  recursive_frac_loop(grid_tuple, part_tuple, p, /*dim*/0, grid_dim, dims,
                      frac_indices, fracs);

  /* Compute the weights and flattened grid indices. */
  recursive_weight_loop(mass, sub_indices, frac_indices,
                        weight_indices, weights, fracs, /*dim*/0, dims,
                        grid_dim);

}

/**
 * @brief Computes an integrated SED for a collection of particles.
 *
 * @param np_stellar_spectra:
 * @param np_total_spectra:
 * @param grid_tuple:
 * @param part_tuple:
 * @param part_mass:
 * @param fesc:
 * @param len_tuple:
 * @param grid_dim:
 * @param npart:
 * @param nlam:
 */
PyObject *compute_integrated_sed(PyObject *self, PyObject *args) {

  const int32_t grid_dim, npart, nlam;
  const double fesc;
  const PyObject *grid_tuple, *part_tuple;
  const PyArrayObject *np_stellar_spectra, *np_total_spectra;
  const PyArrayObject *np_part_mass, *np_grid_dims;

  if(!PyArg_ParseTuple(args, "OOOOOdOiii",
                       &np_stellar_spectra, &np_total_spectra, &grid_tuple,
                       &part_tuple, &np_part_mass, &fesc, &np_grid_dims,
                       &grid_dim, &npart, &nlam))
    return NULL;

  /* Quick check to make sure our inputs are valid. */
  if (grid_dim == 0) return NULL;
  if (npart == 0) return NULL;
  if (nlam == 0) return NULL;

  /* Extract a point32_ter to the spectra grids */
  const double *grid_stellar_spectra = np_stellar_spectra->data;
  /* double *grid_total_spectra = NULL; */
  /* if (np_total_spectra != NULL) */
  const double *grid_total_spectra = np_total_spectra->data;

  /* Set up arrays to hold the SEDs themselves. */
  double *stellar_spectra = malloc(nlam * sizeof(double));
  bzero(stellar_spectra, nlam * sizeof(double));
  double *total_spectra = NULL;
  if (grid_total_spectra != NULL) {
    total_spectra = malloc(nlam * sizeof(double));
    bzero(total_spectra, nlam * sizeof(double));
  }

  /* Extract a point32_ter to the grid dims */
  const int32_t *dims = np_grid_dims->data;

  /* Extract a point32_ter to the particle masses. */
  const double *part_mass = np_part_mass->data;

  /* Compute the number of weights we need. */
  const int32_t nweights = pow(2, grid_dim) + 0.1;

  /* Define an array to hold this particle's weights. */
  double *weights = malloc(nweights * sizeof(double));
  bzero(weights, nweights * sizeof(double));

  /* Set up arrays to store grid indices for the weights, mass fractions
   * and indices.
   * NOTE: the wavelength index on frac_indices is always 0. */
  double *fracs = malloc(grid_dim * sizeof(double));
  int32_t *frac_indices = malloc((grid_dim + 1) * sizeof(int32_t));
  int32_t *weight_indices = malloc(nweights * sizeof(int32_t));
  int32_t *sub_indices = malloc(grid_dim * sizeof(double));
    
  /* Loop over particles. */
  for (int32_t p = 0; p < npart; p++) {

    /* Reset arrays. */
    for (int32_t ind = 0; ind < grid_dim + 1; ind++)
      frac_indices[ind] = 0;
    for (int32_t ind = 0; ind < nweights; ind++)
      weight_indices[ind] = 0;
    for (int32_t ind = 0; ind < grid_dim; ind++) {
      fracs[ind] = 0;
      sub_indices[ind] = 0;
    }

    /* Get the weights and indices for this particle. */
    calculate_weights(grid_tuple, part_tuple, dims, part_mass, p,
                      grid_dim, weight_indices, weights, nlam, fracs,
                      frac_indices, sub_indices);

    /* Loop over weights and their indices. */
    for (int32_t i = 0; i < nweights; i++) {

      /* Get this weight and it's flattened index. */
      const double weight = weights[i];
      const int32_t weight_ind = weight_indices[i];

      /* Add this particle's contribution to... */
      for (int32_t ilam = 0; ilam < nlam; ilam++) {

        /* ... the stellar SED. */
        stellar_spectra[ilam] +=
          grid_stellar_spectra[weight_ind + ilam] * weight;

        /* And the int32_trinsic SED if we have it. */
        if (grid_total_spectra != NULL) {
          total_spectra[ilam] +=
            grid_total_spectra[weight_ind + ilam] * (1 - fesc) * weight;
        }
      }
    }
  }
  
  /* Reconstruct the python array to return. */
  npy_intp np_dims[1] = {nlam,};
  PyArrayObject *out_stellar_spectra =
    (PyArrayObject *) PyArray_SimpleNewFromData(1, np_dims, NPY_FLOAT64,
                                                stellar_spectra);
  out_stellar_spectra->flags = NPY_OWNDATA;

  /* Reconstruct the python array to return. */
  PyArrayObject *out_total_spectra =
    (PyArrayObject *) PyArray_SimpleNewFromData(1, np_dims, NPY_FLOAT64,
                                                total_spectra);
  out_total_spectra->flags = NPY_OWNDATA;
  
  return Py_BuildValue("NN", out_stellar_spectra, out_total_spectra);
}

/**
 * @brief Computes an integrated SED for a collection of particles.
 *
 * @param np_stellar_spectra:
 * @param np_total_spectra:
 * @param grid_tuple:
 * @param part_tuple:
 * @param part_mass:
 * @param fesc:
 * @param len_tuple:
 * @param grid_dim:
 * @param npart:
 * @param nlam:
 */
PyObject *compute_particle_seds(PyObject *self, PyObject *args) {

  const int32_t grid_dim, npart, nlam;
  const double fesc;
  const PyObject *grid_tuple, *part_tuple;
  const PyArrayObject *np_stellar_spectra, *np_total_spectra;
  const PyArrayObject *np_part_mass, *np_grid_dims;

  if(!PyArg_ParseTuple(args, "OOOOOdOiii",
                       &np_stellar_spectra, &np_total_spectra, &grid_tuple,
                       &part_tuple, &np_part_mass, &fesc, &np_grid_dims,
                       &grid_dim, &npart, &nlam))
    return NULL;

  /* Quick check to make sure our inputs are valid. */
  if (grid_dim == 0) return NULL;
  if (npart == 0) return NULL;
  if (nlam == 0) return NULL;

  /* Extract a point64_ter to the spectra grids */
  const double *grid_stellar_spectra = np_stellar_spectra->data;
  /* double *grid_total_spectra = NULL; */
  /* if (np_total_spectra != NULL) */
  const double *grid_total_spectra = np_total_spectra->data;

  /* Set up arrays to hold the SEDs themselves. */
  double *stellar_spectra = malloc(npart * nlam * sizeof(double));
  bzero(stellar_spectra, npart * nlam * sizeof(double));
  double *total_spectra = NULL;
  if (grid_total_spectra != NULL) {
    total_spectra = malloc(npart * nlam * sizeof(double));
    bzero(total_spectra, npart * nlam * sizeof(double));
  }

  /* Extract a point64_ter to the grid dims */
  const int32_t *dims = np_grid_dims->data;

  /* Extract a point32_ter to the particle masses. */
  const double *part_mass = np_part_mass->data;

  /* Compute the number of weights we need. */
  const int32_t nweights = pow(2, grid_dim) + 0.1;

  /* Define an array to hold this particle's weights. */
  double *weights = malloc(nweights * sizeof(double));
  bzero(weights, nweights * sizeof(double));

  /* Set up arrays to store grid indices for the weights, mass fractions
   * and indices.
   * NOTE: the wavelength index on frac_indices is always 0. */
  double *fracs = malloc(grid_dim * sizeof(double));
  int32_t *frac_indices = malloc((grid_dim + 1) * sizeof(int32_t));
  int32_t *weight_indices = malloc(nweights * sizeof(int32_t));
  int32_t *sub_indices = malloc(grid_dim * sizeof(double));
    
  /* Loop over particles. */
  for (int32_t p = 0; p < npart; p++) {

    /* Reset arrays. */
    for (int32_t ind = 0; ind < grid_dim + 1; ind++)
      frac_indices[ind] = 0;
    for (int32_t ind = 0; ind < nweights; ind++)
      weight_indices[ind] = 0;
    for (int32_t ind = 0; ind < grid_dim; ind++) {
      fracs[ind] = 0;
      sub_indices[ind] = 0;
    }

    /* Get the weights and indices for this particle. */
    calculate_weights(grid_tuple, part_tuple, dims, part_mass, p,
                      grid_dim, weight_indices, weights, nlam, fracs,
                      frac_indices, sub_indices);

    /* Loop over weights and their indices. */
    for (int32_t i = 0; i < nweights; i++) {

      /* Get this weight and it's flattened index. */
      const double weight = weights[i];
      const int32_t weight_ind = weight_indices[i];

      /* Add this particle's contribution to... */
      for (int32_t ilam = 0; ilam < nlam; ilam++) {

        /* ... the stellar SED. */
        stellar_spectra[p * nlam + ilam] +=
          grid_stellar_spectra[weight_ind + ilam] * weight;

        /* And the int32_trinsic SED if we have it. */
        if (grid_total_spectra != NULL) {
          total_spectra[p * nlam + ilam] +=
            grid_total_spectra[weight_ind + ilam] * (1 - fesc) * weight;
        }
      }
    }
  }
  
  /* Reconstruct the python array to return. */
  npy_intp np_dims[2] = {npart, nlam};
  PyArrayObject *out_stellar_spectra =
    (PyArrayObject *) PyArray_SimpleNewFromData(2, np_dims, NPY_FLOAT32,
                                                stellar_spectra);
  out_stellar_spectra->flags = NPY_OWNDATA;

  /* Reconstruct the python array to return. */
  PyArrayObject *out_total_spectra =
    (PyArrayObject *) PyArray_SimpleNewFromData(2, np_dims, NPY_FLOAT64,
                                                total_spectra);
  out_total_spectra->flags = NPY_OWNDATA;
  
  return Py_BuildValue("NN", out_stellar_spectra, out_total_spectra);
}

/* Below is all the gubbins needed to make the module importable in Python. */
static PyMethodDef SedMethods[] = {
  {"compute_integrated_sed", compute_integrated_sed, METH_VARARGS,
   "Method for calculating integrated intrinsic spectra."},
  {"compute_particle_seds", compute_particle_seds, METH_VARARGS,
   "Method for calculating particle intrinsic spectra."},
  {NULL, NULL, 0, NULL} 
};

/* Make this importable. */
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "make_sed",                          /* m_name */
        "A module to calculate seds",              /* m_doc */
        -1,                                        /* m_size */
        SedMethods,                                /* m_methods */
        NULL,                                      /* m_reload */
        NULL,                                      /* m_traverse */
        NULL,                                      /* m_clear */
        NULL,                                      /* m_free */
    };

PyMODINIT_FUNC PyInit_make_sed(void) {
    PyObject *m = PyModule_Create(&moduledef);
    import_array();
    return m;
}
