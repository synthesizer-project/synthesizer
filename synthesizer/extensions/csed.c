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

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

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
int get_flat_index(const int *multi_index, const int *dims, const int ndims) {
    int index = 0, stride = 1;
    for (int i = ndims - 1; i >= 0; i--) {
        index += stride * multi_index[i];
        stride *= dims[i];
    }
    return index;
}

void nD_index(int N, int* shape, int flat_index, int* nD_index) {
    for (int i = N-1; i >= 0; i--) {
        nD_index[i] = flat_index % shape[i];
        flat_index = flat_index / shape[i];
    }
}

/**
 * @brief Compute a flat grid index based on the grid dimensions for instances
 *        where the length of each dimension is always 2.
 *
 * @param indices: An array of N-dimensional indices.
 * @param dims: The length of each dimension.
 * @param ndim: The number of dimensions.
 */
int get_flat_index_subarray(const int *multi_index, int ndims) {
    int index = 0, stride = 1;
    for (int i = ndims - 1; i >= 0; i--) {
        index += stride * multi_index[i];
        stride *= 2;
    }
    return index;
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
void recursive_frac_loop(const double *grid_props, const double *part_props,
                         int p, int dim, const int ndim,
                         const int *dims, const int npart,
                         int *frac_indices, double *fracs) {

  /* Are we done yet? */
  if (dim >= ndim) {
    return;
  }
  
  /* Get the grid and particle start indices for this property. */
  int grid_start = 0;
  int part_start = 0;
  for (int jdim = 0; jdim < dim; jdim++) {
    grid_start += dims[jdim];
    part_start += npart;
  }

  /* Get this particle property. */
  const double part_val = part_props[part_start + p];

  /****************************************************************************
   * Get the cells corresponding to this particle and compute the fraction.
   ***************************************************************************/

  /* Define the starting indices. */
  int low = grid_start, high = grid_start + dims[dim] - 1;

  /* Here we need to handle if we are outside the range of values. If so
   * there's no point in searching and we return the edge nearest to the
   * value. */
  if (part_val <= grid_props[low]) {
    low = grid_start;
    fracs[dim] = 0;
  } else if (part_val > grid_props[high]) {
    low = grid_start + dims[dim];
    fracs[dim] = 0;
  } else {

    /* While we don't have a pair of adjacent indices. */
    while ((high - low) > 1) {

      /* Define the midpoint. */
      int mid = low + (high - low) / 2;

      /* Where is the midpoint relative to the value? */
      if (grid_props[mid] <= part_val) {
        low = mid;
      } else {
        high = mid; 
      }
    }

    /* Calculate the fraction. Note, this represents the mass fraction in
     * the high cell. */
    fracs[dim] =
      (part_val - grid_props[low]) / (grid_props[high] - grid_props[low]);
  }

  /* Set these indices. */
  frac_indices[dim] = low - grid_start;

  /* Recurse... */
  recursive_frac_loop(grid_props, part_props, p, dim + 1, ndim, dims, npart,
                      frac_indices, fracs);
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
void recursive_weight_loop(const double mass, int *sub_indices,
                           int *frac_indices, int *low_indices,
                           int *weight_indices,
                           double *weights, double *fracs,
                           int dim, const int *dims,
                           const int ndim) {

  /* Are we done yet? */
  if (dim >= ndim) {

    /* Get the index for this particle in the weights array. */
    const int weight_ind = get_flat_index_subarray(sub_indices, ndim);

    /* Get the flattened index into the grid array. */
    weight_indices[weight_ind] = get_flat_index(frac_indices, dims, ndim + 1);

    /* Check whether we need a weight in this cell. */
    for (int i = 0; i < ndim; i++) {
      if ((sub_indices[i] == 1 && fracs[i] == 0 && frac_indices[i] == 0) ||
          (sub_indices[i] == 1 && fracs[i] == 0 && frac_indices[i] == dims[i])) {
        weights[weight_ind] = 0;
        return;
      }
    }

    /* Compute the weight. */
    weights[weight_ind] = mass;
    for (int i = 0; i < ndim; i++) {

      if (sub_indices[i]) {
        weights[weight_ind] *= fracs[i];
      } else {
        weights[weight_ind] *= (1 - fracs[i]);
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
    frac_indices[dim] = low_indices[dim] + i;
    
    /* Recurse... */
    recursive_weight_loop(mass, sub_indices, frac_indices, low_indices,
                          weight_indices, weights, fracs, dim + 1, dims, ndim);
  }
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

  const int ndim;
  const int npart, nlam;
  const double fesc;
  const PyObject *grid_tuple, *part_tuple;
  const PyArrayObject *np_stellar_spectra, *np_total_spectra;
  const PyArrayObject *np_part_mass, *np_ndims;

  if(!PyArg_ParseTuple(args, "OOOOOdOiii",
                       &np_stellar_spectra, &np_total_spectra, &grid_tuple,
                       &part_tuple, &np_part_mass, &fesc, &np_ndims,
                       &ndim, &npart, &nlam))
    return NULL;

  /* Quick check to make sure our inputs are valid. */
  if (ndim == 0) return NULL;
  if (npart == 0) return NULL;
  if (nlam == 0) return NULL;

  /* Extract a pointer to the spectra grids */
  const double *grid_stellar_spectra = PyArray_DATA(np_stellar_spectra);
  /* double *grid_total_spectra = NULL; */
  /* if (np_total_spectra != NULL) */
  const double *grid_total_spectra = PyArray_DATA(np_total_spectra);

  /* Set up arrays to hold the SEDs themselves. */
  double *stellar_spectra = malloc(nlam * sizeof(double));
  bzero(stellar_spectra, nlam * sizeof(double));
  double *total_spectra = NULL;
  if (grid_total_spectra != NULL) {
    total_spectra = malloc(nlam * sizeof(double));
    bzero(total_spectra, nlam * sizeof(double));
  }

  /* Extract a pointer to the grid dims */
  const int *dims = PyArray_DATA(np_ndims);

  /* Extract a pointer to the particle masses. */
  const double *part_mass = PyArray_DATA(np_part_mass);

  /* Compute the number of weights we need. */
  const int nweights = pow(2, ndim) + 0.1;

  /* Define an array to hold this particle's weights. */
  double *weights = malloc(nweights * sizeof(double));
  bzero(weights, nweights * sizeof(double));

  /* Allocate a single array for grid properties*/
  int nprops = 0;
  for (int dim = 0; dim < ndim; dim++) nprops += dims[dim];
  double *grid_props = malloc(nprops * sizeof(double));

  /* Unpack the grid property arrays into a single contiguous array. */
  for (int idim = 0; idim < ndim; idim++) {
    
    /* Extract the data from the numpy array. */
    const PyArrayObject *np_grid_arr = PyTuple_GetItem(grid_tuple, idim);
    const double *grid_arr = PyArray_DATA(np_grid_arr);

    /* Get the start index for this data. */
    int start = 0;
    for (int jdim = 0; jdim < idim; jdim++) start += dims[jdim];

    /* Assign this data to the property array. */
    for (int ind = start; ind < start + dims[idim]; ind++)
      grid_props[ind] = grid_arr[ind - start];
  }

  /* Allocate a single array for particle properties. */
  double *part_props = malloc(npart * ndim * sizeof(double));

  /* Unpack the particle property arrays into a single contiguous array. */
  for (int idim = 0; idim < ndim; idim++) {
    
    /* Extract the data from the numpy array. */
    const PyArrayObject *np_part_arr = PyTuple_GetItem(part_tuple, idim);
    const double *part_arr = PyArray_DATA(np_part_arr);

    /* Get the start index for this data. */
    int start = 0;
    for (int jdim = 0; jdim < idim; jdim++) start += npart;

    /* Assign this data to the property array. */
    for (int ind = start; ind < start + npart; ind++)
      part_props[ind] = part_arr[ind - start];
  }

  /* Set up arrays to store grid indices for the weights, mass fractions
   * and indices.
   * NOTE: the wavelength index on frac_indices is always 0. */
  double *fracs = malloc(ndim * sizeof(double));
  int *frac_indices = malloc((ndim + 1) * sizeof(int));
  int *low_indices = malloc((ndim + 1) * sizeof(int));
  int *weight_indices = malloc(nweights * sizeof(int));
  int *sub_indices = malloc(ndim * sizeof(double));
    
  /* Loop over particles. */
  for (int p = 0; p < npart; p++) {

    /* Reset arrays. */
    for (int ind = 0; ind < ndim + 1; ind++) {
      frac_indices[ind] = 0;
      low_indices[ind] = 0;
    }
    for (int ind = 0; ind < nweights; ind++)
      weight_indices[ind] = 0;
    for (int ind = 0; ind < ndim; ind++) {
      fracs[ind] = 0;
      sub_indices[ind] = 0;
    }

    /* Get this particle's mass. */
    const double mass = part_mass[p];

    /* Compute grid indices and the mass faction in each grid cell. */
    recursive_frac_loop(grid_props, part_props, p, /*dim*/0, ndim, dims,
                        npart, frac_indices, fracs);

    /* Make copy of the indices of the fraction to avoid double addition. */
    for (int ind = 0; ind < ndim + 1; ind++) {
      low_indices[ind] = frac_indices[ind];
    }

    /* Compute the weights and flattened grid indices. */
    recursive_weight_loop(mass, sub_indices, frac_indices, low_indices,
                          weight_indices, weights, fracs, /*dim*/0, dims,
                          ndim);

    /* Loop over weights and their indices. */
    for (int i = 0; i < nweights; i++) {

      /* Get this weight and it's flattened index. */
      const double weight = weights[i];
      const int spectra_ind = weight_indices[i];

      /* Skip zero weight cells. */
      if (weight == 0) continue;

      /* Add this particle's contribution to... */
      for (int ilam = 0; ilam < nlam; ilam++) {

        /* ... the stellar SED. */
        stellar_spectra[ilam] +=
          grid_stellar_spectra[spectra_ind + ilam] * weight;

        /* And the intrinsic SED if we have it. */
        if (grid_total_spectra != NULL) {
          total_spectra[ilam] +=
            grid_total_spectra[spectra_ind + ilam] * (1 - fesc) * weight;
        }
      }
    }
  }
  
  /* Reconstruct the python array to return. */
  npy_intp np_dims[1] = {nlam,};
  PyArrayObject *out_stellar_spectra =
    (PyArrayObject *) PyArray_SimpleNewFromData(1, np_dims, NPY_FLOAT64,
                                                stellar_spectra);
  /* out_stellar_spectra->flags = NPY_OWNDATA; */

  /* Reconstruct the python array to return. */
  PyArrayObject *out_total_spectra =
    (PyArrayObject *) PyArray_SimpleNewFromData(1, np_dims, NPY_FLOAT64,
                                                total_spectra);
  /* out_total_spectra->flags = NPY_OWNDATA; */
  
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
 * @param ndim:
 * @param npart:
 * @param nlam:
 */
PyObject *compute_particle_seds(PyObject *self, PyObject *args) {}
/* PyObject *compute_particle_seds(PyObject *self, PyObject *args) { */

/*   const int grid_dim; */
/*   const int64_t npart, nlam; */
/*   const double fesc; */
/*   const PyObject *grid_tuple, *part_tuple; */
/*   const PyArrayObject *np_stellar_spectra, *np_total_spectra; */
/*   const PyArrayObject *np_part_mass, *np_grid_dims; */

/*   if(!PyArg_ParseTuple(args, "OOOOOdOiii", */
/*                        &np_stellar_spectra, &np_total_spectra, &grid_tuple, */
/*                        &part_tuple, &np_part_mass, &fesc, &np_grid_dims, */
/*                        &grid_dim, &npart, &nlam)) */
/*     return NULL; */

/*   /\* Quick check to make sure our inputs are valid. *\/ */
/*   if (grid_dim == 0) return NULL; */
/*   if (npart == 0) return NULL; */
/*   if (nlam == 0) return NULL; */

/*   /\* Extract a pointer to the spectra grids *\/ */
/*   const double *grid_stellar_spectra = PyArray_DATA(np_stellar_spectra); */
/*   /\* double *grid_total_spectra = NULL; *\/ */
/*   /\* if (np_total_spectra != NULL) *\/ */
/*   const double *grid_total_spectra = PyArray_DATA(np_total_spectra); */

/*   /\* Set up arrays to hold the SEDs themselves. *\/ */
/*   double *stellar_spectra = malloc(nlam * sizeof(double)); */
/*   bzero(stellar_spectra, nlam * sizeof(double)); */
/*   double *total_spectra = NULL; */
/*   if (grid_total_spectra != NULL) { */
/*     total_spectra = malloc(nlam * sizeof(double)); */
/*     bzero(total_spectra, nlam * sizeof(double)); */
/*   } */

/*   /\* Extract a pointer to the grid dims *\/ */
/*   const int64_t *dims = PyArray_DATA(np_grid_dims); */

/*   /\* Extract a pointer to the particle masses. *\/ */
/*   const double *part_mass = PyArray_DATA(np_part_mass); */

/*   /\* Compute the number of weights we need. *\/ */
/*   const int64_t nweights = (int64_t)(pow(2, grid_dim) + 0.1); */

/*   /\* Define an array to hold this particle's weights. *\/ */
/*   double *weights = malloc(nweights * sizeof(double)); */
/*   bzero(weights, nweights * sizeof(double)); */

/*   /\* Set up arrays to store grid indices for the weights, mass fractions */
/*    * and indices. */
/*    * NOTE: the wavelength index on frac_indices is always 0. *\/ */
/*   double *fracs = malloc(grid_dim * sizeof(double)); */
/*   int64_t *frac_indices = malloc((grid_dim + 1) * sizeof(int64_t)); */
/*   int64_t *weight_indices = malloc(nweights * sizeof(int64_t)); */
/*   int64_t *sub_indices = malloc(grid_dim * sizeof(double)); */
    
/*   /\* Loop over particles. *\/ */
/*   for (int64_t p = 0; p < npart; p++) { */

/*     /\* Reset arrays. *\/ */
/*     for (int64_t ind = 0; ind < grid_dim + 1; ind++) */
/*       frac_indices[ind] = 0; */
/*     for (int64_t ind = 0; ind < nweights; ind++) */
/*       weight_indices[ind] = 0; */
/*     for (int64_t ind = 0; ind < grid_dim; ind++) { */
/*       fracs[ind] = 0; */
/*       sub_indices[ind] = 0; */
/*     } */

/*     /\* Get the weights and indices for this particle. *\/ */
/*     calculate_weights(grid_tuple, part_tuple, dims, part_mass, p, */
/*                       grid_dim, weight_indices, weights, nlam, fracs, */
/*                       frac_indices, sub_indices); */

/*     /\* Loop over weights and their indices. *\/ */
/*     for (int64_t i = 0; i < nweights; i++) { */

/*       /\* Get this weight and it's flattened index. *\/ */
/*       const double weight = weights[i]; */
/*       const int64_t weight_ind = weight_indices[i]; */

/*       /\* Add this particle's contribution to... *\/ */
/*       for (int64_t ilam = 0; ilam < nlam; ilam++) { */

/*         /\* ... the stellar SED. *\/ */
/*         stellar_spectra[p * nlam + ilam] += */
/*           grid_stellar_spectra[weight_ind + ilam] * weight; */

/*         /\* And the int64_trinsic SED if we have it. *\/ */
/*         if (grid_total_spectra != NULL) { */
/*           total_spectra[p * nlam + ilam] += */
/*             grid_total_spectra[weight_ind + ilam] * (1 - fesc) * weight; */
/*         } */
/*       } */
/*     } */
/*   } */
  
/*   /\* Reconstruct the python array to return. *\/ */
/*   npy_intp np_dims[2] = {npart, nlam}; */
/*   PyArrayObject *out_stellar_spectra = */
/*     (PyArrayObject *) PyArray_SimpleNewFromData(2, np_dims, NPY_FLOAT32, */
/*                                                 stellar_spectra); */

/*   /\* Reconstruct the python array to return. *\/ */
/*   PyArrayObject *out_total_spectra = */
/*     (PyArrayObject *) PyArray_SimpleNewFromData(2, np_dims, NPY_FLOAT64, */
/*                                                 total_spectra); */
  
/*   return Py_BuildValue("NN", out_stellar_spectra, out_total_spectra); */
/* } */

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

PyMODINIT_FUNC PyInit_csed(void) {
    PyObject *m = PyModule_Create(&moduledef);
    import_array();
    return m;
}
