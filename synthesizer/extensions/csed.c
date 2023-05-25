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
        if (index < 0) {
          printf("index=%d stride=%d multi_index[i]=%d", index, stride, multi_index[i];);
          fflush(stdout);
        }
        stride *= 2;
    }
    return index;
}

/**
 * @brief Calculates the mass fractions in each right most grid cell along
 *        each dimension.
 *
 * @param grid_tuple: The tuple contain each array of grid properties.
 * @param part_tuple: The tuple containing each array of particle properties.
 * @param p: Index of the current particle.
 * @param ndim: The number of grid dimensions.
 * @param dims: The length of each grid dimension.
 * @param indices: The array for storing N-dimensional grid indicies.
 * @param fracs: The array for storing the mass fractions. NOTE: The left most
 *               grid cell's mass fraction is simply (1 - frac[dim])
 */
void frac_loop(const double *grid_props, const double *part_props,
               int p, const int ndim, const int *dims, const int npart,
               int *frac_indices, double *fracs) {

  /* Loop over dimensions. */
  for (int dim = 0; dim < ndim; dim++) {
    
    /* Get the grid and particle start indices for this property. */
    int grid_start = 0;
    int part_start = 0;
    for (int jdim = 0; jdim < dim; jdim++) {
      grid_start += dims[jdim];
      part_start += npart;
    }

    /* Get this particle property. */
    const double part_val = part_props[part_start + p];

    /**************************************************************************
     * Get the cells corresponding to this particle and compute the fraction.
     *************************************************************************/

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
      int diff = high - low;
      while (diff > 1) {
        
        /* Define the midpoint. */
        int mid = low + floor(diff / 2);
        
        /* Where is the midpoint relative to the value? */
        if (grid_props[mid] < part_val) {
          low = mid;
        } else {
          high = mid; 
        }

        /* Compute the new range. */
        diff = high - low;
      }
      
      /* Calculate the fraction. Note, this represents the mass fraction in
       * the high cell. */
      fracs[dim] =
        (part_val - grid_props[low]) / (grid_props[high] - grid_props[low]);
    }

    /* Set these indices. */
    frac_indices[dim] = low - grid_start;
    
  }
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
  const PyArrayObject *np_grid_spectra;
  const PyArrayObject *np_part_mass, *np_ndims;

  if(!PyArg_ParseTuple(args, "OOOOdOiii",
                       &np_grid_spectra, &grid_tuple, &part_tuple,
                       &np_part_mass, &fesc, &np_ndims, &ndim, &npart, &nlam))
    return NULL;

  /* Quick check to make sure our inputs are valid. */
  if (ndim == 0) return NULL;
  if (npart == 0) return NULL;
  if (nlam == 0) return NULL;

  /* Extract a pointer to the spectra grids */
  const double *grid_spectra = PyArray_DATA(np_grid_spectra);

  /* Set up arrays to hold the SEDs themselves. */
  double *spectra = malloc(nlam * sizeof(double));
  bzero(spectra, nlam * sizeof(double));

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
  unsigned int *frac_indices = malloc((ndim + 1) * sizeof(int));
  unsigned int *low_indices = malloc((ndim + 1) * sizeof(int));
  unsigned int *weight_indices = malloc(nweights * sizeof(int));
  short int *sub_indices = malloc(ndim * sizeof(double));
    
  /* Loop over particles. */
  for (int p = 0; p < npart; p++) {

    /* Reset arrays. */
    for (int ind = 0; ind < nweights; ind++)
      weight_indices[ind] = 0;
    for (int ind = 0; ind < ndim; ind++) {
      fracs[ind] = 0;
      sub_indices[ind] = 0;
    }

    /* Get this particle's mass. */
    const double mass = part_mass[p];

    /* Compute grid indices and the mass faction in each grid cell. */
    frac_loop(grid_props, part_props, p, ndim, dims, npart, frac_indices,
              fracs);
      
    /* Make a copy of the indices of the fraction to avoid double addition. */
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
      if (weight <= 0) continue;

      /* Add this particle's contribution to the spectra */
      for (int ilam = 0; ilam < nlam; ilam++) {

        /* Are we doing total or stellar? */
        spectra[ilam] +=
          grid_spectra[spectra_ind + ilam] * (1 - fesc) * weight; 
      }
    }
  }
  
  /* Reconstruct the python array to return. */
  npy_intp np_dims[1] = {nlam,};
  PyArrayObject *out_spectra =
    (PyArrayObject *) PyArray_SimpleNewFromData(1, np_dims, NPY_FLOAT64,
                                                spectra);
  
  return Py_BuildValue("N", out_spectra);
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
PyObject *compute_particle_seds(PyObject *self, PyObject *args) {
  
  const int ndim;
  const int npart, nlam;
  const double fesc;
  const PyObject *grid_tuple, *part_tuple;
  const PyArrayObject *np_grid_spectra;
  const PyArrayObject *np_part_mass, *np_ndims;

  if(!PyArg_ParseTuple(args, "OOOOdOiii",
                       &np_grid_spectra, &grid_tuple, &part_tuple,
                       &np_part_mass, &fesc, &np_ndims, &ndim, &npart, &nlam))
    return NULL;

  /* Quick check to make sure our inputs are valid. */
  if (ndim == 0) return NULL;
  if (npart == 0) return NULL;
  if (nlam == 0) return NULL;

  /* Extract a pointer to the spectra grids */
  const double *grid_spectra = PyArray_DATA(np_grid_spectra);

  /* Set up arrays to hold the SEDs themselves. */
  double *spectra = malloc(npart * nlam * sizeof(double));
  bzero(spectra, npart * nlam * sizeof(double));

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
  unsigned int *frac_indices = malloc((ndim + 1) * sizeof(int));
  unsigned int *low_indices = malloc((ndim + 1) * sizeof(int));
  unsigned int *weight_indices = malloc(nweights * sizeof(int));
  short int *sub_indices = malloc(ndim * sizeof(double));
    
  /* Loop over particles. */
  for (int p = 0; p < npart; p++) {

    /* Reset arrays. */
    for (int ind = 0; ind < nweights; ind++)
      weight_indices[ind] = 0;
    for (int ind = 0; ind < ndim; ind++) {
      fracs[ind] = 0;
      sub_indices[ind] = 0;
    }

    /* Get this particle's mass. */
    const double mass = part_mass[p];

    /* Compute grid indices and the mass faction in each grid cell. */
    frac_loop(grid_props, part_props, p, ndim, dims, npart, frac_indices,
              fracs);
      
    /* Make a copy of the indices of the fraction to avoid double addition. */
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
      if (weight <= 0) continue;

      /* Add this particle's contribution to the spectra */
      for (int ilam = 0; ilam < nlam; ilam++) {

        /* Are we doing total or stellar? */
        spectra[p * nlam + ilam] +=
          grid_spectra[spectra_ind + ilam] * (1 - fesc) * weight; 
      }
    }
  }
  
  /* Reconstruct the python array to return. */
  npy_intp np_dims[2] = {npart, nlam,};
  PyArrayObject *out_spectra =
    (PyArrayObject *) PyArray_SimpleNewFromData(2, np_dims, NPY_FLOAT64,
                                                spectra);
  
  return Py_BuildValue("N", out_spectra);
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

PyMODINIT_FUNC PyInit_csed(void) {
    PyObject *m = PyModule_Create(&moduledef);
    import_array();
    return m;
}
