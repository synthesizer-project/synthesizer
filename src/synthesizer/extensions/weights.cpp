/******************************************************************************
 * A C module containing all the weights functions common to all particle
 * spectra extensions.
 *****************************************************************************/
/* C includes */
#include <array>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

/* Python includes */
#include <Python.h>

/* Local includes */
#include "cpp_to_python.h"
#include "index_utils.h"
#include "timers.h"
#include "weights.h"

/* Optional openmp include. */
#ifdef WITH_OPENMP
#include <omp.h>
#endif

/**
 * @brief This calculates the grid weights in each grid cell using a cloud
 *        in cell (CIC) approach.
 *
 * This is the serial version of the function. Each particle distributes its
 * weight across 2^ndim neighboring grid cells based on its fractional
 * distance along each axis.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A class containing the particle properties.
 * @param out: The output array. Must have been allocated to grid size.
 */
static void weight_loop_cic_serial(GridProps *grid_props, Particles *parts,
                                   void *out) {

  /* Unpack the grid properties. */
  const std::array<int, MAX_GRID_NDIM> dims = grid_props->dims;
  const int ndim = grid_props->ndim;

  /* The number of sub-cells to iterate over (2^ndim corner combinations). */
  const int num_sub_cells = 1 << ndim;

  /* Set sub_dims to [2, 2, ..., 2] for binary decoding of cell offsets. */
  std::array<int, MAX_GRID_NDIM> sub_dims;
  for (int i = 0; i < ndim; ++i) {
    sub_dims[i] = 2;
  }

  /* Cast the output buffer to a double array. */
  double *out_arr = static_cast<double *>(out);

  /* Loop over particles. */
  for (int p = 0; p < parts->npart; ++p) {

    /* Skip if this particle is masked. */
    if (parts->part_is_masked(p)) {
      continue;
    }

    /* Get this particle's weight. */
    const double weight = parts->get_weight_at(p);

    /* Setup the base cell indices and axis fractions toward upper cells. */
    std::array<int, MAX_GRID_NDIM> part_indices;
    std::array<double, MAX_GRID_NDIM> axis_fracs;

    /* Get the grid indices and cell fractions for this particle. */
    get_part_ind_frac_cic(part_indices, axis_fracs, grid_props, parts, p);

    /* Loop over all 2^ndim surrounding grid cells. */
    for (int icell = 0; icell < num_sub_cells; ++icell) {

      /* Decode icell into a multidimensional binary offset. */
      std::array<int, MAX_GRID_NDIM> subset_ind;
      get_indices_from_flat(icell, ndim, sub_dims, subset_ind);

      /* Compute the target grid cell index and the contribution fraction. */
      double frac = 1.0;
      std::array<int, MAX_GRID_NDIM> frac_ind;
      for (int idim = 0; idim < ndim; ++idim) {
        int offset = subset_ind[idim]; // 0 or 1
        frac *= offset ? axis_fracs[idim] : (1.0 - axis_fracs[idim]);
        frac_ind[idim] = part_indices[idim] + offset;
      }

      /* Skip if contribution is zero (can happen with boundary clamps). */
      if (frac == 0.0) {
        continue;
      }

      /* Convert multi-dimensional index to flat index. */
      int flat_ind = get_flat_index(frac_ind, dims.data(), ndim);

      /* Accumulate the weighted contribution into the grid cell. */
      out_arr[flat_ind] += frac * weight;
    }
  }
}

/**
 * @brief This calculates the grid weights in each grid cell using a cloud
 *        in cell approach.
 *
 * This is the parallel version of the function.
 *
 * Each thread accumulates weights into a private local buffer, which is added
 * into the global output array at the end of the thread’s execution.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A class containing the particle properties.
 * @param out_size: The size of the output array. (This will be allocated within
 *                  this function.)
 * @param out: The output array.
 * @param nthreads: The number of threads to use.
 */
#ifdef WITH_OPENMP
static void weight_loop_cic_omp(GridProps *grid_props, Particles *parts,
                                int out_size, void *out, int nthreads) {

  /* Convert out. */
  double *out_arr = static_cast<double *>(out);

  /* Unpack the grid properties. */
  const std::array<int, MAX_GRID_NDIM> dims = grid_props->dims;
  const int ndim = grid_props->ndim;

  /* Set the sub cell constants we'll use below. */
  const int num_sub_cells = 1 << ndim; // 2^ndim
  std::array<int, MAX_GRID_NDIM> sub_dims;
  for (int i = 0; i < ndim; i++) {
    sub_dims[i] = 2;
  }

#pragma omp parallel num_threads(nthreads)
  {

    /* First let's slice up the particles between the threads. */
    const int npart_per_thread = (parts->npart + nthreads - 1) / nthreads;

    /* Get the thread id. */
    const int tid = omp_get_thread_num();

    /* Get the start and end particle indices for this thread. */
    const int start = tid * npart_per_thread;
    int end = start + npart_per_thread;
    if (end > parts->npart) {
      end = parts->npart;
    }

    /* Allocate a local output array. This avoids race conditions and false
     * sharing. */
    std::vector<double> local_out_arr(out_size, 0.0);

    /* Loop over the particles assigned to this thread. */
    for (int p = start; p < end; p++) {

      /* Skip if this particle is masked. */
      if (parts->part_is_masked(p)) {
        continue;
      }

      /* Get this particle's weight. */
      const double weight = parts->get_weight_at(p);

      /* Setup the index and mass fraction arrays. */
      std::array<int, MAX_GRID_NDIM> part_indices;
      std::array<double, MAX_GRID_NDIM> axis_fracs;

      /* Get the grid indices and cell fractions for the particle. */
      get_part_ind_frac_cic(part_indices, axis_fracs, grid_props, parts, p);

      /* Now loop over this collection of cells collecting and setting their
       * weights. */
      for (int icell = 0; icell < num_sub_cells; icell++) {

        /* Set up some index arrays we'll need. */
        std::array<int, MAX_GRID_NDIM> subset_ind;
        std::array<int, MAX_GRID_NDIM> frac_ind;

        /* Get the multi-dimensional version of icell. */
        get_indices_from_flat(icell, ndim, sub_dims, subset_ind);

        /* Multiply all contributing fractions and get the fraction index
         * in the grid. */
        double frac = 1.0;
        for (int idim = 0; idim < ndim; idim++) {
          int offset = subset_ind[idim]; // 0 or 1
          frac *= offset ? axis_fracs[idim] : (1.0 - axis_fracs[idim]);
          frac_ind[idim] = part_indices[idim] + offset;
        }

        /* Nothing to do if fraction is 0. */
        if (frac == 0.0) {
          continue;
        }

        /* Unravel the indices. */
        const int flat_ind = get_flat_index(frac_ind, dims.data(), ndim);

        /* Store the weight in the thread-local output array. */
        local_out_arr[flat_ind] += frac * weight;
      }
    }

    /* Update the global output array. This is the only critical section. */
#pragma omp critical
    {
      for (int i = 0; i < out_size; i++) {
        out_arr[i] += local_out_arr[i];
      }
    }
  }
}
#endif

/**
 * @brief This calculates the grid weights in each grid cell using a cloud
 *        in cell approach.
 *
 * This is a wrapper which calls the correct function based on the number of
 * threads requested and whether OpenMP is available.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param out_size: The size of the output array. (This will be allocated
 * within this function.)
 * @param out: The output array.
 * @param nthreads: The number of threads to use.
 */
void weight_loop_cic(GridProps *grid_props, Particles *parts, int out_size,
                     void *out, const int nthreads) {

  double start_time = tic();

  /* Call the correct function for the configuration/number of threads. */

#ifdef WITH_OPENMP

  /* If we have multiple threads and OpenMP we can parallelise. */
  if (nthreads > 1) {
    weight_loop_cic_omp(grid_props, parts, out_size, out, nthreads);
  }
  /* Otherwise there's no point paying the OpenMP overhead. */
  else {
    weight_loop_cic_serial(grid_props, parts, out);
  }

#else

  (void)nthreads;

  /* We don't have OpenMP, just call the serial version. */
  weight_loop_cic_serial(grid_props, parts, out);

#endif
  toc("Cloud in Cell weight loop", start_time);
}

/**
 * @brief This calculates the grid weights in each grid cell using a nearest
 *        grid point approach.
 *
 * This is the serial version of the function.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param out: The output array.
 */
static void weight_loop_ngp_serial(GridProps *grid_props, Particles *parts,
                                   void *out) {

  /* Unpack the grid properties. */
  std::array<int, MAX_GRID_NDIM> dims = grid_props->dims;
  const int ndim = grid_props->ndim;

  /* Convert out. */
  double *out_arr = static_cast<double *>(out);

  /* Loop over particles. */
  for (int p = 0; p < parts->npart; p++) {

    /* Skip masked particles. */
    if (parts->part_is_masked(p)) {
      continue;
    }

    /* Get this particle's weight. */
    const double weight = parts->get_weight_at(p);

    /* Setup the index array. */
    std::array<int, MAX_GRID_NDIM> part_indices;

    /* Get the grid indices for the particle. */
    get_part_inds_ngp(part_indices, grid_props, parts, p);

    /* Unravel the indices. */
    int flat_ind = get_flat_index(part_indices, dims.data(), ndim);

    /* Store the weight. */
    out_arr[flat_ind] += weight;
  }
}

/**
 * @brief This calculates the grid weights in each grid cell using a nearest
 *        grid point approach.
 *
 * This is the parallel version of the function.
 *
 * Each thread accumulates weights into a private local buffer, which is added
 * into the global output array at the end of the thread’s execution.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param out_size: The size of the output array. (This will be allocated
 *                  within this function.)
 * @param out: The output array.
 * @param nthreads: The number of threads to use.
 */
#ifdef WITH_OPENMP
static void weight_loop_ngp_omp(GridProps *grid_props, Particles *parts,
                                int out_size, void *out, int nthreads) {

  /* Convert out. */
  double *out_arr = static_cast<double *>(out);

  /* Unpack the grid properties. */
  std::array<int, MAX_GRID_NDIM> dims = grid_props->dims;
  const int ndim = grid_props->ndim;

#pragma omp parallel num_threads(nthreads)
  {

    /* First let's slice up the particles between the threads. */
    const int npart_per_thread = (parts->npart + nthreads - 1) / nthreads;

    /* Get the thread id. */
    const int tid = omp_get_thread_num();

    /* Get the start and end particle indices for this thread. */
    const int start = tid * npart_per_thread;
    int end = start + npart_per_thread;
    if (end > parts->npart) {
      end = parts->npart;
    }

    /* Allocate a local output array. This avoids race conditions and false
     * sharing. */
    std::vector<double> local_out_arr(out_size, 0.0);

    /* Loop over the assigned particle range. */
    for (int p = start; p < end; ++p) {

      /* Skip masked particles. */
      if (parts->part_is_masked(p)) {
        continue;
      }

      /* Get this particle's weight. */
      const double weight = parts->get_weight_at(p);

      /* Setup the index array. */
      std::array<int, MAX_GRID_NDIM> part_indices;

      /* Get the grid indices for the particle. */
      get_part_inds_ngp(part_indices, grid_props, parts, p);

      /* Unravel the indices. */
      int flat_ind = get_flat_index(part_indices, dims.data(), ndim);

      /* Store the weight in the thread-local output array. */
      local_out_arr[flat_ind] += weight;
    }

    /* Update the global output array. This is the only critical section. */
#pragma omp critical
    {
      for (int i = 0; i < out_size; i++) {
        out_arr[i] += local_out_arr[i];
      }
    }
  }
}
#endif

/**
 * @brief This calculates the grid weights in each grid cell using a nearest
 *        grid point approach.
 *
 * This is a wrapper which calls the correct function based on the number of
 * threads requested and whether OpenMP is available.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param out_size: The size of the output array. (This will be allocated
 * within this function.)
 * @param out: The output array.
 * @param nthreads: The number of threads to use.
 */
void weight_loop_ngp(GridProps *grid_props, Particles *parts, int out_size,
                     void *out, const int nthreads) {

  double start_time = tic();

  /* Call the correct function for the configuration/number of threads. */

#ifdef WITH_OPENMP

  /* If we have multiple threads and OpenMP we can parallelise. */
  if (nthreads > 1) {
    weight_loop_ngp_omp(grid_props, parts, out_size, out, nthreads);
  }
  /* Otherwise there's no point paying the OpenMP overhead. */
  else {
    weight_loop_ngp_serial(grid_props, parts, out);
  }

#else

  (void)nthreads;

  /* We don't have OpenMP, just call the serial version. */
  weight_loop_ngp_serial(grid_props, parts, out);

#endif
  toc("Nearest Grid Point weight loop", start_time);
}

/**
 * @brief Compute the weight in each grid cell based on the particles.
 *
 * @param grid_props: The Grid object.
 * @param parts: The object containing the particle properties.
 * @param method: The method to use for assigning weights.
 * @param nthreads: The number of threads to use.
 *
 * @return The weights in each grid cell.
 */
PyObject *compute_grid_weights(PyObject *self, PyObject *args) {

  double start_time = tic();
  double setup_start = tic();

  /* We don't need the self argument but it has to be there. Tell the compiler
   * we don't care. */
  (void)self;

  int ndim, npart, nthreads;
  PyObject *grid_tuple, *part_tuple;
  PyArrayObject *np_part_mass, *np_ndims;
  char *method;

  if (!PyArg_ParseTuple(args, "OOOOiisi", &grid_tuple, &part_tuple,
                        &np_part_mass, &np_ndims, &ndim, &npart, &method,
                        &nthreads))
    return nullptr;

  /* Extract the grid struct. */
  GridProps *grid_props =
      new GridProps(/*np_grid_spectra*/ nullptr, grid_tuple,
                    /*np_lam*/ nullptr, /*np_lam_mask*/ nullptr, 1);

  RETURN_IF_PYERR();

  /* Create the object that holds the particle properties. */
  Particles *part_props = new Particles(np_part_mass, /*np_velocities*/ nullptr,
                                        /*np_mask*/ nullptr, part_tuple, npart);

  RETURN_IF_PYERR();

  /* Allocate the sfzh array to output. */
  double *grid_weights = new double[grid_props->size]();
  if (grid_weights == nullptr) {
    PyErr_SetString(PyExc_MemoryError, "Could not allocate memory for sfzh.");
  }
  RETURN_IF_PYERR();

  toc("Extracting Python data", setup_start);

  /* With everything set up we can compute the weights for each particle using
   * the requested method. */
  if (strcmp(method, "cic") == 0) {
    weight_loop_cic(grid_props, part_props, grid_props->size, grid_weights,
                    nthreads);
  } else if (strcmp(method, "ngp") == 0) {
    weight_loop_ngp(grid_props, part_props, grid_props->size, grid_weights,
                    nthreads);
  } else {
    PyErr_SetString(PyExc_ValueError, "Unknown grid assignment method.");
    return nullptr;
  }

  /* Check we got the output. (Any error messages will already be set) */
  if (grid_weights == nullptr) {
    return nullptr;
  }

  /* Reconstruct the python array to return. */
  std::array<npy_intp, MAX_GRID_NDIM> np_dims;
  for (int idim = 0; idim < grid_props->ndim; idim++) {
    np_dims[idim] = grid_props->dims[idim];
  }

  PyArrayObject *out_weights =
      wrap_array_to_numpy(grid_props->ndim, np_dims.data(), grid_weights);

  /* Clean up memory! */
  delete part_props;
  delete grid_props;

  toc("Computing SFZH", start_time);

  return Py_BuildValue("N", out_weights);
}

/* Below is all the gubbins needed to make the module importable in Python. */
static PyMethodDef WeightMethods[] = {
    {"compute_grid_weights", (PyCFunction)compute_grid_weights, METH_VARARGS,
     "Method for calculating the weights on a grid."},
    {nullptr, nullptr, 0, nullptr}};

/* Make this importable. */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "compute_weights",                                    /* m_name */
    "A module to calculating particle weigths on a grid", /* m_doc */
    -1,                                                   /* m_size */
    WeightMethods,                                        /* m_methods */
    nullptr,                                              /* m_reload */
    nullptr,                                              /* m_traverse */
    nullptr,                                              /* m_clear */
    nullptr,                                              /* m_free */
};

PyMODINIT_FUNC PyInit_weights(void) {
  PyObject *m = PyModule_Create(&moduledef);
  if (numpy_import() < 0) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to import numpy.");
    return NULL;
  }
  return m;
}
