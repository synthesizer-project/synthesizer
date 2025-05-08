/******************************************************************************
 * C extension to calculate line of sight metal surface densities for star
 * particles.
 *****************************************************************************/

/* C headers. */
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* Python headers. */
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>

/* Local includes. */
#include "octree.h"
#include "property_funcs.h"
#include "timers.h"

/**
 * @brief Computes the line of sight surface densities with a loop.
 *
 * This is the serial version of the function that computes the line of sight
 * surface densities for each particle. It uses a simple loop over the star
 * particles and the gas particles. This will be used when the number of gas
 * particles is small enough that making a tree is pointless.
 *
 * @param pos_i The positions of the star particles.
 * @param pos_j The positions of the gas particles.
 * @param smls The smoothing lengths of the gas particles.
 * @param surf_den_vals The surface density values of the gas particles.
 * @param kernel The kernel to use for the calculation.
 * @param surf_dens The array to store the surface densities in.
 * @param npart_i The number of star particles.
 * @param npart_j The number of gas particles.
 * @param kdim The dimension of the kernel.
 * @param threshold The threshold for the kernel.
 */
static void los_loop_serial(const double *pos_i, const double *pos_j,
                            const double *smls, const double *surf_den_vals,
                            const double *kernel, double *surf_dens,
                            const int npart_i, const int npart_j,
                            const int kdim, const double threshold) {

  /* Loop over particle postions. */
  for (int i = 0; i < npart_i; i++) {

    double x = pos_i[i * 3];
    double y = pos_i[i * 3 + 1];
    double z = pos_i[i * 3 + 2];

    for (int j = 0; j < npart_j; j++) {

      /* Get gas particle data. */
      double xj = pos_j[j * 3];
      double yj = pos_j[j * 3 + 1];
      double zj = pos_j[j * 3 + 2];
      double sml = smls[j];
      double surf_den_val = surf_den_vals[j];

      /* Skip straight away if the surface density particle is behind the z
       * position. */
      if (zj < z) {
        continue;
      }

      /* Calculate the projected x and y separations. */
      double dx = xj - x;
      double dy = yj - y;

      /* Calculate the impact parameter. */
      double b = sqrt(dx * dx + dy * dy);

      /* Early skip if the star's line of sight doesn't fall in the gas
       * particles kernel. */
      if (b > (threshold * sml))
        continue;

      /* Find fraction of smoothing length. */
      double q = b / sml;

      /* Get the value of the kernel at q. */
      int index = kdim * q;
      double kvalue = kernel[index];

      /* Finally, compute the dust surface density itself. */
      surf_dens[i] += surf_den_val / (sml * sml) * kvalue;
    }
  }
}

/**
 * @brief Computes the line of sight surface densities with a loop.
 *
 * This is the parallel version of the function that computes the line of sight
 * surface densities for each particle. It uses a simple loop over the star
 * particles and the gas particles. This will be used when the number of gas
 * particles is small enough that making a tree is pointless.
 *
 * @param pos_i The positions of the star particles.
 * @param pos_j The positions of the gas particles.
 * @param smls The smoothing lengths of the gas particles.
 * @param surf_den_vals The surface density values of the gas particles.
 * @param kernel The kernel to use for the calculation.
 * @param surf_dens The array to store the surface densities in.
 * @param npart_i The number of star particles.
 * @param npart_j The number of gas particles.
 * @param kdim The dimension of the kernel.
 * @param threshold The threshold for the kernel.
 * @param nthreads The number of threads to use.
 */
#ifdef WITH_OPENMP
static void los_loop_omp(const double *pos_i, const double *pos_j,
                         const double *smls, const double *surf_den_vals,
                         const double *kernel, double *surf_dens,
                         const int npart_i, const int npart_j, const int kdim,
                         const double threshold, const int nthreads) {

  /* How many particles should each thread get? */
  int nparti_per_thread = npart_i / nthreads;

#pragma omp parallel num_threads(nthreads)
  {

    /* Get the thread number. */
    int tid = omp_get_thread_num();

    /* Get the start and end indices for this thread. */
    int start = tid * nparti_per_thread;
    int end = (tid == nthreads - 1) ? npart_i : (tid + 1) * nparti_per_thread;

    /* Get this threads chunk of the results array to write to. We get a chunk
     * here to avoid cache locality issues. */
    double *surf_dens_thread = synth_malloc((end - start) * sizeof(double),
                                            "surface densities thread");

    /* Loop over particle postions. */
    for (int i = start; i < end; i++) {

      /* Get the relative index. */
      int ii = i - start;

      double x = pos_i[i * 3];
      double y = pos_i[i * 3 + 1];
      double z = pos_i[i * 3 + 2];

      for (int j = 0; j < npart_j; j++) {

        /* Get gas particle data. */
        double xj = pos_j[j * 3];
        double yj = pos_j[j * 3 + 1];
        double zj = pos_j[j * 3 + 2];
        double sml = smls[j];
        double surf_den_val = surf_den_vals[j];

        /* Skip straight away if the surface density particle is behind the z
         * position. */
        if (zj < z) {
          continue;
        }

        /* Calculate the projected x and y separations. */
        double dx = xj - x;
        double dy = yj - y;

        /* Calculate the impact parameter. */
        double b = sqrt(dx * dx + dy * dy);

        /* Early skip if the star's line of sight doesn't fall in the gas
         * particles kernel. */
        if (b > (threshold * sml))
          continue;

        /* Find fraction of smoothing length. */
        double q = b / sml;

        /* Get the value of the kernel at q. */
        int index = kdim * q;
        double kvalue = kernel[index];

        /* Finally, compute the dust surface density itself. */
        surf_dens_thread[ii] += surf_den_val / (sml * sml) * kvalue;
      }
    }

    /* Copy the results back to the main array. */
#pragma omp critical
    {
      for (int i = start; i < end; i++) {
        surf_dens[i] = surf_dens_thread[i - start];
      }
    }
  }
}
#endif

/**
 * @brief Computes the line of sight surface densities with a loop.
 *
 * This is a wrapper function which will call the correct version of the
 * function to compute the line of sight surface densities for each particle
 * based on whether or not OpenMP is available and the number of threads to use.
 *
 * @param pos_i The positions of the star particles.
 * @param pos_j The positions of the gas particles.
 * @param smls The smoothing lengths of the gas particles.
 * @param surf_den_vals The surface density values of the gas particles.
 * @param kernel The kernel to use for the calculation.
 * @param surf_dens The array to store the surface densities in.
 * @param npart_i The number of star particles.
 * @param npart_j The number of gas particles.
 * @param kdim The dimension of the kernel.
 * @param threshold The threshold for the kernel.
 * @param nthreads The number of threads to use.
 */
static void los_loop(const double *pos_i, const double *pos_j,
                     const double *smls, const double *surf_den_vals,
                     const double *kernel, double *surf_dens, const int npart_i,
                     const int npart_j, const int kdim, const double threshold,
                     const int nthreads) {

  double start = tic();

#ifdef WITH_OPENMP

  /* If we have multiple threads and OpenMP we can parallelise. */
  if (nthreads > 1) {
    los_loop_omp(pos_i, pos_j, smls, surf_den_vals, kernel, surf_dens, npart_i,
                 npart_j, kdim, threshold, nthreads);
  } else {
    los_loop_serial(pos_i, pos_j, smls, surf_den_vals, kernel, surf_dens,
                    npart_i, npart_j, kdim, threshold);
  }

#else

  (void)nthreads;

  /* If we don't have OpenMP call the serial version. */
  los_loop_serial(pos_i, pos_j, smls, surf_den_vals, kernel, surf_dens, npart_i,
                  npart_j, kdim, threshold);

#endif
  toc("Loop surface density calculation", start);
}

/**
 * @brief Recursively calculate the line of sight surface densities.
 *
 * This will recurse to the leaves of the cell tree, any cells further than the
 * maximum smoothing length from the position will be skipped. Once in the
 * leaves the particles themselves will be checked to see if their SPH kernel
 * overlaps with the line of sight of the star particle.
 *
 * @param c The cell to calculate the surface densities for.
 * @param x The x position of the star particle.
 * @param y The y position of the star particle.
 * @param z The z position of the star particle.
 * @param threshold The threshold for the kernel.
 * @param kdim The dimension of the kernel.
 * @param kernel The kernel to use for the calculation.
 */
static double calculate_los_recursive(struct cell *c, const double x,
                                      const double y, const double z,
                                      double threshold, int kdim,
                                      const double *kernel) {

  /* Early exit if the cell is entirely behind the position. */
  if (c->loc[2] + c->width < z) {
    return 0;
  }

  /* Early exit if the projected distance between cells is more than the
   * maximum smoothing length in the cell. */
  if (c->max_sml_squ < min_projected_dist2(c, x, y)) {
    return 0;
  }

  /* The line of sight dust surface density. */
  double surf_dens = 0.0;

  /* Is the cell split? */
  if (c->split) {

    /* Ok, so we recurse... */
    for (int ip = 0; ip < 8; ip++) {
      struct cell *cp = &c->progeny[ip];

      /* Skip empty progeny. */
      if (cp->part_count == 0) {
        continue;
      }

      /* Recurse... */
      surf_dens +=
          calculate_los_recursive(cp, x, y, z, threshold, kdim, kernel);
    }

  } else {

    /* We're in a leaf if we get here, unpack the particles. */
    int npart_j = c->part_count;
    struct particle *parts = c->particles;

    /* Loop over the particles adding their contribution. */
    for (int j = 0; j < npart_j; j++) {

      /* Get the particle. */
      struct particle *part = &parts[j];

      /* Skip straight away if the gas particle is behind the star. */
      if (part->pos[2] < z) {
        continue;
      }

      /* Calculate the x and y separations. */
      double dx = part->pos[0] - x;
      double dy = part->pos[1] - y;

      /* Calculate the impact parameter. */
      double b = sqrt(dx * dx + dy * dy);

      /* Early skip if the star's line of sight doesn't fall in the gas
       * particles kernel. */
      if (b > (threshold * part->sml)) {
        continue;
      }

      /* Find fraction of smoothing length. */
      double q = b / part->sml;

      /* Get the value of the kernel at q. */
      int index = kdim * q;
      double kvalue = kernel[index];

      /* Finally, compute the surface density itself. */
      surf_dens += part->surf_den_var / (part->sml * part->sml) * kvalue;
    }
  }

  return surf_dens;
}

/**
 * @brief Computes the line of sight surface densities with a tree.
 *
 * This is the serial version of the function that computes the line of sight
 * surface densities for each particle. It uses a tree to store the gas
 * particles and then traverses the tree to find the particles that are within
 * the kernel of the star particle.
 *
 * @param root The root of the tree.
 * @param pos_i The positions of the star particles.
 * @param smls The smoothing lengths of the gas particles.
 * @param surf_den_vals The surface density values of the gas particles.
 * @param kernel The kernel to use for the calculation.
 * @param surf_dens The array to store the surface densities in.
 * @param npart_i The number of star particles.
 * @param kdim The dimension of the kernel.
 * @param threshold The threshold for the kernel.
 */
static void los_tree_serial(struct cell *root, const double *pos_i,
                            const double *kernel, double *surf_dens,
                            const int npart_i, const int kdim,
                            const double threshold) {

  /* Loop over the particles we are calculating the surface density for. */
  for (int i = 0; i < npart_i; i++) {

    /* Start at the root. We'll recurse through the tree to the leaves
     * skipping all cells out of range of this particle. */
    surf_dens[i] =
        calculate_los_recursive(root, pos_i[i * 3], pos_i[i * 3 + 1],
                                pos_i[i * 3 + 2], threshold, kdim, kernel);
  }
}

/**
 * @brief Computes the line of sight surface densities with a tree.
 *
 * This is the parallel version of the function that computes the line of sight
 * surface densities for each particle. It uses a tree to store the gas
 * particles and then traverses the tree to find the particles that are within
 * the kernel of the star particle.
 *
 * @param root The root of the tree.
 * @param pos_i The positions of the star particles.
 * @param smls The smoothing lengths of the gas particles.
 * @param surf_den_vals The surface density values of the gas particles.
 * @param kernel The kernel to use for the calculation.
 * @param surf_dens The array to store the surface densities in.
 * @param npart_i The number of star particles.
 * @param kdim The dimension of the kernel.
 * @param threshold The threshold for the kernel.
 * @param nthreads The number of threads to use.
 */
#ifdef WITH_OPENMP
static void los_tree_omp(struct cell *root, const double *pos_i,
                         const double *kernel, double *surf_dens,
                         const int npart_i, const int kdim,
                         const double threshold, const int nthreads) {

  /* How many particles should each thread get? */
  int nparti_per_thread = npart_i / nthreads;

#pragma omp parallel num_threads(nthreads)
  {

    /* Get the thread number. */
    int tid = omp_get_thread_num();

    /* Get the start and end indices for this thread. */
    int start = tid * nparti_per_thread;
    int end = (tid == nthreads - 1) ? npart_i : (tid + 1) * nparti_per_thread;

    /* Get this threads chunk of the results array to write to. We get a chunk
     * here to avoid cache locality issues. */
    double *surf_dens_thread = synth_malloc((end - start) * sizeof(double),
                                            "surface densities thread");

    /* Loop over the particles we are calculating the surface density for. */
    for (int i = start; i < end; i++) {

      /* Start at the root. We'll recurse through the tree to the leaves
       * skipping all cells out of range of this particle. */
      surf_dens_thread[i - start] =
          calculate_los_recursive(root, pos_i[i * 3], pos_i[i * 3 + 1],
                                  pos_i[i * 3 + 2], threshold, kdim, kernel);
    }

    /* Copy the results back to the main array. */
#pragma omp critical
    {
      memcpy(&surf_dens[start], surf_dens_thread,
             (end - start) * sizeof(double));
    }
  }
}
#endif

/**
 * @brief Computes the line of sight surface densities with a tree.
 *
 * This is a wrapper function which will call the correct version of the
 * function to compute the line of sight surface densities for each particle
 * based on whether or not OpenMP is available and the number of threads to use.
 *
 * @param root The root of the tree.
 * @param pos_i The positions of the star particles.
 * @param smls The smoothing lengths of the gas particles.
 * @param surf_den_vals The surface density values of the gas particles.
 * @param kernel The kernel to use for the calculation.
 * @param surf_dens The array to store the surface densities in.
 * @param npart_i The number of star particles.
 * @param kdim The dimension of the kernel.
 * @param threshold The threshold for the kernel.
 * @param nthreads The number of threads to use.
 */
static void los_tree(struct cell *root, const double *pos_i,
                     const double *kernel, double *surf_dens, const int npart_i,
                     const int kdim, const double threshold,
                     const int nthreads) {

  double start = tic();

#ifdef WITH_OPENMP

  /* If we have multiple threads and OpenMP we can parallelise. */
  if (nthreads > 1) {
    los_tree_omp(root, pos_i, kernel, surf_dens, npart_i, kdim, threshold,
                 nthreads);
  } else {
    los_tree_serial(root, pos_i, kernel, surf_dens, npart_i, kdim, threshold);
  }

#else

  (void)nthreads;

  /* If we don't have OpenMP call the serial version. */
  los_tree_serial(root, pos_i, kernel, surf_dens, npart_i, kdim, threshold);

#endif
  toc("Recursive surface density calculation", start);
}

/**
 * @brief Computes the line of sight surface densities for each particle.
 *
 * This will calculate the line of sight surface densities for of an arbitrary
 * property of one set of particles for the positions of another set of
 * particles.
 *
 * The kernel is assumed to be a 1D array of values that are
 * evaluated at the separations of the particles. The kernel is assumed to be
 * normalised such that the integral of the kernel over all space is 1.
 *
 * @param np_kernel The kernel to use for the calculation.
 * @param np_pos_i The positions of the star particles.
 */
PyObject *compute_column_density(PyObject *self, PyObject *args) {

  /* We don't need the self argument but it has to be there. Tell the compiler
   * we don't care. */
  (void)self;

  int npart_i, npart_j, kdim, force_loop, min_count, nthreads;
  double threshold;
  PyArrayObject *np_kernel, *np_pos_i, *np_pos_j, *np_smls, *np_surf_den_val;

  if (!PyArg_ParseTuple(args, "OOOOOiiidiii", &np_kernel, &np_pos_i, &np_pos_j,
                        &np_smls, &np_surf_den_val, &npart_i, &npart_j, &kdim,
                        &threshold, &force_loop, &min_count, &nthreads))
    return NULL;

  double start = tic();

  /* Quick check to make sure our inputs are valid. */
  if (npart_i == 0) {
    PyErr_SetString(
        PyExc_ValueError,
        "The number of particles to calculate surface densities for "
        "must be greater than zero.");
    return NULL;
  }
  if (npart_j == 0) {
    PyErr_SetString(PyExc_ValueError,
                    "The number of particles to calculate surface densities "
                    "with must be greater than zero.");
    return NULL;
  }
  if (kdim == 0) {
    PyErr_SetString(PyExc_ValueError,
                    "The kernel dimension must be greater than zero.");
    return NULL;
  }

  /* Extract a pointers to the actual data in the numpy arrays. */
  const double *kernel = extract_data_double(np_kernel, "kernel");
  const double *pos_i = extract_data_double(np_pos_i, "pos_i");
  const double *pos_j = extract_data_double(np_pos_j, "pos_j");
  const double *smls = extract_data_double(np_smls, "smls");
  const double *surf_den_val =
      extract_data_double(np_surf_den_val, "surf_den_val");

  /* Set up arrays to hold the surface densities themselves. */
  double *surf_dens =
      synth_malloc(npart_i * sizeof(double), "surface densities");

  /* No point constructing cells if there isn't enough gas to construct a tree
   * below depth 0. (and loop if we've been told to) */
  if (force_loop || npart_j < min_count) {

    /* Use the simple loop over stars and gas. */
    los_loop(pos_i, pos_j, smls, surf_den_val, kernel, surf_dens, npart_i,
             npart_j, kdim, threshold, nthreads);

    /* Reconstruct the python array to return. */
    npy_intp np_dims[1] = {
        npart_i,
    };
    PyArrayObject *out_surf_dens = (PyArrayObject *)PyArray_SimpleNewFromData(
        1, np_dims, NPY_FLOAT64, surf_dens);

    toc("Calculating surface densities (with a loop)", start);

    return Py_BuildValue("N", out_surf_dens);
  }

  /* Allocate cells array. The first cell will be the root and then we will
   * dynamically nibble off cells for the progeny. */
  int ncells = 1;
  struct cell *root = synth_malloc(sizeof(struct cell), "root cell");

  /* Consturct the cell tree. */
  construct_cell_tree(pos_j, smls, surf_den_val, npart_j, root, ncells,
                      MAX_DEPTH, min_count);

  /* Calculate the surface densities. */
  los_tree(root, pos_i, kernel, surf_dens, npart_i, kdim, threshold, nthreads);

  /* Clean up. */
  cleanup_cell_tree(root);

  /* Reconstruct the python array to return. */
  npy_intp np_dims[1] = {npart_i};
  PyArrayObject *out_surf_dens =
      c_array_to_numpy(1, np_dims, NPY_FLOAT64, surf_dens);

  toc("Calculating surface densities (with cells)", start);

  return Py_BuildValue("N", out_surf_dens);
}

/* Below is all the gubbins needed to make the module importable in Python. */
static PyMethodDef LosMethods[] = {
    {"compute_column_density", (PyCFunction)compute_column_density,
     METH_VARARGS, "Method for calculating line of sight surface densities."},
    {NULL, NULL, 0, NULL}};

/* Make this importable. */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "los_surface_dens",                            /* m_name */
    "A module to calculate los surface densities", /* m_doc */
    -1,                                            /* m_size */
    LosMethods,                                    /* m_methods */
    NULL,                                          /* m_reload */
    NULL,                                          /* m_traverse */
    NULL,                                          /* m_clear */
    NULL,                                          /* m_free */
};

PyMODINIT_FUNC PyInit_column_density(void) {
  PyObject *m = PyModule_Create(&moduledef);
  import_array();
  return m;
}
