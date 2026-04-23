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
#define PY_ARRAY_UNIQUE_SYMBOL SYNTHESIZER_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy_init.h"
#include <Python.h>

/* Local includes. */
#include "cpp_to_python.h"
#include "kernel_utils.h"
#include "octree.h"
#include "property_funcs.h"
#include "timers.h"
#ifdef ATOMIC_TIMING
#include "timers_init.h"
#endif

/**
 * @brief Computes the line of sight surface densities with a loop.
 *
 * This is the serial version of the function that computes the line of sight
 * surface densities for each particle. It uses a simple loop over the star
 * particles and the gas particles. This will be used when the number of gas
 * particles is small enough that making a tree is pointless.
 *
 * @param pos_i The positions of the particles to compute the surface
 *             densities for (e.g. star particles).
 * @param pos_j The positions of the particles to compute the surface
 *             densities from (e.g. gas particles).
 * @param smls The smoothing lengths of the particles to compute the
 *            surface densities from.
 * @param surf_den_vals The surface density values of the particles to compute
 *            the surface densities from.
 * @param kernel The projected LOS kernel lookup table.
 * @param truncated_kernel The truncated LOS kernel lookup table storing
 *        cumulative LOS contributions for inside-kernel paths.
 * @param surf_dens The array to store the surface densities in.
 * @param npart_i The number of star particles.
 * @param npart_j The number of gas particles.
 * @param kdim The number of projected-kernel entries.
 * @param trunc_qdim The number of projected-separation entries in the
 *        truncated kernel table.
 * @param zdim The number of LOS-coordinate entries in the truncated table.
 * @param threshold The threshold for the kernel.
 */
static void los_loop_serial(const double *pos_i, const double *pos_j,
                            const double *smls, const double *surf_den_vals,
                            const double *kernel,
                            const double *truncated_kernel,
                            double *surf_dens, const int npart_i,
                            const int npart_j, const int kdim,
                            const int trunc_qdim,
                            const int zdim, const double threshold) {

  /* Loop over particle postions. */
  for (int i = 0; i < npart_i; i++) {

    double x = pos_i[i * 3];
    double y = pos_i[i * 3 + 1];
    double z = pos_i[i * 3 + 2];

    /* Loop over other particle postions. */
    for (int j = 0; j < npart_j; j++) {

      /* Get gas particle data. */
      double xj = pos_j[j * 3];
      double yj = pos_j[j * 3 + 1];
      double zj = pos_j[j * 3 + 2];
      double sml = smls[j];
      double surf_den_val = surf_den_vals[j];

      /* Skip straight away if the source kernel lies entirely behind the
       * input position. */
      if ((zj - threshold * sml) > z) {
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

      /* If the input lies inside the source kernel we need the truncated LOS
       * contribution. Otherwise the input sees the full projected kernel. */
      double kvalue = 0.0;
      if (z < (zj + threshold * sml)) {
        const double z_trunc = (z - zj) / (threshold * sml);
        kvalue = get_truncated_kernel_value(
            truncated_kernel, trunc_qdim, zdim, q / threshold, z_trunc);
      } else {
        kvalue = get_kernel_value(kernel, kdim, q / threshold);
      }

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
 * @param pos_i The positions of the particles to compute the surface
 *             densities for (e.g. star particles).
 * @param pos_j The positions of the particles to compute the surface
 *             densities from (e.g. gas particles).
 * @param smls The smoothing lengths of the particles to compute the
 *            surface densities from.
 * @param surf_den_vals The surface density values of the particles to compute
 *            the surface densities from.
 * @param kernel The projected LOS kernel lookup table.
 * @param truncated_kernel The truncated LOS kernel lookup table storing
 *        cumulative LOS contributions for inside-kernel paths.
 * @param surf_dens The array to store the surface densities in.
 * @param npart_i The number of star particles.
 * @param npart_j The number of gas particles.
 * @param kdim The number of projected-kernel entries.
 * @param trunc_qdim The number of projected-separation entries in the
 *        truncated kernel table.
 * @param zdim The number of LOS-coordinate entries in the truncated table.
 * @param threshold The threshold for the kernel.
 * @param nthreads The number of threads to use.
 */
#ifdef WITH_OPENMP
static void los_loop_omp(const double *pos_i, const double *pos_j,
                         const double *smls, const double *surf_den_vals,
                         const double *kernel,
                         const double *truncated_kernel, double *surf_dens,
                         const int npart_i, const int npart_j,
                         const int kdim, const int trunc_qdim,
                         const int zdim,
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
    double *surf_dens_thread = new double[end - start]();

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

        /* Skip straight away if the source kernel lies entirely behind the
         * input position. */
        if ((zj - threshold * sml) > z) {
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

        /* If the input lies inside the source kernel we need the truncated LOS
         * contribution. Otherwise the input sees the full projected kernel. */
        double kvalue = 0.0;
        if (z < (zj + threshold * sml)) {
          const double z_trunc = (z - zj) / (threshold * sml);
          kvalue = get_truncated_kernel_value(
              truncated_kernel, trunc_qdim, zdim, q / threshold, z_trunc);
        } else {
          kvalue = get_kernel_value(kernel, kdim, q / threshold);
        }

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

    /* Clean up the thread's chunk of the results array. */
    delete[] surf_dens_thread;
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
 * @param kernel The projected LOS kernel lookup table.
 * @param truncated_kernel The truncated LOS kernel lookup table storing
 *        cumulative LOS contributions for inside-kernel paths.
 * @param surf_dens The array to store the surface densities in.
 * @param npart_i The number of star particles.
 * @param npart_j The number of gas particles.
 * @param kdim The number of projected-kernel entries.
 * @param trunc_qdim The number of projected-separation entries in the
 *        truncated kernel table.
 * @param zdim The number of LOS-coordinate entries in the truncated table.
 * @param threshold The threshold for the kernel.
 * @param nthreads The number of threads to use.
 */
static void los_loop(const double *pos_i, const double *pos_j,
                     const double *smls, const double *surf_den_vals,
                     const double *kernel, const double *truncated_kernel,
                     double *surf_dens, const int npart_i, const int npart_j,
                     const int kdim, const int trunc_qdim,
                     const int zdim, const double threshold,
                     const int nthreads) {

  tic("los_loop");

#ifdef WITH_OPENMP

  /* If we have multiple threads and OpenMP we can parallelise. */
  if (nthreads > 1) {
    los_loop_omp(pos_i, pos_j, smls, surf_den_vals, kernel, truncated_kernel,
                 surf_dens, npart_i, npart_j, kdim, trunc_qdim, zdim, threshold,
                 nthreads);
  } else {
    los_loop_serial(pos_i, pos_j, smls, surf_den_vals, kernel,
                    truncated_kernel, surf_dens, npart_i, npart_j, kdim,
                    trunc_qdim, zdim, threshold);
  }

#else

  (void)nthreads;

  /* If we don't have OpenMP call the serial version. */
  los_loop_serial(pos_i, pos_j, smls, surf_den_vals, kernel,
                  truncated_kernel, surf_dens, npart_i, npart_j, kdim,
                  trunc_qdim, zdim, threshold);

#endif
  toc("los_loop");
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
 * @param kdim The number of projected-kernel entries.
 * @param trunc_qdim The number of projected-separation entries in the
 *        truncated kernel table.
 * @param zdim The number of LOS-coordinate entries in the truncated table.
 * @param kernel The projected LOS kernel lookup table.
 * @param truncated_kernel The truncated LOS kernel lookup table storing
 *        cumulative LOS contributions for inside-kernel paths.
 */
static double calculate_los_recursive(struct cell *c, const double x,
                                      const double y, const double z,
                                      double threshold, int kdim,
                                      int trunc_qdim, int zdim,
                                      const double *kernel,
                                      const double *truncated_kernel) {

  /* Early exit if the cell is entirely behind the position. */
  if ((c->loc[2] - threshold * sqrt(c->max_sml_squ)) > z) {
    return 0;
  }

  /* Early exit if the projected distance between cells is more than the
   * maximum smoothing length in the cell. */
  if (c->max_sml_squ * (threshold * threshold) < min_projected_dist2(c, x, y)) {
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
      surf_dens += calculate_los_recursive(
          cp, x, y, z, threshold, kdim, trunc_qdim, zdim, kernel,
          truncated_kernel);
    }

  } else {

    /* We're in a leaf if we get here, unpack the particles. */
    int npart_j = c->part_count;
    struct particle *parts = c->particles;

    /* Loop over the particles adding their contribution. */
    for (int j = 0; j < npart_j; j++) {

      /* Get the particle. */
      struct particle *part = &parts[j];

      /* Skip straight away if the source kernel lies entirely behind the
       * input position. */
      if ((part->pos[2] - threshold * part->sml) > z) {
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

      /* If the input lies inside the source kernel we need the truncated LOS
       * contribution. Otherwise the input sees the full projected kernel. */
      double kvalue = 0.0;
      if (z < (part->pos[2] + threshold * part->sml)) {
        const double z_trunc = (z - part->pos[2]) / (threshold * part->sml);
        kvalue = get_truncated_kernel_value(
            truncated_kernel, trunc_qdim, zdim, q / threshold, z_trunc);
      } else {
        kvalue = get_kernel_value(kernel, kdim, q / threshold);
      }

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
 * @param kernel The projected LOS kernel lookup table.
 * @param truncated_kernel The truncated LOS kernel lookup table storing
 *        cumulative LOS contributions for inside-kernel paths.
 * @param surf_dens The array to store the surface densities in.
 * @param npart_i The number of star particles.
 * @param kdim The number of projected-kernel entries.
 * @param trunc_qdim The number of projected-separation entries in the
 *        truncated kernel table.
 * @param zdim The number of LOS-coordinate entries in the truncated table.
 * @param threshold The threshold for the kernel.
 */
static void los_tree_serial(struct cell *root, const double *pos_i,
                            const double *kernel,
                            const double *truncated_kernel, double *surf_dens,
                            const int npart_i, const int kdim,
                            const int trunc_qdim, const int zdim,
                            const double threshold) {

  /* Loop over the particles we are calculating the surface density for. */
  for (int i = 0; i < npart_i; i++) {

      /* Start at the root. We'll recurse through the tree to the leaves
       * skipping all cells out of range of this particle. */
    surf_dens[i] = calculate_los_recursive(
        root, pos_i[i * 3], pos_i[i * 3 + 1], pos_i[i * 3 + 2], threshold,
        kdim, trunc_qdim, zdim, kernel, truncated_kernel);
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
 * @param kernel The projected LOS kernel lookup table.
 * @param truncated_kernel The truncated LOS kernel lookup table storing
 *        cumulative LOS contributions for inside-kernel paths.
 * @param surf_dens The array to store the surface densities in.
 * @param npart_i The number of star particles.
 * @param kdim The number of projected-kernel entries.
 * @param trunc_qdim The number of projected-separation entries in the
 *        truncated kernel table.
 * @param zdim The number of LOS-coordinate entries in the truncated table.
 * @param threshold The threshold for the kernel.
 * @param nthreads The number of threads to use.
 */
#ifdef WITH_OPENMP
static void los_tree_omp(struct cell *root, const double *pos_i,
                         const double *kernel,
                         const double *truncated_kernel, double *surf_dens,
                         const int npart_i, const int kdim,
                         const int trunc_qdim, const int zdim,
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
    double *surf_dens_thread = new double[end - start]();

    /* Loop over the particles we are calculating the surface density for. */
    for (int i = start; i < end; i++) {

      /* Start at the root. We'll recurse through the tree to the leaves
       * skipping all cells out of range of this particle. */
      surf_dens_thread[i - start] = calculate_los_recursive(
          root, pos_i[i * 3], pos_i[i * 3 + 1], pos_i[i * 3 + 2], threshold,
          kdim, trunc_qdim, zdim, kernel, truncated_kernel);
    }

    /* Copy the results back to the main array. */
#pragma omp critical
    {
      memcpy(&surf_dens[start], surf_dens_thread,
             (end - start) * sizeof(double));
    }

    /* Clean up the thread-local buffer once its results have been copied. */
    delete[] surf_dens_thread;
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
 * @param kernel The projected LOS kernel lookup table.
 * @param truncated_kernel The truncated LOS kernel lookup table storing
 *        cumulative LOS contributions for inside-kernel paths.
 * @param surf_dens The array to store the surface densities in.
 * @param npart_i The number of star particles.
 * @param kdim The number of projected-kernel entries.
 * @param trunc_qdim The number of projected-separation entries in the
 *        truncated kernel table.
 * @param zdim The number of LOS-coordinate entries in the truncated table.
 * @param threshold The threshold for the kernel.
 * @param nthreads The number of threads to use.
 */
static void los_tree(struct cell *root, const double *pos_i,
                     const double *kernel, const double *truncated_kernel,
                     double *surf_dens, const int npart_i, const int kdim,
                     const int trunc_qdim, const int zdim,
                     const double threshold,
                     const int nthreads) {

  tic("los_tree");

#ifdef WITH_OPENMP

  /* If we have multiple threads and OpenMP we can parallelise. */
  if (nthreads > 1) {
    los_tree_omp(root, pos_i, kernel, truncated_kernel, surf_dens, npart_i,
                 kdim, trunc_qdim, zdim, threshold, nthreads);
  } else {
    los_tree_serial(root, pos_i, kernel, truncated_kernel, surf_dens, npart_i,
                    kdim, trunc_qdim, zdim, threshold);
  }

#else

  (void)nthreads;

  /* If we don't have OpenMP call the serial version. */
  los_tree_serial(root, pos_i, kernel, truncated_kernel, surf_dens, npart_i,
                  kdim, trunc_qdim, zdim, threshold);

#endif
  toc("los_tree");
}

/**
 * @brief Computes the line of sight surface densities for each particle.
 *
 * This will calculate the line of sight surface densities for of an arbitrary
 * property of one set of particles for the positions of another set of
 * particles.
 *
 * The projected kernel is assumed to be a 1D array of values evaluated at the
 * separations of the particles. The truncated kernel is assumed to be a 2D
 * lookup table tabulated in projected separation and LOS truncation
 * coordinate. Both are assumed to be normalised consistently with the Python
 * ``Kernel`` helper.
 *
 * @param np_kernel The projected LOS kernel lookup table.
 * @param np_truncated_kernel The truncated LOS kernel lookup table.
 * @param np_pos_i The positions of the star particles.
 */
PyObject *compute_column_density(PyObject *self, PyObject *args) {

  /* We don't need the self argument but it has to be there. Tell the compiler
   * we don't care. */
  (void)self;

  int npart_i, npart_j, kdim, trunc_qdim, zdim, force_loop, min_count,
      nthreads;
  double threshold;
  PyArrayObject *np_kernel, *np_truncated_kernel, *np_pos_i, *np_pos_j,
      *np_smls, *np_surf_den_val;

  if (!PyArg_ParseTuple(args, "OOOOOOiiiiidiii", &np_kernel,
                         &np_truncated_kernel, &np_pos_i, &np_pos_j, &np_smls,
                         &np_surf_den_val, &npart_i, &npart_j, &kdim,
                         &trunc_qdim, &zdim, &threshold, &force_loop,
                         &min_count, &nthreads))
    return NULL;

  tic("compute_column_density");

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
  if (zdim == 0) {
    PyErr_SetString(PyExc_ValueError,
                    "The truncated kernel dimension must be greater than "
                    "zero.");
    return NULL;
  }
  if (trunc_qdim == 0) {
    PyErr_SetString(PyExc_ValueError,
                    "The truncated kernel q-dimension must be greater than "
                    "zero.");
    return NULL;
  }

  /* Extract a pointers to the actual data in the numpy arrays. */
  const double *kernel = extract_data_double(np_kernel, "kernel");
  const double *truncated_kernel =
      extract_data_double(np_truncated_kernel, "truncated_kernel");
  const double *pos_i = extract_data_double(np_pos_i, "pos_i");
  const double *pos_j = extract_data_double(np_pos_j, "pos_j");
  const double *smls = extract_data_double(np_smls, "smls");
  const double *surf_den_val =
      extract_data_double(np_surf_den_val, "surf_den_val");

  /* One of the data extractions failed and set a Python error. Return NULL
   * to propagate the exception back to Python. */
  if (kernel == NULL || truncated_kernel == NULL || pos_i == NULL ||
      pos_j == NULL || smls == NULL || surf_den_val == NULL) {
    return NULL;
  }

  /* Create the output array. */
  npy_intp np_dims[1] = {npart_i};
  PyArrayObject *np_surf_dens =
      (PyArrayObject *)PyArray_ZEROS(1, np_dims, NPY_DOUBLE, 0);
  double *surf_dens = static_cast<double *>(PyArray_DATA(np_surf_dens));

  /* No point constructing cells if there isn't enough gas to construct a tree
   * below depth 0. (and loop if we've been told to) */
  if (force_loop || npart_j < min_count) {

    /* Use the simple loop over stars and gas. */
    los_loop(pos_i, pos_j, smls, surf_den_val, kernel, truncated_kernel,
             surf_dens, npart_i, npart_j, kdim, trunc_qdim, zdim, threshold,
             nthreads);

    toc("compute_column_density");

    return Py_BuildValue("N", np_surf_dens);
  }

  /* Allocate cells array. The first cell will be the root and then we will
   * dynamically nibble off cells for the progeny. */
  int ncells = 1;
  struct cell *root = new struct cell;

  /* Consturct the cell tree. */
  construct_cell_tree(pos_j, smls, surf_den_val, npart_j, root, ncells,
                      MAX_DEPTH, min_count);

  /* Calculate the surface densities. */
  los_tree(root, pos_i, kernel, truncated_kernel, surf_dens, npart_i, kdim,
           trunc_qdim, zdim, threshold, nthreads);

  /* Clean up. */
  cleanup_cell_tree(root);

  toc("compute_column_density");

  return Py_BuildValue("N", np_surf_dens);
}

/**
 * @brief Compute one pairwise smoothed-input LOS contribution.
 *
 * Each input/source particle pair is mapped onto the precomputed overlap table
 * `G(q, u, eta)` and contributes exactly once.
 *
 * The runtime pair coordinates are defined using the support radii of the input
 * and source particles:
 *
 *   q   = b / (R_i + R_j)
 *   u   = (z_i - z_j) / (R_i + R_j)
 *   eta = h_i / h_j
 *
 * where `R = threshold * h` is the effective support radius used by the LOS
 * calculation. The overlap table then returns the kernel-averaged fraction of
 * the source LOS kernel seen by the input particle.
 *
 * @param xi The x position of the input particle.
 * @param yi The y position of the input particle.
 * @param zi The z position of the input particle.
 * @param hi The smoothing length of the input particle.
 * @param xj The x position of the source particle.
 * @param yj The y position of the source particle.
 * @param zj The z position of the source particle.
 * @param hj The smoothing length of the source particle.
 * @param surf_den_val The source value to accumulate.
 * @param overlap_kernel The overlap kernel lookup table.
 * @param q_grid The q-axis of the overlap table.
 * @param u_grid The u-axis of the overlap table.
 * @param eta_grid The eta-axis of the overlap table.
 * @param qdim The number of q-grid entries.
 * @param udim The number of u-grid entries.
 * @param etadim The number of eta-grid entries.
 * @param threshold The support threshold in units of the smoothing length.
 *
 * @return The LOS surface density contribution for this pair.
 */
static inline double calculate_smoothed_pair_contribution(
    const double xi, const double yi, const double zi, const double hi,
    const double xj, const double yj, const double zj, const double hj,
    const double surf_den_val, const double *overlap_kernel,
    const double *q_grid, const double *u_grid, const double *eta_grid,
    const int qdim, const int udim, const int etadim, const double threshold) {

  /* Reject pathological zero-support pairs before doing any further work. */
  if (hj <= 0.0) {
    return 0.0;
  }

  /* Convert smoothing lengths to the support radii used by the LOS solver. */
  const double support_i = threshold * hi;
  const double support_j = threshold * hj;
  const double support_sum = support_i + support_j;
  if (support_sum <= 0.0) {
    return 0.0;
  }

  /* Compute the projected particle-centre separation. We stay in squared form
   * for the first overlap rejection to avoid an unnecessary square root in the
   * common no-overlap case. */
  const double dx = xj - xi;
  const double dy = yj - yi;
  const double b2 = dx * dx + dy * dy;
  const double support_sum2 = support_sum * support_sum;

  /* If the projected supports do not overlap, the kernel average is zero. */
  if (b2 >= support_sum2) {
    return 0.0;
  }

  /* If the source kernel lies entirely behind the input kernel there is no
   * foreground contribution anywhere within the input support. */
  if ((zj - support_j) >= (zi + support_i)) {
    return 0.0;
  }

  /* Map this pair onto the tabulated overlap coordinates. */
  const double b = sqrt(b2);
  const double q = b / support_sum;
  const double u = (zi - zj) / support_sum;
  const double eta = hi / hj;

  /* The table contains the kernel-averaged LOS contribution. Only the source
   * normalisation remains to be applied at runtime. */
  return surf_den_val / (hj * hj) *
         get_overlap_kernel_value(overlap_kernel, q_grid, u_grid, eta_grid,
                                  qdim, udim, etadim, q, u, eta);
}

/**
 * @brief Recursively calculate smoothed-input LOS surface densities.
 *
 * The source tree is traversed once per input particle. Internal nodes are
 * pruned when even the largest possible source support in that node cannot
 * overlap the input support in projection or along the line of sight. Once we
 * reach a leaf, each source particle contributes exactly once via the overlap
 * table.
 *
 * @param c The cell to traverse.
 * @param xi The x position of the input particle.
 * @param yi The y position of the input particle.
 * @param zi The z position of the input particle.
 * @param hi The smoothing length of the input particle.
 * @param overlap_kernel The overlap kernel lookup table.
 * @param q_grid The q-axis of the overlap table.
 * @param u_grid The u-axis of the overlap table.
 * @param eta_grid The eta-axis of the overlap table.
 * @param qdim The number of q-grid entries.
 * @param udim The number of u-grid entries.
 * @param etadim The number of eta-grid entries.
 * @param threshold The support threshold in units of the smoothing length.
 *
 * @return The LOS surface density contribution from this node.
 */
static double calculate_los_recursive_smoothed(
    struct cell *c, const double xi, const double yi, const double zi,
    const double hi, const double *overlap_kernel, const double *q_grid,
    const double *u_grid, const double *eta_grid, const int qdim,
    const int udim, const int etadim, const double threshold) {

  /* The input-particle support radius is fixed along this traversal. */
  const double support_i = threshold * hi;

  /* The node can only contain source particles up to this support radius. */
  const double support_max = threshold * sqrt(c->max_sml_squ);

  /* Reject nodes whose source kernels are entirely behind the full input
   * support. */
  const double cell_z_min = c->loc[2] - support_max;
  if (cell_z_min >= (zi + support_i)) {
    return 0.0;
  }

  /* Reject nodes whose projected bounding box stays further away than the
   * combined input and maximum source support. */
  const double support_sum = support_i + support_max;
  if ((support_sum * support_sum) < min_projected_dist2(c, xi, yi)) {
    return 0.0;
  }

  /* Accumulate the contribution from this node. */
  double surf_dens = 0.0;

  if (c->split) {

    /* Internal nodes are traversed recursively until we reach leaves, where we
     * can evaluate the exact pairwise overlap with each source particle. */
    for (int ip = 0; ip < 8; ip++) {
      struct cell *cp = &c->progeny[ip];

      if (cp->part_count == 0) {
        continue;
      }

      surf_dens += calculate_los_recursive_smoothed(
          cp, xi, yi, zi, hi, overlap_kernel, q_grid, u_grid, eta_grid, qdim,
          udim, etadim, threshold);
    }

  } else {

    /* Once in a leaf we are back to the same one-pair-at-a-time structure as
     * the simple loop implementation. */
    const int npart_j = c->part_count;
    struct particle *parts = c->particles;

    for (int j = 0; j < npart_j; j++) {
      struct particle *part = &parts[j];

      surf_dens += calculate_smoothed_pair_contribution(
          xi, yi, zi, hi, part->pos[0], part->pos[1], part->pos[2], part->sml,
          part->surf_den_var, overlap_kernel, q_grid, u_grid, eta_grid, qdim,
          udim, etadim, threshold);
    }
  }

  return surf_dens;
}

/**
 * @brief Compute smoothed-input LOS surface densities with a loop.
 *
 * For each input particle we loop over source particles, reject non-overlapping
 * pairs, interpolate the overlap table once, and apply the source prefactor.
 *
 * @param pos_i The positions of the input particles.
 * @param input_smls The smoothing lengths of the input particles.
 * @param pos_j The positions of the source particles.
 * @param smls The smoothing lengths of the source particles.
 * @param surf_den_vals The source values to accumulate.
 * @param overlap_kernel The overlap kernel lookup table.
 * @param q_grid The q-axis of the overlap table.
 * @param u_grid The u-axis of the overlap table.
 * @param eta_grid The eta-axis of the overlap table.
 * @param surf_dens The array to store the surface densities in.
 * @param npart_i The number of input particles.
 * @param npart_j The number of source particles.
 * @param qdim The number of q-grid entries.
 * @param udim The number of u-grid entries.
 * @param etadim The number of eta-grid entries.
 * @param threshold The support threshold in units of the smoothing length.
 */
static void los_loop_smoothed_serial(
    const double *pos_i, const double *input_smls, const double *pos_j,
    const double *smls, const double *surf_den_vals,
    const double *overlap_kernel, const double *q_grid, const double *u_grid,
    const double *eta_grid, double *surf_dens, const int npart_i,
    const int npart_j, const int qdim, const int udim, const int etadim,
    const double threshold) {

  /* Loop over the input particles. */
  for (int i = 0; i < npart_i; i++) {

    const double xi = pos_i[i * 3];
    const double yi = pos_i[i * 3 + 1];
    const double zi = pos_i[i * 3 + 2];
    const double hi = input_smls[i];

    /* Each source particle contributes at most once to this input particle. */
    for (int j = 0; j < npart_j; j++) {
      surf_dens[i] += calculate_smoothed_pair_contribution(
          xi, yi, zi, hi, pos_j[j * 3], pos_j[j * 3 + 1], pos_j[j * 3 + 2],
          smls[j], surf_den_vals[j], overlap_kernel, q_grid, u_grid, eta_grid,
          qdim, udim, etadim, threshold);
    }
  }
}

#ifdef WITH_OPENMP
/**
 * @brief Parallel smoothed-input LOS loop implementation.
 *
 * The input particles are split into contiguous chunks and each thread writes
 * to a thread-local result buffer before copying back to the shared output
 * array.
 *
 * @param pos_i The positions of the input particles.
 * @param input_smls The smoothing lengths of the input particles.
 * @param pos_j The positions of the source particles.
 * @param smls The smoothing lengths of the source particles.
 * @param surf_den_vals The source values to accumulate.
 * @param overlap_kernel The overlap kernel lookup table.
 * @param q_grid The q-axis of the overlap table.
 * @param u_grid The u-axis of the overlap table.
 * @param eta_grid The eta-axis of the overlap table.
 * @param surf_dens The array to store the surface densities in.
 * @param npart_i The number of input particles.
 * @param npart_j The number of source particles.
 * @param qdim The number of q-grid entries.
 * @param udim The number of u-grid entries.
 * @param etadim The number of eta-grid entries.
 * @param threshold The support threshold in units of the smoothing length.
 * @param nthreads The number of threads to use.
 */
static void los_loop_smoothed_omp(
    const double *pos_i, const double *input_smls, const double *pos_j,
    const double *smls, const double *surf_den_vals,
    const double *overlap_kernel, const double *q_grid, const double *u_grid,
    const double *eta_grid, double *surf_dens, const int npart_i,
    const int npart_j, const int qdim, const int udim, const int etadim,
    const double threshold, const int nthreads) {

  int nparti_per_thread = npart_i / nthreads;

#pragma omp parallel num_threads(nthreads)
  {
    int tid = omp_get_thread_num();
    int start = tid * nparti_per_thread;
    int end = (tid == nthreads - 1) ? npart_i : (tid + 1) * nparti_per_thread;

    double *surf_dens_thread = new double[end - start]();

    los_loop_smoothed_serial(&pos_i[start * 3], &input_smls[start], pos_j, smls,
                             surf_den_vals, overlap_kernel, q_grid, u_grid,
                             eta_grid, surf_dens_thread, end - start, npart_j,
                             qdim, udim, etadim, threshold);

#pragma omp critical
    {
      memcpy(&surf_dens[start], surf_dens_thread,
             sizeof(double) * (end - start));
    }

    delete[] surf_dens_thread;
  }
}
#endif

/**
 * @brief Wrapper around the smoothed-input LOS loop implementations.
 *
 * @param pos_i The positions of the input particles.
 * @param input_smls The smoothing lengths of the input particles.
 * @param pos_j The positions of the source particles.
 * @param smls The smoothing lengths of the source particles.
 * @param surf_den_vals The source values to accumulate.
 * @param overlap_kernel The overlap kernel lookup table.
 * @param q_grid The q-axis of the overlap table.
 * @param u_grid The u-axis of the overlap table.
 * @param eta_grid The eta-axis of the overlap table.
 * @param surf_dens The array to store the surface densities in.
 * @param npart_i The number of input particles.
 * @param npart_j The number of source particles.
 * @param qdim The number of q-grid entries.
 * @param udim The number of u-grid entries.
 * @param etadim The number of eta-grid entries.
 * @param threshold The support threshold in units of the smoothing length.
 * @param nthreads The number of threads to use.
 */
static void los_loop_smoothed(const double *pos_i, const double *input_smls,
                              const double *pos_j, const double *smls,
                              const double *surf_den_vals,
                              const double *overlap_kernel,
                              const double *q_grid, const double *u_grid,
                              const double *eta_grid, double *surf_dens,
                              const int npart_i, const int npart_j,
                              const int qdim, const int udim, const int etadim,
                              const double threshold, const int nthreads) {

  tic("Loop surface density calculation with smoothed inputs");

#ifdef WITH_OPENMP
  if (nthreads > 1) {
    los_loop_smoothed_omp(pos_i, input_smls, pos_j, smls, surf_den_vals,
                          overlap_kernel, q_grid, u_grid, eta_grid, surf_dens,
                          npart_i, npart_j, qdim, udim, etadim, threshold,
                          nthreads);
  } else {
    los_loop_smoothed_serial(pos_i, input_smls, pos_j, smls, surf_den_vals,
                             overlap_kernel, q_grid, u_grid, eta_grid,
                             surf_dens, npart_i, npart_j, qdim, udim, etadim,
                             threshold);
  }
#else
  (void)nthreads;
  los_loop_smoothed_serial(pos_i, input_smls, pos_j, smls, surf_den_vals,
                           overlap_kernel, q_grid, u_grid, eta_grid, surf_dens,
                           npart_i, npart_j, qdim, udim, etadim, threshold);
#endif

  toc("Loop surface density calculation with smoothed inputs");
}

/**
 * @brief Compute smoothed-input LOS surface densities with a tree.
 *
 * Python prepares arrays, the extension traverses the source tree once per
 * input particle, and each source particle contributes exactly once at the
 * leaves.
 *
 * @param root The root of the source-particle tree.
 * @param pos_i The positions of the input particles.
 * @param input_smls The smoothing lengths of the input particles.
 * @param overlap_kernel The overlap kernel lookup table.
 * @param q_grid The q-axis of the overlap table.
 * @param u_grid The u-axis of the overlap table.
 * @param eta_grid The eta-axis of the overlap table.
 * @param surf_dens The array to store the surface densities in.
 * @param npart_i The number of input particles.
 * @param qdim The number of q-grid entries.
 * @param udim The number of u-grid entries.
 * @param etadim The number of eta-grid entries.
 * @param threshold The support threshold in units of the smoothing length.
 */
static void los_tree_smoothed_serial(
    struct cell *root, const double *pos_i, const double *input_smls,
    const double *overlap_kernel, const double *q_grid, const double *u_grid,
    const double *eta_grid, double *surf_dens, const int npart_i,
    const int qdim, const int udim, const int etadim, const double threshold) {

  for (int i = 0; i < npart_i; i++) {
    surf_dens[i] = calculate_los_recursive_smoothed(
        root, pos_i[i * 3], pos_i[i * 3 + 1], pos_i[i * 3 + 2], input_smls[i],
        overlap_kernel, q_grid, u_grid, eta_grid, qdim, udim, etadim,
        threshold);
  }
}

#ifdef WITH_OPENMP
/**
 * @brief Parallel smoothed-input LOS tree implementation.
 *
 * @param root The root of the source-particle tree.
 * @param pos_i The positions of the input particles.
 * @param input_smls The smoothing lengths of the input particles.
 * @param overlap_kernel The overlap kernel lookup table.
 * @param q_grid The q-axis of the overlap table.
 * @param u_grid The u-axis of the overlap table.
 * @param eta_grid The eta-axis of the overlap table.
 * @param surf_dens The array to store the surface densities in.
 * @param npart_i The number of input particles.
 * @param qdim The number of q-grid entries.
 * @param udim The number of u-grid entries.
 * @param etadim The number of eta-grid entries.
 * @param threshold The support threshold in units of the smoothing length.
 * @param nthreads The number of threads to use.
 */
static void los_tree_smoothed_omp(struct cell *root, const double *pos_i,
                                  const double *input_smls,
                                  const double *overlap_kernel,
                                  const double *q_grid, const double *u_grid,
                                  const double *eta_grid, double *surf_dens,
                                  const int npart_i, const int qdim,
                                  const int udim, const int etadim,
                                  const double threshold, const int nthreads) {

  int nparti_per_thread = npart_i / nthreads;

#pragma omp parallel num_threads(nthreads)
  {
    int tid = omp_get_thread_num();
    int start = tid * nparti_per_thread;
    int end = (tid == nthreads - 1) ? npart_i : (tid + 1) * nparti_per_thread;

    double *surf_dens_thread = new double[end - start]();

    los_tree_smoothed_serial(root, &pos_i[start * 3], &input_smls[start],
                             overlap_kernel, q_grid, u_grid, eta_grid,
                             surf_dens_thread, end - start, qdim, udim, etadim,
                             threshold);

#pragma omp critical
    {
      memcpy(&surf_dens[start], surf_dens_thread,
             sizeof(double) * (end - start));
    }

    delete[] surf_dens_thread;
  }
}
#endif

/**
 * @brief Wrapper around the smoothed-input LOS tree implementations.
 *
 * @param root The root of the source-particle tree.
 * @param pos_i The positions of the input particles.
 * @param input_smls The smoothing lengths of the input particles.
 * @param overlap_kernel The overlap kernel lookup table.
 * @param q_grid The q-axis of the overlap table.
 * @param u_grid The u-axis of the overlap table.
 * @param eta_grid The eta-axis of the overlap table.
 * @param surf_dens The array to store the surface densities in.
 * @param npart_i The number of input particles.
 * @param qdim The number of q-grid entries.
 * @param udim The number of u-grid entries.
 * @param etadim The number of eta-grid entries.
 * @param threshold The support threshold in units of the smoothing length.
 * @param nthreads The number of threads to use.
 */
static void los_tree_smoothed(struct cell *root, const double *pos_i,
                              const double *input_smls,
                              const double *overlap_kernel,
                              const double *q_grid, const double *u_grid,
                              const double *eta_grid, double *surf_dens,
                              const int npart_i, const int qdim, const int udim,
                              const int etadim, const double threshold,
                              const int nthreads) {

  tic("Recursive surface density calculation with smoothed inputs");

#ifdef WITH_OPENMP
  if (nthreads > 1) {
    los_tree_smoothed_omp(root, pos_i, input_smls, overlap_kernel, q_grid,
                          u_grid, eta_grid, surf_dens, npart_i, qdim, udim,
                          etadim, threshold, nthreads);
  } else {
    los_tree_smoothed_serial(root, pos_i, input_smls, overlap_kernel, q_grid,
                             u_grid, eta_grid, surf_dens, npart_i, qdim, udim,
                             etadim, threshold);
  }
#else
  (void)nthreads;
  los_tree_smoothed_serial(root, pos_i, input_smls, overlap_kernel, q_grid,
                           u_grid, eta_grid, surf_dens, npart_i, qdim, udim,
                           etadim, threshold);
#endif

  toc("Recursive surface density calculation with smoothed inputs");
}

/**
 * @brief Compute smoothed-input LOS surface densities for each particle.
 *
 * This will calculate the line-of-sight surface density of an arbitrary
 * property of one set of particles for the positions of another set of
 * particles, averaging the answer across the support of each input particle.
 *
 * The smoothed-input path uses the precomputed overlap kernel table together
 * with its q, u, and eta coordinate axes. The public Python layer prepares
 * those arrays, and this extension dispatches to either the pairwise loop or
 * the tree walk.
 *
 * @param np_overlap_kernel The overlap kernel lookup table.
 * @param np_q_grid The q-axis of the overlap table.
 * @param np_u_grid The u-axis of the overlap table.
 * @param np_eta_grid The eta-axis of the overlap table.
 * @param np_pos_i The positions of the input particles.
 */
PyObject *compute_column_density_smoothed(PyObject *self, PyObject *args) {

  /* We don't need the self argument but it has to be there. Tell the compiler
   * we don't care. */
  (void)self;

  int npart_i, npart_j, qdim, udim, etadim, force_loop, min_count, nthreads;
  double threshold;
  PyArrayObject *np_overlap_kernel, *np_q_grid, *np_u_grid, *np_eta_grid,
      *np_pos_i, *np_input_smls, *np_pos_j, *np_smls, *np_surf_den_val;

  if (!PyArg_ParseTuple(args, "OOOOOOOOOiiiiidiii", &np_overlap_kernel,
                        &np_q_grid, &np_u_grid, &np_eta_grid, &np_pos_i,
                        &np_input_smls, &np_pos_j, &np_smls, &np_surf_den_val,
                        &npart_i, &npart_j, &qdim, &udim, &etadim, &threshold,
                        &force_loop, &min_count, &nthreads)) {
    return NULL;
  }

  tic("Calculating smoothed surface densities");

  /* Quick check to make sure our inputs are valid. */
  if (npart_i == 0) {
    PyErr_SetString(
        PyExc_ValueError,
        "The number of particles to calculate smoothed surface densities for "
        "must be greater than zero.");
    return NULL;
  }
  if (npart_j == 0) {
    PyErr_SetString(PyExc_ValueError,
                    "The number of particles to calculate smoothed surface "
                    "densities with must be greater than zero.");
    return NULL;
  }
  if (qdim == 0 || udim == 0 || etadim == 0) {
    PyErr_SetString(PyExc_ValueError,
                    "All overlap kernel dimensions must be greater than "
                    "zero.");
    return NULL;
  }

  /* Extract pointers to the actual data in the numpy arrays. */
  const double *overlap_kernel =
      extract_data_double(np_overlap_kernel, "overlap_kernel");
  const double *q_grid = extract_data_double(np_q_grid, "q_grid");
  const double *u_grid = extract_data_double(np_u_grid, "u_grid");
  const double *eta_grid = extract_data_double(np_eta_grid, "eta_grid");
  const double *pos_i = extract_data_double(np_pos_i, "pos_i");
  const double *input_smls = extract_data_double(np_input_smls, "input_smls");
  const double *pos_j = extract_data_double(np_pos_j, "pos_j");
  const double *smls = extract_data_double(np_smls, "smls");
  const double *surf_den_val =
      extract_data_double(np_surf_den_val, "surf_den_val");

  /* One of the data extractions failed and set a Python error. Return NULL
   * to propagate the exception back to Python. */
  if (overlap_kernel == NULL || q_grid == NULL || u_grid == NULL ||
      eta_grid == NULL || pos_i == NULL || input_smls == NULL ||
      pos_j == NULL || smls == NULL || surf_den_val == NULL) {
    return NULL;
  }

  /* Create the output array. */
  npy_intp np_dims[1] = {npart_i};
  PyArrayObject *np_surf_dens =
      (PyArrayObject *)PyArray_ZEROS(1, np_dims, NPY_DOUBLE, 0);
  double *surf_dens = static_cast<double *>(PyArray_DATA(np_surf_dens));

  /* No point constructing a source tree if there are too few source particles
   * to justify it, or if we have explicitly been asked to use the loop path. */
  if (force_loop || npart_j < min_count) {

    /* Use the simple pairwise loop over input and source particles. */
    tic("Dispatching smoothed LOS loop path");
    los_loop_smoothed(pos_i, input_smls, pos_j, smls, surf_den_val,
                      overlap_kernel, q_grid, u_grid, eta_grid, surf_dens,
                      npart_i, npart_j, qdim, udim, etadim, threshold,
                      nthreads);
    toc("Dispatching smoothed LOS loop path");

    toc("Calculating smoothed surface densities");

    return Py_BuildValue("N", np_surf_dens);
  }

  /* Allocate the root cell. The tree construction routine will dynamically
   * allocate progeny cells beneath this root as required. */
  tic("Constructing smoothed LOS source tree");
  int ncells = 1;
  int maxdepth = MAX_DEPTH;
  struct cell *root = new struct cell[1];

  /* Construct the source-particle cell tree. */
  construct_cell_tree(pos_j, smls, surf_den_val, npart_j, root, ncells,
                      maxdepth, min_count);
  toc("Constructing smoothed LOS source tree");

  /* Calculate the smoothed LOS surface densities using the tree. */
  tic("Dispatching smoothed LOS tree path");
  los_tree_smoothed(root, pos_i, input_smls, overlap_kernel, q_grid, u_grid,
                    eta_grid, surf_dens, npart_i, qdim, udim, etadim, threshold,
                    nthreads);
  toc("Dispatching smoothed LOS tree path");

  /* Clean up the source-particle cell tree. */
  tic("Cleaning up smoothed LOS source tree");
  cleanup_cell_tree(root);
  toc("Cleaning up smoothed LOS source tree");

  toc("Calculating smoothed surface densities");

  return Py_BuildValue("N", np_surf_dens);
}

/* Below is all the gubbins needed to make the module importable in Python. */
static PyMethodDef LosMethods[] = {
    {"compute_column_density", (PyCFunction)compute_column_density,
     METH_VARARGS, "Method for calculating line of sight surface densities."},
    {"compute_column_density_smoothed",
     (PyCFunction)compute_column_density_smoothed, METH_VARARGS,
     "Method for calculating line of sight surface densities with smoothed "
     "input particles."},
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
  if (m == NULL)
    return NULL;
  if (numpy_import() < 0) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to import numpy.");
    Py_DECREF(m);
    return NULL;
  }
#ifdef ATOMIC_TIMING
  if (import_toc_capsule() < 0) {
    Py_DECREF(m);
    return NULL;
  }
#endif
  return m;
}
