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
#include "octree.h"
#include "property_funcs.h"
#include "timers.h"
#ifdef ATOMIC_TIMING
#include "timers_init.h"
#endif

/* Forward declaration for the projected-kernel tree recursion used by the
 * smoothed-input tree traversal when a whole node is entirely in front. */
static double calculate_los_recursive(struct cell *c, const double x,
                                      const double y, const double z,
                                      double threshold, int kdim,
                                      const double *kernel);

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

    /* Loop over other particle postions. */
    for (int j = 0; j < npart_j; j++) {

      /* Get gas particle data. */
      double xj = pos_j[j * 3];
      double yj = pos_j[j * 3 + 1];
      double zj = pos_j[j * 3 + 2];
      double sml = smls[j];
      double surf_den_val = surf_den_vals[j];

      /* Skip straight away if the surface density particle is behind the z
       * position. */
      if (zj > z) {
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

      /* Get the value of the kernel at q (handling q =1). */
      int index = std::min(kdim - 1, static_cast<int>(q * kdim));
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
 * @param pos_i The positions of the particles to compute the surface
 *             densities for (e.g. star particles).
 * @param pos_j The positions of the particles to compute the surface
 *             densities from (e.g. gas particles).
 * @param smls The smoothing lengths of the particles to compute the
 *            surface densities from.
 * @param surf_den_vals The surface density values of the particles to compute
 *            the surface densities from.
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

        /* Skip straight away if the surface density particle is behind the z
         * position. */
        if (zj > z) {
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

        /* Get the value of the kernel at q (handling q =1). */
        int index = std::min(kdim - 1, static_cast<int>(q * kdim));
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

  tic("Loop surface density calculation");

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
  toc("Loop surface density calculation");
}

/**
 * @brief Get a kernel lookup value at a dimensionless radius.
 *
 * @param kernel The 1D kernel lookup table.
 * @param kdim The number of entries in the kernel lookup table.
 * @param q The dimensionless radius.
 * @return The interpolated kernel value.
 */
static double get_kernel_value(const double *kernel, const int kdim,
                               const double q) {

  if (q < 0.0 || q >= 1.0) {
    return 0.0;
  }

  const double scaled = q * (kdim - 1);
  const int index = static_cast<int>(scaled);
  const int next_index = std::min(kdim - 1, index + 1);
  const double frac = scaled - index;

  return kernel[index] + frac * (kernel[next_index] - kernel[index]);
}

/**
 * @brief Get a truncated LOS kernel value.
 *
 * @param kernel The 2D truncated LOS kernel lookup table.
 * @param kdim The number of impact-parameter entries.
 * @param zdim The number of LOS-coordinate entries.
 * @param q The dimensionless impact parameter.
 * @param z The dimensionless LOS coordinate.
 * @return The interpolated cumulative LOS kernel value.
 */
static double get_truncated_kernel_value(const double *kernel, const int kdim,
                                         const int zdim, const double q,
                                         const double z) {

  if (q < 0.0 || q >= 1.0) {
    return 0.0;
  }

  const double clamped_z = std::max(-1.0, std::min(1.0, z));

  const double scaled_q = q * (kdim - 1);
  const int q_index = static_cast<int>(scaled_q);
  const int q_next = std::min(kdim - 1, q_index + 1);
  const double q_frac = scaled_q - q_index;

  const double scaled_z = 0.5 * (clamped_z + 1.0) * (zdim - 1);
  const int z_index = static_cast<int>(scaled_z);
  const int z_next = std::min(zdim - 1, z_index + 1);
  const double z_frac = scaled_z - z_index;

  const double v00 = kernel[q_index * zdim + z_index];
  const double v01 = kernel[q_index * zdim + z_next];
  const double v10 = kernel[q_next * zdim + z_index];
  const double v11 = kernel[q_next * zdim + z_next];

  const double vz0 = v00 + z_frac * (v01 - v00);
  const double vz1 = v10 + z_frac * (v11 - v10);

  return vz0 + q_frac * (vz1 - vz0);
}

/**
 * @brief Computes the smoothed LOS contribution of one source particle.
 *
 * @param x The x-coordinate of the input sample point.
 * @param y The y-coordinate of the input sample point.
 * @param z The z-coordinate of the input sample point.
 * @param hj The smoothing length of the source particle.
 * @param support_j The support radius of the source particle.
 * @param xj The x-coordinate of the source particle.
 * @param yj The y-coordinate of the source particle.
 * @param zj The z-coordinate of the source particle.
 * @param surf_den_val The source surface density value.
 * @param projected_kernel The full projected LOS kernel lookup table.
 * @param truncated_kernel The truncated LOS kernel lookup table.
 * @param kdim The number of impact-parameter entries.
 * @param zdim The number of LOS-coordinate entries.
 * @return The source contribution at the input sample point.
 */
static double calculate_smoothed_source_contribution(
    const double x, const double y, const double z, const double hj,
    const double support_j, const double xj, const double yj, const double zj,
    const double surf_den_val, const double *projected_kernel,
    const double *truncated_kernel, const int kdim, const int zdim) {

  const double dx = xj - x;
  const double dy = yj - y;
  const double b = sqrt(dx * dx + dy * dy);

  if (b >= support_j) {
    return 0.0;
  }

  const double q = b / support_j;
  const double prefactor = surf_den_val / (hj * hj);
  const double z_min = zj - support_j;
  const double z_max = zj + support_j;

  /* If the whole source kernel is in front of the sample point we can use the
   * full projected kernel lookup. */
  if (z >= z_max) {
    return prefactor * get_kernel_value(projected_kernel, kdim, q);
  }

  /* If the whole source kernel is behind the sample point there is no
   * foreground contribution. */
  if (z <= z_min) {
    return 0.0;
  }

  /* Otherwise the sample point cuts through the source kernel and we need the
   * truncated foreground contribution. */
  const double z_upper = (z - zj) / support_j;
  return prefactor * get_truncated_kernel_value(truncated_kernel, kdim, zdim, q,
                                                z_upper);
}

/**
 * @brief Recursively calculate smoothed LOS surface density contributions.
 *
 * @param c The cell to calculate the surface densities for.
 * @param x The x position of the input sample point.
 * @param y The y position of the input sample point.
 * @param z The z position of the input sample point.
 * @param threshold The support threshold in units of the smoothing length.
 * @param kdim The dimension of the kernel tables.
 * @param zdim The dimension of the LOS-coordinate table.
 * @param projected_kernel The projected kernel lookup table.
 * @param truncated_kernel The truncated LOS kernel lookup table.
 * @return The LOS surface density contribution for this sample point.
 */
static double calculate_los_recursive_smoothed(
    struct cell *c, const double x, const double y, const double z,
    const double threshold, const int kdim, const int zdim,
    const double *projected_kernel, const double *truncated_kernel) {

  const double max_sml = sqrt(c->max_sml_squ);
  const double support_max = threshold * max_sml;
  const double cell_z_min = c->loc[2] - support_max;
  const double cell_z_max = c->loc[2] + c->width + support_max;

  /* Early exit if the expanded node is entirely behind the sample point. */
  if (cell_z_min >= z) {
    return 0.0;
  }

  /* Early exit if the projected distance between the point and cell is more
   * than the maximum support radius in the cell. */
  if ((threshold * threshold * c->max_sml_squ) < min_projected_dist2(c, x, y)) {
    return 0.0;
  }

  /* If the whole node is in front of the sample point we can use the existing
   * projected-kernel recursion. */
  if (cell_z_max <= z) {
    return calculate_los_recursive(c, x, y, z, threshold, kdim,
                                   projected_kernel);
  }

  double surf_dens = 0.0;

  if (c->split) {
    for (int ip = 0; ip < 8; ip++) {
      struct cell *cp = &c->progeny[ip];

      if (cp->part_count == 0) {
        continue;
      }

      surf_dens += calculate_los_recursive_smoothed(
          cp, x, y, z, threshold, kdim, zdim, projected_kernel,
          truncated_kernel);
    }
  } else {
    int npart_j = c->part_count;
    struct particle *parts = c->particles;

    for (int j = 0; j < npart_j; j++) {
      struct particle *part = &parts[j];

      surf_dens += calculate_smoothed_source_contribution(
          x, y, z, part->sml, threshold * part->sml, part->pos[0], part->pos[1],
          part->pos[2], part->surf_den_var, projected_kernel, truncated_kernel,
          kdim, zdim);
    }
  }

  return surf_dens;
}

/**
 * @brief Computes smoothed-input LOS surface densities with a loop.
 *
 * This correctness-first implementation handles only the brute-force loop path
 * for threshold=1 and evaluates the exact foreground contribution by
 * integrating the source kernel in front of each sample point in the input
 * kernel.
 *
 * @param pos_i The positions of the input particles.
 * @param input_smls The smoothing lengths of the input particles.
 * @param pos_j The positions of the contributing particles.
 * @param smls The smoothing lengths of the contributing particles.
 * @param surf_den_vals The source values to accumulate.
 * @param radial_kernel The 1D radial kernel lookup table.
 * @param truncated_kernel The truncated LOS kernel lookup table.
 * @param surf_dens The array to store the surface densities in.
 * @param npart_i The number of input particles.
 * @param npart_j The number of contributing particles.
 * @param kdim The dimension of the radial kernel table.
 * @param zdim The dimension of the LOS coordinate table.
 */
static void los_loop_smoothed_serial(
    const double *pos_i, const double *input_smls, const double *pos_j,
    const double *smls, const double *surf_den_vals,
    const double *projected_kernel, const double *radial_kernel,
    const double *truncated_kernel, double *surf_dens, const int npart_i,
    const int npart_j, const int kdim, const int zdim,
    const double threshold) {

  const int nxy = 16;
  const int nz = 16;
  const double dxy = 2.0 / nxy;
  const double dz = 2.0 / nz;

  for (int i = 0; i < npart_i; i++) {

    const double hi = input_smls[i];
    const double support_i = threshold * hi;
    const double xi = pos_i[i * 3];
    const double yi = pos_i[i * 3 + 1];
    const double zi = pos_i[i * 3 + 2];

    double particle_surf_dens = 0.0;
    double weight_sum = 0.0;

    for (int ix = 0; ix < nxy; ix++) {
      const double qx = -1.0 + (ix + 0.5) * dxy;

      for (int iy = 0; iy < nxy; iy++) {
        const double qy = -1.0 + (iy + 0.5) * dxy;

        for (int iz = 0; iz < nz; iz++) {
          const double qz = -1.0 + (iz + 0.5) * dz;
          const double qr = sqrt(qx * qx + qy * qy + qz * qz);

          if (qr >= 1.0) {
            continue;
          }

          const double input_weight = get_kernel_value(radial_kernel, kdim, qr);
          if (input_weight == 0.0) {
            continue;
          }

          const double x = xi + support_i * qx;
          const double y = yi + support_i * qy;
          const double z = zi + support_i * qz;

          double sample_surf_dens = 0.0;

          for (int j = 0; j < npart_j; j++) {
            const double hj = smls[j];
            const double support_j = threshold * hj;
            const double xj = pos_j[j * 3];
            const double yj = pos_j[j * 3 + 1];
            const double zj = pos_j[j * 3 + 2];
            const double surf_den_val = surf_den_vals[j];

            sample_surf_dens += calculate_smoothed_source_contribution(
                x, y, z, hj, support_j, xj, yj, zj, surf_den_val,
                projected_kernel, truncated_kernel, kdim, zdim);
          }

          particle_surf_dens += input_weight * sample_surf_dens;
          weight_sum += input_weight;
        }
      }
    }

    if (weight_sum > 0.0) {
      particle_surf_dens /= weight_sum;
    }

    surf_dens[i] = particle_surf_dens;
  }
}

#ifdef WITH_OPENMP
static void los_loop_smoothed_omp(
    const double *pos_i, const double *input_smls, const double *pos_j,
    const double *smls, const double *surf_den_vals,
    const double *projected_kernel, const double *radial_kernel,
    const double *truncated_kernel, double *surf_dens, const int npart_i,
    const int npart_j, const int kdim, const int zdim,
    const double threshold, const int nthreads) {

  int nparti_per_thread = npart_i / nthreads;

#pragma omp parallel num_threads(nthreads)
  {
    int tid = omp_get_thread_num();
    int start = tid * nparti_per_thread;
    int end = (tid == nthreads - 1) ? npart_i : (tid + 1) * nparti_per_thread;

    double *surf_dens_thread = new double[end - start]();

    los_loop_smoothed_serial(
        &pos_i[start * 3], &input_smls[start], pos_j, smls, surf_den_vals,
        projected_kernel, radial_kernel, truncated_kernel, surf_dens_thread,
        end - start, npart_j, kdim, zdim, threshold);

#pragma omp critical
    { memcpy(&surf_dens[start], surf_dens_thread,
             sizeof(double) * (end - start)); }

    delete[] surf_dens_thread;
  }
}
#endif

/**
 * @brief Computes smoothed-input LOS surface densities with a loop.
 *
 * @param pos_i The positions of the input particles.
 * @param input_smls The smoothing lengths of the input particles.
 * @param pos_j The positions of the contributing particles.
 * @param smls The smoothing lengths of the contributing particles.
 * @param surf_den_vals The source values to accumulate.
 * @param projected_kernel The full projected LOS kernel lookup table.
 * @param radial_kernel The 1D radial kernel lookup table.
 * @param truncated_kernel The truncated LOS kernel lookup table.
 * @param surf_dens The array to store the surface densities in.
 * @param npart_i The number of input particles.
 * @param npart_j The number of contributing particles.
 * @param kdim The dimension of the kernel lookup tables.
 * @param zdim The dimension of the LOS coordinate table.
 * @param threshold The support threshold in units of the smoothing length.
 * @param nthreads The number of threads to use.
 */
static void los_loop_smoothed(
    const double *pos_i, const double *input_smls, const double *pos_j,
    const double *smls, const double *surf_den_vals,
    const double *projected_kernel, const double *radial_kernel,
    const double *truncated_kernel, double *surf_dens, const int npart_i,
    const int npart_j, const int kdim, const int zdim,
    const double threshold, const int nthreads) {

  tic("Loop surface density calculation with smoothed inputs");

#ifdef WITH_OPENMP
  if (nthreads > 1) {
    los_loop_smoothed_omp(pos_i, input_smls, pos_j, smls, surf_den_vals,
                          projected_kernel, radial_kernel, truncated_kernel,
                          surf_dens, npart_i, npart_j, kdim, zdim, threshold,
                          nthreads);
  } else {
    los_loop_smoothed_serial(pos_i, input_smls, pos_j, smls, surf_den_vals,
                             projected_kernel, radial_kernel,
                             truncated_kernel, surf_dens, npart_i, npart_j,
                             kdim, zdim, threshold);
  }
#else
  (void)nthreads;
  los_loop_smoothed_serial(pos_i, input_smls, pos_j, smls, surf_den_vals,
                           projected_kernel, radial_kernel, truncated_kernel,
                           surf_dens, npart_i, npart_j, kdim, zdim, threshold);
#endif

  toc("Loop surface density calculation with smoothed inputs");
}

/**
 * @brief Computes smoothed-input LOS surface densities with a tree.
 *
 * @param root The root of the tree.
 * @param pos_i The positions of the input particles.
 * @param input_smls The smoothing lengths of the input particles.
 * @param projected_kernel The full projected LOS kernel lookup table.
 * @param radial_kernel The 1D radial kernel lookup table.
 * @param truncated_kernel The truncated LOS kernel lookup table.
 * @param surf_dens The array to store the surface densities in.
 * @param npart_i The number of input particles.
 * @param kdim The dimension of the kernel lookup tables.
 * @param zdim The dimension of the LOS coordinate table.
 * @param threshold The support threshold in units of the smoothing length.
 */
static void los_tree_smoothed_serial(
    struct cell *root, const double *pos_i, const double *input_smls,
    const double *projected_kernel, const double *radial_kernel,
    const double *truncated_kernel, double *surf_dens, const int npart_i,
    const int kdim, const int zdim, const double threshold) {

  const int nxy = 16;
  const int nz = 16;
  const double dxy = 2.0 / nxy;
  const double dz = 2.0 / nz;

  for (int i = 0; i < npart_i; i++) {
    const double hi = input_smls[i];
    const double support_i = threshold * hi;
    const double xi = pos_i[i * 3];
    const double yi = pos_i[i * 3 + 1];
    const double zi = pos_i[i * 3 + 2];

    double particle_surf_dens = 0.0;
    double weight_sum = 0.0;

    for (int ix = 0; ix < nxy; ix++) {
      const double qx = -1.0 + (ix + 0.5) * dxy;

      for (int iy = 0; iy < nxy; iy++) {
        const double qy = -1.0 + (iy + 0.5) * dxy;

        for (int iz = 0; iz < nz; iz++) {
          const double qz = -1.0 + (iz + 0.5) * dz;
          const double qr = sqrt(qx * qx + qy * qy + qz * qz);

          if (qr >= 1.0) {
            continue;
          }

          const double input_weight = get_kernel_value(radial_kernel, kdim, qr);
          if (input_weight == 0.0) {
            continue;
          }

          const double x = xi + support_i * qx;
          const double y = yi + support_i * qy;
          const double z = zi + support_i * qz;

          const double sample_surf_dens = calculate_los_recursive_smoothed(
              root, x, y, z, threshold, kdim, zdim, projected_kernel,
              truncated_kernel);

          particle_surf_dens += input_weight * sample_surf_dens;
          weight_sum += input_weight;
        }
      }
    }

    if (weight_sum > 0.0) {
      particle_surf_dens /= weight_sum;
    }

    surf_dens[i] = particle_surf_dens;
  }
}

#ifdef WITH_OPENMP
static void los_tree_smoothed_omp(
    struct cell *root, const double *pos_i, const double *input_smls,
    const double *projected_kernel, const double *radial_kernel,
    const double *truncated_kernel, double *surf_dens, const int npart_i,
    const int kdim, const int zdim, const double threshold,
    const int nthreads) {

  int nparti_per_thread = npart_i / nthreads;

#pragma omp parallel num_threads(nthreads)
  {
    int tid = omp_get_thread_num();
    int start = tid * nparti_per_thread;
    int end = (tid == nthreads - 1) ? npart_i : (tid + 1) * nparti_per_thread;

    double *surf_dens_thread = new double[end - start]();

    los_tree_smoothed_serial(root, &pos_i[start * 3], &input_smls[start],
                             projected_kernel, radial_kernel, truncated_kernel,
                             surf_dens_thread, end - start, kdim, zdim,
                             threshold);

#pragma omp critical
    { memcpy(&surf_dens[start], surf_dens_thread,
             sizeof(double) * (end - start)); }

    delete[] surf_dens_thread;
  }
}
#endif

/**
 * @brief Computes smoothed-input LOS surface densities with a tree.
 *
 * @param root The root of the cell tree.
 * @param pos_i The positions of the input particles.
 * @param input_smls The smoothing lengths of the input particles.
 * @param projected_kernel The full projected LOS kernel lookup table.
 * @param radial_kernel The 1D radial kernel lookup table.
 * @param truncated_kernel The truncated LOS kernel lookup table.
 * @param surf_dens The array to store the surface densities in.
 * @param npart_i The number of input particles.
 * @param kdim The dimension of the kernel lookup tables.
 * @param zdim The dimension of the LOS coordinate table.
 * @param threshold The support threshold in units of the smoothing length.
 * @param nthreads The number of threads to use.
 */
static void los_tree_smoothed(
    struct cell *root, const double *pos_i, const double *input_smls,
    const double *projected_kernel, const double *radial_kernel,
    const double *truncated_kernel, double *surf_dens, const int npart_i,
    const int kdim, const int zdim, const double threshold,
    const int nthreads) {

  tic("Recursive surface density calculation with smoothed inputs");

#ifdef WITH_OPENMP
  if (nthreads > 1) {
    los_tree_smoothed_omp(root, pos_i, input_smls, projected_kernel,
                          radial_kernel, truncated_kernel, surf_dens, npart_i,
                          kdim, zdim, threshold, nthreads);
  } else {
    los_tree_smoothed_serial(root, pos_i, input_smls, projected_kernel,
                             radial_kernel, truncated_kernel, surf_dens,
                             npart_i, kdim, zdim, threshold);
  }
#else
  (void)nthreads;
  los_tree_smoothed_serial(root, pos_i, input_smls, projected_kernel,
                           radial_kernel, truncated_kernel, surf_dens, npart_i,
                           kdim, zdim, threshold);
#endif

  toc("Recursive surface density calculation with smoothed inputs");
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
  if (c->loc[2] > z) {
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
      if (part->pos[2] > z) {
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

      /* Get the value of the kernel at q (handling q =1). */
      int index = std::min(kdim - 1, static_cast<int>(q * kdim));
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
    double *surf_dens_thread = new double[end - start]();

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

  tic("Recursive surface density calculation");

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
  toc("Recursive surface density calculation");
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

  tic("Calculating surface densities");

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

  /* One of the data extractions failed and set a Python error. Return NULL
   * to propagate the exception back to Python. */
  if (kernel == NULL || pos_i == NULL || pos_j == NULL || smls == NULL ||
      surf_den_val == NULL) {
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
    los_loop(pos_i, pos_j, smls, surf_den_val, kernel, surf_dens, npart_i,
             npart_j, kdim, threshold, nthreads);

    toc("Calculating surface densities");

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
  los_tree(root, pos_i, kernel, surf_dens, npart_i, kdim, threshold, nthreads);

  /* Clean up. */
  cleanup_cell_tree(root);

  toc("Calculating surface densities");

  return Py_BuildValue("N", np_surf_dens);
}

/**
 * @brief Stub entry point for LOS column densities with smoothed inputs.
 *
 * This path exists so the Python interface and extension data flow can be
 * staged separately from the kernel-overlap implementation.
 *
 * @param self The Python self object.
 * @param args The Python argument tuple.
 * @return Always NULL with a NotImplementedError set.
 */
PyObject *compute_column_density_smoothed(PyObject *self, PyObject *args) {

  /* We don't need the self argument but it has to be there. Tell the compiler
   * we don't care. */
  (void)self;

  int npart_i, npart_j, kdim, zdim, force_loop, min_count, nthreads;
  double threshold;
  PyArrayObject *np_kernel, *np_radial_kernel, *np_truncated_kernel, *np_pos_i,
      *np_input_smls, *np_pos_j, *np_smls, *np_surf_den_val;

  if (!PyArg_ParseTuple(args, "OOOOOOOOiiiidiii", &np_kernel,
                        &np_radial_kernel, &np_truncated_kernel, &np_pos_i,
                        &np_input_smls, &np_pos_j, &np_smls,
                        &np_surf_den_val, &npart_i, &npart_j, &kdim, &zdim,
                        &threshold, &force_loop, &min_count, &nthreads)) {
    return NULL;
  }

  const double *kernel = extract_data_double(np_kernel, "kernel");
  const double *radial_kernel =
      extract_data_double(np_radial_kernel, "radial_kernel");
  const double *truncated_kernel =
      extract_data_double(np_truncated_kernel, "truncated_kernel");
  const double *pos_i = extract_data_double(np_pos_i, "pos_i");
  const double *input_smls = extract_data_double(np_input_smls, "input_smls");
  const double *pos_j = extract_data_double(np_pos_j, "pos_j");
  const double *smls = extract_data_double(np_smls, "smls");
  const double *surf_den_val =
      extract_data_double(np_surf_den_val, "surf_den_val");

  if (kernel == NULL || radial_kernel == NULL || truncated_kernel == NULL ||
      pos_i == NULL || input_smls == NULL || pos_j == NULL || smls == NULL ||
      surf_den_val == NULL) {
    return NULL;
  }

  npy_intp np_dims[1] = {npart_i};
  PyArrayObject *np_surf_dens =
      (PyArrayObject *)PyArray_ZEROS(1, np_dims, NPY_DOUBLE, 0);
  double *surf_dens = static_cast<double *>(PyArray_DATA(np_surf_dens));

  if (force_loop || npart_j < min_count) {
    los_loop_smoothed(pos_i, input_smls, pos_j, smls, surf_den_val, kernel,
                      radial_kernel, truncated_kernel, surf_dens, npart_i,
                      npart_j, kdim, zdim, threshold, nthreads);
    return Py_BuildValue("N", np_surf_dens);
  }

  int ncells = 1;
  int maxdepth = MAX_DEPTH;
  struct cell *root = new struct cell[1];

  construct_cell_tree(pos_j, smls, surf_den_val, npart_j, root, ncells,
                      maxdepth, min_count);

  los_tree_smoothed(root, pos_i, input_smls, kernel, radial_kernel,
                    truncated_kernel, surf_dens, npart_i, kdim, zdim,
                    threshold, nthreads);

  cleanup_cell_tree(root);

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
