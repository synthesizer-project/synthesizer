/******************************************************************************
 * C extension to calculate line of sight metal surface densities for star
 * particles.
 *****************************************************************************/

/* C headers. */
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

/* Python headers. */
#define PY_ARRAY_UNIQUE_SYMBOL SYNTHESIZER_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy_init.h"

#include <Python.h>

/* Local includes. */
#include "cpp_to_python.h"
#include "kernel_extensions/kernel_utils.h"
#include "octree.h"
#include "property_funcs.h"
#include "python_to_cpp.h"
#include "timers.h"
#ifdef ATOMIC_TIMING
#include "timers_init.h"
#endif

/**
 * @brief Build readable names for the column density attribute arrays.
 *
 * @param py_names Optional sequence of attribute names (may be NULL/None).
 * @param n The number of attribute arrays.
 * @return One name per array, e.g. "dust_masses of the absorbing particles",
 *         falling back to "column density attribute 1" etc.
 */
static std::vector<std::string> density_attr_names(PyObject *py_names,
                                                   const size_t n) {
  std::vector<std::string> names;
  names.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    std::string name = "column density attribute " + std::to_string(i + 1);
    if (py_names != NULL && py_names != Py_None &&
        PySequence_Check(py_names) &&
        static_cast<Py_ssize_t>(i) < PySequence_Size(py_names)) {
      PyObject *item = PySequence_GetItem(py_names, i);
      const char *utf8 = (item != NULL && PyUnicode_Check(item))
                             ? PyUnicode_AsUTF8(item)
                             : NULL;
      if (utf8 != NULL) {
        name = std::string(utf8) + " of the absorbing particles";
      } else {
        /* Names are only for error messages, so fall back quietly. */
        PyErr_Clear();
      }
      Py_XDECREF(item);
    }
    names.push_back(name);
  }
  return names;
}

/**
 * @brief Validate a Python sequence of source-property arrays.
 *
 * The returned array pointers are borrowed from ``values``. The Python call
 * argument keeps every array alive until the extension returns, so this helper
 * can expose their existing buffers without stacking or copying particle data.
 *
 * @param values Tuple or list containing one array per requested property.
 * @param npart Number of source particles expected in each array.
 * @param arrays Output vector populated with borrowed NumPy array pointers.
 * @return True when every sequence entry has the required one-dimensional
 *         shape, otherwise false with a Python exception set.
 */
static bool unpack_density_arrays(PyObject *values, const int npart,
                                  std::vector<PyArrayObject *> &arrays) {
  if (!PyTuple_Check(values) && !PyList_Check(values)) {
    PyErr_SetString(PyExc_TypeError,
                    "surf_den_vals must be a tuple or list of arrays.");
    return false;
  }

  PyObject *sequence = PySequence_Fast(values, "invalid surf_den_vals");
  if (sequence == NULL) return false;

  const Py_ssize_t count = PySequence_Fast_GET_SIZE(sequence);
  if (count == 0) {
    Py_DECREF(sequence);
    PyErr_SetString(PyExc_ValueError,
                    "surf_den_vals must contain at least one array.");
    return false;
  }

  arrays.reserve(static_cast<size_t>(count));
  for (Py_ssize_t index = 0; index < count; index++) {
    PyObject *item = PySequence_Fast_GET_ITEM(sequence, index);
    if (!PyArray_Check(item)) {
      Py_DECREF(sequence);
      PyErr_SetString(PyExc_TypeError,
                      "surf_den_vals entries must be NumPy arrays.");
      return false;
    }
    PyArrayObject *array = reinterpret_cast<PyArrayObject *>(item);
    if (PyArray_NDIM(array) != 1 || PyArray_DIM(array, 0) != npart) {
      Py_DECREF(sequence);
      PyErr_SetString(PyExc_ValueError,
                      "Each surf_den_vals array must have one value per "
                      "source particle.");
      return false;
    }
    arrays.push_back(array);
  }

  Py_DECREF(sequence);
  return true;
}

/**
 * @brief Computes the line of sight surface densities with a loop.
 *
 * This is the serial version of the function that computes the line of sight
 * surface densities for each particle. It uses a simple loop over the star
 * particles and the gas particles. This will be used when the number of gas
 * particles is small enough that making a tree is pointless.
 *
 * @tparam Real The floating-point type (float or double).
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
 * @param nattrs The number of source properties evaluated together.
 * @param kdim The number of projected-kernel entries.
 * @param trunc_qdim The number of projected-separation entries in the
 *        truncated kernel table.
 * @param zdim The number of LOS-coordinate entries in the truncated table.
 * @param threshold The threshold for the kernel.
 *
 * Accumulation happens in double precision regardless of Real so that
 * summing many (potentially float32) gas particle contributions along a
 * single line of sight doesn't accumulate float32 rounding error.
 */
template <typename PartReal, typename KernelReal, typename ValueReal>
static void los_loop_serial(
    const FloatView pos_i, const PartReal *pos_j, const PartReal *smls,
    const std::vector<const ValueReal *> &surf_den_vals,
    const KernelReal *kernel, const KernelReal *truncated_kernel,
    double *surf_dens, const int npart_i, const int npart_j, const int nattrs,
    const int kdim, const int trunc_qdim, const int zdim,
    const PartReal threshold) {

  /* Loop over particle postions. */
  for (int i = 0; i < npart_i; i++) {

    PartReal x = pos_i.get<PartReal>(i * 3);
    PartReal y = pos_i.get<PartReal>(i * 3 + 1);
    PartReal z = pos_i.get<PartReal>(i * 3 + 2);

    /* Loop over other particle postions. */
    for (int j = 0; j < npart_j; j++) {

      /* Get gas particle data. */
      PartReal xj = pos_j[j * 3];
      PartReal yj = pos_j[j * 3 + 1];
      PartReal zj = pos_j[j * 3 + 2];
      PartReal sml = smls[j];

      /* Skip straight away if the source kernel lies entirely behind the
       * input position. */
      if ((zj - threshold * sml) > z) {
        continue;
      }

      /* Calculate the projected x and y separations. */
      PartReal dx = xj - x;
      PartReal dy = yj - y;

      /* Calculate the impact parameter. */
      PartReal b = sqrt(dx * dx + dy * dy);

      /* Early skip if the star's line of sight doesn't fall in the gas
       * particles kernel. */
      if (b > (threshold * sml)) continue;

      /* Find fraction of smoothing length. */
      PartReal q = b / sml;

      /* If the input lies inside the source kernel we need the truncated LOS
       * contribution. Otherwise the input sees the full projected kernel. */
      KernelReal kvalue = static_cast<KernelReal>(0.0);
      if (z < (zj + threshold * sml)) {
        const PartReal z_trunc = (z - zj) / (threshold * sml);
        kvalue = get_truncated_kernel_value<KernelReal>(
            truncated_kernel, trunc_qdim, zdim,
            static_cast<KernelReal>(q / threshold),
            static_cast<KernelReal>(z_trunc));
      } else {
        kvalue = get_kernel_value<KernelReal>(
            kernel, kdim, static_cast<KernelReal>(q / threshold));
      }

      const double weight =
          static_cast<double>(kvalue) / static_cast<double>(sml * sml);
      for (int iattr = 0; iattr < nattrs; iattr++) {
        surf_dens[iattr * npart_i + i] +=
            static_cast<double>(surf_den_vals[iattr][j]) * weight;
      }
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
 * @tparam Real The floating-point type.
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
 * @param nattrs The number of source properties evaluated together.
 * @param kdim The number of projected-kernel entries.
 * @param trunc_qdim The number of projected-separation entries in the
 *        truncated kernel table.
 * @param zdim The number of LOS-coordinate entries in the truncated table.
 * @param threshold The threshold for the kernel.
 * @param nthreads The number of threads to use.
 */
#ifdef WITH_OPENMP
template <typename PartReal, typename KernelReal, typename ValueReal>
static void los_loop_omp(const FloatView pos_i, const PartReal *pos_j,
                         const PartReal *smls,
                         const std::vector<const ValueReal *> &surf_den_vals,
                         const KernelReal *kernel,
                         const KernelReal *truncated_kernel, double *surf_dens,
                         const int npart_i, const int npart_j,
                         const int nattrs, const int kdim,
                         const int trunc_qdim, const int zdim,
                         const PartReal threshold, const int nthreads) {

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
     * here to avoid cache locality issues. Accumulation happens in double
     * precision to avoid float32 rounding error over many gas particles. */
    const int thread_npart = end - start;
    std::vector<double> surf_dens_thread(nattrs * thread_npart, 0.0);

    /* Loop over particle postions. */
    for (int i = start; i < end; i++) {

      /* Get the relative index. */
      int ii = i - start;

      PartReal x = pos_i.get<PartReal>(i * 3);
      PartReal y = pos_i.get<PartReal>(i * 3 + 1);
      PartReal z = pos_i.get<PartReal>(i * 3 + 2);

      for (int j = 0; j < npart_j; j++) {

        /* Get gas particle data. */
        PartReal xj = pos_j[j * 3];
        PartReal yj = pos_j[j * 3 + 1];
        PartReal zj = pos_j[j * 3 + 2];
        PartReal sml = smls[j];

        /* Skip straight away if the source kernel lies entirely behind the
         * input position. */
        if ((zj - threshold * sml) > z) {
          continue;
        }

        /* Calculate the projected x and y separations. */
        PartReal dx = xj - x;
        PartReal dy = yj - y;

        /* Calculate the impact parameter. */
        PartReal b = sqrt(dx * dx + dy * dy);

        /* Early skip if the star's line of sight doesn't fall in the gas
         * particles kernel. */
        if (b > (threshold * sml)) continue;

        /* Find fraction of smoothing length. */
        PartReal q = b / sml;

        /* If the input lies inside the source kernel we need the truncated LOS
         * contribution. Otherwise the input sees the full projected kernel. */
        KernelReal kvalue = static_cast<KernelReal>(0.0);
        if (z < (zj + threshold * sml)) {
          const PartReal z_trunc = (z - zj) / (threshold * sml);
          kvalue = get_truncated_kernel_value<KernelReal>(
              truncated_kernel, trunc_qdim, zdim,
              static_cast<KernelReal>(q / threshold),
              static_cast<KernelReal>(z_trunc));
        } else {
          kvalue = get_kernel_value<KernelReal>(
              kernel, kdim, static_cast<KernelReal>(q / threshold));
        }

        const double weight =
            static_cast<double>(kvalue) / static_cast<double>(sml * sml);
        for (int iattr = 0; iattr < nattrs; iattr++) {
          surf_dens_thread[iattr * thread_npart + ii] +=
              static_cast<double>(surf_den_vals[iattr][j]) * weight;
        }
      }
    }

    /* Copy the results back to the main array. */
#pragma omp critical
    {
      for (int iattr = 0; iattr < nattrs; iattr++) {
        memcpy(&surf_dens[iattr * npart_i + start],
               &surf_dens_thread[iattr * thread_npart],
               thread_npart * sizeof(double));
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
 * based on whether or not OpenMP is available and the number of threads to
 * use.
 *
 * @tparam Real The floating-point type (float or double).
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
 * @param nattrs The number of source properties evaluated together.
 * @param kdim The number of projected-kernel entries.
 * @param trunc_qdim The number of projected-separation entries in the
 *        truncated kernel table.
 * @param zdim The number of LOS-coordinate entries in the truncated table.
 * @param threshold The threshold for the kernel.
 * @param nthreads The number of threads to use.
 */
template <typename PartReal, typename KernelReal, typename ValueReal>
static void los_loop(const FloatView pos_i, const PartReal *pos_j,
                     const PartReal *smls,
                     const std::vector<const ValueReal *> &surf_den_vals,
                     const KernelReal *kernel,
                     const KernelReal *truncated_kernel, double *surf_dens,
                     const int npart_i, const int npart_j, const int nattrs,
                     const int kdim, const int trunc_qdim, const int zdim,
                     const PartReal threshold, const int nthreads) {

  tic("column_density.point.loop");

#ifdef WITH_OPENMP

  /* If we have multiple threads and OpenMP we can parallelise. */
  if (nthreads > 1) {
    los_loop_omp<PartReal, KernelReal, ValueReal>(
        pos_i, pos_j, smls, surf_den_vals, kernel, truncated_kernel, surf_dens,
        npart_i, npart_j, nattrs, kdim, trunc_qdim, zdim, threshold, nthreads);
  } else {
    los_loop_serial<PartReal, KernelReal, ValueReal>(
        pos_i, pos_j, smls, surf_den_vals, kernel, truncated_kernel, surf_dens,
        npart_i, npart_j, nattrs, kdim, trunc_qdim, zdim, threshold);
  }

#else

  (void)nthreads;

  /* If we don't have OpenMP call the serial version. */
  los_loop_serial<PartReal, KernelReal, ValueReal>(
      pos_i, pos_j, smls, surf_den_vals, kernel, truncated_kernel, surf_dens,
      npart_i, npart_j, nattrs, kdim, trunc_qdim, zdim, threshold);

#endif
  toc("column_density.point.loop");
}

/**
 * @brief Recursively calculate the line of sight surface densities.
 *
 * This will recurse to the leaves of the cell tree, any cells further than the
 * maximum smoothing length from the position will be skipped. Once in the
 * leaves the particles themselves will be checked to see if their SPH kernel
 * overlaps with the line of sight of the star particle.
 *
 * @tparam Real The floating-point type (float or double).
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
 * @param surf_den_vals Existing source-property buffers, one per attribute.
 * @param surf_dens Attribute-major output accumulation buffer.
 * @param output_index Target-particle index within the output buffer.
 * @param output_stride Number of target particles between attribute rows.
 *
 * Accumulation happens in double precision regardless of Real so that
 * summing many (potentially float32) gas particle contributions along a
 * single line of sight doesn't accumulate float32 rounding error.
 */
template <typename PartReal, typename KernelReal, typename ValueReal>
static void calculate_los_recursive(
    struct cell<PartReal> *c, PartReal x, PartReal y, PartReal z,
    PartReal threshold, int kdim, int trunc_qdim, int zdim,
    const KernelReal *kernel, const KernelReal *truncated_kernel,
    const std::vector<const ValueReal *> &surf_den_vals, double *surf_dens,
    const int output_index, const int output_stride) {

  /* Early exit if the cell is entirely behind the position. */
  if ((c->loc[2] - threshold * sqrt(c->max_sml_squ)) > z) {
    return;
  }

  /* Early exit if the projected distance between cells is more than the
   * maximum smoothing length in the cell. */
  if (c->max_sml_squ * (threshold * threshold) <
      min_projected_dist2<PartReal>(c, x, y)) {
    return;
  }

  /* Is the cell split? */
  if (c->split) {

    /* Ok, so we recurse... */
    for (int ip = 0; ip < 8; ip++) {
      struct cell<PartReal> *cp = &c->progeny[ip];

      /* Skip empty progeny. */
      if (cp->part_count == 0) {
        continue;
      }

      /* Recurse... */
      calculate_los_recursive<PartReal, KernelReal, ValueReal>(
          cp, x, y, z, threshold, kdim, trunc_qdim, zdim, kernel,
          truncated_kernel, surf_den_vals, surf_dens, output_index,
          output_stride);
    }

  } else {

    /* We're in a leaf if we get here, unpack the particles. */
    int npart_j = c->part_count;
    struct particle<PartReal> *parts = c->particles;

    /* Loop over the particles adding their contribution. */
    for (int j = 0; j < npart_j; j++) {

      /* Get the particle. */
      struct particle<PartReal> *part = &parts[j];

      /* Skip straight away if the source kernel lies entirely behind the
       * input position. */
      if ((part->pos[2] - threshold * part->sml) > z) {
        continue;
      }

      /* Calculate the x and y separations. */
      PartReal dx = part->pos[0] - x;
      PartReal dy = part->pos[1] - y;

      /* Calculate the impact parameter. */
      PartReal b = sqrt(dx * dx + dy * dy);

      /* Early skip if the star's line of sight doesn't fall in the gas
       * particles kernel. */
      if (b > (threshold * part->sml)) {
        continue;
      }

      /* Find fraction of smoothing length. */
      PartReal q = b / part->sml;

      /* If the input lies inside the source kernel we need the truncated LOS
       * contribution. Otherwise the input sees the full projected kernel. */
      KernelReal kvalue = static_cast<KernelReal>(0.0);
      if (z < (part->pos[2] + threshold * part->sml)) {
        const PartReal z_trunc = (z - part->pos[2]) / (threshold * part->sml);
        kvalue = get_truncated_kernel_value<KernelReal>(
            truncated_kernel, trunc_qdim, zdim,
            static_cast<KernelReal>(q / threshold),
            static_cast<KernelReal>(z_trunc));
      } else {
        kvalue = get_kernel_value<KernelReal>(
            kernel, kdim, static_cast<KernelReal>(q / threshold));
      }

      const double weight = static_cast<double>(kvalue) /
                            static_cast<double>(part->sml * part->sml);
      for (size_t iattr = 0; iattr < surf_den_vals.size(); iattr++) {
        surf_dens[iattr * output_stride + output_index] +=
            static_cast<double>(surf_den_vals[iattr][part->index]) * weight;
      }
    }
  }
}

/**
 * @brief Computes the line of sight surface densities with a tree.
 *
 * This is the serial version of the function that computes the line of sight
 * surface densities for each particle. It uses a tree to store the gas
 * particles and then traverses the tree to find the particles that are within
 * the kernel of the star particle.
 *
 * @tparam Real The floating-point type (float or double).
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
template <typename PartReal, typename KernelReal, typename ValueReal>
static void los_tree_serial(
    struct cell<PartReal> *root, const FloatView pos_i,
    const KernelReal *kernel, const KernelReal *truncated_kernel,
    const std::vector<const ValueReal *> &surf_den_vals, double *surf_dens,
    const int npart_i, const int kdim, const int trunc_qdim, const int zdim,
    const PartReal threshold) {

  /* Loop over the particles we are calculating the surface density for. */
  for (int i = 0; i < npart_i; i++) {

    /* Start at the root. We'll recurse through the tree to the leaves
     * skipping all cells out of range of this particle. */
    calculate_los_recursive<PartReal, KernelReal, ValueReal>(
        root, pos_i.get<PartReal>(i * 3), pos_i.get<PartReal>(i * 3 + 1),
        pos_i.get<PartReal>(i * 3 + 2), threshold, kdim, trunc_qdim, zdim,
        kernel, truncated_kernel, surf_den_vals, surf_dens, i, npart_i);
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
 * @tparam Real The floating-point type (float or double).
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
template <typename PartReal, typename KernelReal, typename ValueReal>
static void los_tree_omp(struct cell<PartReal> *root, const FloatView pos_i,
                         const KernelReal *kernel,
                         const KernelReal *truncated_kernel,
                         const std::vector<const ValueReal *> &surf_den_vals,
                         double *surf_dens, const int npart_i, const int kdim,
                         const int trunc_qdim, const int zdim,
                         const PartReal threshold, const int nthreads) {

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
     * here to avoid cache locality issues. calculate_los_recursive already
     * accumulates in double, so this buffer stays double too. */
    const int thread_npart = end - start;
    const int nattrs = static_cast<int>(surf_den_vals.size());
    std::vector<double> surf_dens_thread(nattrs * thread_npart, 0.0);

    /* Loop over the particles we are calculating the surface density for. */
    for (int i = start; i < end; i++) {

      /* Start at the root. We'll recurse through the tree to the leaves
       * skipping all cells out of range of this particle. */
      calculate_los_recursive<PartReal, KernelReal, ValueReal>(
          root, pos_i.get<PartReal>(i * 3), pos_i.get<PartReal>(i * 3 + 1),
          pos_i.get<PartReal>(i * 3 + 2), threshold, kdim, trunc_qdim, zdim,
          kernel, truncated_kernel, surf_den_vals, surf_dens_thread.data(),
          i - start, thread_npart);
    }

    /* Copy the results back to the main array. */
#pragma omp critical
    {
      for (int iattr = 0; iattr < nattrs; iattr++) {
        memcpy(&surf_dens[iattr * npart_i + start],
               &surf_dens_thread[iattr * thread_npart],
               thread_npart * sizeof(double));
      }
    }
  }
}
#endif

/**
 * @brief Computes the line of sight surface densities with a tree.
 *
 * This is a wrapper function which will call the correct version of the
 * function to compute the line of sight surface densities for each particle
 * based on whether or not OpenMP is available and the number of threads to
 * use.
 *
 * @tparam Real The floating-point type (float or double).
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
template <typename PartReal, typename KernelReal, typename ValueReal>
static void los_tree(struct cell<PartReal> *root, const FloatView pos_i,
                     const KernelReal *kernel,
                     const KernelReal *truncated_kernel,
                     const std::vector<const ValueReal *> &surf_den_vals,
                     double *surf_dens, const int npart_i, const int kdim,
                     const int trunc_qdim, const int zdim,
                     const PartReal threshold, const int nthreads) {

  tic("column_density.point.tree_query");

#ifdef WITH_OPENMP

  /* If we have multiple threads and OpenMP we can parallelise. */
  if (nthreads > 1) {
    los_tree_omp<PartReal, KernelReal, ValueReal>(
        root, pos_i, kernel, truncated_kernel, surf_den_vals, surf_dens,
        npart_i, kdim, trunc_qdim, zdim, threshold, nthreads);
  } else {
    los_tree_serial<PartReal, KernelReal, ValueReal>(
        root, pos_i, kernel, truncated_kernel, surf_den_vals, surf_dens,
        npart_i, kdim, trunc_qdim, zdim, threshold);
  }

#else

  (void)nthreads;

  /* If we don't have OpenMP call the serial version. */
  los_tree_serial<PartReal, KernelReal, ValueReal>(
      root, pos_i, kernel, truncated_kernel, surf_den_vals, surf_dens, npart_i,
      kdim, trunc_qdim, zdim, threshold);

#endif
  toc("column_density.point.tree_query");
}

/**
 * @brief Templated implementation of column density computation.
 *
 * @tparam Real The floating-point type (float or double).
 */
template <typename KernelReal, typename PartReal, typename ValueReal,
          typename OutT>
static PyObject *compute_column_density_impl(
    PyObject *self, PyArrayObject *np_kernel,
    PyArrayObject *np_truncated_kernel, PyArrayObject *np_pos_i,
    PyArrayObject *np_pos_j, PyArrayObject *np_smls,
    const std::vector<PyArrayObject *> &np_surf_den_vals, int npart_i,
    int npart_j, int kdim, int trunc_qdim, int zdim, double threshold_double,
    int force_loop, int min_count, int nthreads) {

  (void)self;

  const PartReal threshold = static_cast<PartReal>(threshold_double);

  const KernelReal *kernel = extract_data<KernelReal>(np_kernel, "kernel");
  const KernelReal *truncated_kernel =
      extract_data<KernelReal>(np_truncated_kernel, "truncated_kernel");
  const FloatView pos_i(np_pos_i);
  const PartReal *pos_j = extract_data<PartReal>(np_pos_j, "pos_j");
  const PartReal *smls = extract_data<PartReal>(np_smls, "smls");
  /* Keep pointers into the Python-owned source arrays. The argument tuple owns
   * these buffers for the duration of this call, so no particle data is
   * copied.
   */
  std::vector<const ValueReal *> surf_den_vals;
  surf_den_vals.reserve(np_surf_den_vals.size());
  for (PyArrayObject *array : np_surf_den_vals) {
    const ValueReal *values =
        extract_data<ValueReal>(array, "surf_den_vals entry");
    if (values == NULL) return NULL;
    surf_den_vals.push_back(values);
  }

  if (kernel == NULL || truncated_kernel == NULL || pos_i.data == NULL ||
      pos_j == NULL || smls == NULL) {
    return NULL;
  }

  const int typenum = NumpyTypenum<OutT>::typenum;
  const int nattrs = static_cast<int>(surf_den_vals.size());
  /* Store attributes by row so each unit-aware Python result is contiguous. */
  npy_intp np_dims[2] = {nattrs, npart_i};
  PyArrayObject *np_surf_dens =
      (PyArrayObject *)PyArray_ZEROS(2, np_dims, typenum, 0);
  if (np_surf_dens == NULL) {
    PyErr_NoMemory();
    return NULL;
  }
  OutT *surf_dens = static_cast<OutT *>(PyArray_DATA(np_surf_dens));

  /* Accumulate in double regardless of the output precision so that summing
   * many (potentially float32) gas particle contributions along a single
   * line of sight doesn't accumulate float32 rounding error. The double
   * buffer is folded into the Real output array once at the end. */
  std::vector<double> surf_dens_accum(nattrs * npart_i, 0.0);

  if (force_loop || npart_j < min_count) {
    los_loop<PartReal, KernelReal, ValueReal>(
        pos_i, pos_j, smls, surf_den_vals, kernel, truncated_kernel,
        surf_dens_accum.data(), npart_i, npart_j, nattrs, kdim, trunc_qdim,
        zdim, threshold, nthreads);
  } else {
    int ncells = 1;
    struct cell<PartReal> *root = new struct cell<PartReal>;

    construct_cell_tree<PartReal>(pos_j, smls, smls, npart_j, root, ncells,
                                  MAX_DEPTH, min_count);

    los_tree<PartReal, KernelReal, ValueReal>(
        root, pos_i, kernel, truncated_kernel, surf_den_vals,
        surf_dens_accum.data(), npart_i, kdim, trunc_qdim, zdim, threshold,
        nthreads);

    tic("column_density.point.tree_cleanup");
    cleanup_cell_tree<PartReal>(root);
    toc("column_density.point.tree_cleanup");
  }

  tic("column_density.point.pack_output");
  for (int i = 0; i < nattrs * npart_i; i++) {
    surf_dens[i] = static_cast<OutT>(surf_dens_accum[i]);
  }
  toc("column_density.point.pack_output");

  return Py_BuildValue("N", np_surf_dens);
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
      *np_smls;
  PyObject *py_surf_den_vals;
  PyObject *out_dtype = Py_None;
  PyObject *py_attr_names = Py_None;

  if (!PyArg_ParseTuple(args, "OOOOOOiiiiidiii|OO", &np_kernel,
                        &np_truncated_kernel, &np_pos_i, &np_pos_j, &np_smls,
                        &py_surf_den_vals, &npart_i, &npart_j, &kdim,
                        &trunc_qdim, &zdim, &threshold, &force_loop,
                        &min_count, &nthreads, &out_dtype, &py_attr_names))
    return NULL;

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

  std::vector<PyArrayObject *> np_surf_den_vals;
  if (!unpack_density_arrays(py_surf_den_vals, npart_j, np_surf_den_vals)) {
    return NULL;
  }

  PyArrayObject *kernel_arrays[] = {np_kernel, np_truncated_kernel};
  const char *kernel_names[] = {
      "kernel table (Kernel.get_kernel())",
      "truncated kernel table (Kernel.get_truncated_los_kernel())"};
  int kernel_typenum = -1;
  if (!is_matching_float_dtypes(kernel_arrays, kernel_names, 2,
                                &kernel_typenum)) {
    return NULL;
  }

  /* The particles receiving column densities are read once each, so they
   * can have their own precision without joining the templated groups. */
  PyArrayObject *target_arrays[] = {np_pos_i};
  const char *target_names[] = {
      "coordinates of the particles receiving column densities"};
  int target_typenum = -1;
  if (!is_matching_float_dtypes(target_arrays, target_names, 1,
                                &target_typenum)) {
    return NULL;
  }

  PyArrayObject *particle_arrays[] = {np_pos_j, np_smls};
  const char *particle_names[] = {
      "coordinates of the absorbing particles",
      "smoothing_lengths of the absorbing particles"};
  int particle_typenum = -1;
  if (!is_matching_float_dtypes(particle_arrays, particle_names, 2,
                                &particle_typenum)) {
    return NULL;
  }

  std::vector<std::string> value_labels =
      density_attr_names(py_attr_names, np_surf_den_vals.size());
  std::vector<const char *> value_names;
  for (const std::string &label : value_labels) {
    value_names.push_back(label.c_str());
  }
  int value_typenum = -1;
  if (!is_matching_float_dtypes(np_surf_den_vals.data(), value_names.data(),
                                static_cast<int>(np_surf_den_vals.size()),
                                &value_typenum)) {
    return NULL;
  }

  const int output_typenum =
      out_dtype == Py_None ? value_typenum
                           : resolve_output_typenum(out_dtype, "out_dtype");
  if (output_typenum < 0) return NULL;

  return dispatch_float(kernel_typenum, [&](auto k) -> PyObject * {
    return dispatch_float(particle_typenum, [&](auto p) -> PyObject * {
      return dispatch_float(value_typenum, [&](auto v) -> PyObject * {
        return dispatch_float(output_typenum, [&](auto o) -> PyObject * {
          using KernelReal = decltype(k);
          using PartReal = decltype(p);
          using ValueReal = decltype(v);
          using OutT = decltype(o);
          return compute_column_density_impl<KernelReal, PartReal, ValueReal,
                                             OutT>(
              self, np_kernel, np_truncated_kernel, np_pos_i, np_pos_j,
              np_smls, np_surf_den_vals, npart_i, npart_j, kdim, trunc_qdim,
              zdim, threshold, force_loop, min_count, nthreads);
        });
      });
    });
  });
}

/**
 * @brief Compute one pairwise smoothed-input LOS contribution.
 *
 * Each input/source particle pair is mapped onto the precomputed overlap table
 * `G(q, u, eta)` and contributes exactly once.
 *
 * The runtime pair coordinates are defined using the support radii of the
 * input and source particles:
 *
 *   q   = b / (R_i + R_j)
 *   u   = (z_i - z_j) / (R_i + R_j)
 *   eta = h_i / h_j
 *
 * where `R = threshold * h` is the effective support radius used by the LOS
 * calculation. The overlap table then returns the kernel-averaged fraction of
 * the source LOS kernel seen by the input particle.
 *
 * @tparam Real The floating-point type (float or double).
 * @param xi The x position of the input particle.
 * @param yi The y position of the input particle.
 * @param zi The z position of the input particle.
 * @param hi The smoothing length of the input particle.
 * @param xj The x position of the source particle.
 * @param yj The y position of the source particle.
 * @param zj The z position of the source particle.
 * @param hj The smoothing length of the source particle.
 * @param overlap_kernel The overlap kernel lookup table.
 * @param q_grid The q-axis of the overlap table.
 * @param u_grid The u-axis of the overlap table.
 * @param eta_grid The eta-axis of the overlap table.
 * @param qdim The number of q-grid entries.
 * @param udim The number of u-grid entries.
 * @param etadim The number of eta-grid entries.
 * @param threshold The support threshold in units of the smoothing length.
 *
 * @return The geometric LOS weight for this pair. Source-property values are
 *         applied by the caller so all properties reuse this calculation.
 */
template <typename PartReal, typename KernelReal>
static inline double calculate_smoothed_pair_contribution(
    const PartReal xi, const PartReal yi, const PartReal zi, const PartReal hi,
    const PartReal xj, const PartReal yj, const PartReal zj, const PartReal hj,
    const KernelReal *overlap_kernel, const KernelReal *q_grid,
    const KernelReal *u_grid, const KernelReal *eta_grid, const int qdim,
    const int udim, const int etadim, const PartReal threshold) {

  /* Reject pathological zero-support pairs before doing any further work. */
  if (hj <= static_cast<PartReal>(0.0)) {
    return 0.0;
  }

  /* Convert smoothing lengths to the support radii used by the LOS solver. */
  const PartReal support_i = threshold * hi;
  const PartReal support_j = threshold * hj;
  const PartReal support_sum = support_i + support_j;
  if (support_sum <= static_cast<PartReal>(0.0)) {
    return 0.0;
  }

  /* Compute the projected particle-centre separation. We stay in squared form
   * for the first overlap rejection to avoid an unnecessary square root in the
   * common no-overlap case. */
  const PartReal dx = xj - xi;
  const PartReal dy = yj - yi;
  const PartReal b2 = dx * dx + dy * dy;
  const PartReal support_sum2 = support_sum * support_sum;

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
  const PartReal b = sqrt(b2);
  const PartReal q = b / support_sum;
  const PartReal u = (zi - zj) / support_sum;
  const PartReal eta = hi / hj;

  /* The table contains the kernel-averaged LOS contribution. Only the source
   * normalisation remains to be applied at runtime. */
  return static_cast<double>(get_overlap_kernel_value<KernelReal>(
             overlap_kernel, q_grid, u_grid, eta_grid, qdim, udim, etadim,
             static_cast<KernelReal>(q), static_cast<KernelReal>(u),
             static_cast<KernelReal>(eta))) /
         static_cast<double>(hj * hj);
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
 * @tparam Real The floating-point type (float or double).
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
 * @param surf_den_vals Existing source-property buffers, one per attribute.
 * @param surf_dens Attribute-major output accumulation buffer.
 * @param output_index Target-particle index within the output buffer.
 * @param output_stride Number of target particles between attribute rows.
 *
 * @return The LOS surface density contribution from this node.
 *
 * Accumulation happens in double precision regardless of Real so that
 * summing many (potentially float32) source particle contributions along a
 * single line of sight doesn't accumulate float32 rounding error.
 */
template <typename PartReal, typename KernelReal, typename ValueReal>
static void calculate_los_recursive_smoothed(
    struct cell<PartReal> *c, const PartReal xi, const PartReal yi,
    const PartReal zi, const PartReal hi, const KernelReal *overlap_kernel,
    const KernelReal *q_grid, const KernelReal *u_grid,
    const KernelReal *eta_grid, const int qdim, const int udim,
    const int etadim, const PartReal threshold,
    const std::vector<const ValueReal *> &surf_den_vals, double *surf_dens,
    const int output_index, const int output_stride) {

  /* The input-particle support radius is fixed along this traversal. */
  const PartReal support_i = threshold * hi;

  /* The node can only contain source particles up to this support radius. */
  const PartReal support_max = threshold * sqrt(c->max_sml_squ);

  /* Reject nodes whose source kernels are entirely behind the full input
   * support. */
  const PartReal cell_z_min = c->loc[2] - support_max;
  if (cell_z_min >= (zi + support_i)) {
    return;
  }

  /* Reject nodes whose projected bounding box stays further away than the
   * combined input and maximum source support. */
  const PartReal support_sum = support_i + support_max;
  if ((support_sum * support_sum) < min_projected_dist2<PartReal>(c, xi, yi)) {
    return;
  }

  if (c->split) {

    /* Internal nodes are traversed recursively until we reach leaves, where we
     * can evaluate the exact pairwise overlap with each source particle. */
    for (int ip = 0; ip < 8; ip++) {
      struct cell<PartReal> *cp = &c->progeny[ip];

      if (cp->part_count == 0) {
        continue;
      }

      calculate_los_recursive_smoothed<PartReal, KernelReal, ValueReal>(
          cp, xi, yi, zi, hi, overlap_kernel, q_grid, u_grid, eta_grid, qdim,
          udim, etadim, threshold, surf_den_vals, surf_dens, output_index,
          output_stride);
    }

  } else {

    /* Once in a leaf we are back to the same one-pair-at-a-time structure as
     * the simple loop implementation. */
    const int npart_j = c->part_count;
    struct particle<PartReal> *parts = c->particles;

    for (int j = 0; j < npart_j; j++) {
      struct particle<PartReal> *part = &parts[j];

      const double weight =
          calculate_smoothed_pair_contribution<PartReal, KernelReal>(
              xi, yi, zi, hi, part->pos[0], part->pos[1], part->pos[2],
              part->sml, overlap_kernel, q_grid, u_grid, eta_grid, qdim, udim,
              etadim, threshold);
      for (size_t iattr = 0; iattr < surf_den_vals.size(); iattr++) {
        surf_dens[iattr * output_stride + output_index] +=
            static_cast<double>(surf_den_vals[iattr][part->index]) * weight;
      }
    }
  }
}

/**
 * @brief Compute smoothed-input LOS surface densities with a loop.
 *
 * For each input particle we loop over source particles, reject
 * non-overlapping pairs, interpolate the overlap table once, and apply the
 * source prefactor.
 *
 * @tparam Real The floating-point type (float or double).
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
template <typename PartReal, typename KernelReal, typename ValueReal>
static void los_loop_smoothed_serial(
    const FloatView pos_i, const FloatView input_smls, const PartReal *pos_j,
    const PartReal *smls, const std::vector<const ValueReal *> &surf_den_vals,
    const KernelReal *overlap_kernel, const KernelReal *q_grid,
    const KernelReal *u_grid, const KernelReal *eta_grid, double *surf_dens,
    const int npart_i, const int npart_j, const int qdim, const int udim,
    const int etadim, const PartReal threshold) {

  /* Loop over the input particles. */
  for (int i = 0; i < npart_i; i++) {

    const PartReal xi = pos_i.get<PartReal>(i * 3);
    const PartReal yi = pos_i.get<PartReal>(i * 3 + 1);
    const PartReal zi = pos_i.get<PartReal>(i * 3 + 2);
    const PartReal hi = input_smls.get<PartReal>(i);

    /* Each source particle contributes at most once to this input particle.
     * Accumulate in double so summing many source particles doesn't build up
     * float32 rounding error. */
    for (int j = 0; j < npart_j; j++) {
      const double weight =
          calculate_smoothed_pair_contribution<PartReal, KernelReal>(
              xi, yi, zi, hi, pos_j[j * 3], pos_j[j * 3 + 1], pos_j[j * 3 + 2],
              smls[j], overlap_kernel, q_grid, u_grid, eta_grid, qdim, udim,
              etadim, threshold);
      for (size_t iattr = 0; iattr < surf_den_vals.size(); iattr++) {
        surf_dens[iattr * npart_i + i] +=
            static_cast<double>(surf_den_vals[iattr][j]) * weight;
      }
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
 * @tparam Real The floating-point type (float or double).
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
template <typename PartReal, typename KernelReal, typename ValueReal>
static void los_loop_smoothed_omp(
    const FloatView pos_i, const FloatView input_smls, const PartReal *pos_j,
    const PartReal *smls, const std::vector<const ValueReal *> &surf_den_vals,
    const KernelReal *overlap_kernel, const KernelReal *q_grid,
    const KernelReal *u_grid, const KernelReal *eta_grid, double *surf_dens,
    const int npart_i, const int npart_j, const int qdim, const int udim,
    const int etadim, const PartReal threshold, const int nthreads) {

  int nparti_per_thread = npart_i / nthreads;

#pragma omp parallel num_threads(nthreads)
  {
    int tid = omp_get_thread_num();
    int start = tid * nparti_per_thread;
    int end = (tid == nthreads - 1) ? npart_i : (tid + 1) * nparti_per_thread;

    const int thread_npart = end - start;
    const int nattrs = static_cast<int>(surf_den_vals.size());
    std::vector<double> surf_dens_thread(nattrs * thread_npart, 0.0);

    los_loop_smoothed_serial<PartReal, KernelReal, ValueReal>(
        pos_i.offset(start * 3), input_smls.offset(start), pos_j, smls,
        surf_den_vals, overlap_kernel, q_grid, u_grid, eta_grid,
        surf_dens_thread.data(), thread_npart, npart_j, qdim, udim, etadim,
        threshold);

#pragma omp critical
    {
      for (int iattr = 0; iattr < nattrs; iattr++) {
        memcpy(&surf_dens[iattr * npart_i + start],
               &surf_dens_thread[iattr * thread_npart],
               sizeof(double) * thread_npart);
      }
    }
  }
}
#endif

/**
 * @brief Wrapper around the smoothed-input LOS loop implementations.
 *
 * @tparam Real The floating-point type (float or double).
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
template <typename PartReal, typename KernelReal, typename ValueReal>
static void los_loop_smoothed(
    const FloatView pos_i, const FloatView input_smls, const PartReal *pos_j,
    const PartReal *smls, const std::vector<const ValueReal *> &surf_den_vals,
    const KernelReal *overlap_kernel, const KernelReal *q_grid,
    const KernelReal *u_grid, const KernelReal *eta_grid, double *surf_dens,
    const int npart_i, const int npart_j, const int qdim, const int udim,
    const int etadim, const PartReal threshold, const int nthreads) {

  tic("column_density.smoothed.loop");

#ifdef WITH_OPENMP
  if (nthreads > 1) {
    los_loop_smoothed_omp<PartReal, KernelReal, ValueReal>(
        pos_i, input_smls, pos_j, smls, surf_den_vals, overlap_kernel, q_grid,
        u_grid, eta_grid, surf_dens, npart_i, npart_j, qdim, udim, etadim,
        threshold, nthreads);
  } else {
    los_loop_smoothed_serial<PartReal, KernelReal, ValueReal>(
        pos_i, input_smls, pos_j, smls, surf_den_vals, overlap_kernel, q_grid,
        u_grid, eta_grid, surf_dens, npart_i, npart_j, qdim, udim, etadim,
        threshold);
  }
#else
  (void)nthreads;
  los_loop_smoothed_serial<PartReal, KernelReal, ValueReal>(
      pos_i, input_smls, pos_j, smls, surf_den_vals, overlap_kernel, q_grid,
      u_grid, eta_grid, surf_dens, npart_i, npart_j, qdim, udim, etadim,
      threshold);
#endif

  toc("column_density.smoothed.loop");
}

/**
 * @brief Compute smoothed-input LOS surface densities with a tree.
 *
 * Python prepares arrays, the extension traverses the source tree once per
 * input particle, and each source particle contributes exactly once at the
 * leaves.
 *
 * @tparam Real The floating-point type (float or double).
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
template <typename PartReal, typename KernelReal, typename ValueReal>
static void los_tree_smoothed_serial(
    struct cell<PartReal> *root, const FloatView pos_i,
    const FloatView input_smls, const KernelReal *overlap_kernel,
    const KernelReal *q_grid, const KernelReal *u_grid,
    const KernelReal *eta_grid,
    const std::vector<const ValueReal *> &surf_den_vals, double *surf_dens,
    const int npart_i, const int qdim, const int udim, const int etadim,
    const PartReal threshold) {

  for (int i = 0; i < npart_i; i++) {
    calculate_los_recursive_smoothed<PartReal, KernelReal, ValueReal>(
        root, pos_i.get<PartReal>(i * 3), pos_i.get<PartReal>(i * 3 + 1),
        pos_i.get<PartReal>(i * 3 + 2), input_smls.get<PartReal>(i),
        overlap_kernel, q_grid, u_grid, eta_grid, qdim, udim, etadim,
        threshold, surf_den_vals, surf_dens, i, npart_i);
  }
}

#ifdef WITH_OPENMP
/**
 * @brief Parallel smoothed-input LOS tree implementation.
 *
 * @tparam Real The floating-point type (float or double).
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
template <typename PartReal, typename KernelReal, typename ValueReal>
static void los_tree_smoothed_omp(
    struct cell<PartReal> *root, const FloatView pos_i,
    const FloatView input_smls, const KernelReal *overlap_kernel,
    const KernelReal *q_grid, const KernelReal *u_grid,
    const KernelReal *eta_grid,
    const std::vector<const ValueReal *> &surf_den_vals, double *surf_dens,
    const int npart_i, const int qdim, const int udim, const int etadim,
    const PartReal threshold, const int nthreads) {

  int nparti_per_thread = npart_i / nthreads;

#pragma omp parallel num_threads(nthreads)
  {
    int tid = omp_get_thread_num();
    int start = tid * nparti_per_thread;
    int end = (tid == nthreads - 1) ? npart_i : (tid + 1) * nparti_per_thread;

    const int thread_npart = end - start;
    const int nattrs = static_cast<int>(surf_den_vals.size());
    std::vector<double> surf_dens_thread(nattrs * thread_npart, 0.0);

    los_tree_smoothed_serial<PartReal, KernelReal, ValueReal>(
        root, pos_i.offset(start * 3), input_smls.offset(start),
        overlap_kernel, q_grid, u_grid, eta_grid, surf_den_vals,
        surf_dens_thread.data(), thread_npart, qdim, udim, etadim, threshold);

#pragma omp critical
    {
      for (int iattr = 0; iattr < nattrs; iattr++) {
        memcpy(&surf_dens[iattr * npart_i + start],
               &surf_dens_thread[iattr * thread_npart],
               sizeof(double) * thread_npart);
      }
    }
  }
}
#endif

/**
 * @brief Wrapper around the smoothed-input LOS tree implementations.
 *
 * @tparam Real The floating-point type (float or double).
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
template <typename PartReal, typename KernelReal, typename ValueReal>
static void los_tree_smoothed(
    struct cell<PartReal> *root, const FloatView pos_i,
    const FloatView input_smls, const KernelReal *overlap_kernel,
    const KernelReal *q_grid, const KernelReal *u_grid,
    const KernelReal *eta_grid,
    const std::vector<const ValueReal *> &surf_den_vals, double *surf_dens,
    const int npart_i, const int qdim, const int udim, const int etadim,
    const PartReal threshold, const int nthreads) {

  tic("column_density.smoothed.tree_query");

#ifdef WITH_OPENMP
  if (nthreads > 1) {
    los_tree_smoothed_omp<PartReal, KernelReal, ValueReal>(
        root, pos_i, input_smls, overlap_kernel, q_grid, u_grid, eta_grid,
        surf_den_vals, surf_dens, npart_i, qdim, udim, etadim, threshold,
        nthreads);
  } else {
    los_tree_smoothed_serial<PartReal, KernelReal, ValueReal>(
        root, pos_i, input_smls, overlap_kernel, q_grid, u_grid, eta_grid,
        surf_den_vals, surf_dens, npart_i, qdim, udim, etadim, threshold);
  }
#else
  (void)nthreads;
  los_tree_smoothed_serial<PartReal, KernelReal, ValueReal>(
      root, pos_i, input_smls, overlap_kernel, q_grid, u_grid, eta_grid,
      surf_den_vals, surf_dens, npart_i, qdim, udim, etadim, threshold);
#endif

  toc("column_density.smoothed.tree_query");
}

/**
 * @brief Templated implementation of smoothed column density computation.
 *
 * @tparam Real The floating-point type (float or double).
 */
template <typename KernelReal, typename PartReal, typename ValueReal,
          typename OutT>
static PyObject *compute_column_density_smoothed_impl(
    PyObject *self, PyArrayObject *np_overlap_kernel, PyArrayObject *np_q_grid,
    PyArrayObject *np_u_grid, PyArrayObject *np_eta_grid,
    PyArrayObject *np_pos_i, PyArrayObject *np_input_smls,
    PyArrayObject *np_pos_j, PyArrayObject *np_smls,
    const std::vector<PyArrayObject *> &np_surf_den_vals, int npart_i,
    int npart_j, int qdim, int udim, int etadim, double threshold_double,
    int force_loop, int min_count, int nthreads) {

  (void)self;

  const PartReal threshold = static_cast<PartReal>(threshold_double);

  const KernelReal *overlap_kernel =
      extract_data<KernelReal>(np_overlap_kernel, "overlap_kernel");
  const KernelReal *q_grid = extract_data<KernelReal>(np_q_grid, "q_grid");
  const KernelReal *u_grid = extract_data<KernelReal>(np_u_grid, "u_grid");
  const KernelReal *eta_grid =
      extract_data<KernelReal>(np_eta_grid, "eta_grid");
  const FloatView pos_i(np_pos_i);
  const FloatView input_smls(np_input_smls);
  const PartReal *pos_j = extract_data<PartReal>(np_pos_j, "pos_j");
  const PartReal *smls = extract_data<PartReal>(np_smls, "smls");
  /* Keep pointers into the Python-owned source arrays. The argument tuple owns
   * these buffers for the duration of this call, so no particle data is
   * copied.
   */
  std::vector<const ValueReal *> surf_den_vals;
  surf_den_vals.reserve(np_surf_den_vals.size());
  for (PyArrayObject *array : np_surf_den_vals) {
    const ValueReal *values =
        extract_data<ValueReal>(array, "surf_den_vals entry");
    if (values == NULL) return NULL;
    surf_den_vals.push_back(values);
  }

  if (overlap_kernel == NULL || q_grid == NULL || u_grid == NULL ||
      eta_grid == NULL || pos_i.data == NULL || input_smls.data == NULL ||
      pos_j == NULL || smls == NULL) {
    return NULL;
  }

  const int typenum = NumpyTypenum<OutT>::typenum;
  const int nattrs = static_cast<int>(surf_den_vals.size());
  /* Store attributes by row so each unit-aware Python result is contiguous. */
  npy_intp np_dims[2] = {nattrs, npart_i};
  PyArrayObject *np_surf_dens =
      (PyArrayObject *)PyArray_ZEROS(2, np_dims, typenum, 0);
  if (np_surf_dens == NULL) {
    PyErr_NoMemory();
    return NULL;
  }
  OutT *surf_dens = static_cast<OutT *>(PyArray_DATA(np_surf_dens));

  /* Accumulate in double regardless of the output precision so that summing
   * many (potentially float32) source particle contributions along a single
   * line of sight doesn't accumulate float32 rounding error. The double
   * buffer is folded into the Real output array once at the end. */
  std::vector<double> surf_dens_accum(nattrs * npart_i, 0.0);

  /* No point constructing a source tree if there are too few source particles
   * to justify it, or if we have explicitly been asked to use the loop path.
   */
  if (force_loop || npart_j < min_count) {

    /* Use the simple pairwise loop over input and source particles. */
    los_loop_smoothed<PartReal, KernelReal, ValueReal>(
        pos_i, input_smls, pos_j, smls, surf_den_vals, overlap_kernel, q_grid,
        u_grid, eta_grid, surf_dens_accum.data(), npart_i, npart_j, qdim, udim,
        etadim, threshold, nthreads);

  } else {

    /* Allocate the root cell. The tree construction routine will dynamically
     * allocate progeny cells beneath this root as required. */
    int ncells = 1;
    int maxdepth = MAX_DEPTH;
    struct cell<PartReal> *root = new struct cell<PartReal>;

    /* Construct the source-particle cell tree. */
    construct_cell_tree<PartReal>(pos_j, smls, smls, npart_j, root, ncells,
                                  maxdepth, min_count);

    /* Calculate the smoothed LOS surface densities using the tree. */
    los_tree_smoothed<PartReal, KernelReal, ValueReal>(
        root, pos_i, input_smls, overlap_kernel, q_grid, u_grid, eta_grid,
        surf_den_vals, surf_dens_accum.data(), npart_i, qdim, udim, etadim,
        threshold, nthreads);

    /* Clean up the source-particle cell tree. */
    tic("column_density.smoothed.tree_cleanup");
    cleanup_cell_tree<PartReal>(root);
    toc("column_density.smoothed.tree_cleanup");
  }

  tic("column_density.smoothed.pack_output");
  for (int i = 0; i < nattrs * npart_i; i++) {
    surf_dens[i] = static_cast<OutT>(surf_dens_accum[i]);
  }
  toc("column_density.smoothed.pack_output");

  return Py_BuildValue("N", np_surf_dens);
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
      *np_pos_i, *np_input_smls, *np_pos_j, *np_smls;
  PyObject *py_surf_den_vals;
  PyObject *out_dtype = Py_None;
  PyObject *py_attr_names = Py_None;

  if (!PyArg_ParseTuple(args, "OOOOOOOOOiiiiidiii|OO", &np_overlap_kernel,
                        &np_q_grid, &np_u_grid, &np_eta_grid, &np_pos_i,
                        &np_input_smls, &np_pos_j, &np_smls, &py_surf_den_vals,
                        &npart_i, &npart_j, &qdim, &udim, &etadim, &threshold,
                        &force_loop, &min_count, &nthreads, &out_dtype,
                        &py_attr_names)) {
    return NULL;
  }

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

  std::vector<PyArrayObject *> np_surf_den_vals;
  if (!unpack_density_arrays(py_surf_den_vals, npart_j, np_surf_den_vals)) {
    return NULL;
  }

  PyArrayObject *kernel_arrays[] = {np_overlap_kernel, np_q_grid, np_u_grid,
                                    np_eta_grid};
  const char *kernel_names[] = {
      "overlap kernel table (Kernel.get_overlap_kernel())",
      "overlap kernel q grid", "overlap kernel u grid",
      "overlap kernel eta grid"};
  int kernel_typenum = -1;
  if (!is_matching_float_dtypes(kernel_arrays, kernel_names, 4,
                                &kernel_typenum)) {
    return NULL;
  }

  /* The particles receiving column densities are read once each, so they
   * can have their own precision without joining the templated groups. */
  PyArrayObject *target_arrays[] = {np_pos_i, np_input_smls};
  const char *target_names[] = {
      "coordinates of the particles receiving column densities",
      "smoothing_lengths of the particles receiving column densities"};
  int target_typenum = -1;
  if (!is_matching_float_dtypes(target_arrays, target_names, 2,
                                &target_typenum)) {
    return NULL;
  }

  PyArrayObject *particle_arrays[] = {np_pos_j, np_smls};
  const char *particle_names[] = {
      "coordinates of the absorbing particles",
      "smoothing_lengths of the absorbing particles"};
  int particle_typenum = -1;
  if (!is_matching_float_dtypes(particle_arrays, particle_names, 2,
                                &particle_typenum)) {
    return NULL;
  }

  std::vector<std::string> value_labels =
      density_attr_names(py_attr_names, np_surf_den_vals.size());
  std::vector<const char *> value_names;
  for (const std::string &label : value_labels) {
    value_names.push_back(label.c_str());
  }
  int value_typenum = -1;
  if (!is_matching_float_dtypes(np_surf_den_vals.data(), value_names.data(),
                                static_cast<int>(np_surf_den_vals.size()),
                                &value_typenum)) {
    return NULL;
  }

  const int output_typenum =
      out_dtype == Py_None ? value_typenum
                           : resolve_output_typenum(out_dtype, "out_dtype");
  if (output_typenum < 0) return NULL;

  return dispatch_float(kernel_typenum, [&](auto k) -> PyObject * {
    return dispatch_float(particle_typenum, [&](auto p) -> PyObject * {
      return dispatch_float(value_typenum, [&](auto v) -> PyObject * {
        return dispatch_float(output_typenum, [&](auto o) -> PyObject * {
          using KernelReal = decltype(k);
          using PartReal = decltype(p);
          using ValueReal = decltype(v);
          using OutT = decltype(o);
          return compute_column_density_smoothed_impl<KernelReal, PartReal,
                                                      ValueReal, OutT>(
              self, np_overlap_kernel, np_q_grid, np_u_grid, np_eta_grid,
              np_pos_i, np_input_smls, np_pos_j, np_smls, np_surf_den_vals,
              npart_i, npart_j, qdim, udim, etadim, threshold, force_loop,
              min_count, nthreads);
        });
      });
    });
  });
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
  if (m == NULL) return NULL;
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
