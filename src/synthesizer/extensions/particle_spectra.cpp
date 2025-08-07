/******************************************************************************
 * C extension to calculate SEDs for star particles.
 * Calculates weights on an arbitrary dimensional grid given the mass.
 *****************************************************************************/
/* C includes */
#include <algorithm>
#include <array>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

/* Python includes */
#define PY_ARRAY_UNIQUE_SYMBOL SYNTHESIZER_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy_init.h"
#include <Python.h>

/* Local includes */
#include "cpp_to_python.h"
#include "grid_props.h"
#include "macros.h"
#include "part_props.h"
#include "property_funcs.h"
#include "reductions.h"
#include "timers.h"
#include "weights.h"

/**
 * @brief Get the grid indices and fractions for a particle using CIC.
 *
 * This function computes the indices of the grid cells that a particle
 * occupies, along with the fractions of the particle's mass in each cell.
 *
 * This is the serial version of the function.
 *
 * @param GridProps: The properties of the grid.
 * @param parts: The particle properties.
 */
static void get_particle_indices_and_fracs_serial(GridProps *grid_props,
                                                  Particles *parts) {

  /* Unpack the grid properties. */
  const int ndim = grid_props->ndim;
  int nlam = grid_props->nlam;

  // Pre-allocate exactly npart slots to avoid resizing
  parts->grid_indices.resize(parts->npart);
  parts->grid_fracs.resize(parts->npart * ndim);

  /* Loop over particles. */
  for (int p = 0; p < parts->npart; p++) {

    /* Skip masked particles. */
    if (parts->part_is_masked(p)) {
      parts->grid_indices[p] = -1;
      for (int idim = 0; idim < ndim; idim++) {
        parts->grid_fracs[p * ndim + idim] = 0.0;
      }
      continue;
    }

    /* Get the grid indices and cell fractions for the particle. */
    std::array<int, MAX_GRID_NDIM> part_indices;
    std::array<double, MAX_GRID_NDIM> axis_fracs;
    get_part_ind_frac_cic(part_indices, axis_fracs, grid_props, parts, p);

    /* Compute base linear index for this particle */
    const int grid_ind = grid_props->ravel_grid_index(part_indices);

    /* Store the grid indices and fractions in the particle class. */
    parts->grid_indices[p] = grid_ind;

    /* Store the per‐dimension fraction in the particle class. */
    for (int idim = 0; idim < ndim; idim++) {
      parts->grid_fracs[p * ndim + idim] = axis_fracs[idim];
    }
  }
}

/**
 * @brief Get the grid indices and fractions for a particle using CIC.
 *
 * This function computes the indices of the grid cells that a particle
 * occupies, along with the fractions of the particle's mass in each cell.
 *
 * This is the parallel version of the function.
 *
 * @param grid_props: The properties of the grid.
 * @param parts: The particle properties.
 * @param nthreads: The number of threads to use for parallel processing.
 */
static void get_particle_indices_and_fracs_parallel(GridProps *grid_props,
                                                    Particles *parts,
                                                    int nthreads) {
  const int ndim = grid_props->ndim;
  const int npart = parts->npart;

  // Pre-allocate exactly npart slots so we can write into slices on
  // each thread without resizing.
  parts->grid_indices.resize(npart);
  parts->grid_fracs.resize(npart * ndim);

#pragma omp parallel for num_threads(nthreads) schedule(static)
  // Loop over particles in parallel
  for (int p = 0; p < npart; p++) {

    // Skip masked particles
    if (parts->part_is_masked(p)) {
      parts->grid_indices[p] = -1;
      for (int d = 0; d < ndim; d++) {
        parts->grid_fracs[p * ndim + d] = 0.0;
      }
      continue;
    }

    // Get the grid indices and cell fractions for the particle.
    std::array<int, MAX_GRID_NDIM> part_indices;
    std::array<double, MAX_GRID_NDIM> axis_fracs;
    get_part_ind_frac_cic(part_indices, axis_fracs, grid_props, parts, p);

    // Compute base linear index for this particle
    int grid_ind = grid_props->ravel_grid_index(part_indices);
    parts->grid_indices[p] = grid_ind;

    // Store the per‐dimension fractions
    for (int d = 0; d < ndim; d++) {
      parts->grid_fracs[p * ndim + d] = axis_fracs[d];
    }
  }
}

/**
 * @brief Calculate the grid indices and fractions for all particles.
 *
 * This is a wrapper function that calls the correct version based on
 * the number of threads requested or whether OpenMP is available.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param nthreads: The number of threads to use.
 */
void get_particle_indices_and_fracs(GridProps *grid_props, Particles *parts,
                                    int nthreads) {

  double start = tic();

#ifdef WITH_OPENMP
  if (nthreads > 1) {
    get_particle_indices_and_fracs_parallel(grid_props, parts, nthreads);
  } else {
    get_particle_indices_and_fracs_serial(grid_props, parts);
  }
#else
  get_particle_indices_and_fracs_serial(grid_props, parts);
#endif /* WITH_OPENMP */

  toc("Finding particle grid indices and fractions", start);
}

/**
 * @brief This calculates particle spectra using a cloud in cell approach.
 *
 * This is the serial version of the function.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 */
static void spectra_loop_cic_serial(GridProps *grid_props, Particles *parts,
                                    double *part_spectra) {
  /* Unpack the grid properties. */
  const int ndim = grid_props->ndim;
  int nlam = grid_props->nlam;

  /* Calculate the number of cell in a patch of the grid (2^ndim). */
  int ncells = 1 << ndim;

  /* Set up fixed sub-dimensions array (always {2, 2, ..., 2}) */
  std::array<int, MAX_GRID_NDIM> sub_dims;
  for (int idim = 0; idim < ndim; idim++) {
    sub_dims[idim] = 2;
  }

  /* Precompute sub-cell offsets and linear offsets once */
  struct SubCell {
    std::array<int, MAX_GRID_NDIM> offs;
    int linoff;
  };
  std::vector<SubCell> subcells(ncells);
  {
    std::array<int, MAX_GRID_NDIM> subset_ind;
    for (int ic = 0; ic < ncells; ic++) {
      get_indices_from_flat(ic, ndim, sub_dims, subset_ind);
      subcells[ic].offs = subset_ind;
      /* ravel_grid_index on the offset gives the linear offset */
      subcells[ic].linoff = grid_props->ravel_grid_index(subset_ind);
    }
  }

  /* Precompute unmasked wavelengths */
  std::vector<int> good_lams;
  good_lams.reserve(nlam);
  for (int ilam = 0; ilam < nlam; ilam++) {
    if (!grid_props->lam_is_masked(ilam)) {
      good_lams.push_back(ilam);
    }
  }

  /* Loop over particles. */
  for (int p = 0; p < parts->npart; p++) {

    /* Skip masked particles. */
    if (parts->part_is_masked(p)) {
      continue;
    }

    /* Compute base linear index for this particle */
    const int base_linidx = parts->grid_indices[p];

    /* Cache particle weight once */
    const double w_p = parts->get_weight_at(p);

    /* Loop over sub-cells collecting their weighted contributions. */
    for (int icell = 0; icell < ncells; icell++) {
      const auto &sc = subcells[icell];

      /* Compute the CIC fraction */
      double frac = 1.0;
      for (int idim = 0; idim < ndim; idim++) {
        if (sc.offs[idim]) {
          frac *= parts->grid_fracs[p * ndim + idim];
        } else {
          frac *= (1.0 - parts->grid_fracs[p * ndim + idim]);
        }
      }
      if (frac == 0.0) {
        continue;
      }

      /* Define the weighted contribution from this cell. */
      const double weight = frac * w_p;

      /* Compute grid cell index via base + precomputed offset */
      const int grid_ind = base_linidx + sc.linoff;

      /* Add this grid cell's contribution to the spectra. */
      for (int jl = 0, J = (int)good_lams.size(); jl < J; jl++) {
        const int ilam = good_lams[jl];
        const double spec_val = grid_props->get_spectra_at(grid_ind, ilam);
        const int idx = p * nlam + ilam;

        /* Fused multiply-add for precision */
        part_spectra[idx] = std::fma(spec_val, weight, part_spectra[idx]);
      }
    }
  }
}

/**
 * @brief This calculates the grid weights in each grid cell using a cloud
 *        in cell approach.
 *
 * This is the parallel version of the function.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 * @param part_spectra: The per-particle output array.
 * @param nthreads: The number of threads to use.
 */
#ifdef WITH_OPENMP
static void spectra_loop_cic_omp(GridProps *grid_props, Particles *parts,
                                 double *part_spectra, int nthreads) {
  /* Unpack the grid properties. */
  const int ndim = grid_props->ndim;
  const int nlam = grid_props->nlam;
  const int ncells = 1 << ndim;

  /* Subset dimensions are always 2 (low and high side). */
  std::array<int, MAX_GRID_NDIM> sub_dims;
  for (int i = 0; i < ndim; i++) {
    sub_dims[i] = 2;
  }

  /* Precompute sub-cell offsets and linear offsets once */
  struct SubCell {
    std::array<int, MAX_GRID_NDIM> offs;
    int linoff;
  };
  std::vector<SubCell> subcells(ncells);
  {
    std::array<int, MAX_GRID_NDIM> subset_ind;
    for (int ic = 0; ic < ncells; ic++) {
      get_indices_from_flat(ic, ndim, sub_dims, subset_ind);
      subcells[ic].offs = subset_ind;
      /* ravel_grid_index on the offset gives the linear offset */
      subcells[ic].linoff = grid_props->ravel_grid_index(subset_ind);
    }
  }

  /* Precompute unmasked wavelengths */
  std::vector<int> good_lams;
  good_lams.reserve(nlam);
  for (int ilam = 0; ilam < nlam; ilam++) {
    if (!grid_props->lam_is_masked(ilam)) {
      good_lams.push_back(ilam);
    }
  }

#pragma omp parallel num_threads(nthreads)
  {

    /* Split the work evenly across threads (no single particle is more
     * expensive than another). */
    int nparts_per_thread = parts->npart / nthreads;

    /* What thread is this? */
    int tid = omp_get_thread_num();

    /* Get the start and end indices for this thread. */
    int start_idx = tid * nparts_per_thread;
    int end_idx =
        (tid == nthreads - 1) ? parts->npart : start_idx + nparts_per_thread;

    /* Get this threads part of the output array. */
    double *__restrict local_part_spectra = part_spectra + start_idx * nlam;

    /* Get an array that we'll put each particle's spectra into. */
    std::vector<double> this_part_spectra(nlam, 0.0);

    for (int p = start_idx; p < end_idx; p++) {

      /* Skip masked particles. */
      if (parts->part_is_masked(p)) {
        continue;
      }

      /* Compute base linear index for this particle */
      const int base_linidx = parts->grid_indices[p];

      /* Cache particle weight once */
      const double w_p = parts->get_weight_at(p);

      /* Loop over sub-cells collecting their weighted contributions. */
      for (int icell = 0; icell < ncells; icell++) {
        const auto &sc = subcells[icell];

        /* Compute the CIC fraction */
        double frac = 1.0;
        for (int idim = 0; idim < ndim; idim++) {
          if (sc.offs[idim]) {
            frac *= parts->grid_fracs[p * ndim + idim];
          } else {
            frac *= (1.0 - parts->grid_fracs[p * ndim + idim]);
          }
        }
        if (frac == 0.0) {
          continue;
        }

        /* Define the weighted contribution from this cell. */
        const double weight = frac * w_p;

        /* Compute grid cell index via base + precomputed offset */
        const int grid_ind = base_linidx + sc.linoff;

        /* Add this grid cell's contribution to the spectra. */
        for (int jl = 0, J = (int)good_lams.size(); jl < J; jl++) {
          const int ilam = good_lams[jl];
          const double spec_val = grid_props->get_spectra_at(grid_ind, ilam);

          /* Write into the local spectra array for this thread. */
          this_part_spectra[ilam] =
              std::fma(spec_val, weight, this_part_spectra[ilam]);
        }
      }

      /* Copy the entire spectrum at once  into the output array. */
      memcpy(local_part_spectra + (p - start_idx) * nlam,
             this_part_spectra.data(), nlam * sizeof(double));

      /* Reset the local spectra for this particle. */
      std::fill(this_part_spectra.begin(), this_part_spectra.end(), 0.0);
    }
  }
}
#endif /* WITH_OPENMP */

/**
 * @brief This calculates particle spectra using a cloud in cell approach.
 *
 * This is a wrapper which calls the correct function based on the number of
 * threads requested and whether OpenMP is available.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 * @param nthreads: The number of threads to use.
 */
void spectra_loop_cic(GridProps *grid_props, Particles *parts,
                      double *part_spectra, const int nthreads) {

  /* First get the grid indices and fractions for all particles. */
  get_particle_indices_and_fracs(grid_props, parts, nthreads);

  double start_time = tic();

  /* Call the correct function for the configuration/number of threads. */

#ifdef WITH_OPENMP

  /* If we have multiple threads and OpenMP we can parallelise. */
  if (nthreads > 1) {
    spectra_loop_cic_omp(grid_props, parts, part_spectra, nthreads);
  }
  /* Otherwise there's no point paying the OpenMP overhead. */
  else {
    spectra_loop_cic_serial(grid_props, parts, part_spectra);
  }

#else

  (void)nthreads;

  /* We don't have OpenMP, just call the serial version. */
  spectra_loop_cic_serial(grid_props, parts, part_spectra);

#endif
  toc("Cloud in Cell particle spectra loop", start_time);
}

/**
 * @brief Get the grid indices for a particle using Nearest Grid Point (NGP).
 *
 * This is the serial version of the function.
 *
 * @param GridProps: The properties of the grid.
 * @param parts: The particle properties.
 */
static void get_particle_indices_serial(GridProps *grid_props,
                                        Particles *parts) {

  /* Unpack the grid properties. */
  const int ndim = grid_props->ndim;
  int nlam = grid_props->nlam;

  // Pre-allocate exactly npart slots to avoid resizing
  parts->grid_indices.resize(parts->npart);

  /* Loop over particles. */
  for (int p = 0; p < parts->npart; p++) {

    /* Skip masked particles. */
    if (parts->part_is_masked(p)) {
      parts->grid_indices[p] = -1;
      continue;
    }

    /* Get the grid indices and cell fractions for the particle. */
    std::array<int, MAX_GRID_NDIM> part_indices;
    get_part_inds_ngp(part_indices, grid_props, parts, p);

    /* Compute the flattened grid index for this particle and store it. */
    const int grid_ind = grid_props->ravel_grid_index(part_indices);
    parts->grid_indices[p] = grid_ind;
  }
}

/**
 * @brief Get the grid indices for a particle using Nearest Grid Point (NGP).
 *
 * This is the parallel version of the function.
 *
 * @param grid_props: The properties of the grid.
 * @param parts: The particle properties.
 * @param nthreads: The number of threads to use for parallel processing.
 */
static void get_particle_indices_parallel(GridProps *grid_props,
                                          Particles *parts, int nthreads) {
  const int ndim = grid_props->ndim;
  const int npart = parts->npart;

  // Pre-allocate exactly npart slots so we can write into slices on
  // each thread without resizing.
  parts->grid_indices.resize(npart);

#pragma omp parallel for num_threads(nthreads) schedule(static)
  // Loop over particles in parallel
  for (int p = 0; p < npart; p++) {

    // Skip masked particles
    if (parts->part_is_masked(p)) {
      parts->grid_indices[p] = -1;
      continue;
    }

    // Get the grid indices for the particle.
    std::array<int, MAX_GRID_NDIM> part_indices;
    get_part_inds_ngp(part_indices, grid_props, parts, p);

    // Compute the flattened grid index for this particle and store it.
    int grid_ind = grid_props->ravel_grid_index(part_indices);
    parts->grid_indices[p] = grid_ind;
  }
}

/**
 * @brief Calculate the grid indices for all particles.
 *
 * This is a wrapper function that calls the correct version based on
 * the number of threads requested or whether OpenMP is available.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param nthreads: The number of threads to use.
 */
void get_particle_indices(GridProps *grid_props, Particles *parts,
                          int nthreads) {

  double start = tic();

#ifdef WITH_OPENMP
  if (nthreads > 1) {
    get_particle_indices_parallel(grid_props, parts, nthreads);
  } else {
    get_particle_indices_serial(grid_props, parts);
  }
#else
  get_particle_indices_serial(grid_props, parts);
#endif /* WITH_OPENMP */

  toc("Finding particle grid indices", start);
}

/**
 * @brief This calculates particle spectra using a nearest grid point
 *        approach.
 *
 * This is the serial version of the function.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 * @param part_spectra: The per-particle output array.
 */
static void spectra_loop_ngp_serial(GridProps *grid_props, Particles *parts,
                                    double *part_spectra) {
  /* Unpack the grid properties. */
  const int nlam = grid_props->nlam;

  /* Loop over particles. */
  for (int p = 0; p < parts->npart; p++) {

    /* Skip masked particles. */
    if (parts->part_is_masked(p)) {
      continue;
    }

    /* Get the weight's index. */
    const int grid_ind = parts->grid_indices[p];

    /* Get the weight of this particle. */
    const double weight = parts->get_weight_at(p);

    /* Add this grid cell's contribution to the spectra */
    for (int ilam = 0; ilam < nlam; ilam++) {

      /* Skip if this wavelength is masked. */
      if (grid_props->lam_is_masked(ilam)) {
        continue;
      }

      /* Get the spectra value at this index and wavelength. */
      const double spec_val = grid_props->get_spectra_at(grid_ind, ilam);

      /* Assign to this particle's spectra array. */
      part_spectra[p * nlam + ilam] = spec_val * weight;
    }
  }
}

/**
 * @brief This calculates particle spectra using a nearest grid point approach.
 *
 * This is the parallel version of the function.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 * @param part_spectra: The per-particle output array.
 * @param nthreads: The number of threads to use.
 */
#ifdef WITH_OPENMP
static void spectra_loop_ngp_omp(GridProps *grid_props, Particles *parts,
                                 double *part_spectra, int nthreads) {
  /* Unpack the grid properties. */
  const int nlam = grid_props->nlam;

  /* Precompute unmasked wavelengths */
  std::vector<int> good_lams;
  good_lams.reserve(nlam);
  for (int ilam = 0; ilam < nlam; ilam++) {
    if (!grid_props->lam_is_masked(ilam)) {
      good_lams.push_back(ilam);
    }
  }

#pragma omp parallel num_threads(nthreads)
  {
    /* Split the work evenly across threads (no single particle is more
     * expensive than another). */
    int nparts_per_thread = parts->npart / nthreads;

    /* What thread is this? */
    int tid = omp_get_thread_num();

    /* Get the start and end indices for this thread. */
    int start_idx = tid * nparts_per_thread;
    int end_idx =
        (tid == nthreads - 1) ? parts->npart : start_idx + nparts_per_thread;

    /* Get this threads part of the output array. */
    double *__restrict local_part_spectra = part_spectra + start_idx * nlam;

    /* Get an array that we'll put each particle's spectra into. */
    std::vector<double> this_part_spectra(nlam, 0.0);

    /* Loop over particles. */
    for (int p = start_idx; p < end_idx; p++) {

      /* Skip masked particles. */
      if (parts->part_is_masked(p)) {
        continue;
      }

      /* Get the particle's grid index. */
      const int grid_ind = parts->grid_indices[p];

      /* Get the weight of this particle. */
      const double weight = parts->get_weight_at(p);

      /* Add this grid cell's contribution to the spectra */
      for (int jl = 0, J = (int)good_lams.size(); jl < J; jl++) {

        /* Get the wavelength index. */
        const int ilam = good_lams[jl];

        /* Get the spectra value at this index and wavelength. */
        const double spec_val = grid_props->get_spectra_at(grid_ind, ilam);

        /* Assign to this particle's spectra array. */
        this_part_spectra[ilam] = spec_val * weight;
      }

      /* Copy the entire spectrum at once into the output array. */
      memcpy(local_part_spectra + (p - start_idx) * nlam,
             this_part_spectra.data(), nlam * sizeof(double));

      /* No reset needed as we overwrite the whole array each time and the
       * wavelength mask never changes. */
    }
  }
}
#endif

/**
 * @brief This calculates particle spectra using a nearest grid point
 * approach.
 *
 * This is a wrapper which calls the correct function based on the number of
 * threads requested and whether OpenMP is available.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 * @param nthreads: The number of threads to use.
 */
void spectra_loop_ngp(GridProps *grid_props, Particles *parts,
                      double *part_spectra, const int nthreads) {

  /* First get the grid indices for all particles. */
  get_particle_indices(grid_props, parts, nthreads);

  double start_time = tic();

  /* Call the correct function for the configuration/number of threads. */

#ifdef WITH_OPENMP

  /* If we have multiple threads and OpenMP we can parallelise. */
  if (nthreads > 1) {
    spectra_loop_ngp_omp(grid_props, parts, part_spectra, nthreads);
  }
  /* Otherwise there's no point paying the OpenMP overhead. */
  else {
    spectra_loop_ngp_serial(grid_props, parts, part_spectra);
  }

#else

  (void)nthreads;

  /* We don't have OpenMP, just call the serial version. */
  spectra_loop_ngp_serial(grid_props, parts, part_spectra);

#endif
  toc("Nearest Grid Point particle spectra loop", start_time);
}

/**
 * @brief Computes an integrated SED for a collection of particles.
 *
 * @param np_grid_spectra: The SPS spectra array.
 * @param grid_tuple: The tuple containing arrays of grid axis properties.
 * @param part_tuple: The tuple of particle property arrays (in the same
 * order as grid_tuple).
 * @param np_part_mass: The particle mass array.
 * @param np_velocities: The velocities array.
 * @param np_ndims: The size of each grid axis.
 * @param ndim: The number of grid axes.
 * @param npart: The number of particles.
 * @param nlam: The number of wavelength elements.
 * @param vel_shift: bool flag whether to consider doppler shift in spectra
 * computation. Defaults to False
 * @param c: speed of light
 */
PyObject *compute_particle_seds(PyObject *self, PyObject *args) {
  double start_time = tic();

  /* We don't need the self argument but it has to be there. Tell the
   * compiler we don't care. */
  (void)self;

  int ndim, npart, nlam, nthreads;
  PyObject *grid_tuple, *part_tuple;
  PyArrayObject *np_grid_spectra;
  PyArrayObject *np_part_mass, *np_ndims;
  PyArrayObject *np_mask, *np_lam_mask;
  char *method;

  if (!PyArg_ParseTuple(args, "OOOOOiiisiOO", &np_grid_spectra, &grid_tuple,
                        &part_tuple, &np_part_mass, &np_ndims, &ndim, &npart,
                        &nlam, &method, &nthreads, &np_mask, &np_lam_mask)) {
    return NULL;
  }

  /* Extract the grid struct. */
  GridProps *grid_props = new GridProps(np_grid_spectra, grid_tuple,
                                        /*np_lam*/ nullptr, np_lam_mask, nlam);
  RETURN_IF_PYERR();

  /* Create the object that holds the particle properties. */
  Particles *part_props = new Particles(np_part_mass, /*np_velocities*/ NULL,
                                        np_mask, part_tuple, npart);
  RETURN_IF_PYERR();

  double out_tic = tic();

  /* Define the output dimensions. */
  npy_intp np_int_dims[1] = {nlam};
  npy_intp np_part_dims[2] = {npart, nlam};

  /* Allocate the spectra. */
  PyArrayObject *np_spectra =
      (PyArrayObject *)PyArray_ZEROS(1, np_int_dims, NPY_DOUBLE, 0);
  double *spectra = static_cast<double *>(PyArray_DATA(np_spectra));
  PyArrayObject *np_part_spectra =
      (PyArrayObject *)PyArray_ZEROS(2, np_part_dims, NPY_DOUBLE, 0);
  double *part_spectra = static_cast<double *>(PyArray_DATA(np_part_spectra));

  toc("Setting up output arrays", out_tic);

  /* With everything set up we can compute the spectra for each particle
   * using the requested method. */
  if (strcmp(method, "cic") == 0) {
    spectra_loop_cic(grid_props, part_props, part_spectra, nthreads);
  } else if (strcmp(method, "ngp") == 0) {
    spectra_loop_ngp(grid_props, part_props, part_spectra, nthreads);
  } else {
    PyErr_Format(PyExc_ValueError, "Unknown grid assignment method (%s).",
                 method);
    return NULL;
  }
  RETURN_IF_PYERR();

  /* Reduce the per-particle spectra to the integrated spectra. */
  reduce_spectra(spectra, part_spectra, nlam, npart, nthreads);

  /* Clean up memory! */
  delete part_props;
  delete grid_props;

  /* Construct the output tuple. */
  PyObject *out_tuple = Py_BuildValue("NN", np_part_spectra, np_spectra);

  toc("Computing particle and integrated lnus", start_time);

  return out_tuple;
}

/* Below is all the gubbins needed to make the module importable in Python. */
static PyMethodDef SedMethods[] = {
    {"compute_particle_seds", (PyCFunction)compute_particle_seds, METH_VARARGS,
     "Method for calculating particle intrinsic spectra."},
    {NULL, NULL, 0, NULL}};

/* Make this importable. */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "make_particle_sed",                   /* m_name */
    "A module to calculate particle seds", /* m_doc */
    -1,                                    /* m_size */
    SedMethods,                            /* m_methods */
    NULL,                                  /* m_reload */
    NULL,                                  /* m_traverse */
    NULL,                                  /* m_clear */
    NULL,                                  /* m_free */
};

PyMODINIT_FUNC PyInit_particle_spectra(void) {
  PyObject *m = PyModule_Create(&moduledef);
  if (numpy_import() < 0) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to import numpy.");
    return NULL;
  }
  return m;
};
