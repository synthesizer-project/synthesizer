/******************************************************************************
 * C extension to calculate SEDs for star particles.
 * Calculates weights on an arbitrary dimensional grid given the mass.
 *****************************************************************************/
/* C includes */
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
 * @brief Find nearest wavelength bin for a given lambda, in a given wavelength
 * array. Used by the spectra loop functions when considering doppler shift
 *
 * Note: binary search returns the index of the upper bin of those that straddle
 * the given lambda.
 */
int get_upper_lam_bin(double lambda, double *grid_wavelengths, int nlam) {
  return binary_search(0, nlam - 1, grid_wavelengths, lambda);
}

/**
 * @brief This calculates particle spectra using a cloud in cell approach.
 * This is the version of the function that accounts for doppler shift.
 * This is the serial version of the function.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 * @param part_spectra: The per-particle output array.
 * @param c: speed of light.
 */
static void shifted_spectra_loop_cic_serial(GridProps *grid_props,
                                            Particles *parts,
                                            double *part_spectra,
                                            const double c) {

  /* Unpack the grid properties. */
  const int ndim = grid_props->ndim;
  size_t nlam = static_cast<size_t>(grid_props->nlam);
  double *wavelength = grid_props->get_lam();
  const int ncells = 1 << ndim;

  /* Get and cast the number of particles. */
  size_t npart = static_cast<size_t>(parts->npart);

  /* Build sub_dims = [2,2,...,2] once */
  std::array<int, MAX_GRID_NDIM> sub_dims;
  for (int d = 0; d < ndim; ++d) {
    sub_dims[d] = 2;
  }

  /* Precompute sub-cell offsets and linear offsets once */
  struct SubCell {
    std::array<int, MAX_GRID_NDIM> offs;
    int linoff;
  };
  std::vector<SubCell> subcells(ncells);
  {
    std::array<int, MAX_GRID_NDIM> tmp;
    for (int ic = 0; ic < ncells; ++ic) {
      get_indices_from_flat(ic, ndim, sub_dims, tmp);
      subcells[ic].offs = tmp;
      subcells[ic].linoff = grid_props->ravel_grid_index(tmp);
    }
  }

  /* Allocate arrays for shifted wavelengths and bin mappings once */
  std::vector<double> shifted_wavelengths(nlam);
  std::vector<int> mapped_indices(nlam);

  /* Loop over particles. */
  for (size_t p = 0; p < npart; ++p) {

    /* Skip masked particles. */
    if (parts->part_is_masked(p)) {
      continue;
    }

    /* Compute shift factor for this particle. */
    const double vel = parts->get_vel_at(p);
    const double shift_factor = 1.0 + vel / c;

    /* Shift wavelengths & map to bins once per particle. */
    for (size_t il = 0; il < nlam; ++il) {
      const double lam_s = wavelength[il] * shift_factor;
      shifted_wavelengths[il] = lam_s;
      mapped_indices[il] = get_upper_lam_bin(lam_s, wavelength, nlam);
    }

    /* Get per-particle grid base index. */
    const int base_lin = parts->grid_indices[p];

    /* Cache particle weight once */
    const double w_p = parts->get_weight_at(p);

    /* Loop over all 2^ndim sub-cells. */
    for (int ic = 0; ic < ncells; ++ic) {
      const auto &sc = subcells[ic];

      /* Compute CIC fraction for this corner */
      double frac = 1.0;
      for (int d = 0; d < ndim; ++d) {
        frac *= sc.offs[d] ? parts->grid_fracs[p * ndim + d]
                           : (1.0 - parts->grid_fracs[p * ndim + d]);
      }
      if (frac == 0.0)
        continue;

      /* Combined weight */
      const double weight = frac * w_p;

      /* Flattened grid index */
      const int grid_i = base_lin + sc.linoff;

      /* Loop over wavelengths (we can't prepare the unmasked wavelengths
       * like we can in the non-shifted case, since the shifted wavelengths
       * are particle-dependent) */
      for (size_t il = 0; il < nlam; ++il) {
        const int ils = mapped_indices[il];
        /* Skip out-of-bounds or masked */
        if (ils <= 0 || static_cast<size_t>(ils) >= nlam || grid_props->lam_is_masked(ils)) {
          continue;
        }

        /* Interpolation fraction between bins */
        const double lam_s = shifted_wavelengths[il];
        const double frac_s = (lam_s - wavelength[ils - 1]) /
                              (wavelength[ils] - wavelength[ils - 1]);

        /* Base spectra value */
        const double gs = grid_props->get_spectra_at(grid_i, il) * weight;

        /* Distribute into particle & global arrays */
        const size_t base_idx = p * nlam;
        part_spectra[base_idx + ils - 1] += (1.0 - frac_s) * gs;
        part_spectra[base_idx + ils] += frac_s * gs;
      }
    }
  }
}

/**
 * @brief This calculates particle spectra using a cloud in cell approach.
 * This is the parallel version of the function that accounts for Doppler shift.
 *
 * Each thread allocates its shift‐mapping buffers once and reuses them for
 * every particle, and all sub‐cell index math is hoisted out of the particle
 * loop.
 *
 * TODO: This currently scales poorly relative to the non-shifted case, since
 * the memory access pattern is not as cache-friendly due to the scattered
 * accesses to the shifted wavelengths and their mapped indices.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 * @param part_spectra: The per-particle output array.
 * @param nthreads: The number of threads to use.
 * @param c: speed of light.
 */
#ifdef WITH_OPENMP
static void shifted_spectra_loop_cic_omp(GridProps *grid_props,
                                         Particles *parts, double *part_spectra,
                                         int nthreads, const double c) {

  /* Unpack the grid properties. */
  const int ndim = grid_props->ndim;
  size_t nlam = static_cast<size_t>(grid_props->nlam);
  double *wavelength = grid_props->get_lam();
  const int ncells = 1 << ndim;

  /* Get and cast the number of particles. */
  size_t npart = static_cast<size_t>(parts->npart);

  /* Build sub_dims = [2,2,...,2] once */
  std::array<int, MAX_GRID_NDIM> sub_dims;
  for (int d = 0; d < ndim; ++d) {
    sub_dims[d] = 2;
  }

  /* Precompute sub-cell offsets and linear offsets once */
  struct SubCell {
    std::array<int, MAX_GRID_NDIM> offs;
    int linoff;
  };
  std::vector<SubCell> subcells(ncells);
  {
    std::array<int, MAX_GRID_NDIM> tmp;
    for (int ic = 0; ic < ncells; ++ic) {
      get_indices_from_flat(ic, ndim, sub_dims, tmp);
      subcells[ic].offs = tmp;
      subcells[ic].linoff = grid_props->ravel_grid_index(tmp);
    }
  }

#pragma omp parallel num_threads(nthreads)
  {
    /* Allocate per-thread shift buffers once */
    std::vector<double> shifted_wavelengths(nlam);
    std::vector<int> mapped_indices(nlam);

    /* Split the work evenly across threads (no single particle is more
     * expensive than another). */
    size_t nparts_per_thread = npart / nthreads;

    /* What thread is this? */
    int tid = omp_get_thread_num();

    /* Get the start and end indices for this thread. */
    size_t start_idx = tid * nparts_per_thread;
    size_t end_idx =
        (tid == nthreads - 1) ? parts->npart : start_idx + nparts_per_thread;

    /* Get this threads part of the output array. */
    double *__restrict local_part_spectra = part_spectra + start_idx * nlam;

    /* Get an array that we'll put each particle's spectra into. */
    std::vector<double> this_part_spectra(nlam, 0.0);

    /* Loop over particles in this thread's range. */
    for (size_t p = start_idx; p < end_idx; p++) {

      /* Skip masked particles. */
      if (parts->part_is_masked(p)) {
        continue;
      }

      /* Compute the Doppler shift factor. */
      const double vel = parts->get_vel_at(p);
      const double shift_factor = 1.0 + vel / c;

      /* Shift wavelengths & map to bins once per particle */
      for (size_t il = 0; il < nlam; ++il) {
        const double lam_s = wavelength[il] * shift_factor;
        shifted_wavelengths[il] = lam_s;
        mapped_indices[il] = get_upper_lam_bin(lam_s, wavelength, nlam);
      }

      /* Compute base linear index and cached weight */
      const int base_lin = parts->grid_indices[p];
      const double w_p = parts->get_weight_at(p);

      /* Loop over all 2^ndim sub-cells */
      for (int ic = 0; ic < ncells; ++ic) {
        const auto &sc = subcells[ic];

        /* Compute CIC fraction for this corner */
        double frac = 1.0;
        for (int d = 0; d < ndim; ++d) {
          frac *= sc.offs[d] ? parts->grid_fracs[p * ndim + d]
                             : (1.0 - parts->grid_fracs[p * ndim + d]);
        }
        if (frac == 0.0)
          continue;

        /* Combined weight */
        const double weight = frac * w_p;
        const int grid_i = base_lin + sc.linoff;

        /* Loop over wavelengths (we can't prepare the unmasked wavelengths like
         * we can in the non-shifted case, since the shifted wavelengths are
         * particle-dependent) */
        for (size_t il = 0; il < nlam; ++il) {
          const int ils = mapped_indices[il];
          /* Skip out-of-bounds or masked bins */
          if (ils <= 0 || static_cast<size_t>(ils) >= nlam || grid_props->lam_is_masked(ils)) {
            continue;
          }

          /* Interpolation fraction */
          const double lam_s = shifted_wavelengths[il];
          const double frac_s = (lam_s - wavelength[ils - 1]) /
                                (wavelength[ils] - wavelength[ils - 1]);

          /* Base spectra contribution */
          const double gs = grid_props->get_spectra_at(grid_i, il) * weight;

          /* Deposit into the thread's part spectra */
          this_part_spectra[ils - 1] =
              std::fma((1.0 - frac_s), gs, this_part_spectra[ils - 1]);
          this_part_spectra[ils] = std::fma(frac_s, gs, this_part_spectra[ils]);
        }
      }

      /* Copy the entire spectrum at once  into the output array. */
      for (size_t il = 0; il < nlam; ++il) {
        local_part_spectra[(p - start_idx) * nlam + il] = this_part_spectra[il];
      }
    }
  }
}
#endif /* WITH_OPENMP */

/**
 * @brief This calculates doppler-shifted particle spectra using a cloud in
 * cell approach.
 *
 * This is a wrapper which calls the correct function based on the number of
 * threads requested and whether OpenMP is available.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 * @param nthreads: The number of threads to use.
 */
void shifted_spectra_loop_cic(GridProps *grid_props, Particles *parts,
                              double *part_spectra, const int nthreads,
                              const double c) {

  /* First get the grid indices and fractions for all particles. */
  get_particle_indices_and_fracs(grid_props, parts, nthreads);

  double start_time = tic();

  /* Call the correct function for the configuration/number of threads. */

#ifdef WITH_OPENMP

  /* If we have multiple threads and OpenMP we can parallelise. */
  if (nthreads > 1) {
    shifted_spectra_loop_cic_omp(grid_props, parts, part_spectra, nthreads, c);
  }
  /* Otherwise there's no point paying the OpenMP overhead. */
  else {
    shifted_spectra_loop_cic_serial(grid_props, parts, part_spectra, c);
  }

#else

  (void)nthreads;

  /* We don't have OpenMP, just call the serial version. */
  shifted_spectra_loop_cic_serial(grid_props, parts, part_spectra, c);

#endif
  toc("Cloud in Cell particle loop", start_time);
}

/**
 * @brief This calculates particle spectra using a nearest grid point
 * approach. This is the version of the function that accounts doppler shift
 * This is the serial version of the function.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 * @param c: speed of light.
 */
static void shifted_spectra_loop_ngp_serial(GridProps *grid_props,
                                            Particles *parts,
                                            double *part_spectra,
                                            const double c) {

  /* Unpack the grid properties. */
  size_t nlam = static_cast<size_t>(grid_props->nlam);
  double *wavelength = grid_props->get_lam();

  /* Get and cast the number of particles. */
  size_t npart = static_cast<size_t>(parts->npart);

  /* Allocate the shifted wavelengths array and the mapped indices array. */
  std::vector<double> shifted_wavelengths(nlam);
  std::vector<int> mapped_indices(nlam);

  /* Loop over particles. */
  for (size_t p = 0; p < npart; p++) {

    /* Skip masked particles. */
    if (parts->part_is_masked(p)) {
      continue;
    }

    /* Get the particle velocity and red/blue shift factor. */
    double vel = parts->get_vel_at(p);
    double shift_factor = 1.0 + vel / c;

    /* Shift wavelengths & map to bins once per particle. */
    for (size_t il = 0; il < nlam; ++il) {
      const double lam_s = wavelength[il] * shift_factor;
      shifted_wavelengths[il] = lam_s;
      mapped_indices[il] = get_upper_lam_bin(lam_s, wavelength, nlam);
    }

    /* Define the weight. */
    double weight = parts->get_weight_at(p);

    /* Get the weight's index. */
    const int grid_ind = parts->grid_indices[p];

    /* Loop over wavelengths (we can't prepare the unmasked wavelengths
     * like we can in the non-shifted case, since the shifted wavelengths
     * are particle-dependent) */
    for (size_t ilam = 0; ilam < nlam; ilam++) {

      /* Get the shifted wavelength and index. */
      int ilam_shifted = mapped_indices[ilam];
      double shifted_lambda = shifted_wavelengths[ilam];

      /* Skip if this wavelength is masked. */
      if (grid_props->lam_is_masked(ilam_shifted)) {
        continue;
      }

      /* Compute the fraction of the shifted wavelength between the two
       * closest wavelength elements. */
      double frac_shifted = 0.0;
      if (ilam_shifted > 0 && static_cast<size_t>(ilam_shifted) <= nlam - 1) {
        frac_shifted =
            (shifted_lambda - wavelength[ilam_shifted - 1]) /
            (wavelength[ilam_shifted] - wavelength[ilam_shifted - 1]);
      } else {
        /* Out of bounds, skip this wavelength */
        continue;
      }

      /* Get the grid spectra value for this wavelength. */
      double grid_spectra_value =
          grid_props->get_spectra_at(grid_ind, ilam) * weight;

      /* Add the contribution to the corresponding wavelength element. */
      size_t idx = p * nlam + ilam_shifted;
      part_spectra[idx - 1] += (1.0 - frac_shifted) * grid_spectra_value;
      part_spectra[idx] += frac_shifted * grid_spectra_value;
    }
  }
}

/**
 * @brief This calculates particle spectra using a nearest grid point approach.
 * This is the version of the function that accounts for doppler shift.
 * This is the parallel version of the function.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 * @param nthreads: The number of threads to use.
 */
#ifdef WITH_OPENMP
static void shifted_spectra_loop_ngp_omp(GridProps *grid_props,
                                         Particles *parts, double *part_spectra,
                                         int nthreads, const double c) {

  /* Unpack the grid properties. */
  size_t nlam = static_cast<size_t>(grid_props->nlam);
  double *wavelength = grid_props->get_lam();

  /* Get and cast the number of particles. */
  size_t npart = static_cast<size_t>(parts->npart);

#pragma omp parallel num_threads(nthreads)
  {

    /* Allocate the shifted wavelengths array and the mapped indices array. */
    std::vector<double> shifted_wavelengths(nlam);
    std::vector<int> mapped_indices(nlam);

    /* Split the work evenly across threads (no single particle is more
     * expensive than another). */
    size_t nparts_per_thread = npart / nthreads;

    /* What thread is this? */
    int tid = omp_get_thread_num();

    /* Get the start and end indices for this thread. */
    size_t start_idx = tid * nparts_per_thread;
    size_t end_idx =
        (tid == nthreads - 1) ? parts->npart : start_idx + nparts_per_thread;

    /* Get this threads part of the output array. */
    double *__restrict local_part_spectra = part_spectra + start_idx * nlam;

    /* Get an array that we'll put each particle's spectra into. */
    std::vector<double> this_part_spectra(nlam, 0.0);

    /* Loop over particles in this thread's range. */
    for (size_t p = start_idx; p < end_idx; p++) {

      /* Skip masked particles. */
      if (parts->part_is_masked(p)) {
        continue;
      }

      /* Get the particle velocity and red/blue shift factor. */
      double vel = parts->get_vel_at(p);
      double shift_factor = 1.0 + vel / c;

      /* Shift wavelengths & map to bins once per particle. */
      for (size_t il = 0; il < nlam; ++il) {
        const double lam_s = wavelength[il] * shift_factor;
        shifted_wavelengths[il] = lam_s;
        mapped_indices[il] = get_upper_lam_bin(lam_s, wavelength, nlam);
      }

      /* Define the weighted contribution from this cell. */
      const double weight = parts->get_weight_at(p);

      /* Get the index of the grid cell. */
      const int grid_ind = parts->grid_indices[p];

      /* Loop over wavelengths (we can't prepare the unmasked wavelengths
       * like we can in the non-shifted case, since the shifted wavelengths
       * are particle-dependent) */
      for (size_t ilam = 0; ilam < nlam; ilam++) {

        /* Get the shifted wavelength and index. */
        int ilam_shifted = mapped_indices[ilam];
        double shifted_lambda = shifted_wavelengths[ilam];

        /* Skip if this wavelength is masked. */
        if (grid_props->lam_is_masked(ilam_shifted)) {
          continue;
        }

        /* Compute the fraction of the shifted wavelength between the two
         * closest wavelength elements. */
        double frac_shifted = 0.0;
        if (ilam_shifted > 0 && static_cast<size_t>(ilam_shifted) <= nlam - 1) {
          frac_shifted =
              (shifted_lambda - wavelength[ilam_shifted - 1]) /
              (wavelength[ilam_shifted] - wavelength[ilam_shifted - 1]);
        } else {
          /* Out of bounds, skip this wavelength */
          continue;
        }

        /* Get the grid spectra value for this wavelength. */
        double grid_spectra_value =
            grid_props->get_spectra_at(grid_ind, ilam) * weight;

        /* Deposit into the thread's part spectra */
        this_part_spectra[ilam_shifted - 1] =
            std::fma((1.0 - frac_shifted), grid_spectra_value,
                     this_part_spectra[ilam_shifted - 1]);
        this_part_spectra[ilam_shifted] = std::fma(
            frac_shifted, grid_spectra_value, this_part_spectra[ilam_shifted]);
      }

      /* Copy the entire spectrum at once  into the output array. */
      memcpy(local_part_spectra + (p - start_idx) * nlam,
             this_part_spectra.data(), nlam * sizeof(double));

      /* Reset the local spectra for this particle. */
      std::fill(this_part_spectra.begin(), this_part_spectra.end(), 0.0);
    }
  }
}
#endif

/**
 * @brief This calculates doppler-shifted particle spectra using a nearest
 * grid point approach.
 *
 * This is a wrapper which calls the correct function based on the number of
 * threads requested and whether OpenMP is available.
 * This is the version of the wrapper that accounts for doppler shift.
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 * @param nthreads: The number of threads to use.
 */
void shifted_spectra_loop_ngp(GridProps *grid_props, Particles *parts,
                              double *part_spectra, const int nthreads,
                              const double c) {

  /* First get the grid indices for all particles. */
  get_particle_indices(grid_props, parts, nthreads);

  double start_time = tic();

  /* Call the correct function for the configuration/number of threads. */

#ifdef WITH_OPENMP

  /* If we have multiple threads and OpenMP we can parallelise. */
  if (nthreads > 1) {
    shifted_spectra_loop_ngp_omp(grid_props, parts, part_spectra, nthreads, c);
  }
  /* Otherwise there's no point paying the OpenMP overhead. */
  else {
    shifted_spectra_loop_ngp_serial(grid_props, parts, part_spectra, c);
  }

#else

  (void)nthreads;

  /* We don't have OpenMP, just call the serial version. */
  shifted_spectra_loop_ngp_serial(grid_props, parts, part_spectra, c);

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
PyObject *compute_part_seds_with_vel_shift(PyObject *self, PyObject *args) {

  double start_time = tic();
  double setup_start = tic();

  /* We don't need the self argument but it has to be there. Tell the
   * compiler we don't care. */
  (void)self;

  int ndim, npart, nlam, nthreads;
  PyObject *grid_tuple, *part_tuple;
  PyObject *py_c;
  PyArrayObject *np_grid_spectra, *np_lam;
  PyArrayObject *np_velocities;
  PyArrayObject *np_part_mass, *np_ndims;
  PyArrayObject *np_mask, *np_lam_mask;
  char *method;

  if (!PyArg_ParseTuple(args, "OOOOOOOiiisiOOO", &np_grid_spectra, &np_lam,
                        &grid_tuple, &part_tuple, &np_part_mass, &np_velocities,
                        &np_ndims, &ndim, &npart, &nlam, &method, &nthreads,
                        &py_c, &np_mask, &np_lam_mask)) {
    return NULL;
  }

  /* Extract the grid struct. */
  GridProps *grid_props =
      new GridProps(np_grid_spectra, grid_tuple, np_lam, np_lam_mask, nlam);
  RETURN_IF_PYERR();

  /* Create the object that holds the particle properties. */
  Particles *part_props =
      new Particles(np_part_mass, np_velocities, np_mask, part_tuple, npart);
  RETURN_IF_PYERR();

  /* Allocate the spectra. */
  double *spectra = new double[grid_props->nlam]();
  double *part_spectra = new double[npart * grid_props->nlam]();

  /* Convert c to double */
  double c = PyFloat_AsDouble(py_c);

  toc("Extracting Python data", setup_start);

  /* With everything set up we can compute the spectra for each particle
   * using the requested method. */
  if (strcmp(method, "cic") == 0) {
    shifted_spectra_loop_cic(grid_props, part_props, part_spectra, nthreads, c);
  } else if (strcmp(method, "ngp") == 0) {
    shifted_spectra_loop_ngp(grid_props, part_props, part_spectra, nthreads, c);
  } else {
    PyErr_Format(PyExc_ValueError, "Unknown grid assignment method (%s).",
                 method);
    return NULL;
  }
  RETURN_IF_PYERR();

  /* Reduce the per-particle spectra to the integrated spectra. */
  reduce_spectra(spectra, part_spectra, nlam, npart, nthreads);

  /* Check we got the spectra sucessfully. (Any error messages will already
   * be set) */
  if (spectra == NULL) {
    return NULL;
  }
  if (part_spectra == NULL) {
    return NULL;
  }

  /* Clean up memory! */
  delete part_props;
  delete grid_props;

  /* Construct the particle spectra output numpy array. */
  npy_intp np_dims[2] = {npart, nlam};
  PyArrayObject *out_part_spectra =
      wrap_array_to_numpy<double>(2, np_dims, part_spectra);

  /* Construct the integrated spectra output numpy array. */
  npy_intp np_dims_int[1] = {nlam};
  PyArrayObject *out_integrated_spectra =
      wrap_array_to_numpy<double>(1, np_dims_int, spectra);

  /* Construct the output tuple. */
  PyObject *out_tuple =
      Py_BuildValue("NN", out_part_spectra, out_integrated_spectra);

  toc("Computing particle and integrated lnus", start_time);

  return out_tuple;
}

/* Below is all the gubbins needed to make the module importable in Python. */
static PyMethodDef SedMethods[] = {
    {"compute_part_seds_with_vel_shift",
     (PyCFunction)compute_part_seds_with_vel_shift, METH_VARARGS,
     "Method for calculating particle intrinsic spectra accounting for "
     "velocity shift."},
    {NULL, NULL, 0, NULL}};

/* Make this importable. */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "make_shifted_particle_sed",                            /* m_name */
    "A module to calculate doppler shifted  particle seds", /* m_doc */
    -1,                                                     /* m_size */
    SedMethods,                                             /* m_methods */
    NULL,                                                   /* m_reload */
    NULL,                                                   /* m_traverse */
    NULL,                                                   /* m_clear */
    NULL,                                                   /* m_free */
};

PyMODINIT_FUNC PyInit_doppler_particle_spectra(void) {
  PyObject *m = PyModule_Create(&moduledef);
  if (numpy_import() < 0) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to import numpy.");
    return NULL;
  }
  return m;
};
