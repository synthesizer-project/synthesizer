/******************************************************************************
 * C extension to calculate SEDs for star particles.
 * Calculates weights on an arbitrary dimensional grid given the mass.
 *****************************************************************************/
/* C includes */
#include <algorithm>
#include <array>
#include <math.h>
#include <new>
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
#include "timers.h"
#ifdef ATOMIC_TIMING
#include "timers_init.h"
#endif
#include "weights.h"

/**
 * @brief This calculates particle spectra using a cloud in cell approach.
 *
 * This is the serial version of the function for grids with a wavelength mask.
 *
 * @tparam Real The floating-point type of the input data.
 * @tparam OutT The floating-point type stored in the output buffer.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param part_spectra: The per-particle output array.
 * @param good_lams: The wavelength indices that are not masked.
 */
template <typename Real, typename OutT>
static void spectra_loop_cic_with_lam_mask_serial(
    GridProps *grid_props, Particles *parts, OutT *part_spectra,
    const std::vector<int> &good_lams) {
  /* Unpack the grid properties. */
  const int ndim = grid_props->ndim;
  const std::array<int, MAX_GRID_NDIM> dims = grid_props->dims;
  size_t nlam = static_cast<size_t>(grid_props->nlam);
  const Real *__restrict grid_spectra = grid_props->get_spectra<Real>();

  /* Get and cast the number of particles. */
  size_t npart = static_cast<size_t>(parts->npart);

  /* Calculate the number of cell in a patch of the grid (2^ndim). */
  int ncells = 1 << ndim;

  /* Store the non-zero cell contributions for each particle. */
  std::vector<const Real *> cell_spectra_ptrs(ncells);
  std::vector<OutT> cell_weights(ncells);

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

  /* Loop over particles. */
  for (size_t p = 0; p < npart; p++) {

    /* Skip masked particles. */
    if (parts->part_is_masked(p)) {
      continue;
    }

    /* Compute base linear index for this particle */
    /* Cache particle weight and base-cell information once. */
    const Real w_p = parts->get_weight_at<Real>(p);
    std::array<int, MAX_GRID_NDIM> part_indices;
    std::array<Real, MAX_GRID_NDIM> axis_fracs;
    get_part_ind_frac_cic<Real>(part_indices, axis_fracs, grid_props, parts,
                                p);
    const int base_linidx = get_flat_index(part_indices, dims.data(), ndim);

    /* Loop over sub-cells collecting their weighted contributions. */
    int nvalid_cells = 0;
    for (int icell = 0; icell < ncells; icell++) {
      const auto &sc = subcells[icell];

      /* Compute the CIC fraction */
      double frac = 1.0;
      for (int idim = 0; idim < ndim; idim++) {
        frac *= sc.offs[idim] ? axis_fracs[idim]
                              : (static_cast<Real>(1) - axis_fracs[idim]);
      }
      if (frac == static_cast<Real>(0)) {
        continue;
      }

      /* Define the weighted contribution from this cell. */
      const OutT weight = static_cast<OutT>(frac) * static_cast<OutT>(w_p);

      /* Compute grid cell index via base + precomputed offset */
      const int grid_ind = base_linidx + sc.linoff;
      cell_spectra_ptrs[nvalid_cells] =
          grid_spectra + static_cast<size_t>(grid_ind) * nlam;
      cell_weights[nvalid_cells] = weight;
      nvalid_cells++;
    }

    /* Get this particle's output row. */
    OutT *__restrict part_spec = part_spectra + p * nlam;

    /* Add all grid cell contributions to the spectra. */
    for (int jl = 0, J = (int)good_lams.size(); jl < J; jl++) {
      const int ilam = good_lams[jl];
      OutT spec_val = static_cast<OutT>(0);
      for (int icell = 0; icell < nvalid_cells; icell++) {
        spec_val = std::fma(static_cast<OutT>(cell_spectra_ptrs[icell][ilam]),
                            cell_weights[icell], spec_val);
      }
      part_spec[ilam] = spec_val;
    }
  }
}

/**
 * @brief This calculates particle spectra using a cloud in cell approach.
 *
 * This is the serial version of the function for grids without a wavelength
 * mask.
 *
 * @tparam Real The floating-point type of the input data.
 * @tparam OutT The floating-point type stored in the output buffer.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param part_spectra: The per-particle output array.
 */
template <typename Real, typename OutT>
static void spectra_loop_cic_no_lam_mask_serial(GridProps *grid_props,
                                                Particles *parts,
                                                OutT *part_spectra) {
  /* Unpack the grid properties. */
  const int ndim = grid_props->ndim;
  const std::array<int, MAX_GRID_NDIM> dims = grid_props->dims;
  size_t nlam = static_cast<size_t>(grid_props->nlam);
  const Real *__restrict grid_spectra = grid_props->get_spectra<Real>();

  /* Get and cast the number of particles. */
  size_t npart = static_cast<size_t>(parts->npart);

  /* Calculate the number of cell in a patch of the grid (2^ndim). */
  int ncells = 1 << ndim;

  /* Store the non-zero cell contributions for each particle. */
  std::vector<const Real *> cell_spectra_ptrs(ncells);
  std::vector<OutT> cell_weights(ncells);

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

  /* Loop over particles. */
  for (size_t p = 0; p < npart; p++) {

    /* Skip masked particles. */
    if (parts->part_is_masked(p)) {
      continue;
    }

    /* Compute base linear index for this particle */
    /* Cache particle weight and base-cell information once. */
    const Real w_p = parts->get_weight_at<Real>(p);
    std::array<int, MAX_GRID_NDIM> part_indices;
    std::array<Real, MAX_GRID_NDIM> axis_fracs;
    get_part_ind_frac_cic<Real>(part_indices, axis_fracs, grid_props, parts,
                                p);
    const int base_linidx = get_flat_index(part_indices, dims.data(), ndim);

    /* Loop over sub-cells collecting their weighted contributions. */
    int nvalid_cells = 0;
    for (int icell = 0; icell < ncells; icell++) {
      const auto &sc = subcells[icell];

      /* Compute the CIC fraction */
      double frac = 1.0;
      for (int idim = 0; idim < ndim; idim++) {
        frac *= sc.offs[idim] ? axis_fracs[idim]
                              : (static_cast<Real>(1) - axis_fracs[idim]);
      }
      if (frac == static_cast<Real>(0)) {
        continue;
      }

      /* Define the weighted contribution from this cell. */
      const OutT weight = static_cast<OutT>(frac) * static_cast<OutT>(w_p);

      /* Compute grid cell index via base + precomputed offset */
      const int grid_ind = base_linidx + sc.linoff;
      cell_spectra_ptrs[nvalid_cells] =
          grid_spectra + static_cast<size_t>(grid_ind) * nlam;
      cell_weights[nvalid_cells] = weight;
      nvalid_cells++;
    }

    /* Get this particle's output row. */
    OutT *__restrict part_spec = part_spectra + p * nlam;

    /* Add all grid cell contributions to the spectra. */
    for (size_t ilam = 0; ilam < nlam; ilam++) {
      OutT spec_val = static_cast<OutT>(0);
      for (int icell = 0; icell < nvalid_cells; icell++) {
        spec_val = std::fma(static_cast<OutT>(cell_spectra_ptrs[icell][ilam]),
                            cell_weights[icell], spec_val);
      }
      part_spec[ilam] = spec_val;
    }
  }
}

/**
 * @brief This calculates particle spectra using a cloud in cell approach.
 *
 * This is the serial wrapper which dispatches to the masked or unmasked
 * wavelength implementation.
 *
 * @tparam Real The floating-point type of the input data.
 * @tparam OutT The floating-point type stored in the output buffer.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param part_spectra: The per-particle output array.
 * @param has_lam_mask: Are we applying a wavelength mask?
 */
template <typename Real, typename OutT>
static void spectra_loop_cic_serial(GridProps *grid_props, Particles *parts,
                                    OutT *part_spectra, bool has_lam_mask) {
  /* If there is no wavelength mask, use the branch-free contiguous loop. */
  if (!has_lam_mask) {
    spectra_loop_cic_no_lam_mask_serial<Real, OutT>(grid_props, parts,
                                                    part_spectra);
    return;
  }

  /* Precompute unmasked wavelengths. */
  const size_t nlam = static_cast<size_t>(grid_props->nlam);
  std::vector<int> good_lams;
  good_lams.reserve(nlam);
  for (size_t ilam = 0; ilam < nlam; ilam++) {
    if (!grid_props->lam_is_masked(ilam)) {
      good_lams.push_back(ilam);
    }
  }

  spectra_loop_cic_with_lam_mask_serial<Real, OutT>(grid_props, parts,
                                                    part_spectra, good_lams);
}

/**
 * @brief This calculates particle spectra using a cloud in cell approach.
 *
 * This is the parallel version of the function for grids with a wavelength
 * mask.
 *
 * @tparam Real The floating-point type of the input data.
 * @tparam OutT The floating-point type stored in the output buffer.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param part_spectra: The per-particle output array.
 * @param nthreads: The number of threads to use.
 * @param good_lams: The wavelength indices that are not masked.
 */
#ifdef WITH_OPENMP
template <typename Real, typename OutT>
static void spectra_loop_cic_with_lam_mask_omp(
    GridProps *grid_props, Particles *parts, OutT *part_spectra,
    int nthreads, const std::vector<int> &good_lams) {
  /* Unpack the grid properties. */
  const int ndim = grid_props->ndim;
  const std::array<int, MAX_GRID_NDIM> dims = grid_props->dims;
  size_t nlam = static_cast<size_t>(grid_props->nlam);
  const Real *__restrict grid_spectra = grid_props->get_spectra<Real>();
  const int ncells = 1 << ndim;

  /* Get and cast the number of particles. */
  size_t npart = static_cast<size_t>(parts->npart);

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

#pragma omp parallel num_threads(nthreads)
  {

    /* Split the work evenly across threads (no single particle is more
     * expensive than another). */
    size_t nparts_per_thread = npart / nthreads;

    /* What thread is this? */
    int tid = omp_get_thread_num();

    /* Get the start and end indices for this thread. */
    size_t start_idx = tid * nparts_per_thread;
    size_t end_idx =
        (tid == nthreads - 1) ? parts->npart : start_idx + nparts_per_thread;

    /* Store the non-zero cell contributions for each particle. */
    std::vector<const Real *> cell_spectra_ptrs(ncells);
    std::vector<OutT> cell_weights(ncells);

    /* Loop over particles in this thread's range. */
    for (size_t p = start_idx; p < end_idx; p++) {

      /* Skip masked particles. */
      if (parts->part_is_masked(p)) {
        continue;
      }

      /* Compute base linear index for this particle */
      /* Cache particle weight and base-cell information once. */
      const Real w_p = parts->get_weight_at<Real>(p);
      std::array<int, MAX_GRID_NDIM> part_indices;
      std::array<Real, MAX_GRID_NDIM> axis_fracs;
      get_part_ind_frac_cic<Real>(part_indices, axis_fracs, grid_props, parts,
                                  p);
      const int base_linidx = get_flat_index(part_indices, dims.data(), ndim);

      /* Loop over sub-cells collecting their weighted contributions. */
      int nvalid_cells = 0;
      for (int icell = 0; icell < ncells; icell++) {
        const auto &sc = subcells[icell];

        /* Compute the CIC fraction */
        double frac = 1.0;
        for (int idim = 0; idim < ndim; idim++) {
          frac *= sc.offs[idim] ? axis_fracs[idim]
                                : (static_cast<Real>(1) - axis_fracs[idim]);
        }
        if (frac == static_cast<Real>(0)) {
          continue;
        }

        /* Define the weighted contribution from this cell. */
        const OutT weight =
            static_cast<OutT>(frac) * static_cast<OutT>(w_p);

        /* Compute grid cell index via base + precomputed offset */
        const int grid_ind = base_linidx + sc.linoff;
        cell_spectra_ptrs[nvalid_cells] =
            grid_spectra + static_cast<size_t>(grid_ind) * nlam;
        cell_weights[nvalid_cells] = weight;
        nvalid_cells++;
      }

      /* Get this particle's output row. */
      OutT *__restrict part_spec = part_spectra + p * nlam;

      /* Add all grid cell contributions to the spectra. */
      for (int jl = 0, J = (int)good_lams.size(); jl < J; jl++) {
        const int ilam = good_lams[jl];
        OutT spec_val = static_cast<OutT>(0);
        for (int icell = 0; icell < nvalid_cells; icell++) {
          spec_val = std::fma(
              static_cast<OutT>(cell_spectra_ptrs[icell][ilam]),
              cell_weights[icell], spec_val);
        }
        part_spec[ilam] = spec_val;
      }
    }
  }
}

/**
 * @brief This calculates particle spectra using a cloud in cell approach.
 *
 * This is the parallel version of the function for grids without a wavelength
 * mask.
 *
 * @tparam Real The floating-point type of the input data.
 * @tparam OutT The floating-point type stored in the output buffer.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param part_spectra: The per-particle output array.
 * @param nthreads: The number of threads to use.
 */
template <typename Real, typename OutT>
static void spectra_loop_cic_no_lam_mask_omp(GridProps *grid_props,
                                             Particles *parts,
                                             OutT *part_spectra,
                                             int nthreads) {
  /* Unpack the grid properties. */
  const int ndim = grid_props->ndim;
  const std::array<int, MAX_GRID_NDIM> dims = grid_props->dims;
  size_t nlam = static_cast<size_t>(grid_props->nlam);
  const Real *__restrict grid_spectra = grid_props->get_spectra<Real>();
  const int ncells = 1 << ndim;

  /* Get and cast the number of particles. */
  size_t npart = static_cast<size_t>(parts->npart);

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

#pragma omp parallel num_threads(nthreads)
  {

    /* Split the work evenly across threads (no single particle is more
     * expensive than another). */
    size_t nparts_per_thread = npart / nthreads;

    /* What thread is this? */
    int tid = omp_get_thread_num();

    /* Get the start and end indices for this thread. */
    size_t start_idx = tid * nparts_per_thread;
    size_t end_idx =
        (tid == nthreads - 1) ? parts->npart : start_idx + nparts_per_thread;

    /* Store the non-zero cell contributions for each particle. */
    std::vector<const Real *> cell_spectra_ptrs(ncells);
    std::vector<OutT> cell_weights(ncells);

    /* Loop over particles in this thread's range. */
    for (size_t p = start_idx; p < end_idx; p++) {

      /* Skip masked particles. */
      if (parts->part_is_masked(p)) {
        continue;
      }

      /* Compute base linear index for this particle */
      /* Cache particle weight and base-cell information once. */
      const Real w_p = parts->get_weight_at<Real>(p);
      std::array<int, MAX_GRID_NDIM> part_indices;
      std::array<Real, MAX_GRID_NDIM> axis_fracs;
      get_part_ind_frac_cic<Real>(part_indices, axis_fracs, grid_props, parts,
                                  p);
      const int base_linidx = get_flat_index(part_indices, dims.data(), ndim);

      /* Loop over sub-cells collecting their weighted contributions. */
      int nvalid_cells = 0;
      for (int icell = 0; icell < ncells; icell++) {
        const auto &sc = subcells[icell];

        /* Compute the CIC fraction */
        double frac = 1.0;
        for (int idim = 0; idim < ndim; idim++) {
          frac *= sc.offs[idim] ? axis_fracs[idim]
                                : (static_cast<Real>(1) - axis_fracs[idim]);
        }
        if (frac == static_cast<Real>(0)) {
          continue;
        }

        /* Define the weighted contribution from this cell. */
        const OutT weight =
            static_cast<OutT>(frac) * static_cast<OutT>(w_p);

        /* Compute grid cell index via base + precomputed offset */
        const int grid_ind = base_linidx + sc.linoff;
        cell_spectra_ptrs[nvalid_cells] =
            grid_spectra + static_cast<size_t>(grid_ind) * nlam;
        cell_weights[nvalid_cells] = weight;
        nvalid_cells++;
      }

      /* Get this particle's output row. */
      OutT *__restrict part_spec = part_spectra + p * nlam;

      /* Add all grid cell contributions to the spectra. */
      for (size_t ilam = 0; ilam < nlam; ilam++) {
        OutT spec_val = static_cast<OutT>(0);
        for (int icell = 0; icell < nvalid_cells; icell++) {
          spec_val = std::fma(
              static_cast<OutT>(cell_spectra_ptrs[icell][ilam]),
              cell_weights[icell], spec_val);
        }
        part_spec[ilam] = spec_val;
      }
    }
  }
}

/**
 * @brief This calculates particle spectra using a cloud in cell approach.
 *
 * This is the parallel wrapper which dispatches to the masked or unmasked
 * wavelength implementation.
 *
 * @tparam Real The floating-point type of the input data.
 * @tparam OutT The floating-point type stored in the output buffer.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param part_spectra: The per-particle output array.
 * @param nthreads: The number of threads to use.
 * @param has_lam_mask: Are we applying a wavelength mask?
 */
template <typename Real, typename OutT>
static void spectra_loop_cic_omp(GridProps *grid_props, Particles *parts,
                                 OutT *part_spectra, int nthreads,
                                 bool has_lam_mask) {
  /* If there is no wavelength mask, use the branch-free contiguous loop. */
  if (!has_lam_mask) {
    spectra_loop_cic_no_lam_mask_omp<Real, OutT>(grid_props, parts,
                                                 part_spectra, nthreads);
    return;
  }

  /* Precompute unmasked wavelengths. */
  const size_t nlam = static_cast<size_t>(grid_props->nlam);
  std::vector<int> good_lams;
  good_lams.reserve(nlam);
  for (size_t ilam = 0; ilam < nlam; ilam++) {
    if (!grid_props->lam_is_masked(ilam)) {
      good_lams.push_back(ilam);
    }
  }

  spectra_loop_cic_with_lam_mask_omp<Real, OutT>(
      grid_props, parts, part_spectra, nthreads, good_lams);
}
#endif /* WITH_OPENMP */

/**
 * @brief This calculates particle spectra using a cloud in cell approach.
 *
 * This is a wrapper which calls the correct function based on the number of
 * threads requested and whether OpenMP is available.
 *
 * @tparam Real The floating-point type of the input data.
 * @tparam OutT The floating-point type stored in the output buffer.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param part_spectra: The per-particle output array.
 * @param nthreads: The number of threads to use.
 * @param has_lam_mask: Are we applying a wavelength mask?
 */
template <typename Real, typename OutT>
void spectra_loop_cic(GridProps *grid_props, Particles *parts,
                      OutT *part_spectra, const int nthreads,
                      bool has_lam_mask) {

  tic("spectra_loop_cic");

  /* Call the correct function for the configuration/number of threads. */

#ifdef WITH_OPENMP

  /* If we have multiple threads and OpenMP we can parallelise. */
  if (nthreads > 1) {
    spectra_loop_cic_omp<Real, OutT>(grid_props, parts, part_spectra,
                                     nthreads, has_lam_mask);
  }
  /* Otherwise there's no point paying the OpenMP overhead. */
  else {
    spectra_loop_cic_serial<Real, OutT>(grid_props, parts, part_spectra,
                                        has_lam_mask);
  }

#else

  (void)nthreads;

  /* We don't have OpenMP, just call the serial version. */
  spectra_loop_cic_serial<Real, OutT>(grid_props, parts, part_spectra,
                                      has_lam_mask);

#endif
  toc("spectra_loop_cic");
}

/**
 * @brief This calculates particle spectra using a nearest grid point
 *        approach.
 *
 * This is the serial version of the function for grids with a wavelength mask.
 *
 * @tparam Real The floating-point type of the input data.
 * @tparam OutT The floating-point type stored in the output buffer.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param part_spectra: The per-particle output array.
 * @param good_lams: The wavelength indices that are not masked.
 */
template <typename Real, typename OutT>
static void spectra_loop_ngp_with_lam_mask_serial(
    GridProps *grid_props, Particles *parts, OutT *part_spectra,
    const std::vector<int> &good_lams) {
  /* Unpack the grid properties. */
  const std::array<int, MAX_GRID_NDIM> dims = grid_props->dims;
  size_t nlam = static_cast<size_t>(grid_props->nlam);
  const Real *__restrict grid_spectra = grid_props->get_spectra<Real>();

  /* Get and cast the number of particles. */
  size_t npart = static_cast<size_t>(parts->npart);

  /* Loop over particles. */
  for (size_t p = 0; p < npart; p++) {

    /* Skip masked particles. */
    if (parts->part_is_masked(p)) {
      continue;
    }

    /* Get the weight's index. */
    std::array<int, MAX_GRID_NDIM> part_indices;
    get_part_inds_ngp<Real>(part_indices, grid_props, parts, p);
    const int grid_ind = get_flat_index(part_indices, dims.data(), grid_props->ndim);

    /* Get the weight of this particle. */
    const OutT weight = static_cast<OutT>(parts->get_weight_at<Real>(p));
    const Real *__restrict cell_spectra =
        grid_spectra + static_cast<size_t>(grid_ind) * nlam;
    OutT *__restrict part_spec = part_spectra + p * nlam;

    /* Add this grid cell's contribution to the spectra */
    for (int jl = 0, J = (int)good_lams.size(); jl < J; jl++) {
      const int ilam = good_lams[jl];
      const OutT spec_val = static_cast<OutT>(cell_spectra[ilam]);

      /* Assign to this particle's spectra array. */
      part_spec[ilam] = spec_val * weight;
    }
  }
}

/**
 * @brief This calculates particle spectra using a nearest grid point
 *        approach.
 *
 * This is the serial version of the function for grids without a wavelength
 * mask.
 *
 * @tparam Real The floating-point type of the input data.
 * @tparam OutT The floating-point type stored in the output buffer.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param part_spectra: The per-particle output array.
 */
template <typename Real, typename OutT>
static void spectra_loop_ngp_no_lam_mask_serial(GridProps *grid_props,
                                                Particles *parts,
                                                OutT *part_spectra) {
  /* Unpack the grid properties. */
  const std::array<int, MAX_GRID_NDIM> dims = grid_props->dims;
  size_t nlam = static_cast<size_t>(grid_props->nlam);
  const Real *__restrict grid_spectra = grid_props->get_spectra<Real>();

  /* Get and cast the number of particles. */
  size_t npart = static_cast<size_t>(parts->npart);

  /* Loop over particles. */
  for (size_t p = 0; p < npart; p++) {

    /* Skip masked particles. */
    if (parts->part_is_masked(p)) {
      continue;
    }

    /* Get the weight's index. */
    std::array<int, MAX_GRID_NDIM> part_indices;
    get_part_inds_ngp<Real>(part_indices, grid_props, parts, p);
    const int grid_ind = get_flat_index(part_indices, dims.data(), grid_props->ndim);

    /* Get the weight of this particle. */
    const OutT weight = static_cast<OutT>(parts->get_weight_at<Real>(p));
    const Real *__restrict cell_spectra =
        grid_spectra + static_cast<size_t>(grid_ind) * nlam;
    OutT *__restrict part_spec = part_spectra + p * nlam;

    /* Add this grid cell's contribution to the spectra */
    for (size_t ilam = 0; ilam < nlam; ilam++) {
      const OutT spec_val = static_cast<OutT>(cell_spectra[ilam]);

      /* Assign to this particle's spectra array. */
      part_spec[ilam] = spec_val * weight;
    }
  }
}

/**
 * @brief This calculates particle spectra using a nearest grid point
 *        approach.
 *
 * This is the serial wrapper which dispatches to the masked or unmasked
 * wavelength implementation.
 *
 * @tparam Real The floating-point type of the input data.
 * @tparam OutT The floating-point type stored in the output buffer.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param part_spectra: The per-particle output array.
 * @param has_lam_mask: Are we applying a wavelength mask?
 */
template <typename Real, typename OutT>
static void spectra_loop_ngp_serial(GridProps *grid_props, Particles *parts,
                                    OutT *part_spectra, bool has_lam_mask) {
  /* If there is no wavelength mask, use the branch-free contiguous loop. */
  if (!has_lam_mask) {
    spectra_loop_ngp_no_lam_mask_serial<Real, OutT>(grid_props, parts,
                                                    part_spectra);
    return;
  }

  /* Precompute unmasked wavelengths. */
  const size_t nlam = static_cast<size_t>(grid_props->nlam);
  std::vector<int> good_lams;
  good_lams.reserve(nlam);
  for (size_t ilam = 0; ilam < nlam; ilam++) {
    if (!grid_props->lam_is_masked(ilam)) {
      good_lams.push_back(ilam);
    }
  }

  spectra_loop_ngp_with_lam_mask_serial<Real, OutT>(grid_props, parts,
                                                    part_spectra, good_lams);
}

/**
 * @brief This calculates particle spectra using a nearest grid point approach.
 *
 * This is the parallel version of the function for grids with a wavelength
 * mask.
 *
 * @tparam Real The floating-point type of the input data.
 * @tparam OutT The floating-point type stored in the output buffer.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param part_spectra: The per-particle output array.
 * @param nthreads: The number of threads to use.
 * @param good_lams: The wavelength indices that are not masked.
 */
#ifdef WITH_OPENMP
template <typename Real, typename OutT>
static void spectra_loop_ngp_with_lam_mask_omp(
    GridProps *grid_props, Particles *parts, OutT *part_spectra,
    int nthreads, const std::vector<int> &good_lams) {
  /* Unpack the grid properties. */
  const std::array<int, MAX_GRID_NDIM> dims = grid_props->dims;
  size_t nlam = static_cast<size_t>(grid_props->nlam);
  const Real *__restrict grid_spectra = grid_props->get_spectra<Real>();

  /* Get and cast the number of particles. */
  size_t npart = static_cast<size_t>(parts->npart);

#pragma omp parallel num_threads(nthreads)
  {
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
    OutT *__restrict local_part_spectra = part_spectra + start_idx * nlam;

    /* Get an array that we'll put each particle's spectra into. */
    std::vector<OutT> this_part_spectra(nlam, static_cast<OutT>(0));

    /* Loop over particles. */
    for (size_t p = start_idx; p < end_idx; p++) {

      /* Skip masked particles. */
      if (parts->part_is_masked(p)) {
        continue;
      }

      /* Get the particle's grid index. */
      std::array<int, MAX_GRID_NDIM> part_indices;
      get_part_inds_ngp<Real>(part_indices, grid_props, parts, p);
      const int grid_ind =
          get_flat_index(part_indices, dims.data(), grid_props->ndim);

      /* Get the weight of this particle. */
      const OutT weight = static_cast<OutT>(parts->get_weight_at<Real>(p));
      const Real *__restrict cell_spectra =
          grid_spectra + static_cast<size_t>(grid_ind) * nlam;

      /* Add this grid cell's contribution to the spectra */
      for (int jl = 0, J = (int)good_lams.size(); jl < J; jl++) {

        /* Get the wavelength index. */
        const int ilam = good_lams[jl];

        /* Get the spectra value at this index and wavelength. */
        const OutT spec_val = static_cast<OutT>(cell_spectra[ilam]);

        /* Assign to this particle's spectra array. */
        this_part_spectra[ilam] = spec_val * weight;
      }

      /* Copy the entire spectrum at once into the output array. */
      memcpy(local_part_spectra + (p - start_idx) * nlam,
             this_part_spectra.data(), nlam * sizeof(OutT));

      /* No reset needed as we overwrite the whole array each time and the
       * wavelength mask never changes. */
    }
  }
}

/**
 * @brief This calculates particle spectra using a nearest grid point approach.
 *
 * This is the parallel version of the function for grids without a wavelength
 * mask.
 *
 * @tparam Real The floating-point type of the input data.
 * @tparam OutT The floating-point type stored in the output buffer.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param part_spectra: The per-particle output array.
 * @param nthreads: The number of threads to use.
 */
template <typename Real, typename OutT>
static void spectra_loop_ngp_no_lam_mask_omp(GridProps *grid_props,
                                             Particles *parts,
                                             OutT *part_spectra,
                                             int nthreads) {
  /* Unpack the grid properties. */
  const std::array<int, MAX_GRID_NDIM> dims = grid_props->dims;
  size_t nlam = static_cast<size_t>(grid_props->nlam);
  const Real *__restrict grid_spectra = grid_props->get_spectra<Real>();

  /* Get and cast the number of particles. */
  size_t npart = static_cast<size_t>(parts->npart);

#pragma omp parallel num_threads(nthreads)
  {
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
    OutT *__restrict local_part_spectra = part_spectra + start_idx * nlam;

    /* Get an array that we'll put each particle's spectra into. */
    std::vector<OutT> this_part_spectra(nlam, static_cast<OutT>(0));

    /* Loop over particles. */
    for (size_t p = start_idx; p < end_idx; p++) {

      /* Skip masked particles. */
      if (parts->part_is_masked(p)) {
        continue;
      }

      /* Get the particle's grid index. */
      std::array<int, MAX_GRID_NDIM> part_indices;
      get_part_inds_ngp<Real>(part_indices, grid_props, parts, p);
      const int grid_ind =
          get_flat_index(part_indices, dims.data(), grid_props->ndim);

      /* Get the weight of this particle. */
      const OutT weight = static_cast<OutT>(parts->get_weight_at<Real>(p));
      const Real *__restrict cell_spectra =
          grid_spectra + static_cast<size_t>(grid_ind) * nlam;

      /* Add this grid cell's contribution to the spectra */
      for (size_t ilam = 0; ilam < nlam; ilam++) {
        const OutT spec_val = static_cast<OutT>(cell_spectra[ilam]);

        /* Assign to this particle's spectra array. */
        this_part_spectra[ilam] = spec_val * weight;
      }

      /* Copy the entire spectrum at once into the output array. */
      memcpy(local_part_spectra + (p - start_idx) * nlam,
             this_part_spectra.data(), nlam * sizeof(OutT));

      /* No reset needed as we overwrite the whole array each time and the
       * wavelength mask never changes. */
    }
  }
}

/**
 * @brief This calculates particle spectra using a nearest grid point approach.
 *
 * This is the parallel wrapper which dispatches to the masked or unmasked
 * wavelength implementation.
 *
 * @tparam Real The floating-point type of the input data.
 * @tparam OutT The floating-point type stored in the output buffer.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param part_spectra: The per-particle output array.
 * @param nthreads: The number of threads to use.
 * @param has_lam_mask: Are we applying a wavelength mask?
 */
template <typename Real, typename OutT>
static void spectra_loop_ngp_omp(GridProps *grid_props, Particles *parts,
                                 OutT *part_spectra, int nthreads,
                                 bool has_lam_mask) {
  /* If there is no wavelength mask, use the branch-free contiguous loop. */
  if (!has_lam_mask) {
    spectra_loop_ngp_no_lam_mask_omp<Real, OutT>(grid_props, parts,
                                                 part_spectra, nthreads);
    return;
  }

  /* Precompute unmasked wavelengths. */
  const size_t nlam = static_cast<size_t>(grid_props->nlam);
  std::vector<int> good_lams;
  good_lams.reserve(nlam);
  for (size_t ilam = 0; ilam < nlam; ilam++) {
    if (!grid_props->lam_is_masked(ilam)) {
      good_lams.push_back(ilam);
    }
  }

  spectra_loop_ngp_with_lam_mask_omp<Real, OutT>(
      grid_props, parts, part_spectra, nthreads, good_lams);
}
#endif

/**
 * @brief This calculates particle spectra using a nearest grid point
 * approach.
 *
 * This is a wrapper which calls the correct function based on the number of
 * threads requested and whether OpenMP is available.
 *
 * @tparam Real The floating-point type of the input data.
 * @tparam OutT The floating-point type stored in the output buffer.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param part_spectra: The per-particle output array.
 * @param nthreads: The number of threads to use.
 * @param has_lam_mask: Are we applying a wavelength mask?
 */
template <typename Real, typename OutT>
void spectra_loop_ngp(GridProps *grid_props, Particles *parts,
                      OutT *part_spectra, const int nthreads,
                      bool has_lam_mask) {

  tic("spectra_loop_ngp");

  /* Call the correct function for the configuration/number of threads. */

#ifdef WITH_OPENMP

  /* If we have multiple threads and OpenMP we can parallelise. */
  if (nthreads > 1) {
    spectra_loop_ngp_omp<Real, OutT>(grid_props, parts, part_spectra,
                                     nthreads, has_lam_mask);
  }
  /* Otherwise there's no point paying the OpenMP overhead. */
  else {
    spectra_loop_ngp_serial<Real, OutT>(grid_props, parts, part_spectra,
                                        has_lam_mask);
  }

#else

  (void)nthreads;

  /* We don't have OpenMP, just call the serial version. */
  spectra_loop_ngp_serial<Real, OutT>(grid_props, parts, part_spectra,
                                      has_lam_mask);

#endif
  toc("spectra_loop_ngp");
}

/**
 * @brief Computes per-particle spectra for a collection of particles.
 *
 * @param np_grid_spectra: The SPS spectra array.
 * @param grid_tuple: The tuple containing arrays of grid axis properties.
 * @param part_tuple: The tuple of particle property arrays (in the same
 * order as grid_tuple).
 * @param np_part_mass: The particle mass array.
 * @param np_ndims: The size of each grid axis.
 * @param ndim: The number of grid axes.
 * @param npart: The number of particles.
 * @param nlam: The number of wavelength elements.
 * @param out_dtype: Requested floating-point dtype for the returned
 *                   per-particle spectra array.
 *
 * @return The per-particle spectra array.
 */
PyObject *compute_particle_seds(PyObject *self, PyObject *args) {
  tic("compute_particle_seds");

  /* We don't need the self argument but it has to be there. Tell the
   * compiler we don't care. */
  (void)self;

  int ndim, npart, nlam, nthreads;
  int has_lam_mask;
  PyObject *grid_tuple, *part_tuple;
  PyObject *out_dtype;
  PyObject *prop_names = NULL;
  PyArrayObject *np_grid_spectra;
  PyArrayObject *np_part_mass, *np_ndims;
  PyArrayObject *np_mask, *np_lam_mask;
  char *method;

  if (!PyArg_ParseTuple(args, "OOOOOiiisiOOpO|O", &np_grid_spectra,
                        &grid_tuple, &part_tuple, &np_part_mass, &np_ndims,
                        &ndim, &npart, &nlam, &method, &nthreads, &np_mask,
                        &np_lam_mask, &has_lam_mask, &out_dtype,
                        &prop_names)) {
    return NULL;
  }

  /* Extract the grid struct. */
  GridProps *grid_props = new GridProps(np_grid_spectra, grid_tuple,
                                        /*np_lam*/ nullptr, np_lam_mask, nlam,
                                        /*np_grid_weights*/ nullptr,
                                        prop_names);
  RETURN_IF_PYERR();

  /* Create the object that holds the particle properties. */
  Particles *part_props = new Particles(np_part_mass, /*np_velocities*/ NULL,
                                        np_mask, part_tuple, prop_names, npart);
  RETURN_IF_PYERR();

  const int grid_typenum = grid_props->get_float_typenum();
  const int part_typenum = part_props->get_float_typenum();
  if (grid_typenum != -1 && part_typenum != -1 && grid_typenum != part_typenum) {
    PyErr_SetString(PyExc_TypeError,
                    "Grid and particle arrays must share the same floating-point dtype.");
    delete part_props;
    delete grid_props;
    return NULL;
  }

  const int input_typenum = grid_typenum != -1 ? grid_typenum : part_typenum;
  const int output_typenum = resolve_output_typenum(out_dtype, "out_dtype");
  if (output_typenum < 0) {
    delete part_props;
    delete grid_props;
    return NULL;
  }

  tic("compute_particle_seds.setup_output_arrays");

  /* Define the output dimensions. */
  npy_intp np_part_dims[2] = {npart, nlam};

  /* Allocate the particle spectra in the requested output precision. */
  PyArrayObject *np_part_spectra = NULL;
  {
    int dispatch_key = (output_typenum == NPY_FLOAT64);

    /* Dispatch: call the matching typed kernel based on the dispatch key. */
    switch (dispatch_key) {
    case 0:
      np_part_spectra = (PyArrayObject *)PyArray_ZEROS(2, np_part_dims,
                                                        NPY_FLOAT32, 0);
      break;
    default:
      np_part_spectra = (PyArrayObject *)PyArray_ZEROS(2, np_part_dims,
                                                        NPY_FLOAT64, 0);
      break;
    }
  }
  if (np_part_spectra == NULL) {
    delete part_props;
    delete grid_props;
    return NULL;
  }

  toc("compute_particle_seds.setup_output_arrays");

  /* With everything set up we can compute the spectra for each particle
   * using the requested method. */
  if (strcmp(method, "cic") == 0) {
    {
      int dispatch_key = ((input_typenum == NPY_FLOAT64) << 1) |
                         (output_typenum == NPY_FLOAT64);

      /* Dispatch: call the matching typed kernel based on the dispatch key. */
      switch (dispatch_key) {
      case 0:
        spectra_loop_cic<float, float>(
            grid_props, part_props,
            static_cast<float *>(PyArray_DATA(np_part_spectra)), nthreads,
            has_lam_mask);
        break;
      case 1:
        spectra_loop_cic<float, double>(
            grid_props, part_props,
            static_cast<double *>(PyArray_DATA(np_part_spectra)), nthreads,
            has_lam_mask);
        break;
      case 2:
        spectra_loop_cic<double, float>(
            grid_props, part_props,
            static_cast<float *>(PyArray_DATA(np_part_spectra)), nthreads,
            has_lam_mask);
        break;
      default:
        spectra_loop_cic<double, double>(
            grid_props, part_props,
            static_cast<double *>(PyArray_DATA(np_part_spectra)), nthreads,
            has_lam_mask);
        break;
      }
    }
  } else if (strcmp(method, "ngp") == 0) {
    {
      int dispatch_key = ((input_typenum == NPY_FLOAT64) << 1) |
                         (output_typenum == NPY_FLOAT64);

      /* Dispatch: call the matching typed kernel based on the dispatch key. */
      switch (dispatch_key) {
      case 0:
        spectra_loop_ngp<float, float>(
            grid_props, part_props,
            static_cast<float *>(PyArray_DATA(np_part_spectra)), nthreads,
            has_lam_mask);
        break;
      case 1:
        spectra_loop_ngp<float, double>(
            grid_props, part_props,
            static_cast<double *>(PyArray_DATA(np_part_spectra)), nthreads,
            has_lam_mask);
        break;
      case 2:
        spectra_loop_ngp<double, float>(
            grid_props, part_props,
            static_cast<float *>(PyArray_DATA(np_part_spectra)), nthreads,
            has_lam_mask);
        break;
      default:
        spectra_loop_ngp<double, double>(
            grid_props, part_props,
            static_cast<double *>(PyArray_DATA(np_part_spectra)), nthreads,
            has_lam_mask);
        break;
      }
    }
  } else {
    PyErr_Format(PyExc_ValueError, "Unknown grid assignment method (%s).",
                 method);
    Py_DECREF(np_part_spectra);
    delete part_props;
    delete grid_props;
    return NULL;
  }
  RETURN_IF_PYERR();

  /* Clean up memory! */
  delete part_props;
  delete grid_props;

  toc("compute_particle_seds");

  return Py_BuildValue("N", np_part_spectra);
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
