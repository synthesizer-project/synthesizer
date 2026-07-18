/******************************************************************************
 * C extension to calculate SEDs for star particles.
 * Calculates weights on an arbitrary dimensional grid given the mass.
 *****************************************************************************/
/* C includes */
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
#include "python_to_cpp.h"
#include "reductions.h"
#include "timers.h"
#ifdef ATOMIC_TIMING
#include "timers_init.h"
#endif
#include "weights.h"

/**
 * @brief Find nearest wavelength bin for a given lambda, in a given wavelength
 * array. Used by the spectra loop functions when considering doppler shift
 *
 * Note: binary search returns the index of the upper bin of those that
 * straddle the given lambda.
 *
 * @tparam SpecReal The spectral floating-point type.
 */
template <typename SpecReal>
int get_upper_lam_bin(SpecReal lambda, const SpecReal *grid_wavelengths,
                      int nlam) {
  return binary_search(0, nlam - 1, grid_wavelengths, lambda);
}

/**
 * @brief This calculates particle spectra using a cloud in cell approach.
 * This is the parallel version of the function that accounts for Doppler
 * shift.
 *
 * Each thread allocates its shift‐mapping buffers once and reuses them for
 * every particle, and all sub‐cell index math is hoisted out of the particle
 * loop.
 *
 * Note: Cell contributions are collected first then fused per wavelength,
 * reducing the outer loop over cells from ncells × nlam scatter-writes to a
 * single accumulation per wavelength, matching the pattern used in the
 * non-shifted CIC path.
 *
 * @tparam PartReal The particle floating-point type.
 * @tparam SpecReal The spectral floating-point type.
 * @tparam OutT The floating-point type stored in the output.
 *
 * @param grid_props: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 * @param part_spectra: The per-particle output array.
 * @param nthreads: The number of threads to use.
 * @param c: speed of light.
 */
template <typename PartReal, typename SpecReal, typename OutT>
static void compute_doppler_particle_seds_impl(GridProps *grid_props,
                                               Particles *part_props,
                                               OutT *part_spectra,
                                               int nthreads, PartReal c,
                                               const char *method) {
  const int ndim = grid_props->ndim;
  const std::array<int, MAX_GRID_NDIM> dims = grid_props->dims;
  const size_t nlam = static_cast<size_t>(grid_props->nlam);
  const SpecReal *wavelength = grid_props->get_lam<SpecReal>();
  const SpecReal *__restrict grid_spectra =
      grid_props->get_spectra<SpecReal>();

  if (strcmp(method, "cic") == 0) {
    const int ncells = 1 << ndim;
    std::array<int, MAX_GRID_NDIM> sub_dims;
    for (int d = 0; d < ndim; ++d) {
      sub_dims[d] = 2;
    }

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

#ifdef WITH_OPENMP
#pragma omp parallel num_threads(nthreads > 1 ? nthreads : 1) if (nthreads > 1)
#endif
    {
      std::vector<SpecReal> shifted_wavelengths(nlam);
      std::vector<int> mapped_indices(nlam);
      std::vector<const SpecReal *> cell_spectra_ptrs(ncells);
      std::vector<OutT> cell_weights(ncells);

#ifdef WITH_OPENMP
      const int thread_count = nthreads > 1 ? nthreads : 1;
      const size_t nparts_per_thread =
          (static_cast<size_t>(part_props->npart) + thread_count - 1) /
          thread_count;
      const int tid = omp_get_thread_num();
      const size_t start_idx = tid * nparts_per_thread;
      const size_t end_idx = std::min(static_cast<size_t>(part_props->npart),
                                      start_idx + nparts_per_thread);
#else
      const size_t start_idx = 0;
      const size_t end_idx = static_cast<size_t>(part_props->npart);
#endif

      for (size_t p = start_idx; p < end_idx; ++p) {
        if (part_props->part_is_masked(p)) {
          continue;
        }

        const PartReal vel = part_props->get_vel_at<PartReal>(p);
        const PartReal shift_factor = static_cast<PartReal>(1) + vel / c;
        for (size_t il = 0; il < nlam; ++il) {
          const SpecReal lam_s =
              wavelength[il] * static_cast<SpecReal>(shift_factor);
          shifted_wavelengths[il] = lam_s;
          mapped_indices[il] = get_upper_lam_bin(lam_s, wavelength, nlam);
        }

        const PartReal w_p = part_props->get_weight_at<PartReal>(p);
        std::array<int, MAX_GRID_NDIM> part_indices;
        std::array<SpecReal, MAX_GRID_NDIM> axis_fracs;
        get_part_ind_frac_cic<PartReal, SpecReal>(part_indices, axis_fracs,
                                                  grid_props, part_props, p);
        const int base_lin = get_flat_index(part_indices, dims.data(), ndim);

        int nvalid_cells = 0;
        for (int ic = 0; ic < ncells; ++ic) {
          const auto &sc = subcells[ic];
          SpecReal frac = static_cast<SpecReal>(1);
          for (int d = 0; d < ndim; ++d) {
            frac *= sc.offs[d] ? axis_fracs[d]
                               : (static_cast<SpecReal>(1) - axis_fracs[d]);
          }
          if (frac == 0.0) {
            continue;
          }

          const int grid_i = base_lin + sc.linoff;
          cell_spectra_ptrs[nvalid_cells] =
              grid_spectra + static_cast<size_t>(grid_i) * nlam;
          cell_weights[nvalid_cells] =
              static_cast<OutT>(frac) * static_cast<OutT>(w_p);
          nvalid_cells++;
        }

        OutT *__restrict p_spec = part_spectra + p * nlam;
        for (size_t il = 0; il < nlam; ++il) {
          OutT total = static_cast<OutT>(0);
          for (int icell = 0; icell < nvalid_cells; ++icell) {
            total = std::fma(static_cast<OutT>(cell_spectra_ptrs[icell][il]),
                             cell_weights[icell], total);
          }

          const int ils = mapped_indices[il];
          if (ils <= 0 || static_cast<size_t>(ils) >= nlam ||
              grid_props->lam_is_masked(ils)) {
            continue;
          }

          const SpecReal lam_s = shifted_wavelengths[il];
          const OutT frac_s =
              static_cast<OutT>((lam_s - wavelength[ils - 1]) /
                                (wavelength[ils] - wavelength[ils - 1]));
          p_spec[ils - 1] += (static_cast<OutT>(1) - frac_s) * total;
          p_spec[ils] += frac_s * total;
        }
      }
    }
    return;
  }

  if (strcmp(method, "ngp") == 0) {
#ifdef WITH_OPENMP
#pragma omp parallel num_threads(nthreads > 1 ? nthreads : 1) if (nthreads > 1)
#endif
    {
      std::vector<SpecReal> shifted_wavelengths(nlam);
      std::vector<int> mapped_indices(nlam);

#ifdef WITH_OPENMP
      const int thread_count = nthreads > 1 ? nthreads : 1;
      const size_t nparts_per_thread =
          (static_cast<size_t>(part_props->npart) + thread_count - 1) /
          thread_count;
      const int tid = omp_get_thread_num();
      const size_t start_idx = tid * nparts_per_thread;
      const size_t end_idx = std::min(static_cast<size_t>(part_props->npart),
                                      start_idx + nparts_per_thread);
#else
      const size_t start_idx = 0;
      const size_t end_idx = static_cast<size_t>(part_props->npart);
#endif

      for (size_t p = start_idx; p < end_idx; ++p) {
        if (part_props->part_is_masked(p)) {
          continue;
        }

        const PartReal vel = part_props->get_vel_at<PartReal>(p);
        const PartReal shift_factor = static_cast<PartReal>(1) + vel / c;
        for (size_t il = 0; il < nlam; ++il) {
          const SpecReal lam_s =
              wavelength[il] * static_cast<SpecReal>(shift_factor);
          shifted_wavelengths[il] = lam_s;
          mapped_indices[il] = get_upper_lam_bin(lam_s, wavelength, nlam);
        }

        const OutT weight =
            static_cast<OutT>(part_props->get_weight_at<PartReal>(p));
        std::array<int, MAX_GRID_NDIM> part_indices;
        get_part_inds_ngp<PartReal, SpecReal>(part_indices, grid_props,
                                              part_props, p);
        const int grid_ind = get_flat_index(part_indices, dims.data(), ndim);
        const SpecReal *__restrict cell_spectra =
            grid_spectra + static_cast<size_t>(grid_ind) * nlam;
        OutT *__restrict p_spec = part_spectra + p * nlam;

        for (size_t ilam = 0; ilam < nlam; ++ilam) {
          const int ilam_shifted = mapped_indices[ilam];
          if (grid_props->lam_is_masked(ilam_shifted)) {
            continue;
          }

          if (!(ilam_shifted > 0 &&
                static_cast<size_t>(ilam_shifted) <= nlam - 1)) {
            continue;
          }

          const SpecReal shifted_lambda = shifted_wavelengths[ilam];
          const OutT frac_shifted = static_cast<OutT>(
              (shifted_lambda - wavelength[ilam_shifted - 1]) /
              (wavelength[ilam_shifted] - wavelength[ilam_shifted - 1]));
          const OutT grid_spectra_value =
              static_cast<OutT>(cell_spectra[ilam]) * weight;

          p_spec[ilam_shifted - 1] +=
              (static_cast<OutT>(1) - frac_shifted) * grid_spectra_value;
          p_spec[ilam_shifted] += frac_shifted * grid_spectra_value;
        }
      }
    }
    return;
  }

  PyErr_Format(PyExc_ValueError, "Unknown grid assignment method (%s).",
               method);
}

/**
 * @brief Computes Doppler-shifted per-particle and integrated spectra.
 *
 * @param np_grid_spectra: The SPS spectra array.
 * @param np_lam: The grid wavelength array.
 * @param grid_tuple: The tuple containing arrays of grid axis properties.
 * @param part_tuple: The tuple of particle property arrays (in the same order
 *                    as grid_tuple).
 * @param np_part_mass: The particle mass array.
 * @param np_velocities: The particle velocity array.
 * @param np_ndims: The size of each grid axis.
 * @param ndim: The number of grid axes.
 * @param npart: The number of particles.
 * @param nlam: The number of wavelength elements.
 * @param method: The grid assignment method.
 * @param nthreads: The number of threads to use.
 * @param py_c: The speed of light in the same units as velocities.
 * @param np_mask: Optional particle mask.
 * @param np_lam_mask: Optional wavelength mask.
 * @param out_dtype: Requested floating-point dtype for the returned spectra.
 *
 * @return A tuple containing the per-particle spectra and integrated spectra.
 */

PyObject *compute_part_seds_with_vel_shift(PyObject *self, PyObject *args) {

  tic("compute_part_seds_with_vel_shift");
  tic("compute_part_seds_with_vel_shift.extract_python_data");

  /* We don't need the self argument but it has to be there. Tell the
   * compiler we don't care. */
  (void)self;

  int ndim, npart, nlam, nthreads;
  PyObject *grid_tuple, *part_tuple;
  PyObject *prop_names = NULL;
  PyObject *py_c, *out_dtype;
  PyArrayObject *np_grid_spectra, *np_lam;
  PyArrayObject *np_velocities;
  PyArrayObject *np_part_mass, *np_ndims;
  PyArrayObject *np_mask, *np_lam_mask;
  char *method;

  if (!PyArg_ParseTuple(args, "OOOOOOOiiisiOOOO|O", &np_grid_spectra, &np_lam,
                        &grid_tuple, &part_tuple, &np_part_mass,
                        &np_velocities, &np_ndims, &ndim, &npart, &nlam,
                        &method, &nthreads, &py_c, &np_mask, &np_lam_mask,
                        &out_dtype, &prop_names)) {
    return NULL;
  }

  /* Extract the grid struct. */
  GridProps *grid_props =
      new GridProps(np_grid_spectra, grid_tuple, np_lam, np_lam_mask, nlam,
                    /*np_grid_weights*/ NULL, prop_names);
  RETURN_IF_PYERR();

  /* Create the object that holds the particle properties. */
  Particles *part_props = new Particles(np_part_mass, np_velocities, np_mask,
                                        part_tuple, prop_names, npart);
  RETURN_IF_PYERR();

  const int grid_typenum = grid_props->get_float_typenum();
  const int part_typenum = part_props->get_float_typenum();
  const int output_typenum = resolve_output_typenum(out_dtype, "out_dtype");
  if (output_typenum < 0) {
    delete part_props;
    delete grid_props;
    return NULL;
  }

  const double c = PyFloat_AsDouble(py_c);

  toc("compute_part_seds_with_vel_shift.extract_python_data");

  /* Allocate the spectra, compute the per-particle doppler shifted spectra,
   * reduce to the integrated spectra, and wrap both in NumPy arrays, all at
   * the dtypes resolved above. */
  npy_intp np_dims[2] = {npart, nlam};
  npy_intp np_dims_int[1] = {nlam};
  PyObject *out_tuple =
      dispatch_float(part_typenum, [&](auto p) -> PyObject * {
        return dispatch_float(grid_typenum, [&](auto g) -> PyObject * {
          return dispatch_float(output_typenum, [&](auto o) -> PyObject * {
            using PartReal = decltype(p);
            using SpecReal = decltype(g);
            using OutT = decltype(o);

            OutT *spectra = new (std::nothrow)
                OutT[static_cast<size_t>(grid_props->nlam)]();
            OutT *part_spectra = new (std::nothrow)
                OutT[static_cast<size_t>(npart) * grid_props->nlam]();
            if (spectra == NULL || part_spectra == NULL) {
              PyErr_SetString(PyExc_MemoryError,
                              "Failed to allocate memory for spectra.");
              delete[] spectra;
              delete[] part_spectra;
              return NULL;
            }

            compute_doppler_particle_seds_impl<PartReal, SpecReal, OutT>(
                grid_props, part_props, part_spectra, nthreads,
                static_cast<PartReal>(c), method);
            if (PyErr_Occurred()) {
              delete[] spectra;
              delete[] part_spectra;
              return NULL;
            }
            reduce_spectra<OutT>(spectra, part_spectra, nlam, npart, nthreads);

            /* Wrapping hands buffer ownership to the NumPy arrays. */
            PyArrayObject *out_part_spectra =
                wrap_array_to_numpy<OutT>(2, np_dims, part_spectra);
            PyArrayObject *out_integrated_spectra =
                wrap_array_to_numpy<OutT>(1, np_dims_int, spectra);
            return Py_BuildValue("NN", out_part_spectra,
                                 out_integrated_spectra);
          });
        });
      });

  /* Clean up memory! */
  delete part_props;
  delete grid_props;

  toc("compute_part_seds_with_vel_shift");

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
