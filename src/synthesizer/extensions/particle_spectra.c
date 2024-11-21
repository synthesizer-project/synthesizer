/******************************************************************************
 * C extension to calculate SEDs for star particles.
 * Calculates weights on an arbitrary dimensional grid given the mass.
 *****************************************************************************/
/* C includes */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Python includes */
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>

/* Local includes */
#include "macros.h"
#include "property_funcs.h"
#include "timers.h"
#include "weights.h"


/** 
    * @brief Find nearest wavelength bin for a given lambda, in a given wavelength array.
    * Used by the spectra loop functions when considering doppler shift
*/
int find_nearest_bin(double lambda, double *grid_wavelengths, int nlam) {
    int nearest_index = 0;
    for (int ilam = 1; ilam < nlam; ilam++) {
        if (fabs(grid_wavelengths[ilam] - lambda) < fabs(grid_wavelengths[nearest_index] - lambda)) {
            nearest_index = ilam;
        }
    }
    return nearest_index;
}

/**
 * @brief This calculates particle spectra using a cloud in cell approach.
 * This is the version of the function that accounts for doppler shift.
 * This is the serial version of the function.
 *
 * @param grid: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 * @param c: speed of light .
 */
static void shifted_spectra_loop_cic_serial(struct grid *grid, struct particles *parts,
                                    double *spectra, const double c) {

  /* Unpack the grid properties. */
  int *dims = grid->dims;
  int ndim = grid->ndim;
  int nlam = grid->nlam;
  double *wavelength = grid->lam;
  double **grid_props = grid->props;
  double *grid_spectra = grid->spectra;


  /* Unpack the particles properties. */
  double *part_masses = parts->mass;
  double **part_props = parts->props;
  double *fesc = parts->fesc;
  double *velocity = parts->velocities;
  int npart = parts->npart;

  /*Array to hold shifted wavelengths */
  double shifted_wavelengths[nlam];

  /* Loop over particles. */
  for (int p = 0; p < npart; p++) {

    /* Get this particle's mass. */
    const double mass = part_masses[p];

    /* Get the particle velocity and shift wl array */
    double vel = velocity[p];
    double shift_factor = 1.0 + vel / c;
    for(int ilam=0; ilam<nlam; ilam++){
      shifted_wavelengths[ilam] = wavelength[ilam] * shift_factor;
    }
    

    /* Setup the index and mass fraction arrays. */
    int part_indices[ndim];
    double axis_fracs[ndim];

    /* Get the grid indices and cell fractions for the particle. */
    get_part_ind_frac_cic(part_indices, axis_fracs, dims, ndim, grid_props,
                          part_props, p);

    /* To combine fractions we will need an array of dimensions for the
     * subset. These are always two in size, one for the low and one for high
     * grid point. */
    int sub_dims[ndim];
    for (int idim = 0; idim < ndim; idim++) {
      sub_dims[idim] = 2;
    }

    /* Now loop over this collection of cells collecting and setting their
     * weights. */
    for (int icell = 0; icell < (int)pow(2, (double)ndim); icell++) {

      /* Set up some index arrays we'll need. */
      int subset_ind[ndim];
      int frac_ind[ndim];

      /* Get the multi-dimensional version of icell. */
      get_indices_from_flat(icell, ndim, sub_dims, subset_ind);

      /* Multiply all contributing fractions and get the fractions index
       * in the grid. */
      double frac = 1;
      for (int idim = 0; idim < ndim; idim++) {
        if (subset_ind[idim] == 0) {
          frac *= (1 - axis_fracs[idim]);
          frac_ind[idim] = part_indices[idim] - 1;
        } else {
          frac *= axis_fracs[idim];
          frac_ind[idim] = part_indices[idim];
        }
      }

      /* Nothing to do if fraction is 0. */
      if (frac == 0) {
        continue;
      }

      /* Define the weight. */
      double weight = frac * mass * (1.0 - fesc[p]);

      /* Get the weight's index. */
      const int grid_ind = get_flat_index(frac_ind, dims, ndim);

      /* Get the spectra ind. */
      int unraveled_ind[ndim + 1];
      get_indices_from_flat(grid_ind, ndim, dims, unraveled_ind);
      unraveled_ind[ndim] = 0;
      int spectra_ind = get_flat_index(unraveled_ind, dims, ndim + 1);

      /* Add this grid cell's contribution to the spectra */
      for (int ilam = 0; ilam < nlam; ilam++) {
        double shifted_lambda = shifted_wavelengths[ilam];
      /*Find nearest wavelength bin*/
        int ilam_shifted = find_nearest_bin(shifted_lambda, wavelength, nlam); 
        if (ilam_shifted >= nlam - 1 || ilam_shifted < 0) {continue}; // jumps out of the loop if out of bounds
      // Interpolate contribution between ilam_shifted and ilam_shifted+1
        double frac_shifted = (shifted_lambda - wavelength[ilam_shifted]) /
                                  (wavelength[ilam_shifted + 1] - wavelength[ilam_shifted]);


        /* Add the contribution to this wavelength. */
        spectra[p * nlam + ilam_shifted] += grid_spectra[spectra_ind + ilam] * (1.0 - frac_shifted) * weight;
        spectra[p * nlam + ilam_shifted + 1] += grid_spectra[spectra_ind + ilam] * frac_shifted * weight;
      }
    }
  }
}

/**
 * @brief This calculates particle spectra using a cloud in cell approach.
 *
 * This is the serial version of the function.
 *
 * @param grid: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 */
static void spectra_loop_cic_serial(struct grid *grid, struct particles *parts,
                                    double *spectra) {

  /* Unpack the grid properties. */
  int *dims = grid->dims;
  int ndim = grid->ndim;
  int nlam = grid->nlam;
  double **grid_props = grid->props;
  double *grid_spectra = grid->spectra;

  /* Unpack the particles properties. */
  double *part_masses = parts->mass;
  double **part_props = parts->props;
  double *fesc = parts->fesc;
  int npart = parts->npart;

  /* Loop over particles. */
  for (int p = 0; p < npart; p++) {

    /* Get this particle's mass. */
    const double mass = part_masses[p];

    /* Setup the index and mass fraction arrays. */
    int part_indices[ndim];
    double axis_fracs[ndim];

    /* Get the grid indices and cell fractions for the particle. */
    get_part_ind_frac_cic(part_indices, axis_fracs, dims, ndim, grid_props,
                          part_props, p);

    /* To combine fractions we will need an array of dimensions for the
     * subset. These are always two in size, one for the low and one for high
     * grid point. */
    int sub_dims[ndim];
    for (int idim = 0; idim < ndim; idim++) {
      sub_dims[idim] = 2;
    }

    /* Now loop over this collection of cells collecting and setting their
     * weights. */
    for (int icell = 0; icell < (int)pow(2, (double)ndim); icell++) {

      /* Set up some index arrays we'll need. */
      int subset_ind[ndim];
      int frac_ind[ndim];

      /* Get the multi-dimensional version of icell. */
      get_indices_from_flat(icell, ndim, sub_dims, subset_ind);

      /* Multiply all contributing fractions and get the fractions index
       * in the grid. */
      double frac = 1;
      for (int idim = 0; idim < ndim; idim++) {
        if (subset_ind[idim] == 0) {
          frac *= (1 - axis_fracs[idim]);
          frac_ind[idim] = part_indices[idim] - 1;
        } else {
          frac *= axis_fracs[idim];
          frac_ind[idim] = part_indices[idim];
        }
      }

      /* Nothing to do if fraction is 0. */
      if (frac == 0) {
        continue;
      }

      /* Define the weight. */
      double weight = frac * mass * (1.0 - fesc[p]);

      /* Get the weight's index. */
      const int grid_ind = get_flat_index(frac_ind, dims, ndim);

      /* Get the spectra ind. */
      int unraveled_ind[ndim + 1];
      get_indices_from_flat(grid_ind, ndim, dims, unraveled_ind);
      unraveled_ind[ndim] = 0;
      int spectra_ind = get_flat_index(unraveled_ind, dims, ndim + 1);

      /* Add this grid cell's contribution to the spectra */
      for (int ilam = 0; ilam < nlam; ilam++) {

        /* Add the contribution to this wavelength. */
        spectra[p * nlam + ilam] += grid_spectra[spectra_ind + ilam] * weight;
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
 * @param grid: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 * @param nthreads: The number of threads to use.
 */
#ifdef WITH_OPENMP
static void spectra_loop_cic_omp(struct grid *grid, struct particles *parts,
                                 double *spectra, int nthreads) {

  /* How many particles should each thread get? */
  int npart_per_thread = (parts->npart + nthreads - 1) / nthreads;

#pragma omp parallel num_threads(nthreads)
  {

    /* Unpack the grid properties. */
    int *dims = grid->dims;
    int ndim = grid->ndim;
    int nlam = grid->nlam;
    double **grid_props = grid->props;
    double *grid_spectra = grid->spectra;

    /* Unpack the particles properties. */
    double *part_masses = parts->mass;
    double **part_props = parts->props;
    double *fesc = parts->fesc;
    int npart = parts->npart;

    /* Get the thread id. */
    int tid = omp_get_thread_num();

    /* Calculate start and end indices for each thread */
    int start = tid * npart_per_thread;
    int end = start + npart_per_thread;
    if (end >= npart) {
      end = npart;
    }
#ifdef WITH_DEBUGGING_CHECKS
    else {
#pragma omp critical
      PyErr_SetString(PyExc_RuntimeError,
                      "Not all particles distributed to threads.");
      free(spectra);
      spectra = NULL;
      return;
    }
#endif

    /* Loop over particles. */
    for (int p = start; p < end; p++) {

      /* Get this particle's mass. */
      const double mass = part_masses[p];

      /* Setup the index and mass fraction arrays. */
      int part_indices[ndim];
      double axis_fracs[ndim];

      /* Get the grid indices and cell fractions for the particle. */
      get_part_ind_frac_cic(part_indices, axis_fracs, dims, ndim, grid_props,
                            part_props, p);

      /* To combine fractions we will need an array of dimensions for the
       * subset. These are always two in size, one for the low and one for high
       * grid point. */
      int sub_dims[ndim];
      for (int idim = 0; idim < ndim; idim++) {
        sub_dims[idim] = 2;
      }

      /* Now loop over this collection of cells collecting and setting their
       * weights. */
      for (int icell = 0; icell < (int)pow(2, (double)ndim); icell++) {

        /* Set up some index arrays we'll need. */
        int subset_ind[ndim];
        int frac_ind[ndim];

        /* Get the multi-dimensional version of icell. */
        get_indices_from_flat(icell, ndim, sub_dims, subset_ind);

        /* Multiply all contributing fractions and get the fractions index
         * in the grid. */
        double frac = 1;
        for (int idim = 0; idim < ndim; idim++) {
          if (subset_ind[idim] == 0) {
            frac *= (1 - axis_fracs[idim]);
            frac_ind[idim] = part_indices[idim] - 1;
          } else {
            frac *= axis_fracs[idim];
            frac_ind[idim] = part_indices[idim];
          }
        }

        if (frac == 0) {
          continue;
        }

        /* Define the weight. */
        double weight = frac * mass * (1.0 - fesc[p]);

        /* Get the weight's index. */
        const int grid_ind = get_flat_index(frac_ind, dims, ndim);

        /* Get the spectra ind. */
        int unraveled_ind[ndim + 1];
        get_indices_from_flat(grid_ind, ndim, dims, unraveled_ind);
        unraveled_ind[ndim] = 0;
        int spectra_ind = get_flat_index(unraveled_ind, dims, ndim + 1);

        /* Add this grid cell's contribution to the spectra */
        for (int ilam = 0; ilam < nlam; ilam++) {

          /* Add the contribution to this wavelength. */
          spectra[p * nlam + ilam] += grid_spectra[spectra_ind + ilam] * weight;
        }
      }
    }
  }
}
#endif


/**
 * @brief This calculates the grid weights in each grid cell using a cloud
 *        in cell approach.
 * This is the version of the function that accounts for doppler shift.
 * This is the parallel version of the function.
 *
 * @param grid: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 * @param nthreads: The number of threads to use.
 * @param c: speed of light
 */
#ifdef WITH_OPENMP
static void shifted_spectra_loop_cic_omp(struct grid *grid, struct particles *parts,
                                 double *spectra, int nthreads, const double c) {

  /* How many particles should each thread get? */
  int npart_per_thread = (parts->npart + nthreads - 1) / nthreads;

#pragma omp parallel num_threads(nthreads)
  {

    /* Unpack the grid properties. */
    int *dims = grid->dims;
    int ndim = grid->ndim;
    int nlam = grid->nlam;
    double *wavelength = grid->lam;
    double **grid_props = grid->props;
    double *grid_spectra = grid->spectra;

    /* Unpack the particles properties. */
    double *part_masses = parts->mass;
    double **part_props = parts->props;
    double *fesc = parts->fesc;
    double *velocity = parts->velocities;
    int npart = parts->npart;

    double shifted_wavelengths[nlam];

    /* Get the thread id. */
    int tid = omp_get_thread_num();

    /* Calculate start and end indices for each thread */
    int start = tid * npart_per_thread;
    int end = start + npart_per_thread;
    if (end >= npart) {
      end = npart;
    }
#ifdef WITH_DEBUGGING_CHECKS
    else {
#pragma omp critical
      PyErr_SetString(PyExc_RuntimeError,
                      "Not all particles distributed to threads.");
      free(spectra);
      spectra = NULL;
      return;
    }
#endif

    /* Loop over particles. */
    for (int p = start; p < end; p++) {

      /* Get this particle's mass. velocity and doppler shift. */
      const double mass = part_masses[p];
      double vel = velocity[p];
      double shift_factor = 1.0 + vel / c;
      // Fill up the shifted lam array
      for(int ilam=0; ilam<nlam; ilam++){
      shifted_wavelengths[ilam] = wavelength[ilam] * shift_factor;
      }

      /* Setup the index and mass fraction arrays. */
      int part_indices[ndim];
      double axis_fracs[ndim];

      /* Get the grid indices and cell fractions for the particle. */
      get_part_ind_frac_cic(part_indices, axis_fracs, dims, ndim, grid_props,
                            part_props, p);

      /* To combine fractions we will need an array of dimensions for the
       * subset. These are always two in size, one for the low and one for high
       * grid point. */
      int sub_dims[ndim];
      for (int idim = 0; idim < ndim; idim++) {
        sub_dims[idim] = 2;
      }

      /* Now loop over this collection of cells collecting and setting their
       * weights. */
      for (int icell = 0; icell < (int)pow(2, (double)ndim); icell++) {

        /* Set up some index arrays we'll need. */
        int subset_ind[ndim];
        int frac_ind[ndim];

        /* Get the multi-dimensional version of icell. */
        get_indices_from_flat(icell, ndim, sub_dims, subset_ind);

        /* Multiply all contributing fractions and get the fractions index
         * in the grid. */
        double frac = 1;
        for (int idim = 0; idim < ndim; idim++) {
          if (subset_ind[idim] == 0) {
            frac *= (1 - axis_fracs[idim]);
            frac_ind[idim] = part_indices[idim] - 1;
          } else {
            frac *= axis_fracs[idim];
            frac_ind[idim] = part_indices[idim];
          }
        }

        if (frac == 0) {
          continue;
        }

        /* Define the weight. */
        double weight = frac * mass * (1.0 - fesc[p]);

        /* Get the weight's index. */
        const int grid_ind = get_flat_index(frac_ind, dims, ndim);

        /* Get the spectra ind. */
        int unraveled_ind[ndim + 1];
        get_indices_from_flat(grid_ind, ndim, dims, unraveled_ind);
        unraveled_ind[ndim] = 0;
        int spectra_ind = get_flat_index(unraveled_ind, dims, ndim + 1);

        /* Add this grid cell's contribution to the spectra */
        for (int ilam = 0; ilam < nlam; ilam++) {
          double shifted_lambda = shifted_wavelengths[ilam];
          int ilam_shifted = find_nearest_bin(shifted_lambda, wavelength, nlam); 
          if (ilam_shifted >= nlam - 1 || ilam_shifted < 0) {continue}; // jumps out of the loop if out of bounds
          // Interpolate contribution between ilam_shifted and ilam_shifted+1
          double frac_shifted = (shifted_lambda - wavelength[ilam_shifted]) / (wavelength[ilam_shifted + 1] - wavelength[ilam_shifted]);

          /* Add the contribution to this wavelength. */
          spectra[p * nlam + ilam_shifted] += grid_spectra[spectra_ind + ilam] * (1.0 - frac_shifted) * weight;
          spectra[p * nlam + ilam_shifted + 1] += grid_spectra[spectra_ind + ilam] * frac_shifted * weight;
        }
      }
    }
  }
}
#endif

/**
 * @brief This calculates particle spectra using a cloud in cell approach.
 *
 * This is a wrapper which calls the correct function based on the number of
 * threads requested and whether OpenMP is available.
 *
 * @param grid: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 * @param nthreads: The number of threads to use.
 */
void spectra_loop_cic(struct grid *grid, struct particles *parts,
                      double *spectra, const int nthreads) {

  double start_time = tic();

  /* Call the correct function for the configuration/number of threads. */

#ifdef WITH_OPENMP

  /* If we have multiple threads and OpenMP we can parallelise. */
  if (nthreads > 1) {
    spectra_loop_cic_omp(grid, parts, spectra, nthreads);
  }
  /* Otherwise there's no point paying the OpenMP overhead. */
  else {
    spectra_loop_cic_serial(grid, parts, spectra);
  }

#else

  (void)nthreads;

  /* We don't have OpenMP, just call the serial version. */
  spectra_loop_cic_serial(grid, parts, spectra);

#endif
  toc("Cloud in Cell particle spectra loop", start_time);
}

/**
 * @brief This calculates doppler-shifted particle spectra using a cloud in cell approach.
 *
 * This is a wrapper which calls the correct function based on the number of
 * threads requested and whether OpenMP is available.
 *
 * @param grid: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 * @param nthreads: The number of threads to use.
 */
void shifted_spectra_loop_cic(struct grid *grid, struct particles *parts,
                      double *spectra, const int nthreads) {

  double start_time = tic();

  /* Call the correct function for the configuration/number of threads. */

#ifdef WITH_OPENMP

  /* If we have multiple threads and OpenMP we can parallelise. */
  if (nthreads > 1) {
    shifted_spectra_loop_cic_omp(grid, parts, spectra, nthreads);
  }
  /* Otherwise there's no point paying the OpenMP overhead. */
  else {
    shifted_spectra_loop_cic_serial(grid, parts, spectra);
  }

#else

  (void)nthreads;

  /* We don't have OpenMP, just call the serial version. */
  shifted_spectra_loop_cic_serial(grid, parts, spectra);

#endif
  toc("Cloud in Cell particle spectra loop", start_time);
}

/**
 * @brief This calculates particle spectra using a nearest grid point approach.
 *
 * This is the serial version of the function.
 *
 * @param grid: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 */
static void spectra_loop_ngp_serial(struct grid *grid, struct particles *parts,
                                    double *spectra) {

  /* Unpack the grid properties. */
  int *dims = grid->dims;
  int ndim = grid->ndim;
  int nlam = grid->nlam;
  double **grid_props = grid->props;
  double *grid_spectra = grid->spectra;

  /* Unpack the particles properties. */
  double *part_masses = parts->mass;
  double **part_props = parts->props;
  double *fesc = parts->fesc;
  int npart = parts->npart;

  /* Loop over particles. */
  for (int p = 0; p < npart; p++) {

    /* Get this particle's mass. */
    const double mass = part_masses[p];

    /* Setup the index array. */
    int part_indices[ndim];

    /* Get the grid indices for the particle */
    get_part_inds_ngp(part_indices, dims, ndim, grid_props, part_props, p);

    /* Define the weight. */
    double weight = mass * (1.0 - fesc[p]);

    /* Get the weight's index. */
    const int grid_ind = get_flat_index(part_indices, dims, ndim);

    /* Get the spectra ind. */
    int unraveled_ind[ndim + 1];
    get_indices_from_flat(grid_ind, ndim, dims, unraveled_ind);
    unraveled_ind[ndim] = 0;
    int spectra_ind = get_flat_index(unraveled_ind, dims, ndim + 1);

    /* Add this grid cell's contribution to the spectra */
    for (int ilam = 0; ilam < nlam; ilam++) {

      /* Add the contribution to this wavelength. */
      spectra[p * nlam + ilam] += grid_spectra[spectra_ind + ilam] * weight;
    }
  }
}


/**
 * @brief This calculates particle spectra using a nearest grid point approach.
 * This is the version of the function that accounts doppler shift
 * This is the serial version of the function.
 *
 * @param grid: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 * @param c: speed of light.
 */
static void shifted_spectra_loop_ngp_serial(struct grid *grid, struct particles *parts,
                                    double *spectra, const double c) {

  /* Unpack the grid properties. */
  int *dims = grid->dims;
  int ndim = grid->ndim;
  int nlam = grid->nlam;
  double *wavelength = grid->lam;
  double **grid_props = grid->props;
  double *grid_spectra = grid->spectra;

  /* Unpack the particles properties. */
  double *part_masses = parts->mass;
  double **part_props = parts->props;
  double *fesc = parts->fesc;
  double *velocity = parts->velocities;
  int npart = parts->npart;
  
  double shifted_wavelengths[nlam];

  /* Loop over particles. */
  for (int p = 0; p < npart; p++) {

    /* Get this particle's mass, velocity and doppler shift. */
    const double mass = part_masses[p];
    double vel = velocity[p];
    double shift_factor = 1.0 + vel / c;
    for(int ilam=0; ilam<nlam; ilam++){
      shifted_wavelengths[ilam] = wavelength[ilam] * shift_factor;
    }
    
    /* Setup the index array. */
    int part_indices[ndim];

    /* Get the grid indices for the particle */
    get_part_inds_ngp(part_indices, dims, ndim, grid_props, part_props, p);

    /* Define the weight. */
    double weight = mass * (1.0 - fesc[p]);

    /* Get the weight's index. */
    const int grid_ind = get_flat_index(part_indices, dims, ndim);

    /* Get the spectra ind. */
    int unraveled_ind[ndim + 1];
    get_indices_from_flat(grid_ind, ndim, dims, unraveled_ind);
    unraveled_ind[ndim] = 0;
    int spectra_ind = get_flat_index(unraveled_ind, dims, ndim + 1);

    /* Add this grid cell's contribution to the spectra */
    for (int ilam = 0; ilam < nlam; ilam++) {
      double shifted_lambda = shifted_wavelengths[ilam];
      int ilam_shifted = find_nearest_bin(shifted_lambda, wavelength, nlam); 
      if (ilam_shifted >= nlam - 1 || ilam_shifted < 0) {continue}; // jumps out of the loop if out of bounds
      double frac_shifted = (shifted_lambda - wavelength[ilam_shifted]) / (wavelength[ilam_shifted + 1] - wavelength[ilam_shifted]);
      
      /* Add the contribution to this wavelength. */
      spectra[p * nlam + ilam_shifted] += grid_spectra[spectra_ind + ilam] * (1.0 - frac_shifted) * weight;
      spectra[p * nlam + ilam_shifted + 1] += grid_spectra[spectra_ind + ilam] * frac_shifted * weight;
    }
  }
}

/**
 * @brief This calculates particle spectra using a nearest grid point approach.
 *
 * This is the serial version of the function.
 *
 * @param grid: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 * @param nthreads: The number of threads to use.
 */
#ifdef WITH_OPENMP
static void spectra_loop_ngp_omp(struct grid *grid, struct particles *parts,
                                 double *spectra, int nthreads) {

  /* How many particles should each thread get? */
  int npart_per_thread = (parts->npart + nthreads - 1) / nthreads;

#pragma omp parallel num_threads(nthreads)
  {

    /* Unpack the grid properties. */
    int *dims = grid->dims;
    int ndim = grid->ndim;
    int nlam = grid->nlam;
    double **grid_props = grid->props;
    double *grid_spectra = grid->spectra;

    /* Unpack the particles properties. */
    double *part_masses = parts->mass;
    double **part_props = parts->props;
    double *fesc = parts->fesc;
    int npart = parts->npart;

    /* Get the thread id. */
    int tid = omp_get_thread_num();

    /* Calculate start and end indices for each thread */
    int start = tid * npart_per_thread;
    int end = start + npart_per_thread;
    if (end >= npart) {
      end = npart;
    }
#ifdef WITH_DEBUGGING_CHECKS
    else {
#pragma omp critical
      PyErr_SetString(PyExc_RuntimeError,
                      "Not all particles distributed to threads.");
      free(spectra);
      spectra = NULL;
      return;
    }
#endif

    /* Loop over particles. */
    for (int p = start; p < end; p++) {

      /* Get this particle's mass. */
      const double mass = part_masses[p];

      /* Setup the index array. */
      int part_indices[ndim];

      /* Get the grid indices for the particle */
      get_part_inds_ngp(part_indices, dims, ndim, grid_props, part_props, p);

      /* Define the weight. */
      double weight = mass * (1.0 - fesc[p]);

      /* Get the weight's index. */
      const int grid_ind = get_flat_index(part_indices, dims, ndim);

      /* Get the spectra ind. */
      int unraveled_ind[ndim + 1];
      get_indices_from_flat(grid_ind, ndim, dims, unraveled_ind);
      unraveled_ind[ndim] = 0;
      int spectra_ind = get_flat_index(unraveled_ind, dims, ndim + 1);

      /* Add this grid cell's contribution to the spectra */
      for (int ilam = 0; ilam < nlam; ilam++) {

        /* Add the contribution to this wavelength. */
        spectra[p * nlam + ilam] += grid_spectra[spectra_ind + ilam] * weight;
      }
    }
  }
}
#endif


/**
 * @brief This calculates particle spectra using a nearest grid point approach.
 * This is the version of the function that accounts for doppler shift.
 * This is the serial version of the function.
 *
 * @param grid: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 * @param nthreads: The number of threads to use.
 */
#ifdef WITH_OPENMP
static void shifted_spectra_loop_ngp_omp(struct grid *grid, struct particles *parts,
                                 double *spectra, int nthreads, const double c) {

  /* How many particles should each thread get? */
  int npart_per_thread = (parts->npart + nthreads - 1) / nthreads;

#pragma omp parallel num_threads(nthreads)
  {

    /* Unpack the grid properties. */
    int *dims = grid->dims;
    int ndim = grid->ndim;
    int nlam = grid->nlam;
    double *wavelength = grid->lam;
    double **grid_props = grid->props;
    double *grid_spectra = grid->spectra;

    /* Unpack the particles properties. */
    double *part_masses = parts->mass;
    double **part_props = parts->props;
    double *fesc = parts->fesc;
    double *velocity = parts->velocities;
    int npart = parts->npart;

    double shifted_wavelengths[nlam];

    /* Get the thread id. */
    int tid = omp_get_thread_num();

    /* Calculate start and end indices for each thread */
    int start = tid * npart_per_thread;
    int end = start + npart_per_thread;
    if (end >= npart) {
      end = npart;
    }
#ifdef WITH_DEBUGGING_CHECKS
    else {
#pragma omp critical
      PyErr_SetString(PyExc_RuntimeError,
                      "Not all particles distributed to threads.");
      free(spectra);
      spectra = NULL;
      return;
    }
#endif

    /* Loop over particles. */
    for (int p = start; p < end; p++) {

      /* Get this particle's mass, velocity and doppler shift contribution. */
      const double mass = part_masses[p];
      double vel = velocity[p];
      double shift_factor = 1.0 + vel / c;
      for (int i=0, i<nlam, i++){
        shifted_wavelengths[i] = wavelength[i] * shift_factor;
      }

      /* Setup the index array. */
      int part_indices[ndim];

      /* Get the grid indices for the particle */
      get_part_inds_ngp(part_indices, dims, ndim, grid_props, part_props, p);

      /* Define the weight. */
      double weight = mass * (1.0 - fesc[p]);

      /* Get the weight's index. */
      const int grid_ind = get_flat_index(part_indices, dims, ndim);

      /* Get the spectra ind. */
      int unraveled_ind[ndim + 1];
      get_indices_from_flat(grid_ind, ndim, dims, unraveled_ind);
      unraveled_ind[ndim] = 0;
      int spectra_ind = get_flat_index(unraveled_ind, dims, ndim + 1);

      /* Add this grid cell's contribution to the spectra */
      for (int ilam = 0; ilam < nlam; ilam++) {
        double shifted_lambda = shifted_wavelengths[ilam];
        /*Find nearest wavelength bin*/
        int ilam_shifted = find_nearest_bin(shifted_lambda, wavelength, nlam); 
        if (ilam_shifted >= nlam - 1 || ilam_shifted < 0) {continue}; // jumps out of the loop if out of bounds
        double frac_shifted = (shifted_lambda - wavelength[ilam_shifted]) /
                                  (wavelength[ilam_shifted + 1] - wavelength[ilam_shifted]);

        /* Add the contribution to this wavelength. */
        spectra[p * nlam + ilam_shifted] += grid_spectra[spectra_ind + ilam] * (1.0 - frac_shifted) * weight;
        spectra[p * nlam + ilam_shifted + 1] += grid_spectra[spectra_ind + ilam] * frac_shifted * weight;
      }
    }
  }
}
#endif



/**
 * @brief This calculates particle spectra using a nearest grid point approach.
 *
 * This is a wrapper which calls the correct function based on the number of
 * threads requested and whether OpenMP is available.
 *
 * @param grid: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 * @param nthreads: The number of threads to use.
 */
void spectra_loop_ngp(struct grid *grid, struct particles *parts,
                      double *spectra, const int nthreads) {

  double start_time = tic();

  /* Call the correct function for the configuration/number of threads. */

#ifdef WITH_OPENMP

  /* If we have multiple threads and OpenMP we can parallelise. */
  if (nthreads > 1) {
    spectra_loop_ngp_omp(grid, parts, spectra, nthreads);
  }
  /* Otherwise there's no point paying the OpenMP overhead. */
  else {
    spectra_loop_ngp_serial(grid, parts, spectra);
  }

#else

  (void)nthreads;

  /* We don't have OpenMP, just call the serial version. */
  spectra_loop_ngp_serial(grid, parts, spectra);

#endif
  toc("Nearest Grid Point particle spectra loop", start_time);
}


/**
 * @brief This calculates doppler-shifted particle spectra using a nearest grid point approach.
 *
 * This is a wrapper which calls the correct function based on the number of
 * threads requested and whether OpenMP is available.
 * This is the version of the wrapper that accounts for doppler shift.
 * @param grid: A struct containing the properties along each grid axis.
 * @param parts: A struct containing the particle properties.
 * @param spectra: The output array.
 * @param nthreads: The number of threads to use.
 */
void shifted_spectra_loop_ngp(struct grid *grid, struct particles *parts,
                      double *spectra, const int nthreads) {

  double start_time = tic();

  /* Call the correct function for the configuration/number of threads. */

#ifdef WITH_OPENMP

  /* If we have multiple threads and OpenMP we can parallelise. */
  if (nthreads > 1) {
    shifted_spectra_loop_ngp_omp(grid, parts, spectra, nthreads);
  }
  /* Otherwise there's no point paying the OpenMP overhead. */
  else {
    shifted_spectra_loop_ngp_serial(grid, parts, spectra);
  }

#else

  (void)nthreads;

  /* We don't have OpenMP, just call the serial version. */
  shifted_spectra_loop_ngp_serial(grid, parts, spectra);

#endif
  toc("Nearest Grid Point particle spectra loop", start_time);
}





/**
 * @brief Computes an integrated SED for a collection of particles.
 *
 * @param np_grid_spectra: The SPS spectra array.
 * @param grid_tuple: The tuple containing arrays of grid axis properties.
 * @param part_tuple: The tuple of particle property arrays (in the same order
 *                    as grid_tuple).
 * @param np_part_mass: The particle mass array.
 * @param fesc: The escape fraction.
 * @param np_velocities: The velocities array.
 * @param np_ndims: The size of each grid axis.
 * @param ndim: The number of grid axes.
 * @param npart: The number of particles.
 * @param nlam: The number of wavelength elements.
 * @param shift: flag whether to consider doppler shift in spectra computation.
 */
PyObject *compute_particle_seds(PyObject *self, PyObject *args) {

  double start_time = tic();
  double setup_start = tic();

  /* We don't need the self argument but it has to be there. Tell the compiler
   * we don't care. */
  (void)self;

  int ndim, npart, nlam, nthreads;
  PyObject *grid_tuple, *part_tuple;
  PyObject vel_shift; // does this have to be a pyobject? can we parse vel_shift = 0 to default to False?
  PyArrayObject *np_grid_spectra;
  PyArrayObject *np_fesc;
  PyArrayObject *np_velocities;
  PyArrayObject *np_part_mass, *np_ndims;
  char *method;

  if (!PyArg_ParseTuple(args, "OOOOOOiiisi", &np_grid_spectra, &grid_tuple,
                        &part_tuple, &np_part_mass, &np_fesc, &np_velocities, &np_ndims, &ndim,
                        &npart, &nlam, &method, &vel_shift, &nthreads))
    return NULL;

  /* Extract the grid struct. */
  struct grid *grid_props = get_spectra_grid_struct(
      grid_tuple, np_ndims, np_grid_spectra, ndim, nlam);
  if (grid_props == NULL) {
    return NULL;
  }

  /* Extract the particle struct. */
  struct particles *part_props =
      get_part_struct(part_tuple, np_part_mass, *np_velocities*, np_fesc, npart, ndim);
  if (part_props == NULL) {
    return NULL;
  }

  /* Allocate the spectra. */
  double *spectra = calloc(npart * nlam, sizeof(double));
  if (spectra == NULL) {
    PyErr_SetString(PyExc_MemoryError,
                    "Could not allocate memory for spectra.");
    return NULL;
  }

  toc("Extracting Python data", setup_start);

  if (!vel_shift) {
    /*No shift*/
    /* With everything set up we can compute the spectra for each particle using
     * the requested method. */
    if (strcmp(method, "cic") == 0) {
      spectra_loop_cic(grid_props, part_props, spectra, nthreads);
    } else if (strcmp(method, "ngp") == 0) {
      spectra_loop_ngp(grid_props, part_props, spectra, nthreads);
    } else {
      PyErr_SetString(PyExc_ValueError, "Unknown grid assignment method (%s).");
      return NULL;
    }
  } else {
    /*With shift shift*/
    /* With everything set up we can compute the spectra for each particle using
     * the requested method. */
    if (strcmp(method, "cic") == 0) {
      spectra_loop_cic_shifted(grid_props, part_props, spectra, nthreads); // make these 1
    } else if (strcmp(method, "ngp") == 0) {
      spectra_loop_ngp_shifted(grid_props, part_props, spectra, nthreads); // 2
    } else {
      PyErr_SetString(PyExc_ValueError, "Unknown grid assignment method (%s).");
      return NULL;
  }

  /* Check we got the spectra sucessfully. (Any error messages will already be
   * set) */
  if (spectra == NULL) {
    return NULL;
  }

  /* Clean up memory! */
  free(part_props);
  free(grid_props);

  /* Reconstruct the python array to return. */
  npy_intp np_dims[2] = {
      npart,
      nlam,
  };
  PyArrayObject *out_spectra = (PyArrayObject *)PyArray_SimpleNewFromData(
      2, np_dims, NPY_FLOAT64, spectra);

  toc("Computing particle SEDs", start_time);

  return Py_BuildValue("N", out_spectra);
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
  import_array();
  return m;
}
