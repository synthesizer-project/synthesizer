/* Standard includes */
#include <cmath>

/* Local includes */
#include "reductions.h"
#include "timers.h"

/**
 * @brief Reduce Npart spectra to integrated spectra.
 *
 * This is a serial version of the function.
 *
 * @param spectra: The output array to accumulate the spectra.
 * @param part_spectra: The per-particle spectra array.
 * @param nlam: The number of wavelengths in the spectra.
 * @param npart: The number of particles.
 */
static void reduce_spectra_serial(double *spectra, double *part_spectra,
                                  int nlam, int npart) {

  /* Cast npart to size_t for safety in the loop. */
  size_t npart_size = static_cast<size_t>(npart);

  /* Loop over particles. */
  for (size_t p = 0; p < npart_size; p++) {
    /* Loop over wavelengths. */
    for (int ilam = 0; ilam < nlam; ilam++) {
      size_t part_spec_ind = p * nlam + ilam;
      /* Use fused multiply-add to accumulate with better precision.
       * Equivalent to: += spec_val * weight, but with a single rounding. */
      spectra[ilam] = std::fma(part_spectra[part_spec_ind], 1.0, spectra[ilam]);
    }
  }
}

/**
 * @brief Reduce Npart spectra to integrated spectra.
 *
 * This is a parallel version of the function.
 *
 * @param spectra: The output array to accumulate the spectra.
 * @param part_spectra: The per-particle spectra array.
 * @param nlam: The number of wavelengths in the spectra.
 * @param npart: The number of particles.
 * @param nthreads: The number of threads to use.
 */
#ifdef WITH_OPENMP
static void reduce_spectra_parallel(double *spectra, double *part_spectra,
                                    int nlam, int npart, int nthreads) {

  /* Cast npart to size_t for safety in the loop. */
  size_t npart_size = static_cast<size_t>(npart);

  /* Loop over particles in parallel. */
#if defined(_OPENMP) && _OPENMP >= 201511
#pragma omp parallel for num_threads(nthreads) reduction(+ : spectra[ : nlam])
  for (size_t p = 0; p < npart_size; p++) {
    for (int ilam = 0; ilam < nlam; ilam++) {
      size_t part_spec_ind = p * nlam + ilam;
      spectra[ilam] += part_spectra[part_spec_ind];
    }
  }
#else // OpenMP < 4.5 or no array reduction support
#pragma omp parallel num_threads(nthreads)
  {
    // Thread-local accumulation to avoid false sharing and atomics
    std::vector<double> local(nlam, 0.0);
#pragma omp for nowait schedule(static)
    for (size_t p = 0; p < npart; p++) {
      for (int ilam = 0; ilam < nlam; ilam++) {
        size_t part_spec_ind = p * nlam + ilam;
        local[ilam] += part_spectra[part_spec_ind];
      }
    }
    // Merge
#pragma omp critical
    {
      for (int ilam = 0; ilam < nlam; ilam++) {
        spectra[ilam] += local[ilam];
      }
    }
  }
#endif // WITH_OPENMP
}
#endif

/**
 * @brief Reduce Npart spectra to integrated spectra.
 *
 * This is a wrapper function that calls the serial or parallel version of the
 * function depending on the number of threads requested or whether OpenMP is
 * available.
 *
 * @param spectra: The output array to accumulate the spectra.
 * @param part_spectra: The per-particle spectra array.
 * @param nlam: The number of wavelengths in the spectra.
 * @param npart: The number of particles.
 * @param nthreads: The number of threads to use.
 */
void reduce_spectra(double *spectra, double *part_spectra, int nlam, int npart,
                    int nthreads) {

  double start_time = tic();
  if (nthreads > 1) {
#ifdef WITH_OPENMP
    reduce_spectra_parallel(spectra, part_spectra, nlam, npart, nthreads);
#else
    reduce_spectra_serial(spectra, part_spectra, nlam, npart);
#endif
  } else {
    reduce_spectra_serial(spectra, part_spectra, nlam, npart);
  }
  toc("Reducing particle spectra", start_time);
}
