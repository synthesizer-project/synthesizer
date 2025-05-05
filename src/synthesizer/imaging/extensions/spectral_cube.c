/******************************************************************************
 * C functions for calculating the value of a stellar particles SPH kernel
 *****************************************************************************/

/* C includes. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Python includes. */
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>

/* Local includes. */
#include "../../extensions/property_funcs.h"
#include "../../extensions/timers.h"

#ifdef WITH_OPENMP
#include <omp.h>
#endif

/**
 * @brief Function to populate a data cube with a kernel in serial.
 *
 * The SPH kernel of a particle (integrated along the z axis) is used to
 * calculate the pixel weight for all spaxels within a stellar particles kernel.
 * Once the kernel value is found at a spaxels position the pixel value is added
 * multiplied by the kernels weight.
 *
 * NOTE: the implementation uses the exact position of a particle, thus
 * accounting for sub pixel positioning.
 *
 * @param sed_values: The particle SEDs.
 * @param smoothing_lengths: The stellar particle smoothing lengths.
 * @param xs: The x coordinates of the particles.
 * @param ys: The y coordinates of the particles.
 * @param kernel: The kernel data (integrated along the z axis and softed by
 *               impact parameter).
 * @param res: The pixel resolution.
 * @param npix_x: The number of spaxels along the x axis.
 * @param npix_y: The number of spaxels along the y axis.
 * @param npart: The number of particles.
 * @param nlam: The number of wavelength elements in the SEDs.
 * @param threshold: The threshold of the SPH kernel.
 * @param kdim: The number of elements in the kernel.
 * @param data_cube: The data cube to be populated.
 */
void populate_smoothed_data_cube_serial(
    const double *sed_values, const double *smoothing_lengths, const double *xs,
    const double *ys, const double *kernel, const double res, const int npix_x,
    const int npix_y, const int npart, const int nlam, const double threshold,
    const int kdim, double *data_cube) {

  /* Loop over positions including the sed */
  for (int ind = 0; ind < npart; ind++) {

    /* Get this particles smoothing length and position */
    const double smooth_length = smoothing_lengths[ind];
    const double x = xs[ind];
    const double y = ys[ind];

    /* Calculate the pixel coordinates of this particle. */
    const int i = x / res;
    const int j = y / res;

    /* How many spaxels are in the smoothing length? Add some buffer. */
    const int delta_pix = ceil(smooth_length / res) + 1;

    /* How many spaxels along kernel axis? */
    const int kernel_cdim = 2 * delta_pix + 1;

    /* Create an empty kernel for this particle. */
    double *part_kernel =
        synth_malloc(kernel_cdim * kernel_cdim * sizeof(double), "part_kernel");

    /* Track the kernel sum for normalisation. */
    double kernel_sum = 0;

    /* Loop over a square aperture around this particle */
    for (int ii = i - delta_pix; ii <= i + delta_pix; ii++) {

      /* Skip out of bounds spaxels. */
      if (ii < 0 || ii >= npix_x)
        continue;

      /* Compute the x separation */
      const double x_dist = (ii * res) + (res / 2) - x;

      for (int jj = j - delta_pix; jj <= j + delta_pix; jj++) {

        /* Skip out of bounds spaxels. */
        if (jj < 0 || jj >= npix_y)
          continue;

        /* Compute the y separation */
        const double y_dist = (jj * res) + (res / 2) - y;

        /* Compute the distance between the centre of this pixel
         * and the particle. */
        const double rsqu = (x_dist * x_dist) + (y_dist * y_dist);

        /* Get the pixel coordinates in the kernel */
        const int iii = ii - (i - delta_pix);
        const int jjj = jj - (j - delta_pix);

        /* Calculate the impact parameter. */
        const double q = sqrt(rsqu) / smooth_length;

        /* Skip gas particles outside the kernel. */
        if (q > threshold)
          continue;

        /* Get the value of the kernel at q. */
        const int index = kdim * q;
        const double kvalue = kernel[index];

        /* Set the value in the kernel. */
        part_kernel[iii * kernel_cdim + jjj] = kvalue;
        kernel_sum += kvalue;
      }
    }

    /* Normalise the kernel */
    if (kernel_sum > 0) {
      for (int n = 0; n < kernel_cdim * kernel_cdim; n++) {
        part_kernel[n] /= kernel_sum;
      }
    }

    /* Loop over a square aperture around this particle */
    for (int ii = i - delta_pix; ii <= i + delta_pix; ii++) {

      /* Skip out of bounds spaxels. */
      if (ii < 0 || ii >= npix_x)
        continue;

      for (int jj = j - delta_pix; jj <= j + delta_pix; jj++) {

        /* Skip out of bounds spaxels. */
        if (jj < 0 || jj >= npix_y)
          continue;

        /* Get the pixel coordinates in the kernel */
        const int iii = ii - (i - delta_pix);
        const int jjj = jj - (j - delta_pix);

        /* Skip empty spaxels. */
        if (part_kernel[iii * kernel_cdim + jjj] == 0)
          continue;

        /* Loop over the wavelength axis. */
        for (int ilam = 0; ilam < nlam; ilam++) {
          unsigned long long data_cube_ind = ilam + nlam * (jj + npix_y * ii);
          unsigned long long sed_ind = (ind * nlam) + ilam;
          data_cube[data_cube_ind] +=
              part_kernel[iii * kernel_cdim + jjj] * sed_values[sed_ind];
        }
      }
    }

    free(part_kernel);
  }
}

/**
 * @brief Function to populate a data cube with a kernel in parallel.
 *
 * The SPH kernel of a particle (integrated along the z axis) is used to
 * calculate the pixel weight for all spaxels within a stellar particles kernel.
 * Once the kernel value is found at a spaxels position the pixel value is added
 * multiplied by the kernels weight.
 *
 * NOTE: the implementation uses the exact position of a particle, thus
 * accounting for sub pixel positioning.
 *
 * We use locks here to avoid cache clashes. This is not the most efficient
 * method, unlike the imaging the setting of a spaxels value is sufficiently
 * costly that the overhead of the locks is not a problem.
 *
 * @param sed_values: The particle SEDs.
 * @param smoothing_lengths: The stellar particle smoothing lengths.
 * @param xs: The x coordinates of the particles.
 * @param ys: The y coordinates of the particles.
 * @param kernel: The kernel data (integrated along the z axis and softed by
 *               impact parameter).
 * @param res: The pixel resolution.
 * @param npix_x: The number of spaxels along the x axis.
 * @param npix_y: The number of spaxels along the y axis.
 * @param npart: The number of particles.
 * @param nlam: The number of wavelength elements in the SEDs.
 * @param threshold: The threshold of the SPH kernel.
 * @param kdim: The number of elements in the kernel.
 * @param data_cube: The data cube to be populated.
 */
#ifdef WITH_OPENMP
void populate_smoothed_data_cube_parallel(
    const double *sed_values, const double *smoothing_lengths, const double *xs,
    const double *ys, const double *kernel, const double res, const int npix_x,
    const int npix_y, const int npart, const int nlam, const double threshold,
    const int kdim, double *data_cube, const int nthreads) {

  /* Set up an array of locks. */
  omp_lock_t *locks =
      (omp_lock_t *)malloc(npix_x * npix_y * sizeof(omp_lock_t));
  for (int i = 0; i < npix_x * npix_y; i++) {
    omp_init_lock(&locks[i]);
  }

  /* Loop over positions including the sed */
#pragma omp parallel for num_threads(nthreads)                                 \
    shared(data_cube, sed_values, smoothing_lengths, xs, ys, kernel, res,      \
               npix_x, npix_y, npart, nlam, threshold, kdim)
  for (int ind = 0; ind < npart; ind++) {

    /* Get this particles smoothing length and position */
    const double smooth_length = smoothing_lengths[ind];
    const double x = xs[ind];
    const double y = ys[ind];

    /* Calculate the pixel coordinates of this particle. */
    const int i = x / res;
    const int j = y / res;

    /* How many spaxels are in the smoothing length? Add some buffer. */
    const int delta_pix = ceil(smooth_length / res) + 1;

    /* How many spaxels along kernel axis? */
    const int kernel_cdim = 2 * delta_pix + 1;

    /* Create an empty kernel for this particle. */
    double *part_kernel =
        synth_malloc(kernel_cdim * kernel_cdim * sizeof(double), "part_kernel");

    /* Track the kernel sum for normalisation. */
    double kernel_sum = 0;

    /* Loop over a square aperture around this particle */
    for (int ii = i - delta_pix; ii <= i + delta_pix; ii++) {

      /* Skip out of bounds spaxels. */
      if (ii < 0 || ii >= npix_x)
        continue;

      /* Compute the x separation */
      const double x_dist = (ii * res) + (res / 2) - x;

      for (int jj = j - delta_pix; jj <= j + delta_pix; jj++) {

        /* Skip out of bounds spaxels. */
        if (jj < 0 || jj >= npix_y)
          continue;

        /* Compute the y separation */
        const double y_dist = (jj * res) + (res / 2) - y;

        /* Compute the distance between the centre of this pixel
         * and the particle. */
        const double rsqu = (x_dist * x_dist) + (y_dist * y_dist);

        /* Get the pixel coordinates in the kernel */
        const int iii = ii - (i - delta_pix);
        const int jjj = jj - (j - delta_pix);

        /* Calculate the impact parameter. */
        const double q = sqrt(rsqu) / smooth_length;

        /* Skip gas particles outside the kernel. */
        if (q > threshold)
          continue;

        /* Get the value of the kernel at q. */
        const int index = kdim * q;
        const double kvalue = kernel[index];

        /* Set the value in the kernel. */
        part_kernel[iii * kernel_cdim + jjj] = kvalue;
        kernel_sum += kvalue;
      }
    }

    /* Normalise the kernel */
    if (kernel_sum > 0) {
      for (int n = 0; n < kernel_cdim * kernel_cdim; n++) {
        part_kernel[n] /= kernel_sum;
      }
    }

    /* Loop over a square aperture around this particle */
    for (int ii = i - delta_pix; ii <= i + delta_pix; ii++) {

      /* Skip out of bounds spaxels. */
      if (ii < 0 || ii >= npix_x)
        continue;

      for (int jj = j - delta_pix; jj <= j + delta_pix; jj++) {

        /* Skip out of bounds spaxels. */
        if (jj < 0 || jj >= npix_y)
          continue;

        /* Get the pixel coordinates in the kernel */
        const int iii = ii - (i - delta_pix);
        const int jjj = jj - (j - delta_pix);

        /* Skip empty spaxels. */
        if (part_kernel[iii * kernel_cdim + jjj] == 0)
          continue;

        /* Loop over the wavelength axis. */
        omp_set_lock(&locks[jj + npix_y * ii]);
        for (int ilam = 0; ilam < nlam; ilam++) {
          unsigned long long data_cube_ind = ilam + nlam * (jj + npix_y * ii);
          unsigned long long sed_ind = (ind * nlam) + ilam;
          data_cube[data_cube_ind] +=
              part_kernel[iii * kernel_cdim + jjj] * sed_values[sed_ind];
        }
        omp_unset_lock(&locks[jj + npix_y * ii]);
      }
    }

    free(part_kernel);
  }
}
#endif

/**
 * @brief Populate a data cube from particle data and a kernel.
 *
 * This is a wrapper function that calls the serial or parallel version of the
 * function depending on the number of threads requested or whether OpenMP is
 * available.
 *
 * @param sed_values: The particle SEDs.
 * @param smoothing_lengths: The stellar particle smoothing lengths.
 * @param xs: The x coordinates of the particles.
 * @param ys: The y coordinates of the particles.
 * @param kernel: The kernel data (integrated along the z axis and softed by
 *               impact parameter).
 * @param res: The pixel resolution.
 * @param npix_x: The number of spaxels along the x axis.
 * @param npix_y: The number of spaxels along the y axis.
 * @param npart: The number of particles.
 * @param nlam: The number of wavelength elements in the SEDs.
 * @param threshold: The threshold of the SPH kernel.
 * @param kdim: The number of elements in the kernel.
 * @param data_cube: The data cube to be populated.
 * @param nthreads: The number of threads to use.
 */
void populate_smoothed_data_cube(
    const double *sed_values, const double *smoothing_lengths, const double *xs,
    const double *ys, const double *kernel, const double res, const int npix_x,
    const int npix_y, const int npart, const int nlam, const double threshold,
    const int kdim, double *data_cube, const int nthreads) {

  double start = tic();

#ifdef WITH_OPENMP
  if (nthreads > 1) {
    populate_smoothed_data_cube_parallel(
        sed_values, smoothing_lengths, xs, ys, kernel, res, npix_x, npix_y,
        npart, nlam, threshold, kdim, data_cube, nthreads);
  } else {
    populate_smoothed_data_cube_serial(sed_values, smoothing_lengths, xs, ys,
                                       kernel, res, npix_x, npix_y, npart, nlam,
                                       threshold, kdim, data_cube);
  }
#else
  populate_smoothed_data_cube_serial(sed_values, smoothing_lengths, xs, ys,
                                     kernel, res, npix_x, npix_y, npart, nlam,
                                     threshold, kdim, data_cube);
#endif
  toc("Populating smoothed data_cube", start);
}

/**
 * @brief Function to populate a data cube in serial.
 *
 * The SPH kernel of a particle (integrated along the z axis) is used to
 * calculate the pixel weight for all spaxels within a stellar particles kernel.
 * Once the kernel value is found at a spaxels position the pixel value is added
 * multiplied by the kernels weight.
 *
 * NOTE: the implementation uses the exact position of a particle, thus
 * accounting for sub pixel positioning.
 *
 * @param sed_values: The particle SEDs.
 * @param xs: The x coordinates of the particles.
 * @param ys: The y coordinates of the particles.
 * @param res: The pixel resolution.
 * @param npix_x: The number of spaxels along the x axis.
 * @param npix_y: The number of spaxels along the y axis.
 * @param npart: The number of particles.
 * @param nlam: The number of wavelength elements in the SEDs.
 * @param data_cube: The data cube to be populated.
 */
void populate_hist_data_cube_serial(const double *sed_values, const double *xs,
                                    const double *ys, const double res,
                                    const int npix_x, const int npix_y,
                                    const int npart, const int nlam,
                                    double *data_cube) {

  /* Loop over positions including the sed */
  for (int ind = 0; ind < npart; ind++) {

    /* Get this particle's position */
    const double x = xs[ind];
    const double y = ys[ind];

    /* Calculate the pixel coordinates of this particle. */
    const int i = x / res;
    const int j = y / res;

    /* Skip if outside the FOV. */
    if (i < 0 || i >= npix_x || j < 0 || j >= npix_y)
      continue;

    /* Loop over the wavelength axis. */
    for (int ilam = 0; ilam < nlam; ilam++) {
      unsigned long long data_cube_ind = ilam + nlam * (j + npix_y * i);
      unsigned long long sed_ind = (ind * nlam) + ilam;
      data_cube[data_cube_ind] += sed_values[sed_ind];
    }
  }
}

/**
 * @brief Function to populate a data cube in parallel.
 *
 * The SPH kernel of a particle (integrated along the z axis) is used to
 * calculate the pixel weight for all spaxels within a stellar particles kernel.
 * Once the kernel value is found at a spaxels position the pixel value is added
 * multiplied by the kernels weight.
 *
 * NOTE: the implementation uses the exact position of a particle, thus
 * accounting for sub pixel positioning.
 *
 * @param sed_values: The particle SEDs.
 * @param xs: The x coordinates of the particles.
 * @param ys: The y coordinates of the particles.
 * @param res: The pixel resolution.
 * @param npix_x: The number of spaxels along the x axis.
 * @param npix_y: The number of spaxels along the y axis.
 * @param npart: The number of particles.
 * @param nlam: The number of wavelength elements in the SEDs.
 * @param data_cube: The data cube to be populated.
 * @param nthreads: The number of threads to use.
 */
#ifdef WITH_OPENMP
void populate_hist_data_cube_parallel(const double *sed_values,
                                      const double *xs, const double *ys,
                                      const double res, const int npix_x,
                                      const int npix_y, const int npart,
                                      const int nlam, double *data_cube,
                                      const int nthreads) {

  /* Set up an array of locks. */
  omp_lock_t *locks =
      (omp_lock_t *)malloc(npix_x * npix_y * sizeof(omp_lock_t));
  for (int i = 0; i < npix_x * npix_y; i++) {
    omp_init_lock(&locks[i]);
  }

  /* Loop over positions including the sed */
#pragma omp parallel for num_threads(nthreads)                                 \
    shared(data_cube, sed_values, xs, ys, res, npix_x, npix_y, npart, nlam)
  for (int ind = 0; ind < npart; ind++) {

    /* Get this particle's position */
    const double x = xs[ind];
    const double y = ys[ind];

    /* Calculate the pixel coordinates of this particle. */
    const int i = x / res;
    const int j = y / res;

    /* Skip if outside the FOV. */
    if (i < 0 || i >= npix_x || j < 0 || j >= npix_y)
      continue;

    /* Loop over the wavelength axis. */
    omp_set_lock(&locks[j + npix_y * i]);
    for (int ilam = 0; ilam < nlam; ilam++) {
      unsigned long long data_cube_ind = ilam + nlam * (j + npix_y * i);
      unsigned long long sed_ind = (ind * nlam) + ilam;
      data_cube[data_cube_ind] += sed_values[sed_ind];
    }
    omp_unset_lock(&locks[j + npix_y * i]);
  }
}
#endif

/**
 * @brief Populate a data cube from particle data and a kernel.
 *
 * This is a wrapper function that calls the serial or parallel version of the
 * function depending on the number of threads requested or whether OpenMP is
 * available.
 *
 * @param sed_values: The particle SEDs.
 * @param xs: The x coordinates of the particles.
 * @param ys: The y coordinates of the particles.
 * @param res: The pixel resolution.
 * @param npix_x: The number of spaxels along the x axis.
 * @param npix_y: The number of spaxels along the y axis.
 * @param npart: The number of particles.
 * @param nlam: The number of wavelength elements in the SEDs.
 * @param data_cube: The data cube to be populated.
 * @param nthreads: The number of threads to use.
 */
void populate_hist_data_cube(const double *sed_values, const double *xs,
                             const double *ys, const double res,
                             const int npix_x, const int npix_y,
                             const int npart, const int nlam, double *data_cube,
                             const int nthreads) {

  double start = tic();

#ifdef WITH_OPENMP
  if (nthreads > 1) {
    populate_hist_data_cube_parallel(sed_values, xs, ys, res, npix_x, npix_y,
                                     npart, nlam, data_cube, nthreads);
  } else {
    populate_hist_data_cube_serial(sed_values, xs, ys, res, npix_x, npix_y,
                                   npart, nlam, data_cube);
  }
#else
  populate_hist_data_cube_serial(sed_values, xs, ys, res, npix_x, npix_y, npart,
                                 nlam, data_cube);
#endif
  toc("Populating histogram data_cube", start);
}
/**
 * @brief Function to compute a data cube from particle data without smoothing.
 *
 * NOTE: the implementation uses the exact position of a particle, thus
 * accounting for sub pixel positioning.
 *
 * @param np_sed_values: The particle SEDs.
 * @param np_xs: The x coordinates of the particles.
 * @param np_ys: The y coordinates of the particles.
 * @param res: The pixel resolution.
 * @param npix_x: The number of spaxels along the x axis.
 * @param npix_y: The number of spaxels along the y axis.
 * @param npart: The number of particles.
 * @param nlam: The number of wavelength elements in the SEDs.
 * @param nthreads: The number of threads to use.
 */
PyObject *make_data_cube_hist(PyObject *self, PyObject *args) {

  /* We don't need the self argument but it has to be there. Tell the compiler
   * we don't care. */
  (void)self;

  double res;
  int npix_x, npix_y, npart, nlam, nthreads;
  PyArrayObject *np_sed_values;
  PyArrayObject *np_xs, *np_ys;

  if (!PyArg_ParseTuple(args, "OOOdiiiii", &np_sed_values, &np_xs, &np_ys, &res,
                        &npix_x, &npix_y, &npart, &nlam, &nthreads))
    return NULL;

  /* Get pointers to the actual data. */
  const double *sed_values = extract_data_double(np_sed_values, "sed_values");
  const double *xs = extract_data_double(np_xs, "xs");
  const double *ys = extract_data_double(np_ys, "ys");

  /* Allocate the data cube. */
  const int npix = npix_x * npix_y;
  double *data_cube = synth_malloc(npix * nlam * sizeof(double), "data_cube");

  /* Populate the data cube. */
  populate_hist_data_cube(sed_values, xs, ys, res, npix_x, npix_y, npart, nlam,
                          data_cube, nthreads);

  /* Construct a numpy python array to return the DATA_CUBE. */
  npy_intp dims[3] = {npix_x, npix_y, nlam};
  PyArrayObject *out_data_cube = (PyArrayObject *)PyArray_SimpleNewFromData(
      3, dims, NPY_FLOAT64, data_cube);

  return Py_BuildValue("N", out_data_cube);
}

/**
 * @brief Function to compute an DATA_CUBE from particle data and a kernel.
 *
 * The SPH kernel of a particle (integrated along the z axis) is used to
 * calculate the spaxel weight for all spaxels within a stellar particles
 * kernel. Once the kernel value is found at a spaxel's position each element of
 * the SED is added to the spaxel mulitplied by the kernels weight.
 *
 * NOTE: the implementation uses the exact position of a particle, thus
 * accounting for sub pixel positioning.
 *
 * @param np_sed_values: The particle SEDs.
 * @param np_smoothing_lengths: The stellar particle smoothing lengths.
 * @param np_xs: The x coordinates of the particles.
 * @param np_ys: The y coordinates of the particles.
 * @param np_kernel: The kernel data (integrated along the z axis and softed by
 *                   impact parameter).
 * @param res: The pixel resolution.
 * @param npix_x: The number of spaxels along the x axis.
 * @param npix_y: The number of spaxels along the y axis.
 * @param npart: The number of particles.
 * @param nlam: The number of wavelength elements in the SEDs.
 * @param threshold: The threshold of the SPH kernel.
 * @param kdim: The number of elements in the kernel.
 * @param nthreads: The number of threads to use.
 */
PyObject *make_data_cube_smooth(PyObject *self, PyObject *args) {

  /* We don't need the self argument but it has to be there. Tell the compiler
   * we don't care. */
  (void)self;

  double res, threshold;
  int npix_x, npix_y, npart, nlam, kdim, nthreads;
  PyArrayObject *np_sed_values, *np_kernel;
  PyArrayObject *np_smoothing_lengths, *np_xs, *np_ys;

  if (!PyArg_ParseTuple(args, "OOOOOdiiiidii", &np_sed_values,
                        &np_smoothing_lengths, &np_xs, &np_ys, &np_kernel, &res,
                        &npix_x, &npix_y, &npart, &nlam, &threshold, &kdim,
                        &nthreads))
    return NULL;

  /* Get pointers to the actual data. */
  const double *sed_values = extract_data_double(np_sed_values, "sed_values");
  const double *smoothing_lengths =
      extract_data_double(np_smoothing_lengths, "smoothing_lengths");
  const double *xs = extract_data_double(np_xs, "xs");
  const double *ys = extract_data_double(np_ys, "ys");
  const double *kernel = extract_data_double(np_kernel, "kernel");

  /* Allocate DATA_CUBE. */
  const int npix = npix_x * npix_y;
  double *data_cube = synth_malloc(npix * nlam * sizeof(double), "data_cube");

  /* Populate the DATA_CUBE. */
  populate_smoothed_data_cube(sed_values, smoothing_lengths, xs, ys, kernel,
                              res, npix_x, npix_y, npart, nlam, threshold, kdim,
                              data_cube, nthreads);

  /* Construct a numpy python array to return the DATA_CUBE. */
  npy_intp dims[3] = {npix_x, npix_y, nlam};
  PyArrayObject *out_data_cube =
      c_array_to_numpy(3, dims, NPY_FLOAT64, data_cube);

  return Py_BuildValue("N", out_data_cube);
}

static PyMethodDef ImageMethods[] = {
    {"make_data_cube_hist", (PyCFunction)make_data_cube_hist, METH_VARARGS,
     "Method for sorting particles into a spectral cube without smoothing."},
    {"make_data_cube_smooth", (PyCFunction)make_data_cube_smooth, METH_VARARGS,
     "Method for smoothing particles into a spectral cube."},
    {NULL, NULL, 0, NULL},
};

/* Make this importable. */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "spectral_cube",                              /* m_name */
    "A module to make data cubes from particles", /* m_doc */
    -1,                                           /* m_size */
    ImageMethods,                                 /* m_methods */
    NULL,                                         /* m_reload */
    NULL,                                         /* m_traverse */
    NULL,                                         /* m_clear */
    NULL,                                         /* m_free */
};

PyMODINIT_FUNC PyInit_spectral_cube(void) {
  PyObject *m = PyModule_Create(&moduledef);
  import_array();
  return m;
}
