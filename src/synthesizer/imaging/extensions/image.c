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
 * @brief Function to populate an image in serial.
 *
 * The SPH kernel of a particle (integrated along the z axis) is used to
 * calculate the pixel weight for all pixels within a stellar particles kernel.
 * Once the kernel value is found at a pixels position the pixel value is added
 * multiplied by the kernels weight.
 *
 * NOTE: the implementation uses the exact position of a particle, thus
 * accounting for sub pixel positioning.
 *
 * @param pix_values: The particle data to be sorted into pixels
 *                    (luminosity/flux/mass etc.).
 * @param smoothing_lengths: The stellar particle smoothing lengths.
 * @param xs: The x coordinates of the particles.
 * @param ys: The y coordinates of the particles.
 * @param kernel: The kernel data (integrated along the z axis and softed by
 *               impact parameter).
 * @param res: The pixel resolution.
 * @param npix_x: The number of pixels along the x axis.
 * @param npix_y: The number of pixels along the y axis.
 * @param npart: The number of particles.
 * @param threshold: The threshold of the SPH kernel.
 * @param kdim: The number of elements in the kernel.
 * @param img: The image to be populated.
 */
void populate_smoothed_image_serial(const double *pix_values,
                                    const double *smoothing_lengths,
                                    const double *xs, const double *ys,
                                    const double *kernel, const double res,
                                    const int npix_x, const int npix_y,
                                    const int npart, const double threshold,
                                    const int kdim, double *img) {

  /* Loop over positions including the sed */
  for (int ind = 0; ind < npart; ind++) {

    /* Get this particles smoothing length and position */
    const double smooth_length = smoothing_lengths[ind];
    const double x = xs[ind];
    const double y = ys[ind];

    /* Calculate the pixel coordinates of this particle. */
    const int i = x / res;
    const int j = y / res;

    /* How many pixels are in the smoothing length? Add some buffer. */
    const int delta_pix = ceil(smooth_length / res) + 1;

    /* How many pixels along kernel axis? */
    const int kernel_cdim = 2 * delta_pix + 1;

    /* Create an empty kernel for this particle. */
    double *part_kernel =
        synth_malloc(kernel_cdim * kernel_cdim * sizeof(double), "part_kernel");

    /* Track the kernel sum for normalisation. */
    double kernel_sum = 0;

    /* Loop over a square aperture around this particle */
    for (int ii = i - delta_pix; ii <= i + delta_pix; ii++) {

      /* Skip out of bounds pixels. */
      if (ii < 0 || ii >= npix_x)
        continue;

      /* Compute the x separation */
      const double x_dist = (ii * res) + (res / 2) - x;

      for (int jj = j - delta_pix; jj <= j + delta_pix; jj++) {

        /* Skip out of bounds pixels. */
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

      /* Skip out of bounds pixels. */
      if (ii < 0 || ii >= npix_x)
        continue;

      for (int jj = j - delta_pix; jj <= j + delta_pix; jj++) {

        /* Skip out of bounds pixels. */
        if (jj < 0 || jj >= npix_y)
          continue;

        /* Get the pixel coordinates in the kernel */
        const int iii = ii - (i - delta_pix);
        const int jjj = jj - (j - delta_pix);

        /* Skip empty pixels. */
        if (part_kernel[iii * kernel_cdim + jjj] == 0)
          continue;

        /* Loop over the wavelength axis. */
        img[jj + npix_y * ii] +=
            part_kernel[iii * kernel_cdim + jjj] * pix_values[ind];
      }
    }

    free(part_kernel);
  }
}

/**
 * @brief Function to populate an image in parallel.
 *
 * The SPH kernel of a particle (integrated along the z axis) is used to
 * calculate the pixel weight for all pixels within a stellar particles kernel.
 * Once the kernel value is found at a pixels position the pixel value is added
 * multiplied by the kernels weight.
 *
 * NOTE: the implementation uses the exact position of a particle, thus
 * accounting for sub pixel positioning.
 *
 * @param pix_values: The particle data to be sorted into pixels
 *                    (luminosity/flux/mass etc.).
 * @param smoothing_lengths: The stellar particle smoothing lengths.
 * @param xs: The x coordinates of the particles.
 * @param ys: The y coordinates of the particles.
 * @param kernel: The kernel data (integrated along the z axis and softed by
 *               impact parameter).
 * @param res: The pixel resolution.
 * @param npix_x: The number of pixels along the x axis.
 * @param npix_y: The number of pixels along the y axis.
 * @param npart: The number of particles.
 * @param threshold: The threshold of the SPH kernel.
 * @param kdim: The number of elements in the kernel.
 * @param img: The image to be populated.
 */
#ifdef WITH_OPENMP
void populate_smoothed_image_parallel(
    const double *pix_values, const double *smoothing_lengths, const double *xs,
    const double *ys, const double *kernel, const double res, const int npix_x,
    const int npix_y, const int npart, const double threshold, const int kdim,
    double *img, const int nthreads) {

  /* Loop over positions including the sed */
#pragma omp parallel for num_threads(nthreads)
  for (int ind = 0; ind < npart; ind++) {

    /* Get this particles smoothing length and position */
    const double smooth_length = smoothing_lengths[ind];
    const double x = xs[ind];
    const double y = ys[ind];

    /* Calculate the pixel coordinates of this particle. */
    const int i = x / res;
    const int j = y / res;

    /* How many pixels are in the smoothing length? Add some buffer. */
    const int delta_pix = ceil(smooth_length / res) + 1;

    /* How many pixels along kernel axis? */
    const int kernel_cdim = 2 * delta_pix + 1;

    /* Create an empty kernel for this particle. */
    double *part_kernel =
        synth_malloc(kernel_cdim * kernel_cdim * sizeof(double), "part_kernel");

    /* Track the kernel sum for normalisation. */
    double kernel_sum = 0;

    /* Loop over a square aperture around this particle */
    for (int ii = i - delta_pix; ii <= i + delta_pix; ii++) {

      /* Skip out of bounds pixels. */
      if (ii < 0 || ii >= npix_x)
        continue;

      /* Compute the x separation */
      const double x_dist = (ii * res) + (res / 2) - x;

      for (int jj = j - delta_pix; jj <= j + delta_pix; jj++) {

        /* Skip out of bounds pixels. */
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

      /* Skip out of bounds pixels. */
      if (ii < 0 || ii >= npix_x)
        continue;

      for (int jj = j - delta_pix; jj <= j + delta_pix; jj++) {

        /* Skip out of bounds pixels. */
        if (jj < 0 || jj >= npix_y)
          continue;

        /* Get the pixel coordinates in the kernel */
        const int iii = ii - (i - delta_pix);
        const int jjj = jj - (j - delta_pix);

        /* Skip empty pixels. */
        if (part_kernel[iii * kernel_cdim + jjj] == 0)
          continue;

#pragma omp critical
        {
          /* Loop over the wavelength axis. */
          img[jj + npix_y * ii] +=
              part_kernel[iii * kernel_cdim + jjj] * pix_values[ind];
        }
      }
    }

    free(part_kernel);
  }
}
#endif

/**
 * @brief Function to populate an image from particle data and a kernel.
 *
 * This is a wrapper function that calls the serial or parallel version of the
 * function depending on the number of threads requested or whether OpenMP is
 * available.
 *
 * @param pix_values: The particle data to be sorted into pixels
 *                    (luminosity/flux/mass etc.).
 * @param smoothing_lengths: The stellar particle smoothing lengths.
 * @param xs: The x coordinates of the particles.
 * @param ys: The y coordinates of the particles.
 * @param kernel: The kernel data (integrated along the z axis and softed by
 *               impact parameter).
 * @param res: The pixel resolution.
 * @param npix_x: The number of pixels along the x axis.
 * @param npix_y: The number of pixels along the y axis.
 * @param npart: The number of particles.
 * @param threshold: The threshold of the SPH kernel.
 * @param kdim: The number of elements in the kernel.
 * @param img: The image to be populated.
 * @param nthreads: The number of threads to use.
 */
void populate_smoothed_image(const double *pix_values,
                             const double *smoothing_lengths, const double *xs,
                             const double *ys, const double *kernel,
                             const double res, const int npix_x,
                             const int npix_y, const int npart,
                             const double threshold, const int kdim,
                             double *img, const int nthreads) {

  double start = tic();

#ifdef WITH_OPENMP
  if (nthreads > 1) {
    populate_smoothed_image_parallel(pix_values, smoothing_lengths, xs, ys,
                                     kernel, res, npix_x, npix_y, npart,
                                     threshold, kdim, img, nthreads);
  } else {
    populate_smoothed_image_serial(pix_values, smoothing_lengths, xs, ys,
                                   kernel, res, npix_x, npix_y, npart,
                                   threshold, kdim, img);
  }
#else
  populate_smoothed_image_serial(pix_values, smoothing_lengths, xs, ys, kernel,
                                 res, npix_x, npix_y, npart, threshold, kdim,
                                 img);
#endif
  toc("Populating smoothed image", start);
}

/**
 * @brief Function to compute an image from particle data and a kernel.
 *
 * The SPH kernel of a particle (integrated along the z axis) is used to
 * calculate the pixel weight for all pixels within a stellar particles
 * kernel. Once the kernel value is found at a pixels position the pixel value
 * is added multiplied by the kernels weight.
 *
 * NOTE: the implementation uses the exact position of a particle, thus
 * accounting for sub pixel positioning.
 *
 * @param np_pix_values: The particle data to be sorted into pixels
 *                       (luminosity/flux/mass etc.).
 * @param np_smoothing_lengths: The stellar particle smoothing lengths.
 * @param np_xs: The x coordinates of the particles.
 * @param np_ys: The y coordinates of the particles.
 * @param np_kernel: The kernel data (integrated along the z axis and softed
 * by impact parameter).
 * @param res: The pixel resolution.
 * @param npix: The number of pixels along an axis.
 * @param npart: The number of particles.
 * @param threshold: The threshold of the SPH kernel.
 * @param kdim: The number of elements in the kernel.
 * @param nthreads: The number of threads to use.
 */
PyObject *make_img(PyObject *self, PyObject *args) {

  /* We don't need the self argument but it has to be there. Tell the compiler
   * we don't care. */
  (void)self;

  double res, threshold;
  int npix_x, npix_y, npart, kdim, nthreads;
  PyArrayObject *np_pix_values, *np_kernel;
  PyArrayObject *np_smoothing_lengths, *np_xs, *np_ys;

  if (!PyArg_ParseTuple(args, "OOOOOdiiidii", &np_pix_values,
                        &np_smoothing_lengths, &np_xs, &np_ys, &np_kernel, &res,
                        &npix_x, &npix_y, &npart, &threshold, &kdim, &nthreads))
    return NULL;

  /* Get pointers to the actual data. */
  const double *pix_values = extract_data_double(np_pix_values, "pix_values");
  const double *smoothing_lengths =
      extract_data_double(np_smoothing_lengths, "smoothing_lengths");
  const double *xs = extract_data_double(np_xs, "xs");
  const double *ys = extract_data_double(np_ys, "ys");
  const double *kernel = extract_data_double(np_kernel, "kernel");

  /* Allocate the image.. */
  const int npix = npix_x * npix_y;
  double *img = synth_malloc(npix * sizeof(double), "image");

  /* Populate the image. */
  populate_smoothed_image(pix_values, smoothing_lengths, xs, ys, kernel, res,
                          npix_x, npix_y, npart, threshold, kdim, img,
                          nthreads);

  /* Construct a numpy python array to return the IFU. */
  npy_intp dims[3] = {npix_x, npix_y};
  PyArrayObject *out_img =
      (PyArrayObject *)PyArray_SimpleNewFromData(2, dims, NPY_FLOAT64, img);

  return Py_BuildValue("N", out_img);
}

static PyMethodDef ImageMethods[] = {
    {"make_img", (PyCFunction)make_img, METH_VARARGS,
     "Method for smoothing particles into an image."},
    {NULL, NULL, 0, NULL},
};

/* Make this importable. */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "image",                                  /* m_name */
    "A module to make images from particles", /* m_doc */
    -1,                                       /* m_size */
    ImageMethods,                             /* m_methods */
    NULL,                                     /* m_reload */
    NULL,                                     /* m_traverse */
    NULL,                                     /* m_clear */
    NULL,                                     /* m_free */
};

PyMODINIT_FUNC PyInit_image(void) {
  PyObject *m = PyModule_Create(&moduledef);
  import_array();
  return m;
}
