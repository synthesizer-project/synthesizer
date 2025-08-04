/******************************************************************************
 * C functions for calculating the value of a stellar particles SPH kernel
 *****************************************************************************/

/* C includes. */
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <math.h>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

/* Python includes. */
#define PY_ARRAY_UNIQUE_SYMBOL SYNTHESIZER_ARRAY_API
#define NO_IMPORT_ARRAY
#include "../../extensions/numpy_init.h"
#include <Python.h>

/* Local includes. */
#include "../../extensions/cpp_to_python.h"
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
 * @param pos: The coordinates of the particles.
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
                                    const double *pos, const double *kernel,
                                    const double res, const int npix_x,
                                    const int npix_y, const int npart,
                                    const double threshold, const int kdim,
                                    double *img, const int nimgs) {

  /* Find the maximum kernel_cdim we'll need. We need this to preallocate the
   * kernel we will populate for each particle. */
  int max_kernel_cdim = 0;
  for (int ind = 0; ind < npart; ind++) {
    const double smooth_length = smoothing_lengths[ind];
    const int delta_pix = ceil(smooth_length / res) + 1;
    const int kernel_cdim_temp = 2 * delta_pix + 1;
    if (kernel_cdim_temp > max_kernel_cdim) {
      max_kernel_cdim = kernel_cdim_temp;
    }
  }

  /* Allocate the particle kernel. */
  double *part_kernel = new double[max_kernel_cdim * max_kernel_cdim];

  /* Loop over positions including the sed */
  for (int ind = 0; ind < npart; ind++) {

    /* Get this particles smoothing length and position */
    const double smooth_length = smoothing_lengths[ind];
    const double x = pos[ind * 3 + 0];
    const double y = pos[ind * 3 + 1];

    /* Calculate the pixel coordinates of this particle. */
    const int i = x / res;
    const int j = y / res;

    /* How many pixels are in the smoothing length? Add some buffer. */
    const int delta_pix = ceil(smooth_length / res) + 1;

    /* How many pixels along kernel axis? */
    const int kernel_cdim = 2 * delta_pix + 1;

    /* Zero the part of the kernel we will use. */
    memset(part_kernel, 0, kernel_cdim * kernel_cdim * sizeof(double));

    /* Track the kernel sum for normalisation. */
    double kernel_sum = 0;

    /* Compute the pixel indices for the kernel. */
    int ii_min = (i - delta_pix) < 0 ? 0 : (i - delta_pix);
    int ii_max = (i + delta_pix) >= npix_x ? npix_x - 1 : (i + delta_pix);
    int jj_min = (j - delta_pix) < 0 ? 0 : (j - delta_pix);
    int jj_max = (j + delta_pix) >= npix_y ? npix_y - 1 : (j + delta_pix);

    /* Calculate the threshold for the kernel. */
    double thresh2 = smooth_length * smooth_length * threshold * threshold;

    /* Loop over a square aperture around this particle */
    for (int ii = ii_min; ii <= ii_max; ii++) {

      /* Compute the x separation */
      const double x_dist = res * (ii + 0.5) - x;

      for (int jj = jj_min; jj <= jj_max; jj++) {

        /* Compute the y separation */
        const double y_dist = res * (jj + 0.5) - y;

        /* Compute the distance between the centre of this pixel
         * and the particle. */
        const double rsqu = (x_dist * x_dist) + (y_dist * y_dist);

        /* Skip particles outside the kernel. */
        if (rsqu > thresh2)
          continue;

        /* Get the pixel coordinates in the kernel */
        const int iii = ii - (i - delta_pix);
        const int jjj = jj - (j - delta_pix);

        /* Calculate the impact parameter. */
        const double q = sqrt(rsqu) / smooth_length;

        /* Get the value of the kernel at q. */
        const int index = kdim * q;
        const double kvalue = kernel[index];

        /* Set the value in the kernel. */
        part_kernel[iii * kernel_cdim + jjj] = kvalue;
        kernel_sum += kvalue;
      }
    }

    /* If the kernel is empty, skip it. */
    if (kernel_sum == 0) {
      continue;
    }

    /* Now add the kernel to the image. */
    for (int ii = ii_min; ii <= ii_max; ii++) {

      for (int jj = jj_min; jj <= jj_max; jj++) {

        /* Get the pixel coordinates in the kernel */
        const int iii = ii - (i - delta_pix);
        const int jjj = jj - (j - delta_pix);

        /* Get the kernel value. */
        const double kvalue = part_kernel[iii * kernel_cdim + jjj] / kernel_sum;

        /* Skip empty pixels. */
        if (kvalue == 0)
          continue;

        /* Add the pixel value to each of the images. */
        for (int nimg = 0; nimg < nimgs; nimg++) {

          /* Get the pixel index in the image (nimg, i, j). */
          const int pix_ind = nimg * npix_x * npix_y + npix_y * ii + jj;

          /* Add the pixel value to this image, pixel values are (Nimg, Npart)
           * in shape. */
          img[pix_ind] += kvalue * pix_values[nimg * npart + ind];
        }
      }
    }
  }
  delete[] part_kernel;
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
 * @param pos: The coordinates of the particles.
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
    const double *pix_values, const double *smoothing_lengths,
    const double *pos, const double *kernel, const double res, const int npix_x,
    const int npix_y, const int npart, const double threshold, const int kdim,
    double *img, const int nimgs, const int nthreads) {

  /* Find the maximum kernel_cdim we'll need */
  int max_kernel_cdim = 0;
  for (int ind = 0; ind < npart; ind++) {
    const double smooth_length = smoothing_lengths[ind];
    const int delta_pix = static_cast<int>(std::ceil(smooth_length / res)) + 1;
    const int kernel_cdim_temp = 2 * delta_pix + 1;
    if (kernel_cdim_temp > max_kernel_cdim) {
      max_kernel_cdim = kernel_cdim_temp;
    }
  }

  const size_t total_pix = static_cast<size_t>(nimgs) * npix_x * npix_y;

  /* Allocate per-thread image arrays */
  std::vector<std::vector<double>> thread_images(nthreads);
  for (int t = 0; t < nthreads; ++t)
    thread_images[t].resize(total_pix, 0.0);

#pragma omp parallel num_threads(nthreads)
  {
    const int tid = omp_get_thread_num();
    double *thread_img = thread_images[tid].data();

    /* Allocate per-thread particle kernel buffer */
    std::vector<double> part_kernel(max_kernel_cdim * max_kernel_cdim, 0.0);

#pragma omp for schedule(dynamic)
    for (int ind = 0; ind < npart; ind++) {

      const double smooth_length = smoothing_lengths[ind];
      const double x = pos[ind * 3 + 0];
      const double y = pos[ind * 3 + 1];

      const int i = static_cast<int>(x / res);
      const int j = static_cast<int>(y / res);

      const int delta_pix =
          static_cast<int>(std::ceil(smooth_length / res)) + 1;
      const int kernel_cdim = 2 * delta_pix + 1;

      std::fill(part_kernel.begin(), part_kernel.end(), 0.0);
      double kernel_sum = 0.0;

      const int ii_min = std::max(i - delta_pix, 0);
      const int ii_max = std::min(i + delta_pix, npix_x - 1);
      const int jj_min = std::max(j - delta_pix, 0);
      const int jj_max = std::min(j + delta_pix, npix_y - 1);

      const double thresh2 =
          smooth_length * smooth_length * threshold * threshold;

      for (int ii = ii_min; ii <= ii_max; ii++) {
        const double x_dist = res * (ii + 0.5) - x;

#pragma omp simd
        for (int jj = jj_min; jj <= jj_max; jj++) {
          const double y_dist = res * (jj + 0.5) - y;
          const double rsqu = x_dist * x_dist + y_dist * y_dist;

          if (rsqu > thresh2)
            continue;

          const int iii = ii - (i - delta_pix);
          const int jjj = jj - (j - delta_pix);

          const double q = std::sqrt(rsqu) / smooth_length;
          const int index = std::min(static_cast<int>(kdim * q), kdim - 1);
          const double kvalue = kernel[index];

          part_kernel[iii * kernel_cdim + jjj] = kvalue;
          kernel_sum += kvalue;
        }
      }

      if (kernel_sum == 0.0)
        continue;

      for (int ii = ii_min; ii <= ii_max; ii++) {

#pragma omp simd
        for (int jj = jj_min; jj <= jj_max; jj++) {

          const int iii = ii - (i - delta_pix);
          const int jjj = jj - (j - delta_pix);
          const double kvalue =
              part_kernel[iii * kernel_cdim + jjj] / kernel_sum;

          if (kvalue == 0.0)
            continue;

          for (int nimg = 0; nimg < nimgs; nimg++) {
            const size_t pix_ind = static_cast<size_t>(nimg) * npix_x * npix_y +
                                   static_cast<size_t>(ii) * npix_y + jj;
            thread_img[pix_ind] += kvalue * pix_values[nimg * npart + ind];
          }
        }
      }
    } // end omp for
  } // end omp parallel

  /* Final reduction into global image */
  for (int t = 0; t < nthreads; ++t) {
    const double *thread_img = thread_images[t].data();
    for (size_t i = 0; i < total_pix; ++i)
      img[i] += thread_img[i];
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
 * @param pos: The coordinates of the particles.
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
                             const double *smoothing_lengths, const double *pos,
                             const double *kernel, const double res,
                             const int npix_x, const int npix_y,
                             const int npart, const double threshold,
                             const int kdim, double *img, const int nimgs,
                             const int nthreads) {

  double start = tic();

#ifdef WITH_OPENMP
  if (nthreads > 1) {
    populate_smoothed_image_parallel(pix_values, smoothing_lengths, pos, kernel,
                                     res, npix_x, npix_y, npart, threshold,
                                     kdim, img, nimgs, nthreads);
  } else {
    populate_smoothed_image_serial(pix_values, smoothing_lengths, pos, kernel,
                                   res, npix_x, npix_y, npart, threshold, kdim,
                                   img, nimgs);
  }
#else
  populate_smoothed_image_serial(pix_values, smoothing_lengths, pos, kernel,
                                 res, npix_x, npix_y, npart, threshold, kdim,
                                 img, nimgs);
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
 * @param np_pos: The coordinates of the particles.
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

  double start_time = tic();

  /* We don't need the self argument but it has to be there. Tell the compiler
   * we don't care. */
  (void)self;

  double res, threshold;
  int npix_x, npix_y, npart, kdim, nthreads, nimgs;
  PyArrayObject *np_pix_values, *np_kernel;
  PyArrayObject *np_smoothing_lengths, *np_pos;

  if (!PyArg_ParseTuple(args, "OOOOdiiidiii", &np_pix_values,
                        &np_smoothing_lengths, &np_pos, &np_kernel, &res,
                        &npix_x, &npix_y, &npart, &threshold, &kdim, &nimgs,
                        &nthreads))
    return NULL;

  /* Get pointers to the actual data. */
  const double *pix_values = extract_data_double(np_pix_values, "pix_values");
  const double *smoothing_lengths =
      extract_data_double(np_smoothing_lengths, "smoothing_lengths");
  const double *pos = extract_data_double(np_pos, "pos");
  const double *kernel = extract_data_double(np_kernel, "kernel");

  toc("Extracting Python data", start_time);

  double out_start = tic();

  /* Create the zeroed image numpy array. */
  npy_intp np_img_dims[3] = {nimgs, npix_x, npix_y};
  PyArrayObject *np_img =
      (PyArrayObject *)PyArray_ZEROS(3, np_img_dims, NPY_DOUBLE, 0);
  double *img = (double *)PyArray_DATA(np_img);

  /* Populate the image. */
  populate_smoothed_image(pix_values, smoothing_lengths, pos, kernel, res,
                          npix_x, npix_y, npart, threshold, kdim, img, nimgs,
                          nthreads);

  toc("Computing smoothed image", start_time);

  return Py_BuildValue("N", np_img);
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
  if (numpy_import() < 0) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to import numpy.");
    return NULL;
  }
  return m;
}
