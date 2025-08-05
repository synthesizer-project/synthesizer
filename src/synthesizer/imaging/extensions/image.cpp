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
#include "../../extensions/octree.h"
#include "../../extensions/property_funcs.h"
#include "../../extensions/timers.h"

#ifdef WITH_OPENMP
#include <omp.h>
#endif

/**
 * @brief Recursively populate a pixel.
 *
 * This will recurse to the leaves of the cell tree, any cells further than the
 * maximum smoothing length from the position will be skipped. Once in the
 * leaves the particles themselves will be checked to see if their SPH kernel
 * overlaps with the line of sight of the star particle.
 *
 * @param c The cell to calculate the surface densities for.
 * @param pix_x The x position of the pixel.
 * @param pix_y The y position of the pixel.
 * @param threshold The threshold for the kernel.
 * @param kdim The dimension of the kernel.
 * @param kernel The kernel to use for the calculation.
 * @param out The output array to populate with the pixel values.
 * @param nimgs The number of images to populate.
 * @param pix_values The pixel values to use for each image.
 */
static void populate_pixel_recursive(const struct cell *c, const double pix_x,
                                     const double pix_y, double threshold,
                                     int kdim, const double *kernel, int npart,
                                     double *out, int nimgs,
                                     const double *pix_values,
                                     const double res) {

  /* Early exit if the projected distance between cells is more than the
   * maximum smoothing length in the cell. */
  if (c->max_sml_squ + 2 * res <
      min_projected_dist2(const_cast<struct cell *>(c), pix_x, pix_y)) {
    return;
  }

  /* Is the cell split? */
  if (c->split) {

    /* Ok, so we recurse... */
    for (int ip = 0; ip < 8; ip++) {
      struct cell *cp = &c->progeny[ip];

      /* Skip empty progeny. */
      if (cp->part_count == 0) {
        continue;
      }

      /* Recurse... */
      populate_pixel_recursive(cp, pix_x, pix_y, threshold, kdim, kernel, npart,
                               out, nimgs, pix_values, res);
    }

  } else {

    /* We're in a leaf if we get here, unpack the particles. */
    int npart = c->part_count;
    struct particle *parts = c->particles;

    /* Loop over the particles adding their contribution to the pixel value. */
    for (int j = 0; j < npart; j++) {

      /* Get the particle. */
      struct particle *part = &parts[j];

      /* Calculate the x and y separations. */
      double dx = part->pos[0] - pix_x - res;
      double dy = part->pos[1] - pix_y - res;

      /* Calculate the impact parameter. */
      double b = sqrt(dx * dx + dy * dy);

      /* Skip if the pixel is outside the kernel. */
      if (b > (threshold * part->sml)) {
        continue;
      }

      /* Find fraction of the smoothing length at this pixel. */
      double q = b / part->sml;

      /* Get the value of the kernel at q (handling q =1). */
      int index = std::min(kdim - 1, static_cast<int>(q * kdim));
      double kvalue = kernel[index];

      /* Finally, compute the pixel value itself across all images. */
      for (int nimg = 0; nimg < nimgs; nimg++) {
        out[nimg] += kvalue * pix_values[nimg * npart + part->index] /
                     (part->sml * part->sml) * res * res;
      }
    }
  }
}

/**
 * @brief Function to populate an image.
 *
 * The SPH kernel of a particle (integrated along the z axis) is used to
 * calculate the pixel weight for all pixels within a stellar particles kernel.
 * Once the kernel value is found at a pixels position the pixel value is added
 * multiplied by the kernels weight.
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
 * @param nimgs: The number of images to populate.
 * @param root: The root of the tree.
 */
void populate_smoothed_image(const double *pix_values, const double *kernel,
                             const double res, const int npix_x,
                             const int npix_y, const int npart,
                             const double threshold, const int kdim,
                             double *img, const int nimgs, struct cell *root,
                             const int nthreads) {

  double start = tic();

#ifdef WITH_OPENMP
  /* If we have multiple threads and OpenMP we can parallelise. */
  if (nthreads > 1) {
    omp_set_num_threads(nthreads);
  } else {
    omp_set_num_threads(1);
  }

#pragma omp parallel for collapse(2) schedule(dynamic)
#endif
  /* Loop over the pixels in the image. */
  for (int i = 0; i < npix_x; i++) {
    for (int j = 0; j < npix_y; j++) {

      /* Calculate the pixel position. */
      const double pix_x = res * (i + 0.5);
      const double pix_y = res * (j + 0.5);

      /* Get the pixel index. */
      int pix_index = i * npix_y + j;

      /* Populate the pixel recursively. */
      populate_pixel_recursive(root, pix_x, pix_y, threshold, kdim, kernel,
                               npart, &img[pix_index * nimgs], nimgs,
                               pix_values, res);
    }
  }

  toc("Populating smoothed image", start);
}

/**
 * @brief Function to compute an image from particle data and a kernel.
 *
 * The SPH kernel of a particle (integrated along the z axis) is used to
 * calculate the pixel weight for all pixels within a stellar particles
 * kernel. Once the kernel value is found at a pixels position the pixel
 * value is added multiplied by the kernels weight.
 *
 * @param np_pix_values: The particle data to be sorted into pixels
 *                       (luminosity/flux/mass etc.).
 * @param np_smoothing_lengths: The stellar particle smoothing lengths.
 * @param np_pos: The coordinates of the particles.
 * @param np_kernel: The kernel data (integrated along the z axis and
 * softed by impact parameter).
 * @param res: The pixel resolution.
 * @param npix: The number of pixels along an axis.
 * @param npart: The number of particles.
 * @param threshold: The threshold of the SPH kernel.
 * @param kdim: The number of elements in the kernel.
 * @param nthreads: The number of threads to use.
 */
PyObject *make_img(PyObject *self, PyObject *args) {

  double start_time = tic();

  /* We don't need the self argument but it has to be there. Tell the
   * compiler we don't care. */
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

  double tree_start = tic();

  /* Allocate cells array. The first cell will be the root and then we
   * will dynamically nibble off cells for the progeny. */
  int ncells = 1;
  struct cell *root = new struct cell;

  /* Consturct the cell tree. */
  construct_cell_tree(pos, smoothing_lengths, smoothing_lengths, npart, root,
                      ncells, MAX_DEPTH, 100);

  toc("Constructing cell tree", tree_start);

  double out_start = tic();

  /* Create the zeroed image numpy array. */
  npy_intp np_img_dims[3] = {npix_x, npix_y, nimgs};
  PyArrayObject *np_img =
      (PyArrayObject *)PyArray_ZEROS(3, np_img_dims, NPY_DOUBLE, 0);
  double *img = (double *)PyArray_DATA(np_img);

  toc("Creating output image", out_start);

  /* Populate the image. */
  populate_smoothed_image(pix_values, kernel, res, npix_x, npix_y, npart,
                          threshold, kdim, img, nimgs, root, nthreads);

  /* Cleanup the cell tree. */
  cleanup_cell_tree(root);

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
