/******************************************************************************
 * C functions for calculating the value of a stellar particles SPH kernel
 *****************************************************************************/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>

/* Define a macro to handle that bzero is non-standard. */
#ifndef bzero
#define bzero(b, len) (memset((b), '\0', (len)), (void)0)
#endif

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
 * @param npix_x: The number of pixels along the x axis.
 * @param npix_y: The number of pixels along the y axis.
 * @param npart: The number of particles.
 * @param nlam: The number of wavelength elements in the SEDs.
 */
PyObject *make_data_cube_hist(PyObject *self, PyObject *args) {

  /* We don't need the self argument but it has to be there. Tell the compiler
   * we don't care. */
  (void)self;

  const double res;
  const int npix_x, npix_y, npart, nlam;
  PyArrayObject *np_sed_values;
  PyArrayObject *np_xs, *np_ys;

  if (!PyArg_ParseTuple(args, "OOOdiiii", &np_sed_values, &np_xs, &np_ys, &res,
                        &npix_x, &npix_y, &npart, &nlam))
    return NULL;

  /* Get pointers to the actual data. */
  const double *sed_values = PyArray_DATA(np_sed_values);
  if (sed_values == NULL) {
    PyErr_SetString(PyExc_ValueError, "Failed to extract sed_values.");
    return NULL;
  }
  const double *xs = PyArray_DATA(np_xs);
  if (xs == NULL) {
    PyErr_SetString(PyExc_ValueError, "Failed to extract xs.");
    return NULL;
  }
  const double *ys = PyArray_DATA(np_ys);
  if (ys == NULL) {
    PyErr_SetString(PyExc_ValueError, "Failed to extract ys.");
    return NULL;
  }

  /* Allocate the data cube. */
  const int npix = npix_x * npix_y;
  double *data_cube = malloc(npix * nlam * sizeof(double));
  if (data_cube == NULL) {
    PyErr_SetString(PyExc_ValueError, "Failed to allocate data_cube.");
    return NULL;
  }
  bzero(data_cube, npix * nlam * sizeof(double));

  /* Loop over positions including the sed */
  for (int ind = 0; ind < npart; ind++) {

    /* Get this particles position */
    const double x = xs[ind];
    const double y = ys[ind];

    /* Calculate the pixel coordinates of this particle. */
    int i = x / res;
    int j = y / res;

    /* Skip particles outside the image */
    if (i < 0 || i >= npix_x || j < 0 || j >= npix_y)
      continue;

    /* Loop over the wavelength axis. */
    for (int ilam = 0; ilam < nlam; ilam++) {
      int data_cube_ind = ilam + nlam * (j + npix_y * i);
      int sed_ind = (ind * nlam) + ilam;
      data_cube[data_cube_ind] += sed_values[sed_ind];
    }
  }

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
 * @param npix_x: The number of pixels along the x axis.
 * @param npix_y: The number of pixels along the y axis.
 * @param npart: The number of particles.
 * @param nlam: The number of wavelength elements in the SEDs.
 * @param threshold: The threshold of the SPH kernel.
 * @param kdim: The number of elements in the kernel.
 */
PyObject *make_data_cube_smooth(PyObject *self, PyObject *args) {

  /* We don't need the self argument but it has to be there. Tell the compiler
   * we don't care. */
  (void)self;

  double res, threshold;
  int npix_x, npix_y, npart, nlam, kdim;
  PyArrayObject *np_sed_values, *np_kernel;
  PyArrayObject *np_smoothing_lengths, *np_xs, *np_ys;

  if (!PyArg_ParseTuple(args, "OOOOOdiiiidi", &np_sed_values,
                        &np_smoothing_lengths, &np_xs, &np_ys, &np_kernel, &res,
                        &npix_x, &npix_y, &npart, &nlam, &threshold, &kdim))
    return NULL;

  /* Get pointers to the actual data. */
  const double *sed_values = PyArray_DATA(np_sed_values);
  if (sed_values == NULL) {
    PyErr_SetString(PyExc_ValueError, "Failed to extract sed_values.");
    return NULL;
  }
  const double *smoothing_lengths = PyArray_DATA(np_smoothing_lengths);
  if (smoothing_lengths == NULL) {
    PyErr_SetString(PyExc_ValueError, "Failed to extract smoothing_lengths.");
    return NULL;
  }
  const double *xs = PyArray_DATA(np_xs);
  if (xs == NULL) {
    PyErr_SetString(PyExc_ValueError, "Failed to extract xs.");
    return NULL;
  }
  const double *ys = PyArray_DATA(np_ys);
  if (ys == NULL) {
    PyErr_SetString(PyExc_ValueError, "Failed to extract ys.");
    return NULL;
  }
  const double *kernel = PyArray_DATA(np_kernel);
  if (kernel == NULL) {
    PyErr_SetString(PyExc_ValueError, "Failed to extract kernel.");
    return NULL;
  }

  /* Allocate DATA_CUBE. */
  const int npix = npix_x * npix_y;
  double *data_cube = malloc(npix * nlam * sizeof(double));
  if (data_cube == NULL) {
    PyErr_SetString(PyExc_ValueError, "Failed to allocate data_cube.");
    return NULL;
  }
  bzero(data_cube, npix * nlam * sizeof(double));

  /* Loop over positions including the sed */
  for (int ind = 0; ind < npart; ind++) {

    /* Get this particles smoothing length and position */
    const double smooth_length = smoothing_lengths[ind];
    const double x = xs[ind];
    const double y = ys[ind];

    /* Calculate the pixel coordinates of this particle. */
    int i = x / res;
    int j = y / res;

    /* How many pixels are in the smoothing length? Add some buffer. */
    int delta_pix = ceil(smooth_length / res) + 1;

    /* How many pixels along kernel axis? */
    int kernel_cdim = 2 * delta_pix + 1;

    /* Create an empty kernel for this particle. */
    double *part_kernel = malloc(kernel_cdim * kernel_cdim * sizeof(double));
    if (part_kernel == NULL) {
      PyErr_SetString(PyExc_MemoryError,
                      "Failed to allocate memory for part_kernel.");
      return NULL;
    }
    bzero(part_kernel, kernel_cdim * kernel_cdim * sizeof(double));

    /* Track the kernel sum for normalisation. */
    double kernel_sum = 0;

    /* Loop over a square aperture around this particle */
    for (int ii = i - delta_pix; ii <= i + delta_pix; ii++) {

      /* Skip out of bounds spaxels. */
      if (ii < 0 || ii >= npix_x)
        continue;

      /* Compute the x separation */
      double x_dist = (ii * res) + (res / 2) - x;

      for (int jj = j - delta_pix; jj <= j + delta_pix; jj++) {

        /* Skip out of bounds spaxels. */
        if (jj < 0 || jj >= npix_y)
          continue;

        /* Compute the y separation */
        double y_dist = (jj * res) + (res / 2) - y;

        /* Compute the distance between the centre of this pixel
         * and the particle. */
        double rsqu = (x_dist * x_dist) + (y_dist * y_dist);

        /* Get the pixel coordinates in the kernel */
        int iii = ii - (i - delta_pix);
        int jjj = jj - (j - delta_pix);

        /* Calculate the impact parameter. */
        double q = sqrt(rsqu) / smooth_length;

        /* Skip gas particles outside the kernel. */
        if (q > threshold)
          continue;

        /* Get the value of the kernel at q. */
        int index = kdim * q;
        double kvalue = kernel[index];

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
        int iii = ii - (i - delta_pix);
        int jjj = jj - (j - delta_pix);

        /* Loop over the wavelength axis. */
        for (int ilam = 0; ilam < nlam; ilam++) {
          int data_cube_ind = ilam + nlam * (jj + npix_y * ii);
          int sed_ind = (ind * nlam) + ilam;
          data_cube[data_cube_ind] +=
              part_kernel[iii * kernel_cdim + jjj] * sed_values[sed_ind];
        }
      }
    }

    free(part_kernel);
  }

  /* Construct a numpy python array to return the DATA_CUBE. */
  npy_intp dims[3] = {npix_x, npix_y, nlam};
  PyArrayObject *out_data_cube = (PyArrayObject *)PyArray_SimpleNewFromData(
      3, dims, NPY_FLOAT64, data_cube);

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
