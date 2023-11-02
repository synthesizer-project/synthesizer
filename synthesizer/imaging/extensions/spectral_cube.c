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
#define bzero(b, len) (memset((b), '\0', (len)), (void)0)

/**
 * @brief Function to compute an IFU from particle data and a kernel.
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
 * @param npix: The number of pixels along an axis.
 * @param npart: The number of particles.
 * @param nlam: The number of wavelength elements in the SEDs.
 * @param threshold: The threshold of the SPH kernel.
 * @param kdim: The number of elements in the kernel.
 */
PyObject *make_ifu(PyObject *self, PyObject *args) {

  const double res, threshold;
  const int npix, npart, nlam, kdim;
  PyArrayObject *np_sed_values, *np_kernel;
  PyArrayObject *np_smoothing_lengths, *np_xs, *np_ys;

  if (!PyArg_ParseTuple(args, "OOOOOdiiidi", &np_sed_values,
                        &np_smoothing_lengths, &np_xs, &np_ys, &np_kernel, &res,
                        &npix, &npart, &nlam, &threshold, &kdim))
    return NULL;

  /* Get pointers to the actual data. */
  const double *sed_values = PyArray_DATA(np_sed_values);
  const double *smoothing_lengths = PyArray_DATA(np_smoothing_lengths);
  const double *xs = PyArray_DATA(np_xs);
  const double *ys = PyArray_DATA(np_ys);
  const double *kernel = PyArray_DATA(np_kernel);

  /* Allocate IFU. */
  double *ifu = malloc(npix * npix * nlam * sizeof(double));
  bzero(ifu, npix * npix * nlam * sizeof(double));

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

    /* Loop over a square aperture around this particle */
    for (int ii = i - delta_pix; ii <= i + delta_pix; ii++) {

      /* Compute the x separation */
      double x_dist = (ii * res) + (res / 2) - x;

      for (int jj = j - delta_pix; jj <= j + delta_pix; jj++) {

        /* Compute the y separation */
        double y_dist = (jj * res) + (res / 2) - y;

        /* Compute the distance between the centre of this pixel
         * and the particle. */
        double rsqu = (x_dist * x_dist) + (y_dist * y_dist);

        /* Get the pixel coordinates in the kernel */
        int iii = ii - (i - delta_pix);
        int jjj = jj - (j - delta_pix);

        /* Calculate the impact parameter. */
        double sml_squ = smooth_length * smooth_length;
        double q = rsqu / sml_squ;

        /* Skip gas particles outside the kernel. */
        if (q > threshold)
          continue;

        /* Get the value of the kernel at q. */
        int index = kdim * q;
        double kvalue = kernel[index];

        /* Loop over the wavelength axis. */
        for (int ilam = 0; ilam < nlam; ilam++) {
          int ifu_ind = ilam + nlam * (jj + npix * ii);
          int sed_ind = (ind * nlam) + ilam;
          ifu[ifu_ind] += kvalue * sed_values[sed_ind];
        }
      }
    }
  }

  /* Construct a numpy python array to return the IFU. */
  npy_intp dims[3] = {npix, npix, nlam};
  PyArrayObject *out_ifu =
      (PyArrayObject *)PyArray_SimpleNewFromData(3, dims, NPY_FLOAT64, ifu);

  return Py_BuildValue("N", out_ifu);
}

static PyMethodDef ImageMethods[] = {
    {"make_ifu", make_ifu, METH_VARARGS,
     "Method for smoothing particles into a spectral cube."},
    {NULL, NULL, 0, NULL},
};

/* Make this importable. */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "spectral_cube",                        /* m_name */
    "A module to make IFUs from particles", /* m_doc */
    -1,                                     /* m_size */
    ImageMethods,                           /* m_methods */
    NULL,                                   /* m_reload */
    NULL,                                   /* m_traverse */
    NULL,                                   /* m_clear */
    NULL,                                   /* m_free */
};

PyMODINIT_FUNC PyInit_spectral_cube(void) {
  PyObject *m = PyModule_Create(&moduledef);
  import_array();
  return m;
}
