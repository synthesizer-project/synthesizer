/******************************************************************************
 * C functions for calculating the value of a stellar particles SPH kernel 
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/ndarraytypes.h>
#include <numpy/ndarrayobject.h>

/* Define a macro to handle that bzero is non-standard. */
#define bzero(b,len) (memset((b), '\0', (len)), (void) 0)

/* Need to define the kernel locally because Python doesn't support header
 * files properly without compiling the C. */
double quintic(double r) {

  if (r < 1 / 3) {
    return 27.0 * (6.4457752 * r * r * r * r * (1.0 - r) -
                   1.4323945 * r * r + 0.17507044);
  } else if (r < 2 / 3) {
    return 27.0 * (3.2228876 * r * r * r * r * (r - 3.0) +
                   10.7429587 * r * r * r - 5.01338071 * r * r +
                   0.5968310366 * r + 0.1352817016);
  } else if (r < 1) {
    return 27.0 * 0.64457752 * (-r * r * r * r * r + 5.0 * r * r * r * r -
                                10.0 * r * r * r + 10.0 * r * r -
                                5.0 * r + 1.0);
  } else {
    return 0;
  }
}

/**
 * @brief 
 * 
 * @param
 */
void sph_kernel_loop(int i, int j, int k, int delta_pix,
                     int npix, double x, double y, double z, double res, 
                     double smooth_length, double *kernel) {
  
  /* Declare variables. */
  int ii, jj, kk, iii, jjj;
  double x_dist, y_dist, z_dist, dist, kernel_val;
      
  /* Define a variable to calculate the kernel sum on the fly. */
  double kernel_sum = 0;

  /* How many pixels along kernel axis? */
  int kernel_cdim = 2 * delta_pix + 1;

  /* Loop over a square aperture around this particle 
   * NOTE: This includes "pixels" in front of and behind the image 
   *       plane since the kernel is by defintion 3D. 
   * TODO: Would be considerably m||e accurate to integrate over the 
   *       kernel in z axis since this is not quantised into pixels 
   *       like the axes in the image plane. */
  for (int ii = i - delta_pix; ii <= i + delta_pix; ii++) {
            
    /* Compute the x separation */
    x_dist = (ii * res) + (res / 2) - x;

    for (int jj = j - delta_pix; jj <= j + delta_pix; jj++) {

      /* Compute the y separation */
      y_dist = (jj * res) + (res / 2) - y;

      for (int kk = k - delta_pix; kk <= k + delta_pix; kk++) {

        /* Skip if outside of image */
        if (ii < 0 || ii >= npix || jj < 0 || jj >= npix) continue;

        /* Compute the z separation */
        z_dist = (kk * res) + (res / 2) - z;

        /* Compute the distance between the centre of this pixel
         * and the particle. */
        dist = sqrt((x_dist * x_dist) + (y_dist * y_dist) + (z_dist * z_dist));

        /* Get the pixel co||dinates in the kernel */
        iii = ii - (i - delta_pix);
        jjj = jj - (j - delta_pix);

        /* Get the value of the kernel here */
        kernel_val = quintic(dist / smooth_length);
        kernel[iii * kernel_cdim + jjj] += kernel_val;
        kernel_sum += kernel_val;
      }
    }
  }

  /* Normalise the kernel */
  if (kernel_sum > 0) {
    for (int n = 0; n < kernel_cdim * kernel_cdim; n++) {
      kernel[n] /= kernel_sum;
    }
  }
}

/**
 * @brief 
 * 
 * @param
 */
PyObject *make_ifu(PyObject *self, PyObject *args) {

  const double res;
  const int npix, npart, nlam;
  PyArrayObject *np_sed_values;
  PyArrayObject *np_smoothing_lengths, *np_xs, *np_ys, *np_zs;

  if(!PyArg_ParseTuple(args, "OOOOOdiii", &np_sed_values,
                       &np_smoothing_lengths, &np_xs,
                       &np_ys, &np_zs, &res, &npix, &npart, &nlam))
    return NULL;

  /* Get pointers to the actual data. */
  double *sed_values = PyArray_DATA(np_sed_values);
  double *smoothing_lengths = PyArray_DATA(np_smoothing_lengths);
  double *xs = PyArray_DATA(np_xs);
  double *ys = PyArray_DATA(np_ys);
  double *zs = PyArray_DATA(np_zs);
  
  /* Allocate IFU. */
  double *ifu = malloc(npix * npix * nlam * sizeof(double));
  bzero(ifu, npix * npix * nlam * sizeof(double));

  /* Loop over positions including the sed */
  for (int ind = 0; ind < npart; ind++) {
    
    /* Get this particles smoothing length and position */
    const double smooth_length = smoothing_lengths[ind];
    const double x = xs[ind];
    const double y = ys[ind];
    const double z = zs[ind];

    /* Calculate the pixel coordinates of this particle. */
    int i = x / res;
    int j = y / res;
    int k = z / res;

    /* How many pixels are in the smoothing length? Add some buffer. */
    int delta_pix = ceil(smooth_length / res) + 1;

    /* How many pixels are in the kernel? */
    int kernel_npix = (2 * delta_pix + 1) * (2 * delta_pix + 1);

    /* Create an empty kernel for this particle. */
    double *kernel = malloc(kernel_npix * sizeof(double));
    bzero(kernel, kernel_npix * sizeof(double));

    /* How many pixels along a kernel axis? */
    int kernel_cdim = 2 * delta_pix + 1;

    /* Calculate this particles sph kernel */
    sph_kernel_loop(i, j, k, delta_pix, npix, x, y, z,
                    res, smooth_length, kernel);

    /* Add each pixel's contribution */
    for (int ii = i - delta_pix; ii < i + delta_pix + 1; ii++) {
      
      /* Skip if outside of image */
      if (ii < 0 || ii >= npix) continue;
        
      for (int jj = j - delta_pix; jj < j + delta_pix + 1; jj++) {
              
        /* Skip if outside of image */
        if (jj < 0 || jj >= npix) continue;
      
        /* Get the kernel coordinates */
        int iii = ii - (i - delta_pix);
        int jjj = jj - (j - delta_pix);

        /* Loop over the wavelength axis. */
        for (int ilam = 0; ilam < nlam; ilam++) {
          ifu[ilam + nlam * (jj + npix * ii)] +=
            kernel[iii * kernel_cdim + jjj] * sed_values[(ind * nlam) + ilam];
        }
      }
    }

    /* Free this kernel ready for the next. */
    free(kernel);
  }

  /* Construct a numpy python array to return the IFU. */
  npy_intp dims[3] = {npix, npix, nlam};
  PyArrayObject *out_ifu =
    (PyArrayObject *) PyArray_SimpleNewFromData(3, dims, NPY_FLOAT64, ifu);
  
  return Py_BuildValue("N", out_ifu);
}

/**
 * @brief 
 * 
 * @param
 */
PyObject *make_img(PyObject *self, PyObject *args) {

  const double res;
  const int npix, npart;
  PyArrayObject *np_pix_values;
  PyArrayObject *np_smoothing_lengths, *np_xs, *np_ys, *np_zs;

  if(!PyArg_ParseTuple(args, "OOOOOdii", &np_pix_values,
                       &np_smoothing_lengths, &np_xs,
                       &np_ys, &np_zs, &res, &npix, &npart))
    return NULL;

  /* Get pointers to the actual data. */
  double *pix_values = PyArray_DATA(np_pix_values);
  double *smoothing_lengths = PyArray_DATA(np_smoothing_lengths);
  double *xs = PyArray_DATA(np_xs);
  double *ys = PyArray_DATA(np_ys);
  double *zs = PyArray_DATA(np_zs);

  /* Allocate IFU. */
  double *img = malloc(npix * npix * sizeof(double));
  bzero(img, npix * npix * sizeof(double));

  /* Loop over positions including the sed */
  for (int ind = 0; ind < npart; ind++) {
    
    /* Get this particles smoothing length and position */
    const double smooth_length = smoothing_lengths[ind];
    const double x = xs[ind];
    const double y = ys[ind];
    const double z = zs[ind];

    /* Calculate the pixel coordinates of this particle. */
    int i = x / res;
    int j = y / res;
    int k = z / res;

    /* How many pixels are in the smoothing length? Add some buffer. */
    int delta_pix = ceil(smooth_length / res) + 1;

    /* How many pixels are in the kernel? */
    int kernel_npix = (2 * delta_pix + 1) * (2 * delta_pix + 1);

    /* Create an empty kernel for this particle. */
    double *kernel = malloc(kernel_npix * sizeof(double));
    bzero(kernel, kernel_npix * sizeof(double));

    /* How many pixels along a kernel axis? */
    int kernel_cdim = 2 * delta_pix + 1;

    /* Calculate this particles sph kernel */
    sph_kernel_loop(i, j, k, delta_pix, npix, x, y, z,
                    res, smooth_length, kernel);

    /* Add each pixel's contribution */
    for (int ii = i - delta_pix; ii < i + delta_pix + 1; ii++) {
      
      /* Skip if outside of image */
      if (ii < 0 || ii >= npix) continue;
        
      for (int jj = j - delta_pix; jj < j + delta_pix + 1; jj++) {
              
        /* Skip if outside of image */
        if (jj < 0 || jj >= npix) continue;
      
        /* Get the kernel coordinates */
        int iii = ii - (i - delta_pix);
        int jjj = jj - (j - delta_pix);

        /* Loop over the wavelength axis. */
        img[jj + npix * ii] +=
          kernel[iii * kernel_cdim + jjj] * pix_values[ind];
      }
    }

    /* Free this kernel ready for the next. */
    free(kernel);
  }

  /* Construct a numpy python array to return the IFU. */
  npy_intp dims[2] = {npix, npix};
  PyArrayObject *out_img =
    (PyArrayObject *) PyArray_SimpleNewFromData(2, dims, NPY_FLOAT64, img);
  
  return Py_BuildValue("N", out_img);
}

static PyMethodDef ImageMethods[] = {
  {"make_ifu", make_ifu, METH_VARARGS,
   "Method for smoothing particles into a spectral cube."},
  {"make_img", make_img, METH_VARARGS,
   "Method for smoothing particles into an image."},
  {NULL, NULL, 0, NULL}, 
};

/* Make this importable. */
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "sph_kernel_calc",                                    /* m_name */
        "A module to make IFUs and images from particles",    /* m_doc */
        -1,                                                   /* m_size */
        ImageMethods,                                         /* m_methods */
        NULL,                                                 /* m_reload */
        NULL,                                                 /* m_traverse */
        NULL,                                                 /* m_clear */
        NULL,                                                 /* m_free */
    };

PyMODINIT_FUNC PyInit_sph_kernel_calc(void) {
    PyObject *m = PyModule_Create(&moduledef);
    import_array();
    return m;
}
