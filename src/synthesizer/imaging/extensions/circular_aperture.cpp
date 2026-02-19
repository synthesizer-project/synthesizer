/******************************************************************************
 * C extension for calculating the overlap area between a circle and a pixel.
 *
 * This is heavily adapted from the circular_aperture module in the photutils
 * package. The original code is available at:
 * github.com/astropy/photutils/tree/ee0dd3ee7b4bc0659f38262b558ca0d6d6836bf5
 * The original code is licensed under a BSD 3-Clause License, which is
 * included in the LICENSE file in the linked repository.
 *****************************************************************************/

/* C includes */
#include <math.h>
#include <stdlib.h>

#ifdef WITH_OPENMP
#include <omp.h>
#endif

/* Python includes */
#define PY_ARRAY_UNIQUE_SYMBOL SYNTHESIZER_ARRAY_API
#define NO_IMPORT_ARRAY
#include "../../extensions/numpy_init.h"
#include <Python.h>

/* Local includes */
#include "../../extensions/data_types.h"
#include "../../extensions/property_funcs.h"
#include "../../extensions/timers.h"
#ifdef ATOMIC_TIMING
#include "../../extensions/timers_init.h"
#endif

/* It's possible we don't have PI... if so define it. */
#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

/**
 * @brief Calculates the square root of a number, treating small negative values
 * as zero.
 *
 * @param x - The input value.
 * @return The square root of x if x >= 0, otherwise 0.
 */
static Float floor_sqrt(Float x) {
  if (x > 0) {
    return sqrt(x);
  } else {
    return 0;
  }
}

/**
 * @brief Calculates the Euclidean distance between two points in a 2D plane.
 *
 * @param x1, y1 - Coordinates of the first point.
 * @param x2, y2 - Coordinates of the second point.
 * @return The Euclidean distance between the two points.
 */
static Float distance(Float x1, Float y1, Float x2, Float y2) {
  return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

/**
 * @brief Calculates the area of a circular arc defined by two points on the
 * circle's perimeter.
 *
 * @param x1, y1 - Coordinates of the first point on the circle's perimeter.
 * @param x2, y2 - Coordinates of the second point on the circle's perimeter.
 * @param r - Radius of the circle.
 * @return The area of the circular arc.
 */
static Float area_arc(Float x1, Float y1, Float x2, Float y2, Float r) {
  Float a = distance(x1, y1, x2, y2);
  Float theta = 2.0 * asin(0.5 * a / r);
  return 0.5 * r * r * (theta - sin(theta));
}

/**
 * @brief Calculates the area of a triangle given its three vertices.
 *
 * @param x1, y1 - Coordinates of the first vertex.
 * @param x2, y2 - Coordinates of the second vertex.
 * @param x3, y3 - Coordinates of the third vertex.
 * @return The area of the triangle.
 */
static Float area_triangle(Float x1, Float y1, Float x2, Float y2,
                            Float x3, Float y3) {
  return 0.5 * fabs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2));
}

/**
 * @brief Core function for calculating the area of overlap between a circle and
 * a rectangle.
 *
 * @param xmin, ymin - Lower-left corner of the rectangle.
 * @param xmax, ymax - Upper-right corner of the rectangle.
 * @param r - Radius of the circle.
 * @return The area of overlap between the circle and the rectangle.
 */
static Float circular_overlap_core(Float xmin, Float ymin, Float xmax,
                                    Float ymax, Float r) {
  Float area, d1, d2, x1, x2, y1, y2;

  if (xmin * xmin + ymin * ymin > r * r) {
    area = 0.0;
  } else if (xmax * xmax + ymax * ymax < r * r) {
    area = (xmax - xmin) * (ymax - ymin);
  } else {
    area = 0.0;
    d1 = floor_sqrt(xmax * xmax + ymin * ymin);
    d2 = floor_sqrt(xmin * xmin + ymax * ymax);
    if (d1 < r && d2 < r) {
      x1 = floor_sqrt(r * r - ymax * ymax);
      y1 = ymax;
      x2 = xmax;
      y2 = floor_sqrt(r * r - xmax * xmax);
      area = ((xmax - xmin) * (ymax - ymin) -
              area_triangle(x1, y1, x2, y2, xmax, ymax) +
              area_arc(x1, y1, x2, y2, r));
    } else if (d1 < r) {
      x1 = xmin;
      y1 = floor_sqrt(r * r - xmin * xmin);
      x2 = xmax;
      y2 = floor_sqrt(r * r - xmax * xmax);
      area = (area_arc(x1, y1, x2, y2, r) +
              area_triangle(x1, y1, x1, ymin, xmax, ymin) +
              area_triangle(x1, y1, x2, ymin, x2, y2));
    } else if (d2 < r) {
      x1 = floor_sqrt(r * r - ymin * ymin);
      y1 = ymin;
      x2 = floor_sqrt(r * r - ymax * ymax);
      y2 = ymax;
      area = (area_arc(x1, y1, x2, y2, r) +
              area_triangle(x1, y1, xmin, y1, xmin, ymax) +
              area_triangle(x1, y1, xmin, y2, x2, y2));
    } else {
      x1 = floor_sqrt(r * r - ymin * ymin);
      y1 = ymin;
      x2 = xmin;
      y2 = floor_sqrt(r * r - xmin * xmin);
      area = (area_arc(x1, y1, x2, y2, r) +
              area_triangle(x1, y1, x2, y2, xmin, ymin));
    }
  }

  return area;
}

/**
 * @brief Calculates the exact area of overlap between a circle and a rectangle.
 *
 * @param xmin, ymin - Lower-left corner of the pixel.
 * @param xmax, ymax - Upper-right corner of the pixel.
 * @param r - Radius of the circle.
 * @return The exact area of overlap between the circle and the rectangle.
 */
static Float circular_overlap_single_exact(Float pix_xmin, Float pix_ymin,
                                            Float pix_xmax, Float pix_ymax,
                                            Float r) {
  if (0.0 <= pix_xmin) {
    if (0.0 <= pix_ymin) {
      return circular_overlap_core(pix_xmin, pix_ymin, pix_xmax, pix_ymax, r);
    } else if (0.0 >= pix_ymax) {
      return circular_overlap_core(-pix_ymax, pix_xmin, -pix_ymin, pix_xmax, r);
    } else {
      return circular_overlap_single_exact(pix_xmin, pix_ymin, pix_xmax, 0.0,
                                           r) +
             circular_overlap_single_exact(pix_xmin, 0.0, pix_xmax, pix_ymax,
                                           r);
    }
  } else if (0.0 >= pix_xmax) {
    if (0.0 <= pix_ymin) {
      return circular_overlap_core(-pix_xmax, pix_ymin, -pix_xmin, pix_ymax, r);
    } else if (0.0 >= pix_ymax) {
      return circular_overlap_core(-pix_xmax, -pix_ymax, -pix_xmin, -pix_ymin,
                                   r);
    } else {
      return circular_overlap_single_exact(pix_xmin, pix_ymin, pix_xmax, 0.0,
                                           r) +
             circular_overlap_single_exact(pix_xmin, 0.0, pix_xmax, pix_ymax,
                                           r);
    }
  } else {
    if (0.0 <= pix_ymin) {
      return circular_overlap_single_exact(pix_xmin, pix_ymin, 0.0, pix_ymax,
                                           r) +
             circular_overlap_single_exact(0.0, pix_ymin, pix_xmax, pix_ymax,
                                           r);
    }
    if (0.0 >= pix_ymax) {
      return circular_overlap_single_exact(pix_xmin, pix_ymin, 0.0, pix_ymax,
                                           r) +
             circular_overlap_single_exact(0.0, pix_ymin, pix_xmax, pix_ymax,
                                           r);
    } else {
      return circular_overlap_single_exact(pix_xmin, pix_ymin, 0.0, 0.0, r) +
             circular_overlap_single_exact(0.0, pix_ymin, pix_xmax, 0.0, r) +
             circular_overlap_single_exact(pix_xmin, 0.0, 0.0, pix_ymax, r) +
             circular_overlap_single_exact(0.0, 0.0, pix_xmax, pix_ymax, r);
    }
  }
}

/**
 * @brief Calculates the signal inside the aperture in serial.
 *
 * @param res - Pixel resolution.
 * @param xmin, ymin - Lower-left corner of the image.
 * @param r - Radius of the aperture.
 * @param nx, ny - Image dimensions.
 * @param img - Image data.
 * @param pixel_radius - Radius of a pixel.
 *
 * @return The signal inside the aperture.
 */
static Float calculate_overlap_serial(const Float res, const Float xmin,
                                       const Float ymin, const Float r,
                                       const int nx, const int ny,
                                       const Float *img,
                                       const Float pixel_radius) {

  /* Define the signal in aperture. */
  Float signal = 0.0;

  /* Loop over pixels and accumalate the pixel weight mulitiplied by pixel
   * values. */
  for (int i = 0; i < nx; i++) {
    Float pxmin = xmin + i * res;
    Float pxcen = pxmin + 0.5 * res;
    Float pxmax = pxmin + res;
    for (int j = 0; j < ny; j++) {
      Float pymin = ymin + j * res;
      Float pycen = pymin + 0.5 * res;
      Float pymax = pymin + res;

      Float frac;

      Float d = sqrt(pxcen * pxcen + pycen * pycen);
      if (d < r - pixel_radius) {
        frac = 1.0;
      } else if (d < r + pixel_radius) {
        frac = circular_overlap_single_exact(pxmin, pymin, pxmax, pymax, r) /
               (res * res);
      } else {
        /* Nothing to do, pixel is outside the aperture. */
        continue;
      }
      signal += frac * img[i * ny + j];
    }
  }

  return signal;
}

/**
 * @brief Calculates the signal inside the aperture in parallel.
 *
 * @param res - Pixel resolution.
 * @param xmin, ymin - Lower-left corner of the image.
 * @param r - Radius of the aperture.
 * @param nx, ny - Image dimensions.
 * @param img - Image data.
 * @param pixel_radius - Radius of a pixel.
 * @param nthreads - Number of threads to use.
 *
 * @return The signal inside the aperture.
 */
static Float calculate_overlap_omp(const Float res, const Float xmin,
                                    const Float ymin, const Float r,
                                    const int nx, const int ny,
                                    const Float *img,
                                    const Float pixel_radius,
                                    const int nthreads) {

  /* Define the signal in aperture. */
  Float signal = 0.0;

  /* Loop over pixels and accumalate the pixel weight mulitiplied by pixel
   * values. */
#pragma omp parallel for num_threads(nthreads) reduction(+ : signal)           \
    schedule(dynamic)
  for (int i = 0; i < nx; i++) {
    Float pxmin = xmin + i * res;
    Float pxcen = pxmin + 0.5 * res;
    Float pxmax = pxmin + res;
    for (int j = 0; j < ny; j++) {
      Float pymin = ymin + j * res;
      Float pycen = pymin + 0.5 * res;
      Float pymax = pymin + res;

      Float frac;

      Float d = sqrt(pxcen * pxcen + pycen * pycen);
      if (d < r - pixel_radius) {
        frac = 1.0;
      } else if (d < r + pixel_radius) {
        frac = circular_overlap_single_exact(pxmin, pymin, pxmax, pymax, r) /
               (res * res);
      } else {
        /* Nothing to do, pixel is outside the aperture. */
        continue;
      }
      signal += frac * img[i * ny + j];
    }
  }

  return signal;
}

/**
 * @brief Calculates the signal inside the aperture.
 *
 * @param res - Pixel resolution.
 * @param r - Radius of the aperture.
 * @param nx, ny - Image dimensions.
 * @param img - Image data.
 * @param cent - Centre of the aperture.
 * @param nthreads - Number of threads to use.
 *
 * @return The signal inside the aperture.
 */
static Float calculate_overlap(const Float res, const Float r, const int nx,
                                const int ny, const Float *img,
                                const Float cent[2], const int nthreads) {

  /* Define some helpful variables. */
  Float xmin = -cent[0] * res;
  Float ymin = -cent[1] * res;
  Float pixel_radius = 0.5 * sqrt(2) * res;

#ifdef WITH_OPENMP
  if (nthreads > 1) {
    return calculate_overlap_omp(res, xmin, ymin, r, nx, ny, img, pixel_radius,
                                 nthreads);
  } else {
    return calculate_overlap_serial(res, xmin, ymin, r, nx, ny, img,
                                    pixel_radius);
  }
#else
  return calculate_overlap_serial(res, xmin, ymin, r, nx, ny, img,
                                  pixel_radius);
#endif
}

/**
 * @brief Calculates the area of overlap between a circle and a pixel grid.
 *
 * @param res - Pixel resolution.
 * @param nx, ny - Grid dimensions.
 * @param r - Radius of the circle.
 * @param np_img - Numpy array representing the image.
 * @param np_cent - Numpy array representing the centre of the circle.
 * @param nthreads - Number of threads to use.
 *
 * @return A numpy array representing the fraction of overlap for each grid
 * cell.
 */
static PyObject *calculate_circular_overlap(PyObject *self, PyObject *args) {

  double start = tic();

  /* We don't need the self argument but it has to be there. Tell the compiler
   * we don't care. */
  (void)self;

  double r_in, res_in;
  int nx, ny, nthreads;
  PyArrayObject *np_img, *np_cent;
  Float signal = 0;

  if (!PyArg_ParseTuple(args, "diidO!O!i", &res_in, &nx, &ny, &r_in,
                        &PyArray_Type, &np_img, &PyArray_Type, &np_cent,
                        &nthreads)) {
    return NULL;
  }

  Float r = (Float)r_in;
  Float res = (Float)res_in;

  /* Unpack the image and centre from the numpy array. */
  const Float *img = extract_data_float(np_img, "img");
  const Float *cent = extract_data_float(np_cent, "cent");
  if (img == NULL || cent == NULL) {
    return NULL;
  }

  /* Is the aperture smaller than a pixel? */
  if (r < 0.5 * res) {
    /* Compute the areas. */
    const Float app_area = M_PI * r * r;
    const Float pix_area = res * res;

    /* Compute the signal based on the fractional area of the aperture. */
    signal = app_area / pix_area * img[(int)cent[0] * ny + (int)cent[1]];
  }

  /* Calculate the signal inside the aperture. */
  if (signal == 0) {
    signal = calculate_overlap(res, r, nx, ny, img, cent, nthreads);
  }

  /* Construct the ouput. */
  PyObject *np_signal = Py_BuildValue("d", (double)signal);

  toc("Calculating signal in aperture", start);

  return np_signal;
}

/* Define the methods for the module */
static PyMethodDef CircularOverlapMethods[] = {
    {"calculate_circular_overlap", calculate_circular_overlap, METH_VARARGS,
     "Calculate the overlap area between a circle and a pixel grid."},
    {NULL, NULL, 0, NULL}};

/* Define the module */
static struct PyModuleDef circularoverlapmodule = {
    PyModuleDef_HEAD_INIT,
    "circular_aperture", // name of module
    "Module for calculating the overlap area between a circle and a pixel "
    "grid.",
    -1,
    CircularOverlapMethods,
    NULL,
    NULL,
    NULL,
    NULL};

/* Define the initialisation function */
PyMODINIT_FUNC PyInit_circular_aperture(void) {
  if (numpy_import() < 0) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to import numpy.");
    return NULL;
  }
  PyObject *m = PyModule_Create(&circularoverlapmodule);
  if (m == NULL)
    return NULL;
#ifdef ATOMIC_TIMING
  if (import_toc_capsule() < 0) {
    Py_DECREF(m);
    return NULL;
  }
#endif
  return m;
}
