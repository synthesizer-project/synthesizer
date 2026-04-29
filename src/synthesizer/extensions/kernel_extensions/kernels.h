/******************************************************************************
 * Common header for the kernel extension modules.
 *
 * This header centralises all includes needed by the kernel table builders.
 *
 * Each source file in the kernel extension must define NO_IMPORT_ARRAY and
 * PY_ARRAY_UNIQUE_SYMBOL before including this header (the bindings file
 * does this explicitly, and this header provides defaults for the others).
 *****************************************************************************/

#ifndef KERNELS_H
#define KERNELS_H

/* C headers. */
#include <math.h>
#include <string.h>

#ifdef WITH_OPENMP
#include <omp.h>
#endif

/* Python headers. */
#ifndef PY_ARRAY_UNIQUE_SYMBOL
#define PY_ARRAY_UNIQUE_SYMBOL SYNTHESIZER_ARRAY_API
#endif
#define NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "numpy_init.h"
#include <Python.h>

/* Local includes. */
#include "integration.h"
#include "kernel_utils.h"
#include "property_funcs.h"
#include "timers.h"
#ifdef ATOMIC_TIMING
#include "timers_init.h"
#endif

/* Kernel function typedef shared across all table builders. */
typedef double (*kernel_func)(double);

/* Declare the Python wrapper functions defined in separate source files.
 * Each compute_* function is implemented in its own .cpp file and
 * referenced by the method table in kernel_bindings.cpp. */
extern PyObject *compute_projected_kernel(PyObject *self, PyObject *args);
extern PyObject *compute_truncated_los_kernel(PyObject *self, PyObject *args);
extern PyObject *compute_overlap_kernel(PyObject *self, PyObject *args);

#endif /* KERNELS_H */
