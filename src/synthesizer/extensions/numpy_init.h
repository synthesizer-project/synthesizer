#ifndef NUMPY_INIT_H_
#define NUMPY_INIT_H_

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>

#pragma GCC diagnostic pop

/* Declare init function. */
int numpy_import();

#endif // NUMPY_INIT_H_
