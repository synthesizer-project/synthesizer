/******************************************************************************
 * A C module containing helper functions for extracting properties from the
 * numpy objects.
 *****************************************************************************/
#ifndef PROPERTY_FUNCS_H_
#define PROPERTY_FUNCS_H_

/* Python includes */
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>

/* Prototypes */
const double *extract_data_double(PyArrayObject *np_arr, char *name);
const int *extract_data_int(PyArrayObject *np_arr, char *name);
const double **extract_grid_props(PyObject *grid_tuple, int ndim, int *dims);
const double **extract_part_props(PyObject *part_tuple, int ndim, int npart);

#endif // PROPERTY_FUNCS_H_
