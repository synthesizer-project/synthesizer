#pragma once

typedef struct _object PyObject;

PyObject *scale_spectra_2d(PyObject *self, PyObject *args);
PyObject *apply_separable_attenuation_2d(PyObject *self, PyObject *args);
PyObject *multiply_rows_by_vector_2d(PyObject *self, PyObject *args);
PyObject *multiply_array_by_vector_1d(PyObject *self, PyObject *args);
