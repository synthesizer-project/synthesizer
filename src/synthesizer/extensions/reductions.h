#pragma once

typedef struct _object PyObject;

// Reduction prototypes
void reduce_spectra(double *spectra, double *part_spectra, int nlam, int npart,
                    int nthreads);

PyObject *reduce_particle_spectra(PyObject *self, PyObject *args);
PyObject *compute_fnu(PyObject *self, PyObject *args);
PyObject *scale_spectra_2d(PyObject *self, PyObject *args);
PyObject *combine_spectra_list_2d(PyObject *self, PyObject *args);
PyObject *apply_separable_attenuation_2d(PyObject *self, PyObject *args);
PyObject *multiply_rows_by_vector_2d(PyObject *self, PyObject *args);
PyObject *multiply_array_by_vector_1d(PyObject *self, PyObject *args);
