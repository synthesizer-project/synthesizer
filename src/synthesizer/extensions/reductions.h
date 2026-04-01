#pragma once

struct _object;
typedef _object PyObject;

// Reduction prototypes
void reduce_spectra(double *spectra, double *part_spectra, int nlam, int npart,
                    int nthreads);

PyObject *reduce_particle_spectra(PyObject *self, PyObject *args);
