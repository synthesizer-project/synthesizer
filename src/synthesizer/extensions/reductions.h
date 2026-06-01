#pragma once

typedef struct _object PyObject;

/* Prototypes for reduction helpers. */
void reduce_spectra(double *spectra, double *part_spectra, int nlam, int npart,
                    int nthreads);

PyObject *reduce_particle_spectra(PyObject *self, PyObject *args);
