#pragma once

typedef struct _object PyObject;

/* Prototypes for reduction helpers. */
template <typename Real, typename OutT = Real>
void reduce_spectra(OutT *spectra, const Real *part_spectra, int nlam,
                    int npart, int nthreads);

PyObject *reduce_particle_spectra(PyObject *self, PyObject *args);
