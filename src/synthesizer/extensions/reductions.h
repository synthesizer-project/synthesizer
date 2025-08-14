#pragma once

// Reduction prototypes
void reduce_spectra(double *spectra, double *part_spectra, int nlam, int npart,
                    int nthreads);
