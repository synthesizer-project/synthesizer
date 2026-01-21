#pragma once

#include "data_types.h"

// Reduction prototypes
void reduce_spectra(FLOAT *spectra, FLOAT *part_spectra, int nlam, int npart,
                    int nthreads);
