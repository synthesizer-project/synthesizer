#pragma once

#include "data_types.h"

// Reduction prototypes
void reduce_spectra(Float *spectra, Float *part_spectra, int nlam, int npart,
                    int nthreads);
