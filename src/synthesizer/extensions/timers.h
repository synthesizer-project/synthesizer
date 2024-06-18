/******************************************************************************
 * C header for timing code execution.
 *****************************************************************************/
#ifndef TIMERS_H_
#define TIMERS_H_

#include <stdio.h>
#include <time.h>

#ifdef WITH_OPENMP
#include <omp.h>
#endif

double tic();
void toc(const char *msg, double start_time);

#endif // TIMERS_H
