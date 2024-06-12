/******************************************************************************
 * C extension for timing code execution.
 *****************************************************************************/
#include <stdio.h>

#ifdef WITH_OPENMP
#include <omp.h>
#define GET_TIME() omp_get_wtime()
#else
#include <time.h>
#define GET_TIME() ((double)clock() / CLOCKS_PER_SEC)
#endif

/* Local includes.*/
#include "timers.h"

/**
 * @brief Start a timer.
 *
 * @return The current time.
 */
double tic() { return GET_TIME(); }

/**
 * @brief Stop a timer and print the elapsed time.
 *
 * @param msg: The message to print.
 * @param start_time: The start time.
 */
void toc(const char *msg, double start_time) {
  double end_time = GET_TIME();
  double elapsed_time = end_time - start_time;
#ifdef WITH_OPENMP
  printf("[Parallel] %s execution time: %f seconds\n", msg, elapsed_time);
#else
  printf("[Serial] %s execution time: %f seconds\n", msg, elapsed_time);
#endif
}
