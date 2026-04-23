/******************************************************************************
 * Shared C-side integration helpers used by Python extensions.
 *****************************************************************************/
#include "integration.h"
#include <math.h>

double trapz_1d(const double *x, const double *y, size_t n) {
  double integral = 0.0;

  if (n < 2) {
    return integral;
  }

  for (size_t j = 0; j < n - 1; ++j) {
    integral += 0.5 * (x[j + 1] - x[j]) * (y[j + 1] + y[j]);
  }

  return integral;
}

double simps_1d(const double *x, const double *y, size_t n) {
  double integral = 0.0;

  if (n < 2) {
    return integral;
  }

  for (size_t j = 0; j < (n - 1) / 2; ++j) {
    const size_t k = 2 * j;
    integral +=
        (x[k + 2] - x[k]) * (y[k] + 4 * y[k + 1] + y[k + 2]) / 6.0;
  }

  if ((n - 1) % 2 != 0) {
    integral += 0.5 * (x[n - 1] - x[n - 2]) * (y[n - 1] + y[n - 2]);
  }

  return integral;
}
