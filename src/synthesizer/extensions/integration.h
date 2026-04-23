#ifndef INTEGRATION_H_
#define INTEGRATION_H_

#include <stddef.h>

/**
 * @brief Integrate one 1D function using composite Simpson's rule.
 *
 * The abscissae are supplied explicitly so this helper can be reused by both
 * generic array integrations and specialised kernel builders. The spacing must
 * be uniform.
 *
 * @param x 1D array of x values.
 * @param y 1D array of y values.
 * @param n Number of samples.
 *
 * @return The integrated value.
 */
double simps_1d(const double *x, const double *y, size_t n);

/**
 * @brief Integrate one 1D function using the trapezoidal rule.
 *
 * @param x 1D array of x values.
 * @param y 1D array of y values.
 * @param n Number of samples.
 *
 * @return The integrated value.
 */
double trapz_1d(const double *x, const double *y, size_t n);

#endif // INTEGRATION_H_
