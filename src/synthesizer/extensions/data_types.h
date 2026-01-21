/******************************************************************************
 * A C++ module containing data type definitions for use in C++ extensions.
 *****************************************************************************/

#ifndef DATA_TYPES_H
#define DATA_TYPES_H

/*----------------------------------------------------------------------------
 * Floating point type
 *----------------------------------------------------------------------------*/

#ifdef SYNTHESIZER_SINGLE_PRECISION
typedef float FLOAT;
#else
typedef double FLOAT;
#endif

/*----------------------------------------------------------------------------
 * NumPy dtype mapping
 *
 * Requires numpy headers to be included before this is used.
 *----------------------------------------------------------------------------*/

#ifdef SYNTHESIZER_SINGLE_PRECISION
#define NPY_FLOAT_T NPY_FLOAT
#define FLOAT_NAME "float32"
#else
#define NPY_FLOAT_T NPY_DOUBLE
#define FLOAT_NAME "float64"
#endif

/*----------------------------------------------------------------------------
 * Size helper
 *----------------------------------------------------------------------------*/

#define FLOAT_BYTES ((int)sizeof(FLOAT))

#endif /* DATA_TYPES_H */
