/******************************************************************************
 * A C++ module containing data type definitions for use in C++ extensions.
 *****************************************************************************/

#ifndef DATA_TYPES_H
#define DATA_TYPES_H

#include <cstdint>

/*----------------------------------------------------------------------------
 * Floating point type
 *----------------------------------------------------------------------------*/

#ifdef SYNTHESIZER_SINGLE_PRECISION
typedef float Float;
typedef int32_t Int;
#define NPY_INT_T NPY_INT32
#define INT_NAME "int32"
#else
typedef double Float;
typedef int64_t Int;
#define NPY_INT_T NPY_INT64
#define INT_NAME "int64"
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

#define FLOAT_BYTES ((int)sizeof(Float))

#endif /* DATA_TYPES_H */
