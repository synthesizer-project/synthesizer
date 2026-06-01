/******************************************************************************
 * C/C++ helpers for validating Python inputs before entering typed kernels.
 *
 * These implementations keep dtype and layout checks close to the extension
 * boundary so hot numerical kernels can work directly with raw typed pointers.
 *****************************************************************************/

/* Local includes */
#include "python_to_cpp.h"

/**
 * @brief Convert a NumPy typenum into a readable dtype string.
 *
 * @param typenum: The NumPy type number.
 *
 * @return The readable dtype name.
 */
const char *typenum_to_string(int typenum) {

  /* Map the NumPy typenum to a readable string for Python errors. */
  switch (typenum) {
  case NPY_FLOAT32:
    return "float32";
  case NPY_FLOAT64:
    return "float64";
  case NPY_INT32:
    return "int32";
  case NPY_INT64:
    return "int64";
  case NPY_BOOL:
    return "bool";
  default:
    return "unsupported dtype";
  }
}

/**
 * @brief Check whether an array is C-contiguous.
 *
 * @param np_arr: The NumPy array to validate.
 * @param name: The name of the NumPy array. (For error messages)
 *
 * @return True if the array is contiguous, false otherwise.
 */
bool is_c_contiguous(PyArrayObject *np_arr, const char *name) {

  /* Reject arrays that would force strided access in the hot kernels. */
  if (!PyArray_IS_C_CONTIGUOUS(np_arr)) {
    PyErr_Format(PyExc_ValueError, "%s must be C-contiguous.", name);
    return false;
  }

  return true;
}

/**
 * @brief Check whether an array is float32 or float64.
 *
 * @param np_arr: The NumPy array to validate.
 * @param name: The name of the NumPy array. (For error messages)
 *
 * @return True if the array has a supported dtype, false otherwise.
 */
bool is_float32_or_float64(PyArrayObject *np_arr, const char *name) {

  /* For the first mixed-precision pass we support only float32 and float64. */
  const int typenum = PyArray_TYPE(np_arr);
  if (typenum != NPY_FLOAT32 && typenum != NPY_FLOAT64) {
    PyErr_Format(PyExc_TypeError,
                 "%s must have dtype float32 or float64 (got %s).", name,
                 typenum_to_string(typenum));
    return false;
  }

  return true;
}

/**
 * @brief Check whether a list of arrays share one floating-point dtype.
 *
 * This helper enforces the initial precision contract used by the migrated
 * extensions: every floating-point input array must be contiguous and must
 * have the same dtype, either float32 or float64.
 *
 * @param arrays: Array of NumPy array pointers to validate.
 * @param names: Matching array names. (For error messages)
 * @param count: The number of arrays to validate.
 * @param resolved_typenum: The shared typenum on success.
 *
 * @return True if all arrays are valid and share one dtype, false otherwise.
 */
bool is_matching_float_dtypes(PyArrayObject **arrays, const char **names,
                              int count, int *resolved_typenum) {

  /* We need at least one array to establish the shared dtype. */
  if (count <= 0) {
    PyErr_SetString(PyExc_ValueError,
                    "At least one array is required for dtype validation.");
    return false;
  }

  /* Walk through the arrays and check that they are all contiguous, have a
   * supported floating-point dtype, and share the same dtype family. */
  int shared_typenum = -1;
  for (int i = 0; i < count; ++i) {

    /* Grab the array and name for this iteration. */
    PyArrayObject *np_arr = arrays[i];
    const char *name = names[i];

    /* Every floating-point input must be contiguous and use a supported
     * precision family before we enter a typed kernel. */
    if (!is_c_contiguous(np_arr, name) ||
        !is_float32_or_float64(np_arr, name)) {
      return false;
    }

    /* What type have we got? */
    const int typenum = PyArray_TYPE(np_arr);

    /* The first array sets the shared dtype we require from the rest. */
    if (shared_typenum == -1) {
      shared_typenum = typenum;
      continue;
    }

    /* Stop early if any input array deviates from the shared dtype. */
    if (typenum != shared_typenum) {
      PyErr_Format(PyExc_TypeError,
                   "%s must share the same floating-point dtype as %s "
                   "(got %s and %s).",
                   name, names[0], typenum_to_string(typenum),
                   typenum_to_string(shared_typenum));
      return false;
    }
  }

  /* Return the shared dtype so the caller can dispatch to the right kernel. */
  *resolved_typenum = shared_typenum;
  return true;
}
