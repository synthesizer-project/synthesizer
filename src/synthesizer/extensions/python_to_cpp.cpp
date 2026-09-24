/******************************************************************************
 * C/C++ helpers for validating Python inputs before entering typed kernels.
 *
 * These implementations keep dtype and layout checks close to the extension
 * boundary so hot numerical kernels can work directly with raw typed pointers.
 *****************************************************************************/

/* Local includes */
#include "python_to_cpp.h"

/* Standard includes */
#include <string>

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
 * @brief Get the readable dtype name of an array (e.g. "float16", "int64").
 *
 * Unlike typenum_to_string this handles every NumPy dtype, so errors can say
 * exactly what the user passed.
 *
 * @param np_arr: The NumPy array.
 *
 * @return The dtype name.
 */
static std::string dtype_name(PyArrayObject *np_arr) {
  PyObject *str =
      PyObject_Str(reinterpret_cast<PyObject *>(PyArray_DESCR(np_arr)));
  if (str == NULL) {
    PyErr_Clear();
    return typenum_to_string(PyArray_TYPE(np_arr));
  }
  const char *utf8 = PyUnicode_AsUTF8(str);
  std::string name =
      utf8 != NULL ? utf8 : typenum_to_string(PyArray_TYPE(np_arr));
  if (utf8 == NULL) {
    PyErr_Clear();
  }
  Py_DECREF(str);
  return name;
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
    PyErr_Format(PyExc_ValueError,
                 "'%s' is not stored contiguously in memory (this usually "
                 "happens after slicing with a step or transposing). Make a "
                 "contiguous copy with np.ascontiguousarray(arr) before "
                 "passing it in.",
                 name);
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

  /* We support only float32 and float64. */
  const int typenum = PyArray_TYPE(np_arr);
  if (typenum != NPY_FLOAT32 && typenum != NPY_FLOAT64) {
    PyErr_Format(PyExc_TypeError,
                 "'%s' has dtype %s, but Synthesizer only accepts float32 or "
                 "float64 arrays. Convert it with arr.astype(np.float64) (or "
                 "np.float32 to save memory).",
                 name, dtype_name(np_arr).c_str());
    return false;
  }

  return true;
}

/**
 * @brief Check whether a NumPy typenum is one of the supported float dtypes.
 *
 * This is the low-level predicate behind the shared mixed-precision boundary
 * helpers. We currently support only float32 and float64 extension inputs.
 *
 * @param typenum: The NumPy typenum to validate.
 *
 * @return True if the typenum is float32 or float64, false otherwise.
 */
bool is_supported_float_typenum(int typenum) {
  return typenum == NPY_FLOAT32 || typenum == NPY_FLOAT64;
}

/**
 * @brief Choose the promoted floating-point typenum for two inputs.
 *
 * This follows the extension boundary promotion rule used for mixed float32 /
 * float64 inputs: if either input is float64, the promoted dtype is float64;
 * otherwise float32 is preserved.
 *
 * @param lhs: The first NumPy typenum.
 * @param rhs: The second NumPy typenum.
 *
 * @return The promoted NumPy typenum.
 */
int promoted_float_typenum(int lhs, int rhs) {
  return (lhs == NPY_FLOAT64 || rhs == NPY_FLOAT64) ? NPY_FLOAT64
                                                    : NPY_FLOAT32;
}

/**
 * @brief Check whether a list of arrays share one floating-point dtype.
 *
 * Every array must be contiguous and must have the same dtype, either float32
 * or float64. On failure the error lists every array with its dtype and says
 * exactly which arrays need converting, so users can fix all of them at once.
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

  /* Report every array with an unsupported dtype in one go. */
  std::string unsupported;
  for (int i = 0; i < count; ++i) {
    if (!is_supported_float_typenum(PyArray_TYPE(arrays[i]))) {
      unsupported +=
          "\n    " + std::string(names[i]) + ": " + dtype_name(arrays[i]);
    }
  }
  if (!unsupported.empty()) {
    PyErr_Format(PyExc_TypeError,
                 "Synthesizer only accepts float32 or float64 arrays, but "
                 "these have a different dtype:%s\nConvert them with "
                 "arr.astype(np.float64) (or np.float32 to save memory).",
                 unsupported.c_str());
    return false;
  }

  /* Every input must be contiguous before we hand out raw pointers. */
  int n32 = 0;
  for (int i = 0; i < count; ++i) {
    if (!is_c_contiguous(arrays[i], names[i])) {
      return false;
    }
    n32 += PyArray_TYPE(arrays[i]) == NPY_FLOAT32;
  }

  /* All the same? Then we're done. */
  if (n32 == 0 || n32 == count) {
    *resolved_typenum = PyArray_TYPE(arrays[0]);
    return true;
  }

  /* Mixed precision. Suggest converting the minority to the majority, and
   * float32 up to float64 on a tie so no precision is lost. */
  const bool to64 = n32 <= count - n32;
  const int from_typenum = to64 ? NPY_FLOAT32 : NPY_FLOAT64;
  const char *to_name = to64 ? "float64" : "float32";
  const char *from_name = to64 ? "float32" : "float64";
  std::string listing;
  std::string offenders;
  int n_offenders = 0;
  for (int i = 0; i < count; ++i) {
    const int typenum = PyArray_TYPE(arrays[i]);
    listing +=
        "\n    " + std::string(names[i]) + ": " + typenum_to_string(typenum);
    if (typenum == from_typenum) {
      offenders += (offenders.empty() ? "" : ", ") + std::string(names[i]);
      n_offenders++;
    }
  }
  PyErr_Format(
      PyExc_TypeError,
      "These arrays are used together and must all have the same "
      "precision (all float32 or all float64), but they are "
      "mixed:%s\nTo fix this, convert %s (%s) to %s, e.g. "
      "arr = arr.astype(np.%s), or convert all of them to %s. "
      "Synthesizer never converts arrays for you because that "
      "would silently create copies of potentially very large "
      "arrays.",
      listing.c_str(),
      n_offenders == 1 ? "the one mismatched array" : "the mismatched arrays",
      offenders.c_str(), to_name, to_name, from_name);
  return false;
}
