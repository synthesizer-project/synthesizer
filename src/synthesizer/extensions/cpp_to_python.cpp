#define PY_ARRAY_UNIQUE_SYMBOL SYNTHESIZER_ARRAY_API
#define NO_IMPORT_ARRAY
#include "cpp_to_python.h"

#include "floating_point_utils.h"
#include "numpy_init.h"
#include "python_to_cpp.h"

#include <cstdio>

/**
 * @brief Resolve and validate a requested floating-point output dtype.
 *
 * @param dtype_obj: Python dtype-like object to parse.
 * @param argument_name: The Python argument name. (For error messages)
 *
 * @return The resolved NumPy typenum, or -1 on failure.
 */
int resolve_output_typenum(PyObject *dtype_obj, const char *argument_name) {

  PyArray_Descr *descr = NULL;

  /* Let NumPy parse dtype-like objects such as np.float32 or np.dtype("f4").
   */
  if (!PyArray_DescrConverter(dtype_obj, &descr)) {
    PyErr_Format(PyExc_TypeError,
                 "%s must be a NumPy dtype or floating-point type.",
                 argument_name);
    return -1;
  }

  /* Extract the resolved typenum and release the temporary descriptor. */
  const int typenum = descr->type_num;
  Py_DECREF(descr);

  /* Only float32 and float64 are valid output dtypes for this pass. */
  if (typenum != NPY_FLOAT32 && typenum != NPY_FLOAT64) {
    PyErr_Format(PyExc_TypeError,
                 "%s must be np.float32 or np.float64 (got %s).",
                 argument_name, typenum_to_string(typenum));
    return -1;
  }

  return typenum;
}

/**
 * @brief Check that the given PyObject* is either None or a NumPy array.
 *
 * @param obj  The Python object to check (e.g. from argument parsing).
 * @param name Optional name to include in the error message.
 *
 * @return nullptr if obj is Py_None, or a PyArrayObject* if it's a valid
 * array. Returns nullptr and sets a Python error if the type is invalid.
 */
PyArrayObject *array_or_none(PyObject *obj, const char *name) {
  if (obj == Py_None) {
    return nullptr;
  }

  if (!PyArray_Check(obj)) {
    PyErr_Format(PyExc_TypeError, "%s must be a NumPy array or None", name);
    return nullptr;
  }

  return reinterpret_cast<PyArrayObject *>(obj);
}

/**
 * @brief Reset the precision overflow flags before running a kernel.
 */
void reset_precision_flags() {
  weight_rescaled.store(false, std::memory_order_relaxed);
  output_overflowed.store(false, std::memory_order_relaxed);
}

/**
 * @brief Raise warnings for any precision overflows flagged by a kernel.
 *
 * @param out_dtype_name: The name of the output dtype (for the message).
 *
 * @return False if a warning was turned into an exception (e.g. by
 * warnings.simplefilter("error")), true otherwise.
 */
bool warn_precision_flags(const char *out_dtype_name) {
  if (output_overflowed.load(std::memory_order_relaxed)) {
    char msg[512];
    snprintf(msg, sizeof(msg),
             "Some output values are too large to be stored at %s and have "
             "overflowed to inf. Use float64 outputs (out_dtype=np.float64) "
             "or smaller internal units for this quantity.",
             out_dtype_name);
    if (PyErr_WarnEx(PyExc_RuntimeWarning, msg, 1) < 0) {
      return false;
    }
  } else if (weight_rescaled.load(std::memory_order_relaxed)) {
    char msg[512];
    snprintf(msg, sizeof(msg),
             "Some particle weights are too large to be stored at %s. They "
             "were applied at float64 instead, so the results are correct, "
             "but these particles take a slower path. Consider smaller "
             "internal units for the weight variable.",
             out_dtype_name);
    if (PyErr_WarnEx(PyExc_RuntimeWarning, msg, 1) < 0) {
      return false;
    }
  }
  return true;
}
