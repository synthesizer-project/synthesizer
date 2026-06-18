#define PY_ARRAY_UNIQUE_SYMBOL SYNTHESIZER_ARRAY_API
#define NO_IMPORT_ARRAY
#include "cpp_to_python.h"

#include "numpy_init.h"

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
    if (!PyErr_Occurred()) {
      PyErr_Format(PyExc_TypeError,
                   "%s must be a NumPy dtype or floating-point type.",
                   argument_name);
    }
    return -1;
  }

  /* Extract the resolved typenum and release the temporary descriptor. */
  const int typenum = descr->type_num;
  Py_DECREF(descr);

  /* Only float32 and float64 are valid output dtypes for this pass. */
  if (typenum != NPY_FLOAT32 && typenum != NPY_FLOAT64) {
    PyErr_Format(PyExc_TypeError, "%s must be float32 or float64.",
                 argument_name);
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
