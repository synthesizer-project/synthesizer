/******************************************************************************
 * A C module containing a simple function to check if ATOMIC_TIMING is enabled.
 *****************************************************************************/
#include <Python.h>

#ifdef ATOMIC_TIMING
#include <atomic>
#endif

/**
 * @brief Check if ATOMIC_TIMING is enabled.
 *
 * @param self: The module.
 * @param args: The arguments.
 */
static PyObject *check_atomic_timing(PyObject *self, PyObject *args) {
  (void)self;
  (void)args;
#ifdef ATOMIC_TIMING
  Py_RETURN_TRUE;
#else
  Py_RETURN_FALSE;
#endif
}

static PyMethodDef AtomicTimingMethods[] = {
    {"check_atomic_timing", check_atomic_timing, METH_VARARGS,
     "Check if ATOMIC_TIMING is enabled."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef atomic_timing_module = {
    PyModuleDef_HEAD_INIT,
    "atomic_timing_check",
    NULL,
    -1,
    AtomicTimingMethods,
    NULL, /* m_reload */
    NULL, /* m_traverse */
    NULL, /* m_clear */
    NULL, /* m_free */
};

PyMODINIT_FUNC PyInit_atomic_timing_check(void) {
  return PyModule_Create(&atomic_timing_module);
}
