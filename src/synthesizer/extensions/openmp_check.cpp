/******************************************************************************
 * A C module containing a simple function to check if OpenMP is enabled.
 *****************************************************************************/
#include <Python.h>

#ifdef WITH_OPENMP
#include <omp.h>
#endif

/**
 * @brief Check if OpenMP is enabled.
 *
 * @param self: The module.
 * @param args: The arguments.
 */
static PyObject *check_openmp(PyObject *self, PyObject *args) {
  (void)self;
  (void)args;
#ifdef WITH_OPENMP
  Py_RETURN_TRUE;
#else
  Py_RETURN_FALSE;
#endif
}

static PyMethodDef OpenMPMethods[] = {
    {"check_openmp", check_openmp, METH_VARARGS, "Check if OpenMP is enabled."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef openmpmodule = {
    PyModuleDef_HEAD_INIT,
    "openmp_check",
    NULL,
    -1,
    OpenMPMethods,
    NULL, /* m_reload */
    NULL, /* m_traverse */
    NULL, /* m_clear */
    NULL, /* m_free */
};

PyMODINIT_FUNC PyInit_openmp_check(void) {
  return PyModule_Create(&openmpmodule);
}
