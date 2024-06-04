
// src/synthesizer/extensions/openmp_check.c
#include <Python.h>

#include <omp.h>

static PyObject *check_openmp(PyObject *self, PyObject *args) {
#ifdef WITH_OPENMP
  Py_RETURN_TRUE;
#else
  Py_RETURN_FALSE;
#endif
}

static PyMethodDef OpenMPMethods[] = {
    {"check_openmp", check_openmp, METH_VARARGS, "Check if OpenMP is enabled."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef openmpmodule = {PyModuleDef_HEAD_INIT, "openmp_check",
                                          NULL, -1, OpenMPMethods};

PyMODINIT_FUNC PyInit_openmp_check(void) {
  return PyModule_Create(&openmpmodule);
}
