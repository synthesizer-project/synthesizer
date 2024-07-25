/******************************************************************************
 * C extension for timing code execution.
 *****************************************************************************/
#include <Python.h>
#include <stdio.h>

#ifdef WITH_OPENMP
#include <omp.h>
#define GET_TIME() omp_get_wtime()
#else
#include <time.h>
#define GET_TIME() ((double)clock() / CLOCKS_PER_SEC)
#endif

/**
 * @brief Start a timer.
 *
 * @return The current time.
 */
double tic() { return GET_TIME(); }

/**
 * @brief Stop a timer and print the elapsed time.
 *
 * @param msg: The message to print.
 * @param start_time: The start time.
 */
void toc(const char *msg, double start_time) {
#ifdef ATOMIC_TIMING
  double end_time = GET_TIME();
  double elapsed_time = end_time - start_time;
#ifdef WITH_OPENMP
  printf("[C] %s took: %f seconds\n", msg, elapsed_time);
#else
  printf("[C] %s took (in serial): %f seconds\n", msg, elapsed_time);
#endif
#endif
}

/* Python wrapper for tic */
static PyObject *py_tic(PyObject *self, PyObject *args) {
  (void)self;
  (void)args;
  return Py_BuildValue("d", tic());
}

/* Python wrapper for toc */
static PyObject *py_toc(PyObject *self, PyObject *args) {
  (void)self;
#ifdef ATOMIC_TIMING
  char *msg;
  double start_time;
  if (!PyArg_ParseTuple(args, "sd", &msg, &start_time))
    return NULL;
  double end_time = GET_TIME();
  double elapsed_time = end_time - start_time;
  printf("[Python] %s took: %f seconds\n", msg, elapsed_time);
#endif
  Py_RETURN_NONE;
}

/* Module method table */
static PyMethodDef TimerMethods[] = {
    {"tic", py_tic, METH_NOARGS, "Start a timer and return the start time."},
    {"toc", py_toc, METH_VARARGS, "Stop the timer and print the elapsed time."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

/* Module definition */
static struct PyModuleDef timermodule = {
    PyModuleDef_HEAD_INIT,
    "timer",                               /* name of module */
    "A module containing timer functions", /* module documentation*/
    -1,                                    /* m_size */
    TimerMethods,                          /* m_methods */
    NULL,                                  /* m_reload */
    NULL,                                  /* m_traverse */
    NULL,                                  /* m_clear */
    NULL,                                  /* m_free */
};

/* Module initialization function */
PyMODINIT_FUNC PyInit_timers(void) { return PyModule_Create(&timermodule); }
