/******************************************************************************
 * C header for timing code execution.
 *****************************************************************************/
#ifndef TIMERS_H_
#define TIMERS_H_

#include <Python.h>
#include <stdio.h>
#include <time.h>

#ifdef WITH_OPENMP
#include <omp.h>
#define GET_TIME() omp_get_wtime()
#else
#include <time.h>
#define GET_TIME() ((double)clock() / CLOCKS_PER_SEC)
#endif

/**
 * @brief Start a timer - inline implementation.
 *
 * @return The current time.
 */
static inline double tic() { return GET_TIME(); }

/**
 * @brief Stop a timer and print elapsed time.
 *
 * When ATOMIC_TIMING is enabled, this function also accumulates timing data
 * by calling back into the Python timers module. This ensures all timing data
 * goes into a single global map regardless of which C++ extension calls toc().
 *
 * @param msg The operation name/message to identify this timing.
 * @param start_time The start time returned by tic().
 */
static inline void toc(const char *msg, double start_time) {
#ifdef ATOMIC_TIMING
  double end_time = GET_TIME();
  double elapsed_time = end_time - start_time;

  // Print for logging/debugging
#ifdef WITH_OPENMP
  printf("[C] %s took: %f seconds\n", msg, elapsed_time);
#else
  printf("[C] %s took (in serial): %f seconds\n", msg, elapsed_time);
#endif

  // Accumulate timing data by calling back to Python timers module
  // This ensures there's only ONE global_timings map
  PyGILState_STATE gstate = PyGILState_Ensure();
  
  PyObject *timers_module = PyImport_ImportModule("synthesizer.extensions.timers");
  if (timers_module != NULL) {
    PyObject *toc_c_func = PyObject_GetAttrString(timers_module, "toc_from_c");
    if (toc_c_func != NULL && PyCallable_Check(toc_c_func)) {
      PyObject *args = Py_BuildValue("(sd)", msg, elapsed_time);
      if (args != NULL) {
        PyObject *result = PyObject_CallObject(toc_c_func, args);
        Py_XDECREF(result);
        Py_DECREF(args);
      }
      Py_DECREF(toc_c_func);
    }
    Py_DECREF(timers_module);
  }
  
  PyGILState_Release(gstate);
#else
  (void)msg;
  (void)start_time;
#endif
}

#endif // TIMERS_H
