/******************************************************************************
 * Helper header for C++ extensions that consume timing functionality.
 *
 * This header provides import_toc_capsule(), which retrieves timer start/stop
 * function pointers from the timers module's PyCapsules and passes them to
 * init_timers() (defined in timers.h).
 *
 * Usage: call import_toc_capsule() once inside your PyInit_* function,
 * AFTER PyModule_Create(). At that point the GIL is naturally held, so
 * all Python API calls here are safe. After init returns, tic()/toc() in
 * timers.h uses the cached raw pointer — no GIL, no Python API.
 *
 * This header deliberately includes Python.h because it is ONLY used at
 * module-init time. It must NOT be included from files that run inside
 * OpenMP parallel regions.
 *****************************************************************************/
#ifndef TIMERS_INIT_H_
#define TIMERS_INIT_H_

#include <Python.h>

/* timers.h provides init_timers(), function pointer typedefs, and
 * capsule names. */
#include "timers.h"

/**
 * @brief Import timer start/stop function pointers from timers module.
 *
 * This function:
 *   1. Imports synthesizer.extensions.timers (Python module).
 *   2. Gets the _tic_start and _toc_stop PyCapsule attributes.
 *   3. Extracts raw function pointers from both capsules.
 *   4. Calls init_timers() to cache them for this shared object.
 *
 * Must be called exactly once per extension module, inside PyInit_*.
 * The GIL is naturally held at that point.
 *
 * If ATOMIC_TIMING is not defined, this is a no-op that always succeeds.
 *
 * @return 0 on success, -1 on failure (with a Python exception set).
 */
static inline int import_toc_capsule(void) {
#ifdef ATOMIC_TIMING
  /* Import the timers module. */
  PyObject *mod = PyImport_ImportModule("synthesizer.extensions.timers");
  if (mod == NULL) {
    return -1;
  }

  /* Get timer start/stop capsules. */
  PyObject *tic_cap = PyObject_GetAttrString(mod, "_tic_start");
  PyObject *toc_cap = PyObject_GetAttrString(mod, "_toc_stop");
  Py_DECREF(mod);
  if (tic_cap == NULL) {
    Py_XDECREF(toc_cap);
    return -1;
  }
  if (toc_cap == NULL) {
    Py_DECREF(tic_cap);
    return -1;
  }

  /* Extract raw timer start function pointer. */
  void *tic_ptr = PyCapsule_GetPointer(tic_cap, TIC_START_CAPSULE_NAME);
  Py_DECREF(tic_cap);
  if (tic_ptr == NULL) {
    return -1;
  }

  /* Extract raw timer stop function pointer. */
  void *toc_ptr = PyCapsule_GetPointer(toc_cap, TOC_STOP_CAPSULE_NAME);
  Py_DECREF(toc_cap);
  if (toc_ptr == NULL) {
    return -1;
  }

  /* Cache for all tic()/toc() calls within this shared object. */
  init_timers((timer_start_fn)tic_ptr, (timer_stop_fn)toc_ptr);
#endif
  return 0;
}

#endif /* TIMERS_INIT_H_ */
