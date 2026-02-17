/******************************************************************************
 * Helper header for C++ extensions that consume timing functionality.
 *
 * This header provides import_toc_capsule(), which retrieves the
 * toc_accumulate function pointer from the timers module's PyCapsule and
 * passes it to init_timers() (defined in timers.h).
 *
 * Usage: call import_toc_capsule() once inside your PyInit_* function,
 * AFTER PyModule_Create(). At that point the GIL is naturally held, so
 * all Python API calls here are safe. After init returns, toc() in
 * timers.h uses the cached raw pointer — no GIL, no Python API.
 *
 * This header deliberately includes Python.h because it is ONLY used at
 * module-init time. It must NOT be included from files that run inside
 * OpenMP parallel regions.
 *****************************************************************************/
#ifndef TIMERS_INIT_H_
#define TIMERS_INIT_H_

#include <Python.h>

/* timers.h provides init_timers(), toc_accumulate_fn typedef, and
 * TOC_ACCUMULATE_CAPSULE_NAME. */
#include "timers.h"

/**
 * @brief Import the toc_accumulate function pointer from the timers module.
 *
 * This function:
 *   1. Imports synthesizer.extensions.timers (Python module).
 *   2. Gets the _toc_accumulate PyCapsule attribute.
 *   3. Extracts the raw function pointer from the capsule.
 *   4. Calls init_timers() to cache it for all TUs in this shared object.
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

  /* Get the _toc_accumulate capsule attribute. */
  PyObject *cap = PyObject_GetAttrString(mod, "_toc_accumulate");
  Py_DECREF(mod);
  if (cap == NULL) {
    return -1;
  }

  /* Extract the raw function pointer. */
  void *ptr = PyCapsule_GetPointer(cap, TOC_ACCUMULATE_CAPSULE_NAME);
  Py_DECREF(cap);
  if (ptr == NULL) {
    return -1;
  }

  /* Cache it for all toc() calls within this shared object. */
  init_timers((toc_accumulate_fn)ptr);
#endif
  return 0;
}

#endif /* TIMERS_INIT_H_ */
