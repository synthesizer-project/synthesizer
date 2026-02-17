/******************************************************************************
 * C++ header for timing code execution.
 *
 * This header is included by ALL C++ extension files that use tic()/toc().
 * It is deliberately free of Python.h and any Python API calls so that
 * toc() can be called from any thread without touching the GIL.
 *
 * When ATOMIC_TIMING is not defined, tic() and toc() are no-ops and the
 * accumulation infrastructure (function pointer, capsule name, init) is
 * compiled out entirely.
 *
 * Architecture (ATOMIC_TIMING enabled):
 *   - toc_accumulate_fn is a typedef for the accumulation function living in
 *     timers.cpp (the timers extension module).
 *   - init_timers() caches a pointer to that function at module-init time
 *     (when the GIL is naturally held). See timers_init.h for the helper
 *     that retrieves the pointer from a PyCapsule.
 *   - At runtime, toc() calls the cached pointer directly — no GIL, no
 *     Python API.
 *
 * The cached pointer is stored as a static local inside an inline (non-static)
 * function. In C++, inline functions have external linkage by default and the
 * ODR guarantees exactly one instance per shared library, so all translation
 * units within the same .so share a single pointer.
 *****************************************************************************/
#ifndef TIMERS_H_
#define TIMERS_H_

#include <time.h>

#ifdef WITH_OPENMP
#include <omp.h>
#define GET_TIME() omp_get_wtime()
#else
#define GET_TIME() ((double)clock() / CLOCKS_PER_SEC)
#endif

/* ---- Accumulation infrastructure (only when ATOMIC_TIMING is enabled) ---- */
#ifdef ATOMIC_TIMING

/**
 * @brief Function pointer type for the accumulation callback.
 *
 * Matches the signature of toc_accumulate() in timers.cpp:
 *   void toc_accumulate(const char *msg, double elapsed, const char *source)
 */
typedef void (*toc_accumulate_fn)(const char *, double, const char *);

/** PyCapsule name for the toc_accumulate function pointer.
 *  Defined here (once) so timers.cpp and timers_init.h both use the
 *  same string. */
#define TOC_ACCUMULATE_CAPSULE_NAME                                            \
  "synthesizer.extensions.timers._toc_accumulate"

/**
 * @brief Access the cached toc_accumulate function pointer.
 *
 * Uses a static local inside a (non-static) inline function so that all
 * translation units linked into the same shared object share a single
 * pointer. This avoids the classic "static-in-header" problem where each
 * .cpp file would get its own independent copy.
 *
 * @return Reference to the cached function pointer.
 */
inline toc_accumulate_fn &_get_cached_toc_fn() {
  static toc_accumulate_fn fn = NULL;
  return fn;
}

/**
 * @brief Store the accumulation function pointer for later use by toc().
 *
 * Must be called once at extension module initialisation (inside PyInit_*)
 * before any toc() calls are made. At that point the GIL is naturally held,
 * so retrieving the PyCapsule is safe.
 *
 * @param fn Pointer to toc_accumulate() obtained from the PyCapsule.
 */
inline void init_timers(toc_accumulate_fn fn) { _get_cached_toc_fn() = fn; }

#endif /* ATOMIC_TIMING */

/**
 * @brief Start a timer — inline implementation.
 *
 * @return The current time.
 */
inline double tic() {
#ifdef ATOMIC_TIMING
  return GET_TIME();
#else
  return 0.0;
#endif
}

/**
 * @brief Stop a timer and accumulate elapsed time.
 *
 * When ATOMIC_TIMING is enabled, this function accumulates timing data
 * by calling the cached function pointer (set by init_timers()). This goes
 * directly to the global_timings map in timers.cpp without any Python API
 * or GIL interaction.
 *
 * When ATOMIC_TIMING is not defined, this is a complete no-op.
 *
 * @param msg The operation name/message to identify this timing.
 * @param start_time The start time returned by tic().
 */
inline void toc(const char *msg, double start_time) {
#ifdef ATOMIC_TIMING
  double end_time = GET_TIME();
  double elapsed_time = end_time - start_time;

  /* Accumulate via the cached function pointer (no GIL required). */
  toc_accumulate_fn fn = _get_cached_toc_fn();
  if (fn != NULL) {
    fn(msg, elapsed_time, "C");
  }
#else
  (void)msg;
  (void)start_time;
#endif
}

#endif /* TIMERS_H_ */
