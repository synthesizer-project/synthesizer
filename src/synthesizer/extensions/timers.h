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
 *   - timer_start_fn/timer_stop_fn are typedefs for start/stop callbacks
 *     living in timers.cpp (the timers extension module).
 *   - init_timers() caches pointers to those functions at module-init time
 *     (when the GIL is naturally held). See timers_init.h for the helper
 *     that retrieves the pointer from a PyCapsule.
 *   - At runtime, tic()/toc() call cached pointers directly — no GIL, no
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
#include <chrono>

#ifdef WITH_OPENMP
#include <omp.h>
#define GET_TIME() omp_get_wtime()
#else
inline double get_wall_time() {
#if defined(CLOCK_MONOTONIC)
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
    return static_cast<double>(ts.tv_sec)
           + static_cast<double>(ts.tv_nsec) * 1.0e-9;
  }
#endif
  auto now = std::chrono::steady_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::duration<double>>(now)
      .count();
}
#define GET_TIME() get_wall_time()
#endif

/* ---- Timer callback infrastructure (ATOMIC_TIMING enabled only) ---- */
#ifdef ATOMIC_TIMING

/**
 * @brief Function pointer type for timer start callback.
 */
typedef void (*timer_start_fn)(const char *, const char *);

/**
 * @brief Function pointer type for timer stop callback.
 */
typedef void (*timer_stop_fn)(const char *, const char *);

/** PyCapsule name for the timer start function pointer.
 *  Defined here (once) so timers.cpp and timers_init.h both use the
 *  same string. */
#define TIC_START_CAPSULE_NAME                                                 \
  "synthesizer.extensions.timers._tic_start"

/** PyCapsule name for the timer stop function pointer. */
#define TOC_STOP_CAPSULE_NAME                                                  \
  "synthesizer.extensions.timers._toc_stop"

/** PyCapsule name for the accumulation helper function pointer. */
#define TOC_ACCUMULATE_CAPSULE_NAME                                            \
  "synthesizer.extensions.timers._toc_accumulate"

/**
 * @brief Access the cached timer start function pointer.
 *
 * Uses a static local inside a (non-static) inline function so that all
 * translation units linked into the same shared object share a single
 * pointer. This avoids the classic "static-in-header" problem where each
 * .cpp file would get its own independent copy.
 *
 * @return Reference to the cached function pointer.
 */
inline timer_start_fn &_get_cached_tic_fn() {
  static timer_start_fn fn = NULL;
  return fn;
}

/**
 * @brief Access the cached timer stop function pointer.
 *
 * @return Reference to the cached function pointer.
 */
inline timer_stop_fn &_get_cached_toc_fn() {
  static timer_stop_fn fn = NULL;
  return fn;
}

/**
 * @brief Store timer callback pointers for later use by tic()/toc().
 *
 * Must be called once at extension module initialisation (inside PyInit_*)
 * before any toc() calls are made. At that point the GIL is naturally held,
 * so retrieving the PyCapsule is safe.
 *
 * @param tic_fn Pointer to timer start callback from PyCapsule.
 * @param toc_fn Pointer to timer stop callback from PyCapsule.
 */
inline void init_timers(timer_start_fn tic_fn, timer_stop_fn toc_fn) {
  _get_cached_tic_fn() = tic_fn;
  _get_cached_toc_fn() = toc_fn;
}

#endif /* ATOMIC_TIMING */

/**
 * @brief Start a timer for a named operation.
 *
 * @param msg The operation name.
 */
inline void tic(const char *msg) {
#ifdef ATOMIC_TIMING
  timer_start_fn fn = _get_cached_tic_fn();
  if (fn != NULL) {
    fn(msg, "C");
  }
#else
  (void)msg;
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
 */
inline void toc(const char *msg) {
#ifdef ATOMIC_TIMING
  /* Stop via cached function pointer (no GIL required). */
  timer_stop_fn fn = _get_cached_toc_fn();
  if (fn != NULL) {
    fn(msg, "C");
  }
#else
  (void)msg;
#endif
}

#endif /* TIMERS_H_ */
