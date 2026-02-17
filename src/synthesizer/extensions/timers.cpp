/******************************************************************************
 * C extension for timing code execution.
 *
 * This module provides the central timing accumulation infrastructure.
 * It owns the global_timings map and exposes a pure-C accumulation function
 * (toc_accumulate) via a PyCapsule so that other C++ extensions can call it
 * directly without acquiring the GIL.
 *
 * Architecture:
 *   - toc_accumulate() is the pure-C++ function that writes to global_timings.
 *   - PyInit_timers() wraps toc_accumulate in a PyCapsule ("_toc_accumulate").
 *   - Other extensions (integrated_spectra, etc.) retrieve the capsule at
 *     their own init time and cache the raw function pointer in timers.h.
 *   - At runtime, timers.h calls the cached pointer directly (no GIL).
 *****************************************************************************/
#include <Python.h>
#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>

#include "timers.h"

/**
 * @brief Structure to hold timing data for a single operation.
 *
 * This structure stores cumulative timing information using atomics for
 * thread-safe accumulation as a defensive measure, in case toc() is
 * called from within OpenMP parallel regions in future. The source field
 * is set once at creation and treated as read-only thereafter.
 *
 * Note: We use atomic<double> for cumulative_time, but fetch_add is only
 * available in C++20. For C++17, we use compare_exchange_weak in a loop.
 */
struct OperationTimingData {
  std::atomic<double> cumulative_time; /**< Sum of all timing measurements */
  std::atomic<int> call_count;         /**< Number of times operation called */
  std::string source; /**< Source of timing: "C" or "Python" */

  /**
   * @brief Default constructor.
   */
  OperationTimingData()
      : cumulative_time(0.0), call_count(0), source("") {}

  /**
   * @brief Constructor with source specification.
   *
   * @param src The source identifier ("C" or "Python").
   */
  OperationTimingData(const std::string &src)
      : cumulative_time(0.0), call_count(0), source(src) {}

  /**
   * @brief Atomically add to cumulative_time.
   *
   * Uses compare-exchange loop for C++17 compatibility.
   *
   * @param value The value to add.
   */
  void add_time(double value) {
    double old_val = cumulative_time.load(std::memory_order_relaxed);
    while (!cumulative_time.compare_exchange_weak(
        old_val, old_val + value, std::memory_order_relaxed)) {
      // Retry if another thread modified it
    }
  }
};

/**
 * @brief Global map storing accumulated timing data for all operations.
 *
 * This map is shared across all threads. A mutex protects ALL map access
 * (find, insert, iterate) since std::unordered_map is not thread-safe for
 * concurrent read/write. The atomic fields on OperationTimingData are a
 * second line of defence for future use when we may move to a concurrent
 * container or fine-grained locking.
 */
static std::unordered_map<std::string, OperationTimingData> global_timings;

/**
 * @brief Mutex protecting all access to global_timings.
 *
 * Held by toc_accumulate() and py_reset_timings(). Currently toc() is
 * only called outside OpenMP parallel regions so the mutex is uncontended,
 * but it guards against future call-site changes.
 */
static std::mutex timings_mutex;

/**
 * @brief Accumulate timing data for an operation (pure C++, no Python API).
 *
 * This is the core accumulation function shared by all extensions via
 * PyCapsule. It does NOT touch the Python API and does NOT require the GIL.
 *
 * Thread safety is provided by a mutex that serialises all map access.
 * Currently toc() is only called outside OpenMP parallel regions so the
 * mutex is uncontended, but it guards against future call-site changes.
 *
 * @param msg The operation name.
 * @param elapsed_time The elapsed time in seconds.
 * @param source The source identifier ("C" or "Python").
 */
extern "C" void toc_accumulate(const char *msg, double elapsed_time,
                               const char *source) {
  std::string operation(msg);
  std::lock_guard<std::mutex> lock(timings_mutex);

  auto it = global_timings.find(operation);
  if (it == global_timings.end()) {
    auto result = global_timings.emplace(
        std::piecewise_construct, std::forward_as_tuple(operation),
        std::forward_as_tuple(std::string(source)));
    it = result.first;
  }

  it->second.add_time(elapsed_time);
  it->second.call_count.fetch_add(1, std::memory_order_relaxed);
}

/* ========================================================================= */
/* Python wrappers                                                           */
/* ========================================================================= */

/* Python wrapper for tic */
static PyObject *py_tic(PyObject *self, PyObject *args) {
  (void)self;
  (void)args;
#ifdef ATOMIC_TIMING
  return Py_BuildValue("d", GET_TIME());
#else
  return Py_BuildValue("d", 0.0);
#endif
}

/**
 * @brief Python wrapper for toc - stop timer and accumulate timing data.
 *
 * This function is called from Python code to stop a timer started with tic().
 * It computes elapsed time and accumulates it with source="Python".
 *
 * When ATOMIC_TIMING is not defined, this is a complete no-op.
 *
 * @param self Module object (unused).
 * @param args Python arguments: (msg, start_time).
 * @return None.
 */
static PyObject *py_toc(PyObject *self, PyObject *args) {
  (void)self;
#ifdef ATOMIC_TIMING
  char *msg;
  double start_time;
  if (!PyArg_ParseTuple(args, "sd", &msg, &start_time))
    return NULL;
  double end_time = GET_TIME();
  double elapsed_time = end_time - start_time;

  /* Accumulate via the shared function (same one exposed by PyCapsule). */
  toc_accumulate(msg, elapsed_time, "Python");
#else
  (void)args;
#endif
  Py_RETURN_NONE;
}

/**
 * @brief Get list of all operation names with accumulated timing data.
 *
 * @param self Module object (unused).
 * @param args Python arguments (none expected).
 * @return Python list of operation names (strings).
 */
static PyObject *py_get_operation_names(PyObject *self, PyObject *args) {
  (void)self;
  (void)args;

  std::lock_guard<std::mutex> lock(timings_mutex);
  // Create Python list
  PyObject *list = PyList_New(global_timings.size());
  size_t i = 0;
  for (const auto &pair : global_timings) {
    PyList_SetItem(list, i++, PyUnicode_FromString(pair.first.c_str()));
  }
  return list;
}

/**
 * @brief Get accumulated timing data for a specific operation.
 *
 * Returns a tuple of (cumulative_time, call_count, source).
 *
 * @param self Module object (unused).
 * @param args Python arguments: (operation_name,).
 * @return Python tuple (cumulative_time, call_count, source) or raises
 * KeyError.
 */
static PyObject *py_get_operation_timings(PyObject *self, PyObject *args) {
  (void)self;
  const char *operation;
  if (!PyArg_ParseTuple(args, "s", &operation))
    return NULL;

  std::lock_guard<std::mutex> lock(timings_mutex);
  // Find operation in map
  auto it = global_timings.find(std::string(operation));
  if (it == global_timings.end()) {
    PyErr_SetString(PyExc_KeyError, operation);
    return NULL;
  }

  // Load atomic values
  double cumulative_time =
      it->second.cumulative_time.load(std::memory_order_relaxed);
  int call_count = it->second.call_count.load(std::memory_order_relaxed);
  const char *source = it->second.source.c_str();

  // Return tuple (cumulative_time, call_count, source)
  return Py_BuildValue("(dis)", cumulative_time, call_count, source);
}

/**
 * @brief Get source ("C" or "Python") for a specific operation.
 *
 * @param self Module object (unused).
 * @param args Python arguments: (operation_name,).
 * @return Python string "C" or "Python", or raises KeyError.
 */
static PyObject *py_get_operation_source(PyObject *self, PyObject *args) {
  (void)self;
  const char *operation;
  if (!PyArg_ParseTuple(args, "s", &operation))
    return NULL;

  std::lock_guard<std::mutex> lock(timings_mutex);
  // Find operation in map
  auto it = global_timings.find(std::string(operation));
  if (it == global_timings.end()) {
    PyErr_SetString(PyExc_KeyError, operation);
    return NULL;
  }

  return PyUnicode_FromString(it->second.source.c_str());
}

/**
 * @brief Reset all accumulated timing data.
 *
 * Clears the global timing map, resetting all accumulated times and counts.
 * This should be called before each profiling run.
 *
 * @param self Module object (unused).
 * @param args Python arguments (none expected).
 * @return None.
 */
static PyObject *py_reset_timings(PyObject *self, PyObject *args) {
  (void)self;
  (void)args;

  std::lock_guard<std::mutex> lock(timings_mutex);
  global_timings.clear();

  Py_RETURN_NONE;
}

/**
 * @brief Testing helper: call toc_accumulate with source="C".
 *
 * This simulates what a C++ extension's toc() call does, allowing tests
 * to verify the full accumulation pipeline without needing real particle
 * data. Takes (operation_name, elapsed_time) as arguments.
 *
 * @param self Module object (unused).
 * @param args Python arguments: (msg, elapsed_time).
 * @return None.
 */
static PyObject *py_test_toc_from_c(PyObject *self, PyObject *args) {
  (void)self;
#ifdef ATOMIC_TIMING
  char *msg;
  double elapsed_time;
  if (!PyArg_ParseTuple(args, "sd", &msg, &elapsed_time))
    return NULL;

  toc_accumulate(msg, elapsed_time, "C");
#else
  (void)args;
#endif
  Py_RETURN_NONE;
}

/* Module method table */
static PyMethodDef TimerMethods[] = {
    {"tic", py_tic, METH_NOARGS, "Start a timer and return the start time."},
    {"toc", py_toc, METH_VARARGS,
     "Stop the timer and accumulate timing data."},
    {"get_operation_names", py_get_operation_names, METH_NOARGS,
     "Get list of all operation names with accumulated timing data."},
    {"get_operation_timings", py_get_operation_timings, METH_VARARGS,
     "Get timing data for an operation as (cumulative_time, call_count, "
     "source)."},
    {"get_operation_source", py_get_operation_source, METH_VARARGS,
     "Get source ('C' or 'Python') for a specific operation."},
    {"reset_timings", py_reset_timings, METH_NOARGS,
     "Clear all accumulated timing data."},
    {"_test_toc_from_c", py_test_toc_from_c, METH_VARARGS,
     "Testing helper: call toc_accumulate with source='C'. "
     "Simulates what a C++ extension's toc() does."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

/* Module definition */
static struct PyModuleDef timermodule = {
    PyModuleDef_HEAD_INIT,
    "timers",                             /* name of module */
    "A module containing timer functions", /* module documentation */
    -1,                                    /* m_size */
    TimerMethods,                          /* m_methods */
    NULL,                                  /* m_reload */
    NULL,                                  /* m_traverse */
    NULL,                                  /* m_clear */
    NULL,                                  /* m_free */
};

/* Module initialization function */
PyMODINIT_FUNC PyInit_timers(void) {
  PyObject *m = PyModule_Create(&timermodule);
  if (m == NULL)
    return NULL;

#ifdef ATOMIC_TIMING
  /* Expose toc_accumulate as a PyCapsule so other extensions can call it
   * directly from C without the GIL. */
  PyObject *capsule = PyCapsule_New((void *)toc_accumulate,
                                    TOC_ACCUMULATE_CAPSULE_NAME, NULL);
  if (capsule == NULL) {
    Py_DECREF(m);
    return NULL;
  }
  if (PyModule_AddObject(m, "_toc_accumulate", capsule) < 0) {
    Py_DECREF(capsule);
    Py_DECREF(m);
    return NULL;
  }
#endif

  return m;
}
