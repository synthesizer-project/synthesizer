/******************************************************************************
 * C extension for timing code execution.
 *****************************************************************************/
#include <Python.h>
#include <stdio.h>
#include <atomic>
#include <string>
#include <tuple>
#include <unordered_map>

#ifdef WITH_OPENMP
#include <omp.h>
#define GET_TIME() omp_get_wtime()
#else
#include <time.h>
#define GET_TIME() ((double)clock() / CLOCKS_PER_SEC)
#endif

/**
 * @brief Structure to hold timing data for a single operation.
 *
 * This structure stores cumulative timing information using atomics for
 * thread-safe lock-free accumulation. The source field is set once and
 * treated as read-only thereafter.
 *
 * Note: We use atomic<double> for cumulative_time, but fetch_add is only
 * available in C++20. For C++17, we use compare_exchange_weak in a loop.
 */
struct OperationTimingData {
  std::atomic<double> cumulative_time; /**< Sum of all timing measurements */
  std::atomic<int> call_count;         /**< Number of times operation called */
  std::string source; /**< Source of timing: "C" or "Python" */

  /**
   * @brief Constructor to initialize atomic members.
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
 * This map is shared across all threads and uses atomic operations for
 * thread-safe accumulation. Map insertion (for new operations) requires
 * a critical section, but updates to existing entries are lock-free.
 */
static std::unordered_map<std::string, OperationTimingData> global_timings;

/* Python wrapper for tic */
static PyObject *py_tic(PyObject *self, PyObject *args) {
  (void)self;
  (void)args;
  return Py_BuildValue("d", GET_TIME());
}

/**
 * @brief Python wrapper for toc - stop timer and accumulate timing data.
 *
 * This function is called from Python code to stop a timer started with tic().
 * It computes elapsed time, prints it, and accumulates it with source="Python".
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

  // Print for logging/debugging
  printf("[Python] %s took: %f seconds\n", msg, elapsed_time);

  // Accumulate timing data
  std::string operation(msg);

  // Check if this operation exists in the map
  auto it = global_timings.find(operation);
  if (it == global_timings.end()) {
    // First time seeing this operation - need critical section for map
    // insertion
#ifdef WITH_OPENMP
#pragma omp critical(timing_init)
#endif
    {
      // Double-check after acquiring lock
      it = global_timings.find(operation);
      if (it == global_timings.end()) {
        // Use emplace with piecewise_construct to construct in-place
        auto result = global_timings.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(operation),
            std::forward_as_tuple("Python"));
        it = result.first;
      }
    }
  }

  // Atomic updates - lock-free
  it->second.add_time(elapsed_time);
  it->second.call_count.fetch_add(1, std::memory_order_relaxed);
#else
  (void)args;
#endif
  Py_RETURN_NONE;
}

/**
 * @brief Accumulate timing data from C code (called via Python callback).
 *
 * This function is called by the inline toc() function in timers.h when C++
 * extensions need to accumulate timing data. By routing through Python, we
 * ensure all timing data goes into a single global_timings map.
 *
 * @param self Module object (unused).
 * @param args Python arguments: (msg, elapsed_time).
 * @return None.
 */
static PyObject *py_toc_from_c(PyObject *self, PyObject *args) {
  (void)self;
#ifdef ATOMIC_TIMING
  char *msg;
  double elapsed_time;
  if (!PyArg_ParseTuple(args, "sd", &msg, &elapsed_time))
    return NULL;

  // Accumulate timing data with source="C"
  std::string operation(msg);

  // Check if this operation exists in the map
  auto it = global_timings.find(operation);
  if (it == global_timings.end()) {
    // First time seeing this operation - need critical section for map
    // insertion
#ifdef WITH_OPENMP
#pragma omp critical(timing_init)
#endif
    {
      // Double-check after acquiring lock
      it = global_timings.find(operation);
      if (it == global_timings.end()) {
        // Use emplace with piecewise_construct to construct in-place
        auto result = global_timings.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(operation),
            std::forward_as_tuple("C"));
        it = result.first;
      }
    }
  }

  // Atomic updates - lock-free
  it->second.add_time(elapsed_time);
  it->second.call_count.fetch_add(1, std::memory_order_relaxed);
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

  // Clear the map (automatic cleanup, no manual memory management needed)
  global_timings.clear();

  Py_RETURN_NONE;
}

/* Module method table */
static PyMethodDef TimerMethods[] = {
    {"tic", py_tic, METH_NOARGS, "Start a timer and return the start time."},
    {"toc", py_toc, METH_VARARGS,
     "Stop the timer, print elapsed time, and accumulate timing data."},
    {"toc_from_c", py_toc_from_c, METH_VARARGS,
     "Accumulate timing data from C code (elapsed time already computed)."},
    {"get_operation_names", py_get_operation_names, METH_NOARGS,
     "Get list of all operation names with accumulated timing data."},
    {"get_operation_timings", py_get_operation_timings, METH_VARARGS,
     "Get timing data for an operation as (cumulative_time, call_count, "
     "source)."},
    {"get_operation_source", py_get_operation_source, METH_VARARGS,
     "Get source ('C' or 'Python') for a specific operation."},
    {"reset_timings", py_reset_timings, METH_NOARGS,
     "Clear all accumulated timing data."},
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
