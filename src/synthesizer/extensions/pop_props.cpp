/******************************************************************************
 * The Populations class used by the parametric extensions (see pop_props.h).
 *****************************************************************************/
/* Standard includes */
#include <array>
#include <limits>

/* Local includes */
#include "pop_props.h"
#include "property_funcs.h"
#include "python_to_cpp.h"
#include "timers.h"

/**
 * @brief Constructor for the Populations class.
 *
 * This validates the Python population inputs. On failure a Python error is
 * set, which the caller must check for.
 *
 * The edges, masses and mask are ordered by the populations' own (input)
 * bin axes, while log_flags, bin_axes and unit_scales have one entry per
 * grid axis.
 *
 * @param edges_tuple: A tuple of 1D bin edge arrays, one per input axis (in
 *                     linear units).
 * @param np_masses: The masses with shape (npop, nbins_0, ..., nbins_N) in
 *                   the input axis order.
 * @param log_flags: A sequence of booleans, one per grid axis, flagging
 *                   axes whose grid coordinates are log10 values.
 * @param bin_axes: A sequence of ints, one per grid axis, giving the input
 *                  axis matching it (a permutation of 0, ..., ndim - 1).
 * @param unit_scales: A sequence of floats, one per grid axis, converting the
 *                     matching input axis' edges into the grid axis' units.
 * @param mask_obj: The mask, a float64 array of shape (ndim, 2) holding the
 *                  allowed [lo, hi] range (linear units, inclusive) along
 *                  each input axis, or None. The bounds must be finite (use
 *                  the largest finite float for an open bound).
 * @param ndim_: The number of grid axes.
 */
Populations::Populations(PyObject *edges_tuple, PyArrayObject *np_masses,
                         PyObject *log_flags, PyObject *bin_axes,
                         PyObject *unit_scales, PyObject *mask_obj, int ndim_)
    : npop(0), ndim(ndim_), ncells(0), masses_(nullptr) {

  tic("Populations.__init__");

  /* Start with a range covering every finite value along every axis (the
   * mask, if any, narrows these below). */
  mask_lo_.fill(-std::numeric_limits<double>::max());
  mask_hi_.fill(std::numeric_limits<double>::max());

  /* We need one set of edges per input axis and one log flag, bin axis and
   * unit scale per grid axis. */
  if (!PyTuple_Check(edges_tuple) || PyTuple_Size(edges_tuple) != ndim) {
    PyErr_Format(PyExc_ValueError,
                 "edges must be a tuple with one array per grid axis (%d).",
                 ndim);
    return;
  }
  PyObject *per_axis[3] = {log_flags, bin_axes, unit_scales};
  const char *per_axis_names[3] = {"log_flags", "bin_axes", "unit_scales"};
  for (int i = 0; i < 3; i++) {
    if (!PySequence_Check(per_axis[i]) ||
        PySequence_Size(per_axis[i]) != ndim) {
      PyErr_Format(PyExc_ValueError,
                   "%s must be a sequence with one entry per grid axis (%d).",
                   per_axis_names[i], ndim);
      return;
    }
  }

  /* The masses must have a leading population axis then one per axis. */
  if (!PyArray_Check(reinterpret_cast<PyObject *>(np_masses)) ||
      PyArray_NDIM(np_masses) != ndim + 1) {
    PyErr_Format(PyExc_ValueError,
                 "masses must be an array of shape (npop, nbins_0, ..., "
                 "nbins_%d).",
                 ndim - 1);
    return;
  }
  npop = static_cast<int>(PyArray_DIM(np_masses, 0));

  /* Collect the float arrays so we can check they share one dtype. */
  PyArrayObject *float_arrays[MAX_GRID_NDIM + 1] = {NULL};
  const char *float_names[MAX_GRID_NDIM + 1] = {NULL};
  float_arrays[0] = np_masses;
  float_names[0] = "masses";

  /* Validate each grid axis, counting the bins in a single population. */
  std::array<bool, MAX_GRID_NDIM> seen = {};
  ncells = 1;
  for (int idim = 0; idim < ndim; idim++) {

    /* Get the input axis matching this grid axis, each must be used once. */
    PyObject *axis_obj = PySequence_GetItem(bin_axes, idim);
    if (axis_obj == NULL) {
      return;
    }
    const long axis = PyLong_AsLong(axis_obj);
    Py_DECREF(axis_obj);
    if (PyErr_Occurred()) {
      return;
    }
    if (axis < 0 || axis >= ndim || seen[axis]) {
      PyErr_SetString(PyExc_ValueError,
                      "bin_axes must be a permutation of the input axes.");
      return;
    }
    seen[axis] = true;
    bin_axis_[idim] = static_cast<int>(axis);

    /* Get the edges along this axis. */
    PyObject *edges_obj = PyTuple_GetItem(edges_tuple, axis);
    if (edges_obj == NULL || !PyArray_Check(edges_obj) ||
        PyArray_NDIM(reinterpret_cast<PyArrayObject *>(edges_obj)) != 1) {
      PyErr_Format(PyExc_ValueError,
                   "The edges for axis %ld must be a 1D array.", axis);
      return;
    }
    PyArrayObject *np_edges = reinterpret_cast<PyArrayObject *>(edges_obj);
    float_arrays[idim + 1] = np_edges;
    float_names[idim + 1] = "bin edges";

    /* There is one fewer bin than there are edges. */
    const npy_intp nedges = PyArray_DIM(np_edges, 0);
    if (nedges < 2) {
      PyErr_Format(PyExc_ValueError,
                   "The edges for axis %ld must define at least one bin.",
                   axis);
      return;
    }
    nbins[idim] = static_cast<int>(nedges - 1);
    input_nbins_[axis] = nbins[idim];

    /* The masses must have one entry per bin along this axis. */
    if (PyArray_DIM(np_masses, axis + 1) != nbins[idim]) {
      PyErr_Format(PyExc_ValueError,
                   "masses has %ld bins along axis %ld but the edges define "
                   "%d.",
                   static_cast<long>(PyArray_DIM(np_masses, axis + 1)), axis,
                   nbins[idim]);
      return;
    }
    ncells *= nbins[idim];

    /* Get the log flag for this axis. */
    PyObject *flag = PySequence_GetItem(log_flags, idim);
    if (flag == NULL) {
      return;
    }
    const int is_log = PyObject_IsTrue(flag);
    Py_DECREF(flag);
    if (is_log < 0) {
      return;
    }
    log_axis_[idim] = is_log == 1;

    /* Get the unit scale for this axis, which must be positive. */
    PyObject *scale_obj = PySequence_GetItem(unit_scales, idim);
    if (scale_obj == NULL) {
      return;
    }
    unit_scale_[idim] = PyFloat_AsDouble(scale_obj);
    Py_DECREF(scale_obj);
    if (PyErr_Occurred()) {
      return;
    }
    if (!(unit_scale_[idim] > 0.0)) {
      PyErr_SetString(PyExc_ValueError, "unit_scales must be positive.");
      return;
    }
  }

  /* Check the dtypes match and the arrays are contiguous. */
  if (!is_matching_float_dtypes(float_arrays, float_names, ndim + 1,
                                &float_typenum_)) {
    return;
  }
  if (PyArray_TYPE(np_masses) != float_typenum_) {
    PyErr_SetString(PyExc_TypeError,
                    "masses and bin edges must share the same dtype.");
    return;
  }

  /* Store the pointers and check the edges are non-decreasing. */
  masses_ = PyArray_DATA(np_masses);
  for (int idim = 0; idim < ndim; idim++) {
    PyArrayObject *np_edges = reinterpret_cast<PyArrayObject *>(
        PyTuple_GetItem(edges_tuple, bin_axis_[idim]));
    if (PyArray_TYPE(np_edges) != float_typenum_) {
      PyErr_SetString(PyExc_TypeError,
                      "masses and bin edges must share the same dtype.");
      return;
    }
    edges_[idim] = PyArray_DATA(np_edges);

    /* Check the edges are non-decreasing at their dtype (the negated
     * comparison also rejects NaNs). */
    bool sorted = dispatch_float(float_typenum_, [&](auto r) {
      using Real = decltype(r);
      const Real *edges = get_edges<Real>(idim);
      for (int i = 0; i < nbins[idim]; i++) {
        if (!(edges[i + 1] >= edges[i])) {
          return false;
        }
      }
      return true;
    });
    if (!sorted) {
      PyErr_Format(PyExc_ValueError,
                   "The edges for axis %d must be non-decreasing.",
                   bin_axis_[idim]);
      return;
    }
  }

  /* Read the mask's range along each axis if we have one (the struct
   * starts with ranges covering every finite value). */
  if (mask_obj != NULL && mask_obj != Py_None) {
    PyArrayObject *np_mask = array_or_none(mask_obj, "mask");
    if (np_mask == NULL) {
      return;
    }

    /* The mask must be a contiguous (ndim, 2) float64 array. */
    if (PyArray_TYPE(np_mask) != NPY_FLOAT64 || PyArray_NDIM(np_mask) != 2 ||
        PyArray_DIM(np_mask, 0) != ndim || PyArray_DIM(np_mask, 1) != 2) {
      PyErr_Format(PyExc_ValueError,
                   "mask must be a float64 array of shape (%d, 2) holding "
                   "the allowed range along each axis.",
                   ndim);
      return;
    }
    if (!is_c_contiguous(np_mask, "mask")) {
      return;
    }

    /* The mask's rows follow the input axes, store them by grid axis. */
    const double *ranges = static_cast<const double *>(PyArray_DATA(np_mask));
    for (int idim = 0; idim < ndim; idim++) {
      mask_lo_[idim] = ranges[2 * bin_axis_[idim]];
      mask_hi_[idim] = ranges[2 * bin_axis_[idim] + 1];
    }
  }

  toc("Populations.__init__");
}
