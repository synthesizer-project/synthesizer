/******************************************************************************
 * C extension for SPH density and attribute evaluation at query points.
 *
 * This implementation evaluates the SPH density and weighted attributes at
 * arbitrary query points using the shared octree infrastructure. The Python
 * wrapper keeps the same density/attribute semantics and remains the public
 * entry point.
 *****************************************************************************/

/* Python includes */
#define PY_ARRAY_UNIQUE_SYMBOL SYNTHESIZER_ARRAY_API
#define NO_IMPORT_ARRAY
#include <Python.h>

#include <cmath>
#include <memory>
#include <vector>

#include "cpp_to_python.h"
#include "kernel_extensions/kernel_functions.h"
#include "numpy_init.h"
#include "octree.h"
#include "property_funcs.h"
#include "timers.h"
#ifdef ATOMIC_TIMING
#include "timers_init.h"
#endif

/**
 * @brief Lightweight view of one validated attribute array.
 *
 * The SPH evaluator accepts a heterogeneous list of 1D or 2D float64 arrays.
 * Once they have been validated at the Python boundary we keep just the small
 * amount of metadata needed inside the hot query loop.
 */
struct AttributeView {
  PyArrayObject *array;
  const double *data;
  int ndim;
  npy_intp component_count;
};

/**
 * @brief Release the temporary NumPy views created during input validation.
 *
 * @param views The attribute views whose owned NumPy references should be
 *        released.
 */
static void decref_attribute_views(std::vector<AttributeView> &views) {
  for (AttributeView &view : views) {
    Py_XDECREF(view.array);
    view.array = NULL;
    view.data = NULL;
  }
}

/**
 * @brief Compute the minimum possible 3D separation between a point and a
 * cell.
 *
 * @param c The octree cell.
 * @param x The query x coordinate.
 * @param y The query y coordinate.
 * @param z The query z coordinate.
 *
 * @return The squared distance between the query point and the cell bounds.
 */
static double min_squared_dist_to_cell(struct cell *c, const double x,
                                       const double y, const double z) {
  double dx = 0.0;
  double dy = 0.0;
  double dz = 0.0;

  if (!(x > c->loc[0] && x < c->loc[0] + c->width)) {
    dx = fmin(fabs(c->loc[0] - x), fabs(c->loc[0] + c->width - x));
  }
  if (!(y > c->loc[1] && y < c->loc[1] + c->width)) {
    dy = fmin(fabs(c->loc[1] - y), fabs(c->loc[1] + c->width - y));
  }
  if (!(z > c->loc[2] && z < c->loc[2] + c->width)) {
    dz = fmin(fabs(c->loc[2] - z), fabs(c->loc[2] + c->width - z));
  }

  return dx * dx + dy * dy + dz * dz;
}

/**
 * @brief Accumulate the SPH field contribution from one octree branch.
 *
 * The traversal prunes any cell whose entire bounding box lies beyond the
 * largest smoothing-length support stored below that node. Once we reach a
 * leaf, we loop over its particles and add the exact kernel contribution from
 * every particle that overlaps the query point.
 *
 * @param c The current octree cell.
 * @param qx The query x coordinate.
 * @param qy The query y coordinate.
 * @param qz The query z coordinate.
 * @param kernel The analytic SPH kernel function.
 * @param masses The original particle mass array.
 * @param attr_views Metadata for the attribute arrays being accumulated.
 * @param density_out Reference to the density accumulator for this query.
 * @param attr_buffers Output buffers holding weighted attribute sums.
 * @param query_index The row index for this query point in the output buffers.
 */
static void accumulate_sph_query_recursive(
    struct cell *c, const double qx, const double qy, const double qz,
    kernel_func kernel, const double *masses,
    const std::vector<AttributeView> &attr_views, double &density_out,
    const std::vector<double *> &attr_buffers, const npy_intp query_index) {
  if (min_squared_dist_to_cell(c, qx, qy, qz) > c->max_sml_squ) {
    return;
  }

  if (c->split) {
    for (int ip = 0; ip < 8; ++ip) {
      struct cell *cp = &c->progeny[ip];
      if (cp->part_count == 0) {
        continue;
      }
      accumulate_sph_query_recursive(cp, qx, qy, qz, kernel, masses,
                                     attr_views, density_out, attr_buffers,
                                     query_index);
    }
    return;
  }

  struct particle *parts = c->particles;
  const int npart = c->part_count;
  for (int ip = 0; ip < npart; ++ip) {
    const struct particle *part = &parts[ip];
    const double dx = qx - part->pos[0];
    const double dy = qy - part->pos[1];
    const double dz = qz - part->pos[2];
    const double h = part->sml;
    const double q = std::sqrt(dx * dx + dy * dy + dz * dz) / h;
    if (q >= 1.0) {
      continue;
    }

    const int original_index = part->index;
    const double weight = masses[original_index] * kernel(q) / (h * h * h);
    density_out += weight;

    for (size_t ia = 0; ia < attr_views.size(); ++ia) {
      const AttributeView &view = attr_views[ia];
      double *out = attr_buffers[ia];
      if (view.ndim == 1) {
        out[query_index] += weight * view.data[original_index];
        continue;
      }

      for (npy_intp ic = 0; ic < view.component_count; ++ic) {
        out[query_index * view.component_count + ic] +=
            weight * view.data[original_index * view.component_count + ic];
      }
    }
  }
}

/**
 * @brief Evaluate the SPH density field and weighted attributes at query
 * points.
 *
 * Python signature:
 *
 * ``evaluate_sph_density(query_positions, particle_positions,
 * smoothing_lengths, masses, attribute_arrays, kernel_name,
 * maxdepth=16, min_count=8)``
 *
 * @param self The module instance (unused).
 * @param args Python arguments containing the query positions, source particle
 *        data, attribute arrays, kernel name, and optional tree parameters.
 *
 * @return A tuple ``(density, weighted_attributes)`` where ``density`` is a
 *         1D float64 array and ``weighted_attributes`` is a tuple of 1D or 2D
 *         float64 arrays matching the input attribute shapes beyond the first
 *         particle axis.
 */
PyObject *evaluate_sph_density(PyObject *self, PyObject *args) {
  tic("evaluate_sph_density_cpp");

  (void)self;

  PyArrayObject *np_query_positions;
  PyArrayObject *np_particle_positions;
  PyArrayObject *np_smoothing_lengths;
  PyArrayObject *np_masses;
  PyObject *attribute_sequence;
  const char *kernel_name;
  int maxdepth = 16;
  int min_count = 8;

  if (!PyArg_ParseTuple(
          args, "O!O!O!O!Os|ii", &PyArray_Type, &np_query_positions,
          &PyArray_Type, &np_particle_positions, &PyArray_Type,
          &np_smoothing_lengths, &PyArray_Type, &np_masses,
          &attribute_sequence, &kernel_name, &maxdepth, &min_count)) {
    return NULL;
  }

  if (PyArray_NDIM(np_query_positions) != 2 ||
      PyArray_DIM(np_query_positions, 1) != 3) {
    PyErr_SetString(PyExc_ValueError,
                    "query_positions must have shape (N_query, 3).");
    return NULL;
  }
  if (PyArray_NDIM(np_particle_positions) != 2 ||
      PyArray_DIM(np_particle_positions, 1) != 3) {
    PyErr_SetString(PyExc_ValueError,
                    "particle_positions must have shape (N_part, 3).");
    return NULL;
  }
  if (PyArray_NDIM(np_smoothing_lengths) != 1) {
    PyErr_SetString(PyExc_ValueError, "smoothing_lengths must be a 1D array.");
    return NULL;
  }
  if (PyArray_NDIM(np_masses) != 1) {
    PyErr_SetString(PyExc_ValueError, "masses must be a 1D array.");
    return NULL;
  }

  if (PyArray_TYPE(np_query_positions) != NPY_DOUBLE ||
      PyArray_TYPE(np_particle_positions) != NPY_DOUBLE ||
      PyArray_TYPE(np_smoothing_lengths) != NPY_DOUBLE ||
      PyArray_TYPE(np_masses) != NPY_DOUBLE) {
    PyErr_SetString(PyExc_TypeError,
                    "All core SPH arrays must have dtype float64.");
    return NULL;
  }
  if (!PyArray_IS_C_CONTIGUOUS(np_query_positions) ||
      !PyArray_IS_C_CONTIGUOUS(np_particle_positions) ||
      !PyArray_IS_C_CONTIGUOUS(np_smoothing_lengths) ||
      !PyArray_IS_C_CONTIGUOUS(np_masses)) {
    PyErr_SetString(PyExc_ValueError,
                    "All core SPH arrays must be C-contiguous.");
    return NULL;
  }

  const npy_intp n_query = PyArray_DIM(np_query_positions, 0);
  const npy_intp n_part = PyArray_DIM(np_particle_positions, 0);
  if (PyArray_DIM(np_smoothing_lengths, 0) != n_part ||
      PyArray_DIM(np_masses, 0) != n_part) {
    PyErr_SetString(PyExc_ValueError,
                    "particle_positions, smoothing_lengths, and masses must "
                    "share the same leading dimension.");
    return NULL;
  }
  if (maxdepth < 1) {
    PyErr_SetString(PyExc_ValueError, "maxdepth must be >= 1.");
    return NULL;
  }
  if (min_count < 1) {
    PyErr_SetString(PyExc_ValueError, "min_count must be >= 1.");
    return NULL;
  }

  kernel_func kernel = get_kernel_function(kernel_name);
  if (kernel == NULL) {
    PyErr_SetString(PyExc_ValueError, "Kernel name not defined.");
    return NULL;
  }

  PyObject *attribute_fast = PySequence_Fast(
      attribute_sequence, "attribute_arrays must be a sequence of arrays.");
  if (attribute_fast == NULL) {
    return NULL;
  }

  std::vector<AttributeView> attr_views;
  attr_views.reserve(PySequence_Fast_GET_SIZE(attribute_fast));

  PyObject **attribute_items = PySequence_Fast_ITEMS(attribute_fast);
  const Py_ssize_t n_attr = PySequence_Fast_GET_SIZE(attribute_fast);
  for (Py_ssize_t i = 0; i < n_attr; ++i) {
    PyArrayObject *np_attr = (PyArrayObject *)PyArray_FROM_OTF(
        attribute_items[i], NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (np_attr == NULL) {
      decref_attribute_views(attr_views);
      Py_DECREF(attribute_fast);
      return NULL;
    }

    if (PyArray_NDIM(np_attr) != 1 && PyArray_NDIM(np_attr) != 2) {
      PyErr_SetString(PyExc_ValueError,
                      "Each attribute array must be 1D or 2D.");
      Py_DECREF(np_attr);
      decref_attribute_views(attr_views);
      Py_DECREF(attribute_fast);
      return NULL;
    }
    if (PyArray_DIM(np_attr, 0) != n_part) {
      PyErr_SetString(PyExc_ValueError,
                      "Each attribute array must share the particle axis.");
      Py_DECREF(np_attr);
      decref_attribute_views(attr_views);
      Py_DECREF(attribute_fast);
      return NULL;
    }

    AttributeView view;
    view.array = np_attr;
    view.data = static_cast<const double *>(PyArray_DATA(np_attr));
    view.ndim = PyArray_NDIM(np_attr);
    view.component_count = view.ndim == 1 ? 1 : PyArray_DIM(np_attr, 1);
    attr_views.push_back(view);
  }

  const double *query_positions =
      extract_data_double(np_query_positions, "query_positions");
  const double *particle_positions =
      extract_data_double(np_particle_positions, "particle_positions");
  const double *smoothing_lengths =
      extract_data_double(np_smoothing_lengths, "smoothing_lengths");
  const double *masses = extract_data_double(np_masses, "masses");
  if (PyErr_Occurred()) {
    decref_attribute_views(attr_views);
    Py_DECREF(attribute_fast);
    return NULL;
  }

  std::unique_ptr<double[]> density_buffer(new double[n_query]());
  std::vector<std::unique_ptr<double[]> > attr_buffers;
  attr_buffers.reserve(attr_views.size());
  for (const AttributeView &view : attr_views) {
    attr_buffers.emplace_back(new double[n_query * view.component_count]());
  }

  /* The existing octree builder expects one scalar value per particle even
   * though this evaluator only needs positions, smoothing lengths, and the
   * original particle indices stored on each tree node. */
  std::unique_ptr<double[]> surf_den_dummy(new double[n_part]());
  struct cell *root = new struct cell;
  construct_cell_tree(particle_positions, smoothing_lengths,
                      surf_den_dummy.get(), static_cast<int>(n_part), root, 1,
                      maxdepth, min_count);

  /* Keep raw row pointers to the output buffers so the recursive walker can
   * write contributions without unpacking smart pointers at every hit. */
  std::vector<double *> attr_buffer_ptrs;
  attr_buffer_ptrs.reserve(attr_buffers.size());
  for (size_t ia = 0; ia < attr_buffers.size(); ++ia) {
    attr_buffer_ptrs.push_back(attr_buffers[ia].get());
  }

  /* Query points are independent, so we parallelise over them exactly as we do
   * in other particle/image kernels when OpenMP is available. Each thread only
   * touches its own output row, while the octree itself is read-only. */
#ifdef WITH_OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (npy_intp iq = 0; iq < n_query; ++iq) {
    const double qx = query_positions[iq * 3];
    const double qy = query_positions[iq * 3 + 1];
    const double qz = query_positions[iq * 3 + 2];
    accumulate_sph_query_recursive(root, qx, qy, qz, kernel, masses,
                                   attr_views, density_buffer[iq],
                                   attr_buffer_ptrs, iq);
  }

  cleanup_cell_tree(root);

  npy_intp density_dims[1] = {n_query};
  PyArrayObject *np_density =
      wrap_array_to_numpy<double>(1, density_dims, std::move(density_buffer));
  if (np_density == NULL) {
    decref_attribute_views(attr_views);
    Py_DECREF(attribute_fast);
    return NULL;
  }

  PyObject *output_attrs = PyTuple_New(attr_views.size());
  if (output_attrs == NULL) {
    Py_DECREF(np_density);
    decref_attribute_views(attr_views);
    Py_DECREF(attribute_fast);
    return NULL;
  }

  for (size_t ia = 0; ia < attr_views.size(); ++ia) {
    const AttributeView &view = attr_views[ia];
    PyArrayObject *np_attr_out = NULL;
    if (view.ndim == 1) {
      npy_intp dims[1] = {n_query};
      np_attr_out =
          wrap_array_to_numpy<double>(1, dims, std::move(attr_buffers[ia]));
    } else {
      npy_intp dims[2] = {n_query, view.component_count};
      np_attr_out =
          wrap_array_to_numpy<double>(2, dims, std::move(attr_buffers[ia]));
    }
    if (np_attr_out == NULL) {
      Py_DECREF(output_attrs);
      Py_DECREF(np_density);
      decref_attribute_views(attr_views);
      Py_DECREF(attribute_fast);
      return NULL;
    }
    PyTuple_SET_ITEM(output_attrs, ia, (PyObject *)np_attr_out);
  }

  decref_attribute_views(attr_views);
  Py_DECREF(attribute_fast);

  toc("evaluate_sph_density_cpp");
  return Py_BuildValue("NN", np_density, output_attrs);
}

static PyMethodDef SphDensityMethods[] = {
    {"evaluate_sph_density", (PyCFunction)evaluate_sph_density, METH_VARARGS,
     "Evaluate the SPH density field and weighted attributes."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "sph_density",
    "SPH density evaluation helpers for particle resampling.",
    -1,
    SphDensityMethods,
    NULL,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC PyInit_sph_density(void) {
  PyObject *m = PyModule_Create(&moduledef);
  if (m == NULL) {
    return NULL;
  }
  if (numpy_import() < 0) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to import numpy.");
    Py_DECREF(m);
    return NULL;
  }
#ifdef ATOMIC_TIMING
  if (import_toc_capsule() < 0) {
    Py_DECREF(m);
    return NULL;
  }
#endif
  return m;
}
