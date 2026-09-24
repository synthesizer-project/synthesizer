/******************************************************************************
 * C++ extension for depositing particle attributes onto meshes.
 *
 * Particles are scattered onto either a uniform grid or an adaptively refined
 * octree mesh (a uniform grid of root cells, each of which may be recursively
 * split into eight children). Two deposition modes are supported:
 *
 *   - Smoothed: each particle is spread over the leaves overlapping its
 *     compact SPH kernel. The kernel integral over each overlapping leaf is
 *     estimated by tensor-product Gauss-Legendre quadrature over the
 *     intersection of the leaf with the kernel's bounding box.
 *   - Cloud-in-cell (CIC): each particle is a uniform cube with the width of
 *     the leaf containing it, deposited by exact overlap volume.
 *
 * In both modes each particle's weights are normalised to sum to one, so every
 * particle deposits exactly its value regardless of quadrature error. This
 * makes deposition conservative to floating-point precision.
 *
 * Any number of attributes is deposited in a single loop over the particles:
 * the weights are computed once per particle and applied to every attribute.
 *
 * Threading uses a race-free colouring of blocks of root cells (see
 * deposit_all). The summation order never depends on the number of threads,
 * so results are bit-identical for any thread count.
 *
 * Tree layout (only present for refined meshes):
 *   first_child[node]: index of the first of 8 contiguous children, or -1 for
 *                      a leaf. Nodes 0..nroot-1 are the root cells in C order.
 *   node_depth[node]:  refinement depth (0 for root cells).
 *   node_ijk[node]:    integer lower-corner coordinates at the node's depth,
 *                      i.e. lower corner = origin + ijk * res / 2**depth.
 *   node_leaf[node]:   index of the node in the leaf arrays, or -1 if split.
 * Children are stored in octant order (ox << 2 | oy << 1 | oz), so a
 * depth-first walk over the root cells visits leaves in Morton order within
 * each root cell, which is the order leaves are numbered in.
 *****************************************************************************/

/* C/C++ headers. */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <type_traits>
#include <vector>

#ifdef WITH_OPENMP
#include <omp.h>
#endif

/* Python headers. */
#define PY_ARRAY_UNIQUE_SYMBOL SYNTHESIZER_ARRAY_API
#define NO_IMPORT_ARRAY
#include "numpy_init.h"

#include <Python.h>

/* Local includes. */
#include "cpp_to_python.h"
#include "kernel_extensions/kernel_functions.h"
#include "property_funcs.h"
#include "python_to_cpp.h"
#include "timers.h"
#ifdef ATOMIC_TIMING
#include "timers_init.h"
#endif

/* Gauss-Legendre nodes and weights on [-1, 1] for 2, 4 and 8 points. The
 * kernels are smooth polynomials inside their support, so a handful of nodes
 * integrates them far more accurately than midpoint sampling. */
static const double GL2_X[2] = {-0.5773502691896257, 0.5773502691896257};
static const double GL2_W[2] = {1.0, 1.0};
static const double GL4_X[4] = {-0.8611363115940526, -0.33998104358485626,
                                0.33998104358485626, 0.8611363115940526};
static const double GL4_W[4] = {0.34785484513745346, 0.6521451548625465,
                                0.6521451548625465, 0.34785484513745346};
static const double GL8_X[8] = {-0.9602898564975362, -0.7966664774136267,
                                -0.525532409916329,  -0.18343464249564984,
                                0.18343464249564984, 0.525532409916329,
                                0.7966664774136267,  0.9602898564975362};
static const double GL8_W[8] = {0.10122853629037652, 0.22238103445337445,
                                0.31370664587788716, 0.3626837833783618,
                                0.3626837833783618,  0.31370664587788716,
                                0.22238103445337445, 0.10122853629037652};

/**
 * @brief A read-only view of mesh geometry shared by all deposition routines.
 *
 * A uniform mesh has no tree arrays (first_child == nullptr); every root cell
 * is then a leaf and its leaf index equals its root index.
 */
struct MeshView {
  double origin[3];
  double res;
  int64_t dims[3];
  int64_t nroot;
  int64_t nleaf;
  const int64_t *first_child;
  const int32_t *depth;
  const int64_t *ijk;
  const int64_t *node_leaf;
};

/**
 * @brief A leaf overlapped by a particle and the intersection of the leaf
 *        with the particle's support box.
 */
struct Candidate {
  int64_t leaf;
  double lo[3];
  double hi[3];
  double weight;
};

/**
 * @brief Per-thread scratch buffers reused across particles.
 */
struct Scratch {
  std::vector<Candidate> cands;
  std::vector<int64_t> stack;
  std::vector<double> vals;
};

/**
 * @brief Clamp a floating-point cell coordinate to a valid integer index.
 *
 * @param f The (possibly out of range) floating-point index.
 * @param n The number of cells along the axis.
 *
 * @return The clamped index in [0, n - 1].
 */
static inline int64_t clamp_index(const double f, const int64_t n) {
  if (!(f >= 0.0)) return 0;
  const int64_t i = static_cast<int64_t>(f);
  return i >= n ? n - 1 : i;
}

/**
 * @brief Get the lower corner and width of a node.
 *
 * @param m The mesh view.
 * @param node The node index.
 * @param lo Output lower corner.
 *
 * @return The node width.
 */
static inline double node_box(const MeshView &m, const int64_t node,
                              double lo[3]) {
  if (m.first_child == nullptr) {
    /* Uniform mesh: decode the C-order root index. */
    const int64_t i = node / (m.dims[1] * m.dims[2]);
    const int64_t j = (node / m.dims[2]) % m.dims[1];
    const int64_t k = node % m.dims[2];
    lo[0] = m.origin[0] + i * m.res;
    lo[1] = m.origin[1] + j * m.res;
    lo[2] = m.origin[2] + k * m.res;
    return m.res;
  }
  const double w = std::ldexp(m.res, -m.depth[node]);
  for (int a = 0; a < 3; a++) {
    lo[a] = m.origin[a] + static_cast<double>(m.ijk[3 * node + a]) * w;
  }
  return w;
}

/**
 * @brief Is this node a leaf?
 */
static inline bool is_leaf(const MeshView &m, const int64_t node) {
  return m.first_child == nullptr || m.first_child[node] < 0;
}

/**
 * @brief Get the leaf index of a leaf node.
 */
static inline int64_t leaf_index(const MeshView &m, const int64_t node) {
  return m.first_child == nullptr ? node : m.node_leaf[node];
}

/**
 * @brief Get the root cell index containing a point (clamped to the domain).
 */
static inline int64_t root_containing(const MeshView &m, const double x[3]) {
  const int64_t i = clamp_index((x[0] - m.origin[0]) / m.res, m.dims[0]);
  const int64_t j = clamp_index((x[1] - m.origin[1]) / m.res, m.dims[1]);
  const int64_t k = clamp_index((x[2] - m.origin[2]) / m.res, m.dims[2]);
  return (i * m.dims[1] + j) * m.dims[2] + k;
}

/**
 * @brief Get the range of root cells along one axis overlapped by a support.
 *
 * Both the thread scheduler and the weight calculation use this function, so
 * the cells a particle touches are always exactly the cells it was scheduled
 * for (this is what makes the block colouring race free).
 *
 * @param m The mesh view.
 * @param x The particle position.
 * @param r The support radius.
 * @param a The axis.
 * @param lo Output first root index.
 * @param hi Output last root index.
 */
static inline void root_range(const MeshView &m, const double x[3],
                              const double r, const int a, int64_t &lo,
                              int64_t &hi) {
  lo = clamp_index((x[a] - r - m.origin[a]) / m.res, m.dims[a]);
  hi = clamp_index((x[a] + r - m.origin[a]) / m.res, m.dims[a]);
}

/**
 * @brief Find the leaf node containing a point (clamped to the domain).
 *
 * @param m The mesh view.
 * @param x The point.
 * @param width Output width of the containing leaf.
 *
 * @return The node index of the containing leaf.
 */
static inline int64_t leaf_node_containing(const MeshView &m,
                                           const double x[3], double &width) {
  int64_t node = root_containing(m, x);
  double lo[3];
  width = node_box(m, node, lo);
  while (!is_leaf(m, node)) {
    /* Pick the octant on the same side of the node centre as the point. */
    const double half = 0.5 * width;
    int oct = 0;
    if (x[0] >= lo[0] + half) oct |= 4;
    if (x[1] >= lo[1] + half) oct |= 2;
    if (x[2] >= lo[2] + half) oct |= 1;
    node = m.first_child[node] + oct;
    width = node_box(m, node, lo);
  }
  return node;
}

/**
 * @brief Compute the support radius of a particle.
 *
 * Smoothed particles are supported within their smoothing length. CIC
 * particles are cubes with the width of the leaf containing them, so the
 * "radius" is half that width.
 */
static inline double support_radius(const MeshView &m, const double x[3],
                                    const double h, const bool as_points) {
  if (!as_points) return h;
  double width;
  leaf_node_containing(m, x, width);
  return 0.5 * width;
}

/**
 * @brief Pick the Gauss-Legendre rule for one axis of an intersection box.
 *
 * Narrow extents (relative to h) see little kernel variation and need few
 * nodes; the widest possible extent (2h) gets 8.
 *
 * @param ext The extent of the box along the axis.
 * @param h The smoothing length.
 * @param x Output pointer to the nodes.
 * @param w Output pointer to the weights.
 *
 * @return The number of nodes.
 */
static inline int gauss_rule(const double ext, const double h,
                             const double *&x, const double *&w) {
  if (ext <= 0.25 * h) {
    x = GL2_X;
    w = GL2_W;
    return 2;
  }
  if (ext <= h) {
    x = GL4_X;
    w = GL4_W;
    return 4;
  }
  x = GL8_X;
  w = GL8_W;
  return 8;
}

/**
 * @brief Estimate the (unnormalised) kernel mass inside a box.
 *
 * The box is the intersection of a leaf with the kernel's bounding box, so it
 * is never wider than 2h along any axis. We integrate W(|r - x| / h) over it
 * with a tensor-product Gauss-Legendre rule chosen per axis by gauss_rule.
 * The kernel normalisation (1 / h^3) cancels when the particle's weights are
 * normalised, so it is not applied here.
 *
 * @param c The candidate holding the intersection box.
 * @param x The particle position.
 * @param h The particle smoothing length.
 * @param kernel The dimensionless kernel function W(q), zero for q > 1.
 *
 * @return The estimated kernel integral over the box.
 */
static inline double kernel_box_integral(const Candidate &c, const double x[3],
                                         const double h,
                                         kernel_func<double> kernel) {
  /* Map each axis' nodes into the box, relative to the particle. */
  const double *gx[3], *gw[3];
  int n[3];
  double offs[3][8];
  for (int a = 0; a < 3; a++) {
    const double half = 0.5 * (c.hi[a] - c.lo[a]);
    const double mid = 0.5 * (c.hi[a] + c.lo[a]) - x[a];
    n[a] = gauss_rule(2.0 * half, h, gx[a], gw[a]);
    for (int i = 0; i < n[a]; i++) offs[a][i] = mid + half * gx[a][i];
  }

  const double inv_h = 1.0 / h;
  double sum = 0.0;
  for (int i = 0; i < n[0]; i++) {
    const double dx2 = offs[0][i] * offs[0][i];
    for (int j = 0; j < n[1]; j++) {
      const double dxy2 = dx2 + offs[1][j] * offs[1][j];
      const double wij = gw[0][i] * gw[1][j];
      for (int k = 0; k < n[2]; k++) {
        const double r2 = dxy2 + offs[2][k] * offs[2][k];
        sum += wij * gw[2][k] * kernel(std::sqrt(r2) * inv_h);
      }
    }
  }

  /* The weights on [-1, 1] sum to 2 per axis, hence the factor of 1/8. */
  const double volume =
      (c.hi[0] - c.lo[0]) * (c.hi[1] - c.lo[1]) * (c.hi[2] - c.lo[2]);
  return 0.125 * sum * volume;
}

/**
 * @brief Compute the normalised leaf weights of a single particle.
 *
 * On return s.cands holds every leaf receiving a share of the particle and the
 * weights sum to exactly one (up to rounding).
 *
 * @param m The mesh view.
 * @param x The particle position.
 * @param h The smoothing length (ignored for CIC).
 * @param as_points Whether to use CIC rather than the smoothing kernel.
 * @param kernel The kernel function (unused for CIC).
 * @param s The per-thread scratch buffers.
 */
static void particle_weights(const MeshView &m, const double x[3],
                             const double h, const bool as_points,
                             kernel_func<double> kernel, Scratch &s) {
  s.cands.clear();

  /* A smoothed particle with no extent is a point: it belongs wholly to the
   * leaf containing it. */
  const double r = support_radius(m, x, h, as_points);
  if (!(r > 0.0)) {
    double width;
    const int64_t node = leaf_node_containing(m, x, width);
    s.cands.push_back({leaf_index(m, node), {0, 0, 0}, {0, 0, 0}, 1.0});
    return;
  }

  double blo[3], bhi[3];
  int64_t rlo[3], rhi[3];
  for (int a = 0; a < 3; a++) {
    blo[a] = x[a] - r;
    bhi[a] = x[a] + r;
    root_range(m, x, r, a, rlo[a], rhi[a]);
  }
  const double r2 = r * r;

  /* Walk every root cell overlapping the support box, descending into refined
   * roots, and record the leaves the support actually overlaps. */
  for (int64_t i = rlo[0]; i <= rhi[0]; i++) {
    for (int64_t j = rlo[1]; j <= rhi[1]; j++) {
      for (int64_t k = rlo[2]; k <= rhi[2]; k++) {
        s.stack.clear();
        s.stack.push_back((i * m.dims[1] + j) * m.dims[2] + k);
        while (!s.stack.empty()) {
          const int64_t node = s.stack.back();
          s.stack.pop_back();

          double lo[3];
          const double w = node_box(m, node, lo);

          /* Intersect the node with the support box, skipping nodes with no
           * overlapping volume. */
          Candidate c;
          bool overlaps = true;
          double d2 = 0.0;
          for (int a = 0; a < 3; a++) {
            c.lo[a] = std::max(lo[a], blo[a]);
            c.hi[a] = std::min(lo[a] + w, bhi[a]);
            if (!(c.hi[a] > c.lo[a])) overlaps = false;
            const double d = std::max({lo[a] - x[a], 0.0, x[a] - lo[a] - w});
            d2 += d * d;
          }
          if (!overlaps) continue;

          /* A spherical kernel cannot reach a node further away than h. */
          if (!as_points && d2 >= r2) continue;

          if (is_leaf(m, node)) {
            c.leaf = leaf_index(m, node);
            c.weight = 0.0;
            s.cands.push_back(c);
          } else {
            for (int oct = 0; oct < 8; oct++) {
              s.stack.push_back(m.first_child[node] + oct);
            }
          }
        }
      }
    }
  }

  /* A single overlapped leaf (or none, e.g. a particle outside the domain)
   * receives the whole particle without any integration. */
  if (s.cands.size() <= 1) {
    if (s.cands.empty()) {
      double width;
      const int64_t node = leaf_node_containing(m, x, width);
      s.cands.push_back({leaf_index(m, node), {0, 0, 0}, {0, 0, 0}, 0.0});
    }
    s.cands[0].weight = 1.0;
    return;
  }

  /* Raw weights: exact overlap volume for CIC cubes, sampled kernel integral
   * for smoothed particles. */
  double total = 0.0;
  for (Candidate &c : s.cands) {
    if (as_points) {
      c.weight =
          (c.hi[0] - c.lo[0]) * (c.hi[1] - c.lo[1]) * (c.hi[2] - c.lo[2]);
    } else {
      c.weight = kernel_box_integral(c, x, h, kernel);
    }
    total += c.weight;
  }

  /* Normalising makes the deposit exactly conservative. If sampling missed
   * the kernel entirely (only possible for pathological slivers) fall back to
   * the containing leaf. */
  if (!(total > 0.0)) {
    double width;
    const int64_t node = leaf_node_containing(m, x, width);
    s.cands.clear();
    s.cands.push_back({leaf_index(m, node), {0, 0, 0}, {0, 0, 0}, 1.0});
    return;
  }
  const double inv_total = 1.0 / total;
  for (Candidate &c : s.cands) c.weight *= inv_total;
}

/**
 * @brief Deposit every attribute of one particle onto the mesh.
 *
 * @param acc The accumulation buffer laid out as [leaf][attr] so that all
 *            attributes of a leaf are adjacent in memory.
 * @param counts Optional per-leaf contributor counts (refinement only).
 * @param min_h Optional per-leaf minimum smoothing length (refinement only).
 */
template <typename PartReal, typename ValueReal>
static inline void deposit_particle(
    const MeshView &m, const int64_t p, const PartReal *pos,
    const PartReal *sml, const std::vector<const ValueReal *> &values,
    const bool as_points, kernel_func<double> kernel, Scratch &s, double *acc,
    int64_t *counts, double *min_h) {
  const int nattrs = static_cast<int>(values.size());
  const double x[3] = {static_cast<double>(pos[3 * p]),
                       static_cast<double>(pos[3 * p + 1]),
                       static_cast<double>(pos[3 * p + 2])};
  const double h = as_points ? 0.0 : static_cast<double>(sml[p]);

  particle_weights(m, x, h, as_points, kernel, s);

  /* Gather this particle's attribute values once so the inner loop only
   * touches the contiguous leaf row. */
  for (int a = 0; a < nattrs; a++) {
    s.vals[a] = static_cast<double>(values[a][p]);
  }

  for (const Candidate &c : s.cands) {
    double *row = acc + c.leaf * nattrs;
    for (int a = 0; a < nattrs; a++) row[a] += s.vals[a] * c.weight;
    if (counts != nullptr) counts[c.leaf]++;
    if (min_h != nullptr) min_h[c.leaf] = std::min(min_h[c.leaf], h);
  }
}

/**
 * @brief Deposit all particles onto the mesh.
 *
 * Threading is race free and deterministic. The root grid is divided into
 * cubic blocks of root cells and each particle is assigned to the block
 * containing it. A particle is "local" if its support reaches no further than
 * the neighbouring blocks. Blocks are processed in 27 colours ((bx % 3,
 * by % 3, bz % 3)); blocks of one colour are at least three blocks apart along
 * some axis, so local particles of concurrently processed blocks can never
 * touch the same leaf. Particles are processed in passes: each pass sizes the
 * blocks so that 90% of the remaining particles are local, and leaves the
 * wider particles for the next, coarser pass.
 *
 * The block sizes depend only on the particles and the mesh, and each block's
 * particles are deposited in index order by a single thread, so the order in
 * which contributions are summed into any leaf (and hence the result) does not
 * depend on the number of threads.
 *
 * @param acc Zeroed accumulation buffer of size nleaf * nattrs ([leaf][attr]).
 * @param counts Optional zeroed per-leaf contributor counts.
 * @param min_h Optional per-leaf minimum smoothing lengths (initialised to
 *              the largest finite double).
 */
template <typename PartReal, typename ValueReal>
static void deposit_all(const MeshView &m, const PartReal *pos,
                        const PartReal *sml,
                        const std::vector<const ValueReal *> &values,
                        const int64_t npart, const bool as_points,
                        kernel_func<double> kernel, const int nthreads,
                        double *acc, int64_t *counts, double *min_h) {
  const int nattrs = static_cast<int>(values.size());

  /* Find each particle's range of root cells and home root cell. */
  std::vector<int64_t> lo(3 * npart), hi(3 * npart), mid(3 * npart);
#pragma omp parallel for num_threads(nthreads) schedule(static)
  for (int64_t p = 0; p < npart; p++) {
    const double x[3] = {static_cast<double>(pos[3 * p]),
                         static_cast<double>(pos[3 * p + 1]),
                         static_cast<double>(pos[3 * p + 2])};
    const double h = as_points ? 0.0 : static_cast<double>(sml[p]);
    const double r = std::max(0.0, support_radius(m, x, h, as_points));
    for (int a = 0; a < 3; a++) {
      root_range(m, x, r, a, lo[3 * p + a], hi[3 * p + a]);
      mid[3 * p + a] = clamp_index((x[a] - m.origin[a]) / m.res, m.dims[a]);
    }
  }

  std::vector<int64_t> pending(npart);
  for (int64_t p = 0; p < npart; p++) pending[p] = p;
  std::vector<int64_t> spans, block_of, block_start, order, remaining;
  std::vector<int64_t> colour_blocks[27];
  while (!pending.empty()) {
    const int64_t npending = static_cast<int64_t>(pending.size());

    /* Size the blocks so 90% of the remaining particles span no more than
     * one block along every axis; all such particles are local. */
    spans.resize(npending);
    for (int64_t i = 0; i < npending; i++) {
      const int64_t p = pending[i];
      int64_t span = 1;
      for (int a = 0; a < 3; a++) {
        span = std::max(span, hi[3 * p + a] - lo[3 * p + a] + 1);
      }
      spans[i] = span;
    }
    const int64_t q = (npending * 9) / 10;
    std::nth_element(spans.begin(), spans.begin() + q, spans.end());
    const int64_t bw = std::max<int64_t>(1, spans[q]);
    int64_t nb[3];
    for (int a = 0; a < 3; a++) nb[a] = (m.dims[a] + bw - 1) / bw;
    const int64_t nblock = nb[0] * nb[1] * nb[2];

    /* Assign local particles to blocks and defer the rest. */
    block_of.resize(npending);
    block_start.assign(nblock + 1, 0);
    remaining.clear();
    for (int64_t i = 0; i < npending; i++) {
      const int64_t p = pending[i];
      int64_t b[3];
      bool local = true;
      for (int a = 0; a < 3; a++) {
        b[a] = mid[3 * p + a] / bw;
        local = local && lo[3 * p + a] >= (b[a] - 1) * bw &&
                hi[3 * p + a] < (b[a] + 2) * bw;
      }
      if (local) {
        block_of[i] = (b[0] * nb[1] + b[1]) * nb[2] + b[2];
        block_start[block_of[i] + 1]++;
      } else {
        block_of[i] = -1;
        remaining.push_back(p);
      }
    }

    /* Stable counting sort of the local particles by block. */
    for (int64_t b = 0; b < nblock; b++) block_start[b + 1] += block_start[b];
    order.resize(block_start[nblock]);
    {
      std::vector<int64_t> fill(block_start.begin(), block_start.end() - 1);
      for (int64_t i = 0; i < npending; i++) {
        if (block_of[i] >= 0) order[fill[block_of[i]]++] = pending[i];
      }
    }

    /* Group the occupied blocks by colour. */
    for (auto &blocks : colour_blocks) blocks.clear();
    for (int64_t bx = 0; bx < nb[0]; bx++) {
      for (int64_t by = 0; by < nb[1]; by++) {
        for (int64_t bz = 0; bz < nb[2]; bz++) {
          const int64_t b = (bx * nb[1] + by) * nb[2] + bz;
          if (block_start[b + 1] == block_start[b]) continue;
          colour_blocks[(bx % 3) * 9 + (by % 3) * 3 + bz % 3].push_back(b);
        }
      }
    }

    /* Deposit colour by colour; blocks of one colour run concurrently. */
    for (const auto &blocks : colour_blocks) {
      const int64_t ncolour = static_cast<int64_t>(blocks.size());
      if (ncolour == 0) continue;
#pragma omp parallel num_threads(nthreads)
      {
        Scratch s;
        s.vals.resize(nattrs);
#pragma omp for schedule(dynamic, 1)
        for (int64_t ib = 0; ib < ncolour; ib++) {
          const int64_t b = blocks[ib];
          for (int64_t o = block_start[b]; o < block_start[b + 1]; o++) {
            deposit_particle(m, order[o], pos, sml, values, as_points, kernel,
                             s, acc, counts, min_h);
          }
        }
      }
    }

    pending.swap(remaining);
  }
}

/**
 * @brief Validate a sequence of per-particle value arrays.
 *
 * The returned pointers are borrowed; the Python argument tuple keeps the
 * arrays alive for the duration of the call.
 *
 * @return True on success, false with a Python exception set.
 */
static bool unpack_value_arrays(PyObject *values, const int64_t npart,
                                std::vector<PyArrayObject *> &arrays) {
  if (!PyTuple_Check(values) && !PyList_Check(values)) {
    PyErr_SetString(PyExc_TypeError, "values must be a tuple or list.");
    return false;
  }
  const Py_ssize_t count = PySequence_Size(values);
  if (count <= 0) {
    PyErr_SetString(PyExc_ValueError,
                    "values must contain at least one "
                    "array.");
    return false;
  }
  for (Py_ssize_t i = 0; i < count; i++) {
    /* Borrowed references: lists and tuples both support fast access. */
    PyObject *item = PyTuple_Check(values) ? PyTuple_GET_ITEM(values, i)
                                           : PyList_GET_ITEM(values, i);
    if (!PyArray_Check(item)) {
      PyErr_SetString(PyExc_TypeError, "values entries must be NumPy arrays.");
      return false;
    }
    PyArrayObject *arr = reinterpret_cast<PyArrayObject *>(item);
    if (PyArray_NDIM(arr) != 1 || PyArray_DIM(arr, 0) != npart) {
      PyErr_SetString(PyExc_ValueError,
                      "Each values array must have one entry per particle.");
      return false;
    }
    arrays.push_back(arr);
  }
  return true;
}

/**
 * @brief Validate and read the shared geometry arguments.
 *
 * @return True on success, false with a Python exception set.
 */
static bool parse_geometry(PyObject *py_origin, const double res,
                           PyObject *py_dims, MeshView &m) {
  if (!PyArg_ParseTuple(py_origin, "ddd", &m.origin[0], &m.origin[1],
                        &m.origin[2])) {
    return false;
  }
  long long dims[3];
  if (!PyArg_ParseTuple(py_dims, "LLL", &dims[0], &dims[1], &dims[2])) {
    return false;
  }
  if (!(res > 0.0)) {
    PyErr_SetString(PyExc_ValueError, "res must be positive.");
    return false;
  }
  for (int a = 0; a < 3; a++) {
    if (dims[a] <= 0) {
      PyErr_SetString(PyExc_ValueError, "dims must be positive.");
      return false;
    }
    m.dims[a] = dims[a];
  }
  m.res = res;
  m.nroot = m.dims[0] * m.dims[1] * m.dims[2];
  m.nleaf = m.nroot;
  m.first_child = nullptr;
  m.depth = nullptr;
  m.ijk = nullptr;
  m.node_leaf = nullptr;
  return true;
}

/**
 * @brief Validate the particle arrays and look up the kernel function.
 *
 * @return True on success, false with a Python exception set.
 */
static bool parse_particles(PyArrayObject *np_pos, PyObject *py_sml,
                            const int as_points, const char *kernel_name,
                            int64_t &npart, PyArrayObject *&np_sml,
                            int &part_typenum, kernel_func<double> &kernel) {
  if (PyArray_NDIM(np_pos) != 2 || PyArray_DIM(np_pos, 1) != 3) {
    PyErr_SetString(PyExc_ValueError, "pos must have shape (N, 3).");
    return false;
  }
  npart = PyArray_DIM(np_pos, 0);

  np_sml = nullptr;
  kernel = nullptr;
  if (!as_points) {
    np_sml = array_or_none(py_sml, "sml");
    if (np_sml == nullptr) {
      PyErr_SetString(PyExc_ValueError,
                      "Smoothed deposition requires smoothing lengths.");
      return false;
    }
    if (PyArray_NDIM(np_sml) != 1 || PyArray_DIM(np_sml, 0) != npart) {
      PyErr_SetString(PyExc_ValueError,
                      "sml must have one entry per particle.");
      return false;
    }
    kernel = get_kernel_function<double>(kernel_name);
    if (kernel == nullptr) {
      PyErr_Format(PyExc_ValueError, "Unknown kernel '%s'.", kernel_name);
      return false;
    }
  }

  PyArrayObject *arrays[2] = {np_pos, np_sml};
  const char *names[2] = {"pos", "sml"};
  return is_matching_float_dtypes(arrays, names, np_sml == nullptr ? 1 : 2,
                                  &part_typenum);
}

/**
 * @brief Check a tree array is a contiguous 1D/2D array of the given type.
 */
static bool check_tree_array(PyArrayObject *arr, const int typenum,
                             const npy_intp n, const int ncols,
                             const char *name) {
  const bool ok =
      PyArray_TYPE(arr) == typenum && PyArray_IS_C_CONTIGUOUS(arr) &&
      PyArray_DIM(arr, 0) == n &&
      (ncols == 0 ? PyArray_NDIM(arr) == 1
                  : PyArray_NDIM(arr) == 2 && PyArray_DIM(arr, 1) == ncols);
  if (!ok) {
    PyErr_Format(PyExc_ValueError, "Malformed mesh tree array '%s'.", name);
  }
  return ok;
}

/**
 * @brief Attach a refinement tree tuple (or None) to a mesh view.
 *
 * @param py_tree None for a uniform mesh, otherwise the tuple
 *                (first_child, node_depth, node_ijk, node_leaf, nleaf)
 *                returned by build_refined_mesh. The arrays are borrowed.
 * @param m The mesh view to update.
 *
 * @return True on success, false with a Python exception set.
 */
static bool attach_tree(PyObject *py_tree, MeshView &m) {
  if (py_tree == Py_None) return true;
  PyObject *fc, *dp, *ij, *nl;
  long long nleaf;
  if (!PyArg_ParseTuple(py_tree, "O!O!O!O!L", &PyArray_Type, &fc,
                        &PyArray_Type, &dp, &PyArray_Type, &ij, &PyArray_Type,
                        &nl, &nleaf)) {
    return false;
  }
  PyArrayObject *np_fc = reinterpret_cast<PyArrayObject *>(fc);
  const npy_intp nnode = PyArray_DIM(np_fc, 0);
  if (!check_tree_array(np_fc, NPY_INT64, nnode, 0, "first_child") ||
      !check_tree_array(reinterpret_cast<PyArrayObject *>(dp), NPY_INT32,
                        nnode, 0, "node_depth") ||
      !check_tree_array(reinterpret_cast<PyArrayObject *>(ij), NPY_INT64,
                        nnode, 3, "node_ijk") ||
      !check_tree_array(reinterpret_cast<PyArrayObject *>(nl), NPY_INT64,
                        nnode, 0, "node_leaf")) {
    return false;
  }
  if (nnode < m.nroot || nleaf <= 0) {
    PyErr_SetString(PyExc_ValueError, "Mesh tree does not match dims.");
    return false;
  }
  m.first_child = data_ptr<int64_t>(np_fc);
  m.depth = data_ptr<int32_t>(reinterpret_cast<PyArrayObject *>(dp));
  m.ijk = data_ptr<int64_t>(reinterpret_cast<PyArrayObject *>(ij));
  m.node_leaf = data_ptr<int64_t>(reinterpret_cast<PyArrayObject *>(nl));
  m.nleaf = nleaf;
  return true;
}

/**
 * @brief Typed implementation of deposit_to_mesh.
 */
template <typename PartReal, typename ValueReal, typename OutT>
static PyObject *deposit_impl(const MeshView &m, PyArrayObject *np_pos,
                              PyArrayObject *np_sml,
                              const std::vector<PyArrayObject *> &np_values,
                              const int64_t npart, const bool as_points,
                              kernel_func<double> kernel, const int nthreads) {
  const PartReal *pos = data_ptr<PartReal>(np_pos);
  const PartReal *sml =
      np_sml == nullptr ? nullptr : data_ptr<PartReal>(np_sml);
  std::vector<const ValueReal *> values;
  for (PyArrayObject *arr : np_values)
    values.push_back(data_ptr<ValueReal>(arr));
  const int nattrs = static_cast<int>(values.size());

  /* Accumulate in double in a [leaf][attr] layout for locality, then
   * transpose into the contiguous [attr][leaf] output. */
  std::vector<double> acc;
  try {
    acc.assign(static_cast<size_t>(m.nleaf) * nattrs, 0.0);
  } catch (const std::bad_alloc &) {
    PyErr_NoMemory();
    return nullptr;
  }

  tic("mesh.deposit");
  deposit_all(m, pos, sml, values, npart, as_points, kernel, nthreads,
              acc.data(), nullptr, nullptr);
  toc("mesh.deposit");

  npy_intp dims[2] = {nattrs, static_cast<npy_intp>(m.nleaf)};
  const int out_typenum =
      std::is_same_v<OutT, float> ? NPY_FLOAT32 : NPY_FLOAT64;
  PyArrayObject *out = reinterpret_cast<PyArrayObject *>(
      PyArray_EMPTY(2, dims, out_typenum, 0));
  if (out == nullptr) return nullptr;
  OutT *out_data = data_ptr<OutT>(out);

  tic("mesh.pack_output");
#pragma omp parallel for num_threads(nthreads) schedule(static)
  for (int64_t leaf = 0; leaf < m.nleaf; leaf++) {
    for (int a = 0; a < nattrs; a++) {
      out_data[a * m.nleaf + leaf] =
          static_cast<OutT>(acc[static_cast<size_t>(leaf) * nattrs + a]);
    }
  }
  toc("mesh.pack_output");

  return reinterpret_cast<PyObject *>(out);
}

/**
 * @brief Deposit particle attributes onto a uniform or refined mesh.
 *
 * Python signature:
 *   deposit_to_mesh(pos, sml, values, origin, res, dims, kernel_name,
 *                   as_points, tree, nthreads, out_dtype)
 *
 * @param pos (N, 3) particle positions.
 * @param sml (N,) smoothing lengths, or None when as_points is true. Must
 *            share the dtype of pos.
 * @param values Tuple of (N,) arrays sharing one float dtype.
 * @param origin (x, y, z) lower corner of the mesh domain.
 * @param res Root cell width.
 * @param dims (nx, ny, nz) number of root cells.
 * @param kernel_name Name of the SPH kernel (see kernel_functions.h).
 * @param as_points Use cloud-in-cell deposition instead of the kernel.
 * @param tree None for a uniform mesh, otherwise the tuple
 *             (first_child, node_depth, node_ijk, node_leaf, nleaf) returned
 *             by build_refined_mesh.
 * @param nthreads Number of OpenMP threads.
 * @param out_dtype Output dtype, or None to use the values dtype.
 *
 * @return An (nattrs, nleaf) array. For a uniform mesh nleaf = nx * ny * nz in
 *         C order.
 */
static PyObject *deposit_to_mesh(PyObject *self, PyObject *args) {
  (void)self;

  PyArrayObject *np_pos;
  PyObject *py_sml, *py_values, *py_origin, *py_dims, *py_tree, *out_dtype;
  double res;
  const char *kernel_name;
  int as_points, nthreads;
  if (!PyArg_ParseTuple(args, "O!OOOdOsiOiO", &PyArray_Type, &np_pos, &py_sml,
                        &py_values, &py_origin, &res, &py_dims, &kernel_name,
                        &as_points, &py_tree, &nthreads, &out_dtype)) {
    return nullptr;
  }

  MeshView m;
  if (!parse_geometry(py_origin, res, py_dims, m)) return nullptr;

  int64_t npart;
  PyArrayObject *np_sml;
  int part_typenum;
  kernel_func<double> kernel;
  if (!parse_particles(np_pos, py_sml, as_points, kernel_name, npart, np_sml,
                       part_typenum, kernel)) {
    return nullptr;
  }

  std::vector<PyArrayObject *> np_values;
  if (!unpack_value_arrays(py_values, npart, np_values)) return nullptr;
  std::vector<const char *> value_names(np_values.size(), "values");
  int value_typenum;
  if (!is_matching_float_dtypes(np_values.data(), value_names.data(),
                                static_cast<int>(np_values.size()),
                                &value_typenum)) {
    return nullptr;
  }

  /* Attach the refinement tree if one was supplied. */
  if (!attach_tree(py_tree, m)) return nullptr;

  const int out_typenum = out_dtype == Py_None
                              ? value_typenum
                              : resolve_output_typenum(out_dtype, "out_dtype");
  if (out_typenum < 0) return nullptr;

  return dispatch_float(part_typenum, [&](auto p) -> PyObject * {
    return dispatch_float(value_typenum, [&](auto v) -> PyObject * {
      return dispatch_float(out_typenum, [&](auto o) -> PyObject * {
        return deposit_impl<decltype(p), decltype(v), decltype(o)>(
            m, np_pos, np_sml, np_values, npart, as_points != 0, kernel,
            nthreads);
      });
    });
  });
}

/**
 * @brief The growable tree built during refinement.
 */
struct MeshTree {
  std::vector<int64_t> first_child;
  std::vector<int32_t> depth;
  std::vector<int64_t> ijk;
  std::vector<int64_t> node_leaf;
  int64_t nleaf = 0;

  /**
   * @brief Point a view at the current tree arrays.
   */
  void attach(MeshView &m) const {
    m.first_child = first_child.data();
    m.depth = depth.data();
    m.ijk = ijk.data();
    m.node_leaf = node_leaf.data();
    m.nleaf = nleaf;
  }

  /**
   * @brief Number leaves depth first so leaves are in Morton order within
   *        each root cell and root cells are in C order.
   */
  void number_leaves(const int64_t nroot) {
    nleaf = 0;
    std::vector<int64_t> stack;
    for (int64_t root = 0; root < nroot; root++) {
      stack.push_back(root);
      while (!stack.empty()) {
        const int64_t node = stack.back();
        stack.pop_back();
        if (first_child[node] < 0) {
          node_leaf[node] = nleaf++;
        } else {
          node_leaf[node] = -1;
          /* Push in reverse so octant 0 is visited first. */
          for (int oct = 7; oct >= 0; oct--) {
            stack.push_back(first_child[node] + oct);
          }
        }
      }
    }
  }
};

/**
 * @brief Typed implementation of build_refined_mesh.
 */
template <typename PartReal, typename ValueReal>
static bool build_impl(MeshView &m, MeshTree &tree, PyArrayObject *np_pos,
                       PyArrayObject *np_sml, PyArrayObject *np_refine,
                       const int64_t npart, const bool as_points,
                       kernel_func<double> kernel, const double threshold,
                       const int max_depth, const int nthreads) {
  const PartReal *pos = data_ptr<PartReal>(np_pos);
  const PartReal *sml =
      np_sml == nullptr ? nullptr : data_ptr<PartReal>(np_sml);
  const std::vector<const ValueReal *> values = {
      data_ptr<ValueReal>(np_refine)};

  /* Start from the uniform root grid with every root cell a leaf. */
  tree.first_child.assign(m.nroot, -1);
  tree.depth.assign(m.nroot, 0);
  tree.node_leaf.resize(m.nroot);
  tree.ijk.resize(3 * m.nroot);
  for (int64_t node = 0; node < m.nroot; node++) {
    tree.ijk[3 * node] = node / (m.dims[1] * m.dims[2]);
    tree.ijk[3 * node + 1] = (node / m.dims[2]) % m.dims[1];
    tree.ijk[3 * node + 2] = node % m.dims[2];
  }
  tree.number_leaves(m.nroot);

  /* Refine level by level: deposit the refinement variable onto the current
   * leaves, split every leaf that exceeds the threshold, and repeat until no
   * leaf wants splitting.
   * TODO: every level redeposits all particles; restrict to particles
   * touching split leaves if deep refinement becomes a bottleneck. */
  for (int level = 0; level < max_depth; level++) {
    tree.attach(m);
    std::vector<double> acc(m.nleaf, 0.0);
    std::vector<int64_t> counts(m.nleaf, 0);
    /* Use the largest finite double as "no contributor": the extensions are
     * built with -ffast-math, which assumes infinities never occur. */
    std::vector<double> min_h(m.nleaf, std::numeric_limits<double>::max());
    deposit_all(m, pos, sml, values, npart, as_points, kernel, nthreads,
                acc.data(), counts.data(), min_h.data());

    /* Decide which leaves to split. A leaf is split when its total exceeds
     * the threshold, unless splitting cannot add information: a single
     * contributor, children narrower than the smallest overlapping kernel, or
     * the depth limit. */
    const int64_t nnode = static_cast<int64_t>(tree.first_child.size());
    std::vector<int64_t> split;
    for (int64_t node = 0; node < nnode; node++) {
      if (tree.first_child[node] >= 0) continue;
      const int64_t leaf = tree.node_leaf[node];
      const double child_width = std::ldexp(m.res, -(tree.depth[node] + 1));
      if (acc[leaf] > threshold && counts[leaf] > 1 &&
          tree.depth[node] < max_depth &&
          (as_points || child_width >= min_h[leaf])) {
        split.push_back(node);
      }
    }
    if (split.empty()) break;

    /* Append eight children per split node. */
    for (const int64_t node : split) {
      tree.first_child[node] = static_cast<int64_t>(tree.first_child.size());
      for (int oct = 0; oct < 8; oct++) {
        tree.first_child.push_back(-1);
        tree.depth.push_back(tree.depth[node] + 1);
        tree.ijk.push_back(2 * tree.ijk[3 * node] + ((oct >> 2) & 1));
        tree.ijk.push_back(2 * tree.ijk[3 * node + 1] + ((oct >> 1) & 1));
        tree.ijk.push_back(2 * tree.ijk[3 * node + 2] + (oct & 1));
        tree.node_leaf.push_back(-1);
      }
    }
    tree.number_leaves(m.nroot);
  }
  return true;
}

/**
 * @brief Build an adaptively refined mesh from a refinement variable.
 *
 * Python signature:
 *   build_refined_mesh(pos, sml, refine_values, origin, res, dims,
 *                      kernel_name, as_points, threshold, max_depth, nthreads)
 *
 * A leaf is split into eight children while the deposited total of
 * refine_values inside it exceeds threshold, it has more than one contributing
 * particle, its children would be no narrower than the smallest smoothing
 * length overlapping it (smoothed mode only), and its depth is below
 * max_depth.
 *
 * @return The tuple (first_child, node_depth, node_ijk, node_leaf, nleaf)
 *         describing the tree (see the file header for the layout).
 */
static PyObject *build_refined_mesh(PyObject *self, PyObject *args) {
  (void)self;

  PyArrayObject *np_pos, *np_refine;
  PyObject *py_sml, *py_origin, *py_dims;
  double res, threshold;
  const char *kernel_name;
  int as_points, max_depth, nthreads;
  if (!PyArg_ParseTuple(args, "O!OO!OdOsidii", &PyArray_Type, &np_pos, &py_sml,
                        &PyArray_Type, &np_refine, &py_origin, &res, &py_dims,
                        &kernel_name, &as_points, &threshold, &max_depth,
                        &nthreads)) {
    return nullptr;
  }
  if (max_depth < 0 || max_depth > 60) {
    PyErr_SetString(PyExc_ValueError, "max_depth must be in [0, 60].");
    return nullptr;
  }

  MeshView m;
  if (!parse_geometry(py_origin, res, py_dims, m)) return nullptr;

  int64_t npart;
  PyArrayObject *np_sml;
  int part_typenum;
  kernel_func<double> kernel;
  if (!parse_particles(np_pos, py_sml, as_points, kernel_name, npart, np_sml,
                       part_typenum, kernel)) {
    return nullptr;
  }
  std::vector<PyArrayObject *> np_values = {np_refine};
  const char *value_names[1] = {"refine_values"};
  int value_typenum;
  if (PyArray_NDIM(np_refine) != 1 || PyArray_DIM(np_refine, 0) != npart) {
    PyErr_SetString(PyExc_ValueError,
                    "refine_values must have one entry per particle.");
    return nullptr;
  }
  if (!is_matching_float_dtypes(np_values.data(), value_names, 1,
                                &value_typenum)) {
    return nullptr;
  }

  MeshTree tree;
  try {
    tic("mesh.refine");
    dispatch_float(part_typenum, [&](auto p) {
      return dispatch_float(value_typenum, [&](auto v) {
        return build_impl<decltype(p), decltype(v)>(
            m, tree, np_pos, np_sml, np_refine, npart, as_points != 0, kernel,
            threshold, max_depth, nthreads);
      });
    });
    toc("mesh.refine");
  } catch (const std::bad_alloc &) {
    PyErr_NoMemory();
    return nullptr;
  }

  /* Copy the tree into NumPy arrays. */
  const npy_intp nnode = static_cast<npy_intp>(tree.first_child.size());
  npy_intp dims1[1] = {nnode};
  npy_intp dims2[2] = {nnode, 3};
  PyArrayObject *fc =
      reinterpret_cast<PyArrayObject *>(PyArray_EMPTY(1, dims1, NPY_INT64, 0));
  PyArrayObject *dp =
      reinterpret_cast<PyArrayObject *>(PyArray_EMPTY(1, dims1, NPY_INT32, 0));
  PyArrayObject *ij =
      reinterpret_cast<PyArrayObject *>(PyArray_EMPTY(2, dims2, NPY_INT64, 0));
  PyArrayObject *nl =
      reinterpret_cast<PyArrayObject *>(PyArray_EMPTY(1, dims1, NPY_INT64, 0));
  if (fc == nullptr || dp == nullptr || ij == nullptr || nl == nullptr) {
    Py_XDECREF(fc);
    Py_XDECREF(dp);
    Py_XDECREF(ij);
    Py_XDECREF(nl);
    return nullptr;
  }
  std::copy(tree.first_child.begin(), tree.first_child.end(),
            data_ptr<int64_t>(fc));
  std::copy(tree.depth.begin(), tree.depth.end(), data_ptr<int32_t>(dp));
  std::copy(tree.ijk.begin(), tree.ijk.end(), data_ptr<int64_t>(ij));
  std::copy(tree.node_leaf.begin(), tree.node_leaf.end(),
            data_ptr<int64_t>(nl));

  return Py_BuildValue("NNNNL", fc, dp, ij, nl,
                       static_cast<long long>(tree.nleaf));
}

/**
 * @brief Typed implementation of cell_index.
 */
template <typename PartReal>
static void cell_index_impl(const MeshView &m, const PartReal *points,
                            const int64_t npoints, int64_t *out,
                            const int nthreads) {
#pragma omp parallel for num_threads(nthreads) schedule(static)
  for (int64_t p = 0; p < npoints; p++) {
    const double x[3] = {static_cast<double>(points[3 * p]),
                         static_cast<double>(points[3 * p + 1]),
                         static_cast<double>(points[3 * p + 2])};

    /* Points outside the domain have no cell. The upper faces count as
     * inside so the full closed domain is covered. */
    bool inside = true;
    for (int a = 0; a < 3; a++) {
      const double f = (x[a] - m.origin[a]) / m.res;
      if (!(f >= 0.0 && f <= static_cast<double>(m.dims[a]))) inside = false;
    }
    if (!inside) {
      out[p] = -1;
      continue;
    }

    double width;
    out[p] = leaf_index(m, leaf_node_containing(m, x, width));
  }
}

/**
 * @brief Find the cell (leaf) containing each of a set of points.
 *
 * Python signature:
 *   cell_index(points, origin, res, dims, tree, nthreads)
 *
 * For a uniform mesh this is index arithmetic; for a refined mesh the root
 * cell is found the same way and the tree is descended by octant, so no
 * separate search structure is needed.
 *
 * @param points (N, 3) float32 or float64 positions.
 * @param origin (x, y, z) lower corner of the mesh domain.
 * @param res Root cell width.
 * @param dims (nx, ny, nz) number of root cells.
 * @param tree None or the refinement tree tuple (see deposit_to_mesh).
 * @param nthreads Number of OpenMP threads.
 *
 * @return An (N,) int64 array of flat cell indices (C order for a uniform
 *         mesh, leaf order for a refined mesh), -1 for points outside the
 *         domain.
 */
static PyObject *cell_index(PyObject *self, PyObject *args) {
  (void)self;

  PyArrayObject *np_points;
  PyObject *py_origin, *py_dims, *py_tree;
  double res;
  int nthreads;
  if (!PyArg_ParseTuple(args, "O!OdOOi", &PyArray_Type, &np_points, &py_origin,
                        &res, &py_dims, &py_tree, &nthreads)) {
    return nullptr;
  }

  MeshView m;
  if (!parse_geometry(py_origin, res, py_dims, m)) return nullptr;
  if (!attach_tree(py_tree, m)) return nullptr;

  if (PyArray_NDIM(np_points) != 2 || PyArray_DIM(np_points, 1) != 3) {
    PyErr_SetString(PyExc_ValueError, "points must have shape (N, 3).");
    return nullptr;
  }
  PyArrayObject *arrays[1] = {np_points};
  const char *names[1] = {"points"};
  int typenum;
  if (!is_matching_float_dtypes(arrays, names, 1, &typenum)) return nullptr;
  const int64_t npoints = PyArray_DIM(np_points, 0);

  npy_intp dims[1] = {static_cast<npy_intp>(npoints)};
  PyArrayObject *out =
      reinterpret_cast<PyArrayObject *>(PyArray_EMPTY(1, dims, NPY_INT64, 0));
  if (out == nullptr) return nullptr;

  dispatch_float(typenum, [&](auto p) {
    using PartReal = decltype(p);
    cell_index_impl<PartReal>(m, data_ptr<PartReal>(np_points), npoints,
                              data_ptr<int64_t>(out), nthreads);
  });

  return reinterpret_cast<PyObject *>(out);
}

static PyMethodDef MeshMethods[] = {
    {"deposit_to_mesh", (PyCFunction)deposit_to_mesh, METH_VARARGS,
     "Deposit particle attributes onto a uniform or refined mesh."},
    {"build_refined_mesh", (PyCFunction)build_refined_mesh, METH_VARARGS,
     "Build an adaptively refined mesh from a refinement variable."},
    {"cell_index", (PyCFunction)cell_index, METH_VARARGS,
     "Find the cell containing each of a set of points."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "mesh",
    "Conservative particle-to-mesh deposition.",
    -1,
    MeshMethods,
    NULL,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC PyInit_mesh(void) {
  PyObject *m = PyModule_Create(&moduledef);
  if (m == NULL) return NULL;
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
