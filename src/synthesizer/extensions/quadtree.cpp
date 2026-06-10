/******************************************************************************
 * A module containing the definitions for constructing and manipulating a
 * 2D quadtree for pixel-centric SPH image rendering.
 *****************************************************************************/
#include "quadtree.h"

/* C headers. */
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* C++ headers. */
#include <vector>

/* Python headers. */
#include <Python.h>

/* Local headers. */
#include "property_funcs.h"
#include "timers.h"

/**
 * @brief Recursively populates the quadtree until maxdepth or other stopping
 * criteria are reached.
 *
 * @param c The quadcell to populate.
 * @param ncells The number of cells (incremented as cells are added).
 * @param maxdepth The maximum depth of the tree.
 * @param depth The current depth.
 * @param min_count The minimum number of particles in a leaf cell.
 * @param res The pixel resolution (for G1: sub-pixel early stop).
 */
static void populate_quadtree_recursive(struct quadcell *c, int *ncells,
                                        int maxdepth, int depth, int min_count,
                                        double res) {

  /* Have we reached the bottom? */
  if (depth > maxdepth) {
    return;
  }

  /* Get the particles in this cell. */
  struct particle_2d *particles = c->particles;
  int npart = c->part_count;

  /* No point splitting below the maximum smoothing length, if we have too
   * few particles, or if the cell is smaller than half a pixel (G1:
   * sub-pixel cells provide no spatial discrimination benefit). */
  if (c->width < sqrt(c->max_sml_squ) || npart < min_count ||
      c->width < res / 2.0) {
    return;
  }

  /* Compute the width at this level. */
  double width = c->width / 2;

  /* We need to split... get the progeny. */
  c->split = 1;
  c->progeny = new struct quadcell[4];
  *ncells += 4;
  for (int ip = 0; ip < 4; ip++) {

    /* Get this progeny cell. */
    struct quadcell *cp = &c->progeny[ip];

    /* Set the cell properties. */
    cp->width = width;
    cp->loc[0] = c->loc[0];
    cp->loc[1] = c->loc[1];
    if (ip & 2) cp->loc[0] += cp->width;
    if (ip & 1) cp->loc[1] += cp->width;
    cp->split = 0;
    cp->part_count = 0;
    cp->max_sml_squ = 0;
    cp->depth = depth;
    cp->particles = NULL;
    cp->progeny = NULL;
    cp->maxdepth = depth;
    cp->pos_x = NULL;
    cp->pos_y = NULL;
    cp->sml_arr = NULL;
    cp->sml_squ_arr = NULL;
    cp->indices = NULL;
  }

  /* Loop over particles first counting them. */
  int part_count[4] = {0, 0, 0, 0};
  for (int ipart = 0; ipart < npart; ipart++) {

    /* Get the position of the particle relative to the parent cell. */
    double ipos[2] = {
        particles[ipart].pos[0] - c->loc[0],
        particles[ipart].pos[1] - c->loc[1],
    };

    /* Get the integer cell location. */
    int i = ipos[0] > width;
    int j = ipos[1] > width;

    /* Get the child index (Z-order: j + 2*i). */
    int child_index = j + 2 * i;

    /* Count the particles. */
    part_count[child_index]++;
  }

  /* Allocate the particles. */
  for (int ip = 0; ip < 4; ip++) {
    /* Only allocate non-zero particle counts. */
    if (part_count[ip] == 0) {
      continue;
    }

    /* Allocate the particles. */
    c->progeny[ip].particles = new struct particle_2d[part_count[ip]];
    c->progeny[ip].part_count = 0;
  }

  /* Loop over particles again, this time assigning them. */
  for (int ipart = 0; ipart < npart; ipart++) {

    /* Get the position of the particle relative to the parent cell. */
    double ipos[2] = {
        particles[ipart].pos[0] - c->loc[0],
        particles[ipart].pos[1] - c->loc[1],
    };

    /* Get the integer cell location. */
    int i = ipos[0] > width;
    int j = ipos[1] > width;

    /* Get the child index. */
    int child_index = j + 2 * i;

    /* Assign to the cell. */
    struct quadcell *cp = &c->progeny[child_index];
    cp->particles[cp->part_count++] = particles[ipart];

    /* Updated the maximum smoothing length (unsquared at this stage). */
    if (particles[ipart].sml > cp->max_sml_squ) {
      cp->max_sml_squ = particles[ipart].sml;
    }
  }

#ifdef WITH_DEBUGGING_CHECKS
  /* Check that the particles have been assigned correctly. */
  for (int ip = 0; ip < 4; ip++) {
    struct quadcell *cp = &c->progeny[ip];
    if (cp->part_count != part_count[ip]) {
      printf("Error: Particles not assigned correctly!\n");
    }
  }

  /* Ensure the cell and particle positions agree. */
  for (int ip = 0; ip < 4; ip++) {
    struct quadcell *cp = &c->progeny[ip];
    for (int ipart = 0; ipart < cp->part_count; ipart++) {
      struct particle_2d *pp = &cp->particles[ipart];

      if (pp->pos[0] < cp->loc[0] || pp->pos[0] > cp->loc[0] + cp->width) {
        printf(
            "Error: Particle outside cell bounds in x (c->loc[0] = %f, "
            "c->loc[0] + c->width = %f, pp->pos[0] = %f)!\n",
            cp->loc[0], cp->loc[0] + cp->width, pp->pos[0]);
      }
      if (pp->pos[1] < cp->loc[1] || pp->pos[1] > cp->loc[1] + cp->width) {
        printf(
            "Error: Particle outside cell bounds in y (c->loc[1] = %f, "
            "c->loc[1] + c->width = %f, pp->pos[1] = %f)!\n",
            cp->loc[1], cp->loc[1] + cp->width, pp->pos[1]);
      }
    }
  }
#endif

  /* Recurse... */
  for (int ip = 0; ip < 4; ip++) {
    struct quadcell *cp = &c->progeny[ip];

    /* Skip any empty progeny. */
    if (cp->part_count == 0) {
      continue;
    }

    /* Square the maximum smoothing length. */
    cp->max_sml_squ = cp->max_sml_squ * cp->max_sml_squ;

    /* Go to the next level */
    populate_quadtree_recursive(cp, ncells, maxdepth, depth + 1, min_count,
                                res);

    /* Update the maximum depth. */
    if (cp->maxdepth > c->maxdepth) {
      c->maxdepth = cp->maxdepth;
    }
  }
}

/**
 * @brief Constructs the 2D particles and attaches them to the root cell.
 *
 * @param parts The particles array to populate.
 * @param pos_x The particle x positions.
 * @param pos_y The particle y positions.
 * @param sml The particle smoothing lengths.
 * @param npart The number of particles.
 * @param root The root quadcell.
 */
static void construct_particles_2d(struct particle_2d *parts,
                                   const double *pos_x, const double *pos_y,
                                   const double *sml, const int npart,
                                   struct quadcell *root) {

  tic("construct_particles_2d");

  /* Create an array to hold the 2D bounds of the particle distribution. */
  double bounds[4] = {FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX};

  /* Loop over particles and associate them with the root. We could
   * just attach the pointer but we already need to find the maximum sml in
   * a loop so might as well loop over them as we attach them. */
  for (int ip = 0; ip < npart; ip++) {

    /* Attach the particle properties. */
    parts[ip].pos[0] = pos_x[ip];
    parts[ip].pos[1] = pos_y[ip];
    parts[ip].sml = sml[ip];
    parts[ip].sml_squ = sml[ip] * sml[ip];
    parts[ip].index = ip;

    /* Update the bounds. */
    if (parts[ip].pos[0] < bounds[0]) bounds[0] = parts[ip].pos[0];
    if (parts[ip].pos[0] > bounds[1]) bounds[1] = parts[ip].pos[0];
    if (parts[ip].pos[1] < bounds[2]) bounds[2] = parts[ip].pos[1];
    if (parts[ip].pos[1] > bounds[3]) bounds[3] = parts[ip].pos[1];

    /* Updated the maximum smoothing length (unsquared at this stage). */
    if (parts[ip].sml > root->max_sml_squ) {
      root->max_sml_squ = parts[ip].sml;
    }
  }

  /* Get the cell width based on the bounds we have found. Note that
   * we are assuming a square domain so the maximum width is the width. */
  double width = bounds[1] - bounds[0];
  if (bounds[3] - bounds[2] > width) width = bounds[3] - bounds[2];

  /* Include a small buffer on the width. */
  width *= 1.0001;

  /* Get the geometric mid point. */
  double mid[2] = {0.5 * (bounds[0] + bounds[1]),
                   0.5 * (bounds[2] + bounds[3])};

  /* Recalculate the bounds using the width and midpoint. */
  bounds[0] = mid[0] - (0.5 * width);
  bounds[1] = mid[0] + (0.5 * width);
  bounds[2] = mid[1] - (0.5 * width);
  bounds[3] = mid[1] + (0.5 * width);

  /* Set the root cell properties. */
  root->loc[0] = bounds[0];
  root->loc[1] = bounds[2];
  root->width = width;

  /* Square the maximum smoothing length. */
  root->max_sml_squ = root->max_sml_squ * root->max_sml_squ;

  /* Attach the particles to the root. */
  root->particles = parts;
  root->part_count = npart;

  toc("construct_particles_2d");
}

/**
 * @brief Recursively converts leaf cells from AoS to SoA layout.
 *
 * After construction, leaves store particles in AoS format
 * (struct particle_2d array). This function converts each leaf to SoA
 * (separate arrays for pos_x, pos_y, sml, sml_squ, indices) stored in a
 * single cache-aligned allocation (C2). The AoS array is freed.
 *
 * Internal nodes have all SoA pointers set to NULL.
 *
 * @param c The quadcell to convert.
 */
static void convert_leaves_to_soa(struct quadcell *c) {

  /* Is the cell split? Recurse into progeny. */
  if (c->split) {
    for (int ip = 0; ip < 4; ip++) {
      struct quadcell *cp = &c->progeny[ip];
      if (cp->part_count > 0 || cp->split) {
        convert_leaves_to_soa(cp);
      }
    }
    /* Internal node: ensure SoA pointers are NULL. */
    c->pos_x = NULL;
    c->pos_y = NULL;
    c->sml_arr = NULL;
    c->sml_squ_arr = NULL;
    c->indices = NULL;
    return;
  }

  /* Leaf cell: convert AoS to SoA. */
  int n = c->part_count;
  if (n == 0) {
    c->pos_x = NULL;
    c->pos_y = NULL;
    c->sml_arr = NULL;
    c->sml_squ_arr = NULL;
    c->indices = NULL;
    return;
  }

  /* Compute total block size: 4 double arrays + 1 int array, rounded up
   * to 64 bytes for cache-aligned allocation (C2). */
  size_t double_bytes = n * sizeof(double);
  size_t int_bytes = n * sizeof(int);
  size_t total_bytes = 4 * double_bytes + int_bytes;
  /* Round up to next multiple of 64. */
  total_bytes = (total_bytes + 63) & ~((size_t)63);

  /* Allocate a single cache-aligned block. */
  char *block = (char *)aligned_alloc(64, total_bytes);
  if (block == NULL) {
    printf("Error: Failed to allocate SoA block for leaf cell!\n");
    return;
  }

  /* Set SoA pointers into the block (C1). */
  c->pos_x = (double *)(block);
  c->pos_y = (double *)(block + double_bytes);
  c->sml_arr = (double *)(block + 2 * double_bytes);
  c->sml_squ_arr = (double *)(block + 3 * double_bytes);
  c->indices = (int *)(block + 4 * double_bytes);

  /* Copy data from AoS to SoA. */
  struct particle_2d *parts = c->particles;
  for (int i = 0; i < n; i++) {
    c->pos_x[i] = parts[i].pos[0];
    c->pos_y[i] = parts[i].pos[1];
    c->sml_arr[i] = parts[i].sml;
    c->sml_squ_arr[i] = parts[i].sml_squ;
    c->indices[i] = parts[i].index;
  }

  /* Free the AoS array. */
  delete[] c->particles;
  c->particles = NULL;
}

/**
 * @brief Constructs the quadtree.
 *
 * We use a single cell at the root. This is then split into 4 cells, which
 * are then split into 4 cells each, and so on until we reach the maximum
 * depth or other stopping criteria. After construction, leaf cells are
 * converted from AoS to SoA layout for efficient rendering.
 *
 * @param pos_x The particle x positions.
 * @param pos_y The particle y positions.
 * @param sml The particle smoothing lengths.
 * @param npart The number of particles.
 * @param root The root quadcell (allocated by caller).
 * @param ncells The number of cells (incremented during build).
 * @param maxdepth The maximum depth of the tree.
 * @param min_count The minimum number of particles in a leaf cell.
 * @param res The pixel resolution (for G1: sub-pixel early stop).
 */
void construct_quadtree(const double *pos_x, const double *pos_y,
                        const double *sml, const int npart,
                        struct quadcell *root, int *ncells, int maxdepth,
                        int min_count, double res) {

  tic("construct_quadtree");

  /* Set the root cell properties. */
  root->loc[0] = 0;
  root->loc[1] = 0;
  root->width = 0;
  root->split = 0;
  root->part_count = 0;
  root->max_sml_squ = 0;
  root->depth = 0;
  root->progeny = NULL;
  root->pos_x = NULL;
  root->pos_y = NULL;
  root->sml_arr = NULL;
  root->sml_squ_arr = NULL;
  root->indices = NULL;

  /* Allocate the array of tree particles (AoS). */
  struct particle_2d *parts = new struct particle_2d[npart];

  /* Create the particles and attach them to the root. */
  construct_particles_2d(parts, pos_x, pos_y, sml, npart, root);

  /* And recurse... */
  *ncells = 1;
  populate_quadtree_recursive(root, ncells, maxdepth, 1, min_count, res);

  /* Convert leaf cells to SoA layout (C1, C2). */
  tic("convert_leaves_to_soa");
  convert_leaves_to_soa(root);
  toc("convert_leaves_to_soa");

#ifdef WITH_DEBUGGING_CHECKS
  printf("Constructed quadtree with %d cells\n", *ncells);
  printf("Maximum depth: %d\n", root->maxdepth);
  printf("Cell bounds: x=[%f, %f], y=[%f, %f]\n", root->loc[0],
         root->loc[0] + root->width, root->loc[1], root->loc[1] + root->width);
#endif

  toc("construct_quadtree");
}

/**
 * @brief Clean up the quadtree recursively.
 *
 * Frees SoA blocks for leaf cells, progeny arrays for internal nodes,
 * and the root cell itself.
 *
 * @param c The quadcell to clean up.
 */
void cleanup_quadtree(struct quadcell *c) {

  /* Recurse into progeny first. */
  if (c->split) {
    for (int i = 0; i < 4; i++) {
      cleanup_quadtree(&c->progeny[i]);
    }
    delete[] c->progeny;
  }

  /* Free the SoA block for leaves. The entire block is freed via pos_x
   * since all SoA arrays live in a single allocation (C1/C2). */
  if (c->pos_x != NULL) {
    free(c->pos_x);
    c->pos_x = NULL;
    c->pos_y = NULL;
    c->sml_arr = NULL;
    c->sml_squ_arr = NULL;
    c->indices = NULL;
  }

  /* Free any remaining AoS particles (should not happen after SoA
   * conversion, but handle gracefully). */
  if (c->particles != NULL) {
    delete[] c->particles;
    c->particles = NULL;
  }

  /* Free the root cell itself. */
  if (c->depth == 0) {
    delete c;
  }
}

/**
 * @brief Compute the minimum squared distance between a quadcell and an
 * axis-aligned rectangle.
 *
 * This generalises min_projected_dist2 to rectangular targets (B1),
 * enabling tighter pruning by accounting for pixel extent. Returns zero
 * if the cell and rectangle intersect.
 *
 * @param c The quadcell.
 * @param rx_min The minimum x of the rectangle.
 * @param rx_max The maximum x of the rectangle.
 * @param ry_min The minimum y of the rectangle.
 * @param ry_max The maximum y of the rectangle.
 *
 * @return The minimum squared distance.
 */
double min_dist2_quadcell_to_rect(const struct quadcell *c, double rx_min,
                                  double rx_max, double ry_min,
                                  double ry_max) {

  /* Get the minimum separation along each axis. If the cell and rectangle
   * overlap along an axis, the separation is 0. */
  double dx = 0;
  double dy = 0;

  /* Compute x-axis separation. */
  double cell_x_max = c->loc[0] + c->width;
  if (rx_max < c->loc[0]) {
    /* Rectangle is entirely left of the cell. */
    dx = c->loc[0] - rx_max;
  } else if (rx_min > cell_x_max) {
    /* Rectangle is entirely right of the cell. */
    dx = rx_min - cell_x_max;
  }
  /* Otherwise they overlap in x, dx stays 0. */

  /* Compute y-axis separation. */
  double cell_y_max = c->loc[1] + c->width;
  if (ry_max < c->loc[1]) {
    /* Rectangle is entirely below the cell. */
    dy = c->loc[1] - ry_max;
  } else if (ry_min > cell_y_max) {
    /* Rectangle is entirely above the cell. */
    dy = ry_min - cell_y_max;
  }
  /* Otherwise they overlap in y, dy stays 0. */

  return dx * dx + dy * dy;
}

/**
 * @brief Recursively query the quadtree for particles whose kernels may
 * overlap a pixel rectangle.
 *
 * Traverses the tree, pruning cells that cannot possibly contain a particle
 * whose kernel reaches the pixel (B1). For leaves that pass the prune,
 * the entire SoA particle range is appended as a candidate range (A1).
 *
 * @param c The quadcell to traverse.
 * @param px_min The minimum x of the pixel.
 * @param px_max The maximum x of the pixel.
 * @param py_min The minimum y of the pixel.
 * @param py_max The maximum y of the pixel.
 * @param threshold The kernel support threshold.
 * @param out_leaves Vector to append candidate leaf pointers to.
 * @param out_starts Vector to append start indices to.
 * @param out_counts Vector to append particle counts to.
 */
static void query_quadtree_for_pixel_recursive(
    const struct quadcell *c, double px_min, double px_max, double py_min,
    double py_max, double threshold,
    std::vector<const struct quadcell *> &out_leaves,
    std::vector<int> &out_starts, std::vector<int> &out_counts) {

  /* Prune: can any particle in this subtree reach the pixel? (B1) */
  double thresh_squ = threshold * threshold;
  if (thresh_squ * c->max_sml_squ <
      min_dist2_quadcell_to_rect(c, px_min, px_max, py_min, py_max)) {
    return;
  }

  /* Is the cell split? */
  if (c->split) {

    /* Recurse into non-empty progeny. */
    for (int ip = 0; ip < 4; ip++) {
      const struct quadcell *cp = &c->progeny[ip];

      /* Skip empty progeny. */
      if (cp->part_count == 0 && !cp->split) {
        continue;
      }

      query_quadtree_for_pixel_recursive(cp, px_min, px_max, py_min, py_max,
                                         threshold, out_leaves, out_starts,
                                         out_counts);
    }

  } else {

    /* Leaf cell: all particles are candidates (A1).
     * Per-particle filtering happens in the render loop. */
    if (c->part_count > 0) {
      out_leaves.push_back(c);
      out_starts.push_back(0);
      out_counts.push_back(c->part_count);
    }
  }
}

/**
 * @brief Query the quadtree for particles whose kernels may overlap a pixel.
 *
 * This is the public entry point. It collects candidate leaf ranges for
 * the given pixel rectangle. The renderer then filters per-particle for
 * each sub-pixel sample point.
 *
 * @param c The root quadcell.
 * @param px_min The minimum x of the pixel.
 * @param px_max The maximum x of the pixel.
 * @param py_min The minimum y of the pixel.
 * @param py_max The maximum y of the pixel.
 * @param threshold The kernel support threshold.
 * @param out_leaves Vector to append candidate leaf pointers to.
 * @param out_starts Vector to append start indices to.
 * @param out_counts Vector to append particle counts to.
 */
void query_quadtree_for_pixel(const struct quadcell *c, double px_min,
                              double px_max, double py_min, double py_max,
                              double threshold,
                              std::vector<const struct quadcell *> &out_leaves,
                              std::vector<int> &out_starts,
                              std::vector<int> &out_counts) {

  query_quadtree_for_pixel_recursive(c, px_min, px_max, py_min, py_max,
                                     threshold, out_leaves, out_starts,
                                     out_counts);
}
