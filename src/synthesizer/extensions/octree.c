/******************************************************************************
 * A module containing the defintions for constructing and manipulating an
 * octree.
 *****************************************************************************/
#include "octree.h"

/* C headers. */
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Python headers. */
#include <Python.h>

/* Local headers. */
#include "property_funcs.h"
#include "timers.h"

/**
 * @brief Recursively Populates the cell tree until maxdepth is reached.
 *
 * @param c The cell to populate.
 * @param ncells The number of cells.
 * @param maxdepth The maximum depth of the tree.
 * @param depth The current depth.
 * @param min_count The minimum number of particles in a leaf cell.
 */
static void populate_cell_tree_recursive(struct cell *c, int *ncells,
                                         int maxdepth, int depth,
                                         int min_count) {

  /* Have we reached the bottom? */
  if (depth > maxdepth) {
    PyErr_SetString(PyExc_ValueError, "Maximum depth reached in cell tree!");
    return;
  }

  /* Get the particles in this cell. */
  struct particle *particles = c->particles;
  int npart = c->part_count;

  /* No point splitting below the maximum smoothing length or if we have too
   * few particles. */
  if (c->width < sqrt(c->max_sml_squ) || npart < min_count) {
    return;
  }

  /* Compute the width at this level. */
  double width = c->width / 2;

  /* We need to split... get the progeny. */
  c->split = 1;
  c->progeny = synth_malloc(8 * sizeof(struct cell), "progeny");
  *ncells += 8;
  for (int ip = 0; ip < 8; ip++) {

    /* Get this progeny cell. */
    struct cell *cp = &c->progeny[ip];

    /* Set the cell properties. */
    cp->width = width;
    cp->loc[0] = c->loc[0];
    cp->loc[1] = c->loc[1];
    cp->loc[2] = c->loc[2];
    if (ip & 4)
      cp->loc[0] += cp->width;
    if (ip & 2)
      cp->loc[1] += cp->width;
    if (ip & 1)
      cp->loc[2] += cp->width;
    cp->split = 0;
    cp->part_count = 0;
    cp->max_sml_squ = 0;
    cp->depth = depth;
    cp->particles = NULL;
    cp->progeny = NULL;
    cp->maxdepth = depth;
  }

  /* Loop over particles first counting them. */
  int part_count[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int ipart = 0; ipart < npart; ipart++) {

    /* Get the position of the particle relative to the parent cell. */
    double ipos[3] = {
        particles[ipart].pos[0] - c->loc[0],
        particles[ipart].pos[1] - c->loc[1],
        particles[ipart].pos[2] - c->loc[2],
    };

    /* Get the integer cell location. */
    int i = ipos[0] > width;
    int j = ipos[1] > width;
    int k = ipos[2] > width;

    /* Get the child index. */
    int child_index = k + 2 * (j + 2 * i);

    /* Count the particles. */
    part_count[child_index]++;
  }

  /* Allocate the particles. */
  for (int ip = 0; ip < 8; ip++) {
    /* Only allocate non-zero particle counts. */
    if (part_count[ip] == 0) {
      continue;
    }

    /* Allocate the particles. */
    c->progeny[ip].particles = malloc(part_count[ip] * sizeof(struct particle));
    c->progeny[ip].part_count = 0;
  }

  /* Loop over particles again, this time assigning them. */
  for (int ipart = 0; ipart < npart; ipart++) {

    /* Get the position of the particle relative to the parent cell. */
    double ipos[3] = {
        particles[ipart].pos[0] - c->loc[0],
        particles[ipart].pos[1] - c->loc[1],
        particles[ipart].pos[2] - c->loc[2],
    };

    /* Get the integer cell location. */
    int i = ipos[0] > width;
    int j = ipos[1] > width;
    int k = ipos[2] > width;

    /* Get the child index. */
    int child_index = k + 2 * (j + 2 * i);

    /* Assign to the cell. */
    struct cell *cp = &c->progeny[child_index];
    cp->particles[cp->part_count++] = particles[ipart];

    /* Updated the maximum smoothing length. */
    if (particles[ipart].sml > cp->max_sml_squ) {
      cp->max_sml_squ = particles[ipart].sml;
    }
  }

#ifdef WITH_DEBUGGING_CHECKS
  /* Check that the particles have been assigned correctly. */
  for (int ip = 0; ip < 8; ip++) {
    struct cell *cp = &c->progeny[ip];
    if (cp->part_count != part_count[ip]) {
      printf("Error: Particles not assigned correctly!\n");
    }
  }

  /* Ensure the cell and particle positions agree. */
  for (int ip = 0; ip < 8; ip++) {
    struct cell *cp = &c->progeny[ip];
    for (int ipart = 0; ipart < cp->part_count; ipart++) {
      struct particle *pp = &cp->particles[ipart];

      if (pp->pos[0] < cp->loc[0] || pp->pos[0] > cp->loc[0] + cp->width) {
        printf("Error: Particle outside cell bounds in x (c->loc[0] = %f, "
               "c->loc[0] + c->width = %f, pp->pos[0] = %f)!\n",
               cp->loc[0], cp->loc[0] + cp->width, pp->pos[0]);
      }
      if (pp->pos[1] < cp->loc[1] || pp->pos[1] > cp->loc[1] + cp->width) {
        printf("Error: Particle outside cell bounds in y (c->loc[1] = %f, "
               "c->loc[1] + c->width = %f, pp->pos[1] = %f)!\n",
               cp->loc[1], cp->loc[1] + cp->width, pp->pos[1]);
      }
      if (pp->pos[2] < cp->loc[2] || pp->pos[2] > cp->loc[2] + cp->width) {
        printf("Error: Particle outside cell bounds in z (c->loc[2] = %f, "
               "c->loc[2] + c->width = %f, pp->pos[2] = %f)!\n",
               cp->loc[2], cp->loc[2] + cp->width, pp->pos[2]);
      }
    }
  }
#endif

  /* Recurse... */
  for (int ip = 0; ip < 8; ip++) {
    struct cell *cp = &c->progeny[ip];

    /* Skip any empty progeny. */
    if (cp->part_count == 0) {
      continue;
    }

    /* Square the maximum smoothing length. */
    cp->max_sml_squ = cp->max_sml_squ * cp->max_sml_squ;

    /* Go to the next level */
    populate_cell_tree_recursive(cp, ncells, maxdepth, depth + 1, min_count);

    /* Update the maximum depth. */
    if (cp->maxdepth > c->maxdepth) {
      c->maxdepth = cp->maxdepth;
    }
  }
}

/**
 * @brief Constructs the particles and attaches them to the root cell.
 *
 * @param particles The particles to construct.
 * @param pos The particle positions.
 * @param sml The particle smoothing lengths.
 * @param surf_den_vals The particle value that will be used when calculating
 * the surface density.
 * @param npart The number of particles.
 * @param root The root cell.
 */
static void construct_particles(struct particle *particles, const double *pos,
                                const double *sml, const double *surf_den_vals,
                                const int npart, struct cell *root) {

  double part_start = tic();

  /* Create an array to hold the bounds of the particle distribution. */
  double bounds[6] = {FLT_MAX, 0, FLT_MAX, 0, FLT_MAX, 0};

  /* Loop over gas particles and associate them with the root. We could
   * just attach the pointer but we already need to find the maximum sml in
   * a loop so might as well loop over them as we attach them. */
  for (int ip = 0; ip < npart; ip++) {

    /* Attach the particle properties */
    particles[ip].pos[0] = pos[ip * 3];
    particles[ip].pos[1] = pos[ip * 3 + 1];
    particles[ip].pos[2] = pos[ip * 3 + 2];
    particles[ip].sml = sml[ip];
    particles[ip].surf_den_var = surf_den_vals[ip];

    /* Update the bounds. */
    for (int i = 0; i < 3; i++) {
      if (particles[ip].pos[i] < bounds[i * 2])
        bounds[i * 2] = particles[ip].pos[i];
      if (particles[ip].pos[i] > bounds[i * 2 + 1])
        bounds[i * 2 + 1] = particles[ip].pos[i];
    }

    /* Updated the maximum smoothing length. */
    if (particles[ip].sml > root->max_sml_squ) {
      root->max_sml_squ = particles[ip].sml;
    }
  }

  /* Get the cell width based on the bounds we have found. Note that
   * we are assuming a cubic domain so the maximum width is the width. */
  double width = bounds[1] - bounds[0];
  if (bounds[3] - bounds[2] > width)
    width = bounds[3] - bounds[2];
  if (bounds[5] - bounds[4] > width)
    width = bounds[5] - bounds[4];

  /* Include a small buffer on the width. */
  width *= 1.0001;

  /* Get the geometric mid point. */
  double mid[3] = {0.5 * (bounds[0] + bounds[1]), 0.5 * (bounds[2] + bounds[3]),
                   0.5 * (bounds[4] + bounds[5])};

  /* Recalculate the bounds using the width and midpoint. */
  for (int i = 0; i < 3; i++) {
    bounds[i * 2] = mid[i] - (0.5 * width);
    bounds[i * 2 + 1] = mid[i] + (0.5 * width);
  }

  /* Set the root cell properties. */
  root->loc[0] = bounds[0];
  root->loc[1] = bounds[2];
  root->loc[2] = bounds[4];
  root->width = width;

  /* Square the maximum smoothing length. */
  root->max_sml_squ = root->max_sml_squ * root->max_sml_squ;

  /* Attach the particles to the root. */
  root->particles = particles;
  root->part_count = npart;

  toc("Particle construction and assignment", part_start);
}

/**
 * @brief Constructs the cell tree.
 *
 * We use a single cell at the root. This is then split into 8 cells, which are
 * then split into 8 cells each, and so on until we reach the maximum depth.
 *
 * @param pos The particle positions.
 * @param sml The particle smoothing lengths.
 * @param surf_den_val The particle value that will be used when calculating
 * the surface density.
 * @param npart The number of particles.
 * @param cells The cells to populate. (Can be expanded as needed.)
 * @param ncells The number of cells.
 * @param tot_cells The total number of cells.
 * @param maxdepth The maximum depth of the tree.
 * @param min_count The minimum number of particles in a leaf cell.
 */
void construct_cell_tree(const double *pos, const double *sml,
                         const double *surf_den_val, const int npart,
                         struct cell *root, int ncells, int maxdepth,
                         int min_count) {

  double cell_tree_start = tic();

  /* Set the root cell properties. */
  root->loc[0] = 0;
  root->loc[1] = 0;
  root->loc[2] = 0;
  root->width = 0;
  root->split = 0;
  root->part_count = 0;
  root->max_sml_squ = 0;
  root->depth = 0;

  /* Allocate the array of tree particles. */
  struct particle *parts =
      synth_malloc(npart * sizeof(struct particle), "particles");

  /* Create the particles and attach them to the root. */
  construct_particles(parts, pos, sml, surf_den_val, npart, root);

  /* And recurse... */
  populate_cell_tree_recursive(root, &ncells, maxdepth, 1, min_count);

#ifdef WITH_DEBUGGING_CHECKS
  printf("Constructed cell tree with %d cells\n", ncells);
  printf("Maximum depth: %d\n", root->maxdepth);
  printf("Cell bounds: %f %f %f %f %f %f\n", root->loc[0],
         root->loc[0] + root->width, root->loc[1], root->loc[1] + root->width,
         root->loc[2], root->loc[2] + root->width);
#endif

  toc("Cell tree construction", cell_tree_start);
}

/**
 * @brief Clean up the cell tree recursively.
 *
 * @param c The cell to clean up.
 */
void cleanup_cell_tree(struct cell *c) {
  if (c->split) {
    for (int i = 0; i < 8; i++) {
      cleanup_cell_tree(&c->progeny[i]);
    }
    free(c->progeny);
  }

  /* Clear up particles if we have them. */
  if (c->part_count > 0) {
    free(c->particles);
  }

  if (c->depth == 0) {
    free(c);
  }
}

/**
 * @brief A function to compute the minimum possible projected distance between
 * two cells.
 *
 * @param c The cell to test.
 * @param x The x-coordinate.
 * @param y The y-coordinate.
 *
 * @return The distance between the cells.
 */
double min_projected_dist2(struct cell *c, double x, double y) {

  /* Get the minimum separation along each axis (if the point is within the cell
   * along an axis then the separation should be 0). */
  double dx = 0;
  double dy = 0;
  if (!(x > c->loc[0] && x < c->loc[0] + c->width)) {
    dx = fmin(fabs(c->loc[0] - x), fabs(c->loc[0] + c->width - x));
  }
  if (!(y > c->loc[1] && y < c->loc[1] + c->width)) {
    dy = fmin(fabs(c->loc[1] - y), fabs(c->loc[1] + c->width - y));
  }

  return dx * dx + dy * dy;
}
