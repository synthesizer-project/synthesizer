/******************************************************************************
 * A module containing the definitions for constructing and manipulating a
 * 2D quadtree for pixel-centric SPH image rendering.
 *****************************************************************************/
#include "quadtree.h"

#include <Python.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#include "property_funcs.h"
#include "timers.h"

/* ------------------------------------------------------------------ */
/*  Recursive subdivision (standard 4-way quadtree)                   */
/* ------------------------------------------------------------------ */

static void populate_quadtree_recursive(struct quadcell *c, int *ncells,
                                        int maxdepth, int depth, int min_count,
                                        double res) {
  if (depth > maxdepth) return;
  struct particle_2d *particles = c->particles;
  int npart = c->part_count;
  if (c->width < sqrt(c->max_sml_squ) || npart < min_count ||
      c->width < res / 2.0)
    return;

  double width = c->width / 2;
  c->split = 1;
  c->progeny = new struct quadcell[4];
  *ncells += 4;
  for (int ip = 0; ip < 4; ip++) {
    struct quadcell *cp = &c->progeny[ip];
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

  int part_count[4] = {0, 0, 0, 0};
  for (int ipart = 0; ipart < npart; ipart++) {
    double ipos[2] = {particles[ipart].pos[0] - c->loc[0],
                      particles[ipart].pos[1] - c->loc[1]};
    part_count[(ipos[1] > width) + 2 * (ipos[0] > width)]++;
  }
  for (int ip = 0; ip < 4; ip++) {
    if (part_count[ip] == 0) continue;
    c->progeny[ip].particles = new struct particle_2d[part_count[ip]];
    c->progeny[ip].part_count = 0;
  }
  for (int ipart = 0; ipart < npart; ipart++) {
    double ipos[2] = {particles[ipart].pos[0] - c->loc[0],
                      particles[ipart].pos[1] - c->loc[1]};
    int ci = (ipos[1] > width) + 2 * (ipos[0] > width);
    struct quadcell *cp = &c->progeny[ci];
    cp->particles[cp->part_count++] = particles[ipart];
    if (particles[ipart].sml > cp->max_sml_squ)
      cp->max_sml_squ = particles[ipart].sml;
  }
  for (int ip = 0; ip < 4; ip++) {
    struct quadcell *cp = &c->progeny[ip];
    if (cp->part_count == 0) continue;
    cp->max_sml_squ = cp->max_sml_squ * cp->max_sml_squ;
    populate_quadtree_recursive(cp, ncells, maxdepth, depth + 1, min_count,
                                res);
    if (cp->maxdepth > c->maxdepth) c->maxdepth = cp->maxdepth;
  }
}

/* ------------------------------------------------------------------ */
/*  SoA conversion                                                    */
/* ------------------------------------------------------------------ */

static void convert_leaves_to_soa(struct quadcell *c) {
  if (c->split) {
    for (int ip = 0; ip < 4; ip++) {
      struct quadcell *cp = &c->progeny[ip];
      if (cp->part_count > 0 || cp->split) convert_leaves_to_soa(cp);
    }
    c->pos_x = NULL;
    c->pos_y = NULL;
    c->sml_arr = NULL;
    c->sml_squ_arr = NULL;
    c->indices = NULL;
    return;
  }
  int n = c->part_count;
  if (n == 0) {
    c->pos_x = NULL;
    c->pos_y = NULL;
    c->sml_arr = NULL;
    c->sml_squ_arr = NULL;
    c->indices = NULL;
    return;
  }
  size_t dbytes = n * sizeof(double), ibytes = n * sizeof(int);
  size_t total = (4 * dbytes + ibytes + 63) & ~((size_t)63);
  char *block = (char *)aligned_alloc(64, total);
  c->pos_x = (double *)block;
  c->pos_y = (double *)(block + dbytes);
  c->sml_arr = (double *)(block + 2 * dbytes);
  c->sml_squ_arr = (double *)(block + 3 * dbytes);
  c->indices = (int *)(block + 4 * dbytes);
  struct particle_2d *parts = c->particles;
  for (int i = 0; i < n; i++) {
    c->pos_x[i] = parts[i].pos[0];
    c->pos_y[i] = parts[i].pos[1];
    c->sml_arr[i] = parts[i].sml;
    c->sml_squ_arr[i] = parts[i].sml_squ;
    c->indices[i] = parts[i].index;
  }
  delete[] c->particles;
  c->particles = NULL;
}

/* ------------------------------------------------------------------ */
/*  Particle creation                                                 */
/* ------------------------------------------------------------------ */

static void construct_particles_2d(struct particle_2d *parts,
                                   const double *pos_x, const double *pos_y,
                                   const double *sml, int npart,
                                   double *bounds) {
  bounds[0] = FLT_MAX;
  bounds[1] = -FLT_MAX;
  bounds[2] = FLT_MAX;
  bounds[3] = -FLT_MAX;
  for (int ip = 0; ip < npart; ip++) {
    parts[ip].pos[0] = pos_x[ip];
    parts[ip].pos[1] = pos_y[ip];
    parts[ip].sml = sml[ip];
    parts[ip].sml_squ = sml[ip] * sml[ip];
    parts[ip].index = ip;
    if (parts[ip].pos[0] < bounds[0]) bounds[0] = parts[ip].pos[0];
    if (parts[ip].pos[0] > bounds[1]) bounds[1] = parts[ip].pos[0];
    if (parts[ip].pos[1] < bounds[2]) bounds[2] = parts[ip].pos[1];
    if (parts[ip].pos[1] > bounds[3]) bounds[3] = parts[ip].pos[1];
  }
}

/* ------------------------------------------------------------------ */
/*  Construction                                                      */
/* ------------------------------------------------------------------ */

struct quadtree *construct_quadtree(const double *pos_x, const double *pos_y,
                                    const double *sml, int npart, int maxdepth,
                                    int min_count, double res, int top_grid) {

  tic("construct_quadtree");

  struct particle_2d *parts = new struct particle_2d[npart];
  double bounds[4];
  construct_particles_2d(parts, pos_x, pos_y, sml, npart, bounds);

  double width = bounds[1] - bounds[0];
  if (bounds[3] - bounds[2] > width) width = bounds[3] - bounds[2];
  width *= 1.0001;
  double mid[2] = {0.5 * (bounds[0] + bounds[1]),
                   0.5 * (bounds[2] + bounds[3])};
  bounds[0] = mid[0] - 0.5 * width;
  bounds[1] = mid[0] + 0.5 * width;
  bounds[2] = mid[1] - 0.5 * width;
  bounds[3] = mid[1] + 0.5 * width;

  int n_top = top_grid * top_grid;
  struct quadtree *tree = new struct quadtree;
  tree->top_grid = top_grid;
  tree->cell_width = width / top_grid;
  tree->grid = new struct quadcell[n_top];
  int ncells = n_top;

  for (int ip = 0; ip < n_top; ip++) {
    struct quadcell *cp = &tree->grid[ip];
    cp->width = tree->cell_width;
    cp->loc[0] = bounds[0] + (ip / top_grid) * tree->cell_width;
    cp->loc[1] = bounds[2] + (ip % top_grid) * tree->cell_width;
    cp->split = 0;
    cp->part_count = 0;
    cp->max_sml_squ = 0;
    cp->depth = 0;
    cp->particles = NULL;
    cp->progeny = NULL;
    cp->maxdepth = 0;
    cp->pos_x = NULL;
    cp->pos_y = NULL;
    cp->sml_arr = NULL;
    cp->sml_squ_arr = NULL;
    cp->indices = NULL;
  }

  int *part_count = new int[n_top]();
  for (int ipart = 0; ipart < npart; ipart++) {
    int gi = (int)((parts[ipart].pos[0] - bounds[0]) / tree->cell_width);
    int gj = (int)((parts[ipart].pos[1] - bounds[2]) / tree->cell_width);
    if (gi < 0)
      gi = 0;
    else if (gi >= top_grid)
      gi = top_grid - 1;
    if (gj < 0)
      gj = 0;
    else if (gj >= top_grid)
      gj = top_grid - 1;
    part_count[gj + gi * top_grid]++;
  }
  for (int ip = 0; ip < n_top; ip++) {
    if (part_count[ip] == 0) continue;
    tree->grid[ip].particles = new struct particle_2d[part_count[ip]];
  }
  for (int ipart = 0; ipart < npart; ipart++) {
    int gi = (int)((parts[ipart].pos[0] - bounds[0]) / tree->cell_width);
    int gj = (int)((parts[ipart].pos[1] - bounds[2]) / tree->cell_width);
    if (gi < 0)
      gi = 0;
    else if (gi >= top_grid)
      gi = top_grid - 1;
    if (gj < 0)
      gj = 0;
    else if (gj >= top_grid)
      gj = top_grid - 1;
    int idx = gj + gi * top_grid;
    struct quadcell *cp = &tree->grid[idx];
    cp->particles[cp->part_count++] = parts[ipart];
    if (parts[ipart].sml > cp->max_sml_squ) cp->max_sml_squ = parts[ipart].sml;
  }
  delete[] part_count;
  delete[] parts;

  for (int ip = 0; ip < n_top; ip++) {
    struct quadcell *cp = &tree->grid[ip];
    if (cp->part_count == 0) continue;
    cp->max_sml_squ = cp->max_sml_squ * cp->max_sml_squ;
    populate_quadtree_recursive(cp, &ncells, maxdepth, 1, min_count, res);
  }

  tic("convert_leaves_to_soa");
  for (int ip = 0; ip < n_top; ip++) convert_leaves_to_soa(&tree->grid[ip]);
  toc("convert_leaves_to_soa");

  toc("construct_quadtree");
  return tree;
}

/* ------------------------------------------------------------------ */
/*  Teardown                                                          */
/* ------------------------------------------------------------------ */

static void cleanup_quadcell(struct quadcell *c) {
  if (c->split) {
    for (int i = 0; i < 4; i++) cleanup_quadcell(&c->progeny[i]);
    delete[] c->progeny;
  }
  if (c->pos_x != NULL) {
    free(c->pos_x);
    c->pos_x = c->pos_y = c->sml_arr = c->sml_squ_arr = NULL;
    c->indices = NULL;
  }
  if (c->particles != NULL) {
    delete[] c->particles;
    c->particles = NULL;
  }
}

void cleanup_quadtree(struct quadtree *tree) {
  int n = tree->top_grid * tree->top_grid;
  for (int i = 0; i < n; i++) cleanup_quadcell(&tree->grid[i]);
  delete[] tree->grid;
  delete tree;
}

/* ------------------------------------------------------------------ */
/*  Rect-to-rect distance (B1)                                        */
/* ------------------------------------------------------------------ */

double min_dist2_quadcell_to_rect(const struct quadcell *c, double rx_min,
                                  double rx_max, double ry_min,
                                  double ry_max) {
  double dx = 0, dy = 0;
  double cx_max = c->loc[0] + c->width;
  if (rx_max < c->loc[0])
    dx = c->loc[0] - rx_max;
  else if (rx_min > cx_max)
    dx = rx_min - cx_max;
  double cy_max = c->loc[1] + c->width;
  if (ry_max < c->loc[1])
    dy = c->loc[1] - ry_max;
  else if (ry_min > cy_max)
    dy = ry_min - cy_max;
  return dx * dx + dy * dy;
}

/* ------------------------------------------------------------------ */
/*  Query                                                             */
/* ------------------------------------------------------------------ */

static void query_quadcell_recursive(
    const struct quadcell *c, double px_min, double px_max, double py_min,
    double py_max, double threshold,
    std::vector<const struct quadcell *> &out_leaves,
    std::vector<int> &out_starts, std::vector<int> &out_counts) {

  double kern_r = threshold * sqrt(c->max_sml_squ);
  double cx_min = c->loc[0] - kern_r;
  double cx_max = c->loc[0] + c->width + kern_r;
  double cy_min = c->loc[1] - kern_r;
  double cy_max = c->loc[1] + c->width + kern_r;
  double dx = 0, dy = 0;
  if (px_max < cx_min)
    dx = cx_min - px_max;
  else if (px_min > cx_max)
    dx = px_min - cx_max;
  if (py_max < cy_min)
    dy = cy_min - py_max;
  else if (py_min > cy_max)
    dy = py_min - cy_max;
  if (dx * dx + dy * dy > 0) return;

  if (c->split) {
    for (int ip = 0; ip < 4; ip++) {
      const struct quadcell *cp = &c->progeny[ip];
      if (cp->part_count > 0 || cp->split)
        query_quadcell_recursive(cp, px_min, px_max, py_min, py_max, threshold,
                                 out_leaves, out_starts, out_counts);
    }
  } else if (c->part_count > 0) {
    out_leaves.push_back(c);
    out_starts.push_back(0);
    out_counts.push_back(c->part_count);
  }
}

void query_quadtree_for_pixel(const struct quadtree *tree, double px_min,
                              double px_max, double py_min, double py_max,
                              double threshold,
                              std::vector<const struct quadcell *> &out_leaves,
                              std::vector<int> &out_starts,
                              std::vector<int> &out_counts) {

  /* Iterate all top-level grid cells.  The per-cell recursive pruning
   * inside query_quadcell_recursive accounts for kernel extension beyond
   * cell bounds, so we don't need to pre-filter the grid here. */
  int n = tree->top_grid * tree->top_grid;
  for (int i = 0; i < n; i++) {
    const struct quadcell *c = &tree->grid[i];
    if (c->part_count > 0 || c->split)
      query_quadcell_recursive(c, px_min, px_max, py_min, py_max, threshold,
                               out_leaves, out_starts, out_counts);
  }
}
