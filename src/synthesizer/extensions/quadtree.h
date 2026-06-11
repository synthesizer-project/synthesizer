/******************************************************************************
 * A header containing the definitions for constructing and manipulating a
 * 2D quadtree for pixel-centric SPH image rendering.
 *****************************************************************************/
#ifndef QUADTREE_H_
#define QUADTREE_H_

/* C headers. */
#include <stdint.h>

#include <vector>

/* Define the maximum tree depth. */
#define QUADTREE_MAX_DEPTH 64

/**
 * @brief A 2D particle used during quadtree construction.
 */
struct particle_2d {
  double pos[2];
  double sml;
  double sml_squ;
  int index;
};

/**
 * @brief A node in the 2D quadtree.
 *
 * Internal nodes have 4 progeny; leaves store particles in SoA layout.
 */
struct quadcell {
  double loc[2];
  double width;
  int split;
  int depth;
  int part_count;

  /* SoA particle storage (renderer) — NULL for internal nodes. */
  double *pos_x;
  double *pos_y;
  double *sml_arr;
  double *sml_squ_arr;
  int *indices;

  /* AoS particle storage (construction only). */
  struct particle_2d *particles;

  double max_sml_squ;
  struct quadcell *progeny; /* 4 children, or NULL */
  int maxdepth;
};

/**
 * @brief Top-level quadtree container.
 *
 * A ``top_grid`` × ``top_grid`` grid of independent quadtrees, each
 * rooted at a grid cell covering a sub-region of the domain.
 */
struct quadtree {
  int top_grid;
  double cell_width;
  struct quadcell *grid; /* top_grid * top_grid cells */
};

/* Prototypes. */
struct quadtree *construct_quadtree(const double *pos_x, const double *pos_y,
                                    const double *sml, int npart, int maxdepth,
                                    int min_count, double res, int top_grid);
void cleanup_quadtree(struct quadtree *tree);
double min_dist2_quadcell_to_rect(const struct quadcell *c, double rx_min,
                                  double rx_max, double ry_min, double ry_max);
void query_quadtree_for_pixel(const struct quadtree *tree, double px_min,
                              double px_max, double py_min, double py_max,
                              double threshold,
                              std::vector<const struct quadcell *> &out_leaves,
                              std::vector<int> &out_starts,
                              std::vector<int> &out_counts);

#endif  // QUADTREE_H_
