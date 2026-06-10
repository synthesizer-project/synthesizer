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
 *
 * This is the AoS (Array of Structures) representation used during tree
 * building. After construction completes, leaf cell data is converted to
 * SoA (Structure of Arrays) layout for cache-friendly rendering.
 */
struct particle_2d {

  /* Position of the particle in the image plane. */
  double pos[2];

  /* Smoothing length of the particle. */
  double sml;

  /* Square of the smoothing length (pre-computed). */
  double sml_squ;

  /*! The index of the particle in the original pix_values array. */
  int index;
};

/**
 * @brief A node in the 2D quadtree.
 *
 * Internal nodes store progeny pointers and have zero particle count.
 * Leaf nodes store particles in SoA layout for efficient rendering.
 * The AoS `particles` pointer is used only during construction and is
 * freed after SoA conversion.
 */
struct quadcell {

  /* Location and width (square cell). */
  double loc[2];
  double width;

  /* Is it split? */
  int split;

  /* How deep? */
  int depth;

  /* Number of particles in this cell (0 for internal nodes after SoA
   * conversion, >0 for leaves). */
  int part_count;

  /* SoA particle storage (used by renderer) — NULL for internal nodes.
   * Allocated as a single contiguous aligned block per leaf. */
  double *pos_x;
  double *pos_y;
  double *sml_arr;
  double *sml_squ_arr;
  int *indices;

  /* AoS particle storage (used during construction only). Freed after
   * SoA conversion. */
  struct particle_2d *particles;

  /* Store the square of the maximum smoothing length in this subtree. */
  double max_sml_squ;

  /* Pointers to cells below this one (4 children: SW, SE, NW, NE). */
  struct quadcell *progeny;

  /* The maximum depth in the cell tree below this cell. */
  int maxdepth;
};

/* Prototypes. */
void construct_quadtree(const double *pos_x, const double *pos_y,
                        const double *sml, const int npart,
                        struct quadcell *root, int *ncells, int maxdepth,
                        int min_count, double res);
void cleanup_quadtree(struct quadcell *c);
double min_dist2_quadcell_to_rect(const struct quadcell *c, double rx_min,
                                  double rx_max, double ry_min, double ry_max);
void query_quadtree_for_pixel(const struct quadcell *c, double px_min,
                              double px_max, double py_min, double py_max,
                              double threshold,
                              std::vector<const struct quadcell *> &out_leaves,
                              std::vector<int> &out_starts,
                              std::vector<int> &out_counts);

#endif  // QUADTREE_H_
