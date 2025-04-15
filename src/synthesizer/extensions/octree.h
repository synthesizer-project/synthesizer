/******************************************************************************
 * A header containing the defintions for constructing and manipulating an
 * octree.
 *****************************************************************************/
#ifndef OCTREE_H_
#define OCTREE_H_

/* C headers. */
#include <stdint.h>

/* Define the maximum tree depth. */
#define MAX_DEPTH 64

/**
 * @brief A particle to be contained in a cell.
 */
struct particle {

  /* Position of the particle. */
  double pos[3];

  /* Smoothing length of the particle. */
  double sml;

  /* Surface density variable. */
  double surf_den_var;
};

/**
 * @brief A cell to contain gas particles.
 */
struct cell {

  /* Location and width */
  double loc[3];
  double width;

  /* Is it split? */
  int split;

  /* How deep? */
  int depth;

  /* Pointers to particles in cell. */
  int part_count;
  struct particle *particles;

  /* Store the square of the maximum smoothing length. */
  double max_sml_squ;

  /* Pointers to cells below this one. */
  struct cell *progeny;

  /* The maximum depth in the cell tree. */
  int maxdepth;
};

/* Prototypes. */
void construct_cell_tree(const double *pos, const double *sml,
                         const double *surf_den_val, const int npart,
                         struct cell *root, int ncells, int maxdepth,
                         int min_count);
void cleanup_cell_tree(struct cell *c);
double min_projected_dist2(struct cell *c, double x, double y);

#endif // OCTREE_H_
