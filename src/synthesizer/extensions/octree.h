/******************************************************************************
 * A header containing the defintions for constructing and manipulating an
 * octree.
 *****************************************************************************/
#ifndef OCTREE_H_
#define OCTREE_H_

/* C headers. */
#include <stdint.h>

/* Local headers. */
#include "data_types.h"

/* Define the maximum tree depth. */
#define MAX_DEPTH 64

/**
 * @brief A particle to be contained in a cell.
 */
struct particle {

  /* Position of the particle. */
  FLOAT pos[3];

  /* Smoothing length of the particle. */
  FLOAT sml;

  /* Surface density variable. */
  FLOAT surf_den_var;

  /*! The index of the particle in the original array. */
  int index;
};

/**
 * @brief A cell to contain gas particles.
 */
struct cell {

  /* Location and width */
  FLOAT loc[3];
  FLOAT width;

  /* Is it split? */
  int split;

  /* How deep? */
  int depth;

  /* Pointers to particles in cell. */
  int part_count;
  struct particle *particles;

  /* Store the square of the maximum smoothing length. */
  FLOAT max_sml_squ;

  /* Pointers to cells below this one. */
  struct cell *progeny;

  /* The maximum depth in the cell tree. */
  int maxdepth;
};

/* Prototypes. */
void construct_cell_tree(const FLOAT *pos, const FLOAT *sml,
                         const FLOAT *surf_den_val, const int npart,
                         struct cell *root, int ncells, int maxdepth,
                         int min_count);
void cleanup_cell_tree(struct cell *c);
FLOAT min_projected_dist2(struct cell *c, FLOAT x, FLOAT y);

#endif // OCTREE_H_
