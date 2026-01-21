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
  Float pos[3];

  /* Smoothing length of the particle. */
  Float sml;

  /* Surface density variable. */
  Float surf_den_var;

  /*! The index of the particle in the original array. */
  int index;
};

/**
 * @brief A cell to contain gas particles.
 */
struct cell {

  /* Location and width */
  Float loc[3];
  Float width;

  /* Is it split? */
  int split;

  /* How deep? */
  int depth;

  /* Pointers to particles in cell. */
  int part_count;
  struct particle *particles;

  /* Store the square of the maximum smoothing length. */
  Float max_sml_squ;

  /* Pointers to cells below this one. */
  struct cell *progeny;

  /* The maximum depth in the cell tree. */
  int maxdepth;
};

/* Prototypes. */
void construct_cell_tree(const Float *pos, const Float *sml,
                         const Float *surf_den_val, const int npart,
                         struct cell *root, int ncells, int maxdepth,
                         int min_count);
void cleanup_cell_tree(struct cell *c);
Float min_projected_dist2(struct cell *c, Float x, Float y);

#endif // OCTREE_H_
