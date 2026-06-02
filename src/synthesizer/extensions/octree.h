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
 *
 * @tparam Real The floating-point type (float or double).
 */
template <typename Real>
struct particle {

  /* Position of the particle. */
  Real pos[3];

  /* Smoothing length of the particle. */
  Real sml;

  /* Surface density variable. */
  Real surf_den_var;

  /*! The index of the particle in the original array. */
  int index;
};

/**
 * @brief A cell to contain gas particles.
 *
 * @tparam Real The floating-point type (float or double).
 */
template <typename Real>
struct cell {

  /* Location and width */
  Real loc[3];
  Real width;

  /* Is it split? */
  int split;

  /* How deep? */
  int depth;

  /* Pointers to particles in cell. */
  int part_count;
  struct particle<Real> *particles;

  /* Store the square of the maximum smoothing length. */
  Real max_sml_squ;

  /* Pointers to cells below this one. */
  struct cell<Real> *progeny;

  /* The maximum depth in the cell tree. */
  int maxdepth;
};

/* Prototypes. */
template <typename Real>
void construct_cell_tree(const Real *pos, const Real *sml,
                         const Real *surf_den_val, const int npart,
                         struct cell<Real> *root, int ncells, int maxdepth,
                         int min_count);
template <typename Real>
void cleanup_cell_tree(struct cell<Real> *c);
template <typename Real>
Real min_projected_dist2(struct cell<Real> *c, Real x, Real y);

#endif // OCTREE_H_
