/******************************************************************************
 * A C module containing all the weights functions common to all particle
 * spectra extensions.
 *****************************************************************************/
/* C includes */
#include <math.h>
#include <string.h>

/* Define the weights struct we'll use as a "sparse matrix". */
typedef struct {

  /* The size of each axis. */
  int *axis_size;

  /* The number of entries. */
  int size;

  /*! The indices of the non-zero elements. (First axis is the element, second
   *  are the indices along each axis) */
  int **indices;

  /*! The values of the non-zero elements. */
  double *values;

  /*! The index of the particle corresponding to a weight. */
  int *part_indices;

} Weights;

/* Prototypes */
Weights *init_weights(int ndim, const int *dims, int size);
void get_indices_from_flat(int flat_ind, int ndim, const int *dims,
                           int *indices);
int get_flat_index(const int *multi_index, const int *dims, const int ndims);
int binary_search(int low, int high, const double *arr, const double val);
Weights *weight_loop_cic(const double **grid_props, const double **part_props,
                         const double *part_masses, const int *dims,
                         const int ndim, const int npart);
Weights *weight_loop_ngp(const double **grid_props, const double **part_props,
                         const double *part_masses, const int *dims,
                         const int ndim, const int npart);
