/******************************************************************************
 * A C module containing all the weights functions common to all particle
 * spectra extensions.
 *****************************************************************************/
/* C includes */
#include <math.h>
#include <string.h>

/* Prototypes */
void get_indices_from_flat(int flat_ind, int ndim, const int *dims,
                           int *indices);
int get_flat_index(const int *multi_index, const int *dims, const int ndims);
int binary_search(int low, int high, const double *arr, const double val);
HashMap *weight_loop_cic(const double **grid_props, const double **part_props,
                         const double *part_masses, const int *dims,
                         const int ndim, const int npart, const int per_part);
HashMap *weight_loop_ngp(const double **grid_props, const double **part_props,
                         const double *part_masses, const int *dims,
                         const int ndim, const int npart, const int per_part);
