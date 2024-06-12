/******************************************************************************
 * A C module containing all the weights functions common to all particle
 * spectra extensions.
 *****************************************************************************/
/* C includes */
#include <math.h>
#include <string.h>

/* Local includes */
#include "property_funcs.h"

/* Define a struct to hold the useful data a generic callback function will
 * need. */
struct callback_data {
  int *indices;
  int *dims;
  int ndim;
  int particle_index;
  int nlam;
  double fesc;
  double *grid_spectra;
  double *grid_lines;
  double *grid_continuum;
};

/* Define a callback function to be used by the weights functions. Of
 * the form:
 * void Func(double weight, void *indices, void *output)
 */
typedef void (*WeightFunc)(double, struct callback_data *, void *);

/* Prototypes */
void get_indices_from_flat(int flat_ind, int ndim, const int *dims,
                           int *indices);
int get_flat_index(const int *multi_index, const int *dims, const int ndims);
int binary_search(int low, int high, const double *arr, const double val);
void weight_loop_cic(struct grid *grid, struct particles *parts, void *out,
                     WeightFunc func);
void weight_loop_ngp(struct grid *grid, struct particles *parts, void *out,
                     WeightFunc func);
