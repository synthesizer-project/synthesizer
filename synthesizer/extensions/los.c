/******************************************************************************
 * C extension to calculate line of sight metal surface densities for star
particles.
/*****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/ndarraytypes.h>
#include <numpy/ndarrayobject.h>

/* Define a macro to handle that bzero is non-standard. */
#define bzero(b,len) (memset((b), '\0', (len)), (void) 0)

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
  double *part_pos;
  double *part_sml;
  double *part_met;
  double *part_mass;
  double *part_dtm;
  double max_sml_squ;

  /* Pointers to cells below this one. */
  struct cell *progeny;
  
};

/**
 * @brief Computes the line of sight metal surface densities when there are a
 *        small number of gas particles. No point building a cell structure
 *        with all the overhead when looping is sub second!
 *
 * @param 
 */
void low_mass_los_loop(const double *star_pos, const double *gas_pos, const double *gas_sml,
                       double *gas_dtm, double *gas_mass, double *gas_met,
                       double *kernel, double *los_dustsds,
                       int nstar, int ngas, int kdim, double threshold) {

  /* Loop over stars */
  for (int istar = 0; istar < nstar; istar++) {

    double star_x = star_pos[istar * 3];
    double star_y = star_pos[istar * 3 + 1];
    double star_z = star_pos[istar * 3 + 2];
    
    for (int igas = 0; igas < ngas; igas++) {

      /* Get gas particle data. */
      double gas_x = gas_pos[igas * 3];
      double gas_y = gas_pos[igas * 3 + 1];
      double gas_z = gas_pos[igas * 3 + 2];
      double sml = gas_sml[igas];
      double mass = gas_mass[igas];
      double dtm = gas_dtm[igas];
      double met = gas_met[igas];

      /* Skip straight away if the gas particle is behind the star. */
      if (gas_z > star_z)
        continue;

      /* Calculate the x and y separations. */
      double x = gas_x - star_x;
      double y = gas_y - star_y;

      /* Early skip if the star doesn't fall in the gas particles kernel. */
      if (abs(x) > (threshold * sml) || abs(y) > (threshold * sml))
        continue;

      /* Convert separation to distance. */
      double rsqu = x * x + y * y;

      /* Calculate the impact parameter. */
      double sml_squ = gas_sml[igas] * gas_sml[igas];
      double q = rsqu / sml_squ;

      /* Skip gas particles outside the kernel. */
      if (q > threshold) continue;

      /* Get the value of the kernel at q. */
      int index = kdim * q;
      double kvalue = kernel[index];

      /* Finally, compute the metal surface density itself. */
      los_dustsds[istar] += dtm * mass * met / sml_squ * kvalue;
      
      
    }
  }
  
}

/**
 * @brief Recursively Populates the cell tree until maxdepth is reached.
 *
 * @param 
 */
void populate_cell_tree_recursive(struct cell *c,
                                  struct cell *cells, int ncells,
                                  int tot_cells, int maxdepth,
                                  double *centre, int depth, double *bounds) {

  /* printf("At depth %d with ngas=%d ncells=%d tot_cells=%d\n", */
  /*        depth, c->part_count, ncells, tot_cells); */

  /* Have we reached the bottom? */
  if (depth > maxdepth)
    return;

  /* Get the particles in this cell. */
  double *gas_pos = c->part_pos;
  double *gas_sml = c->part_sml;
  double *gas_dtm = c->part_dtm;
  double *gas_mass = c->part_mass;
  double *gas_met = c->part_met;
  int ngas = c->part_count;

  /* Do we need to split? */
  if (ngas < 100)
    return;

  /* Compute the width at this level. */
  double width = c->width / 2;

  /* We need to split... get the progeny. */
  c->split = 1;
  c->progeny = malloc(8 * sizeof(struct cell));
  for (int ip = 0; ip < 8; ip++) {

    /* Ensure we have allocated cells. */
    if (ncells > tot_cells) {

      /* Allocate the cells. */
      struct cell *new_cells = malloc(8 * 8 * sizeof(struct cell));

      /* Ensure the allocation went smoothly */
      if (new_cells == NULL)
          error("Failed to dynamically allocate more cells in the tree!");

      /* Intialise the cells at 0. */
      bzero(new_cells, 8 * 8 * sizeof(struct cell));

      /* Attach the cells. */
      cells[ncells] = *new_cells;
      tot_cells += 8 * 8;
    }

    /* Nibble off a cell */
    c->progeny[ip] = cells[ncells++];
    struct cell *cp = &c->progeny[ip];

    /* Set the cell properties. */
    cp->width = width;
    cp->loc[0] = c->loc[0];
    cp->loc[1] = c->loc[1];
    cp->loc[2] = c->loc[2];
    if (ip & 4) cp->loc[0] += cp->width;
    if (ip & 2) cp->loc[1] += cp->width;
    if (ip & 1) cp->loc[2] += cp->width;
    cp->split = 0;
    cp->part_count = 0;
    cp->max_sml_squ = 0;
    cp->depth = depth;

    /* Allocate the particle data with the max possible space we could need. */
    cp->part_pos = malloc(ngas * 3 * sizeof(double));
    cp->part_sml = malloc(ngas * sizeof(double));
    cp->part_met = malloc(ngas * sizeof(double));
    cp->part_mass = malloc(ngas * sizeof(double));
    cp->part_dtm = malloc(ngas * sizeof(double));

  }

  /* Loop over gas particles and associate them if they are inside this
   * progeny. */
  for (int igas = 0; igas < ngas; igas++) {

    /* Get the gas particle position relative to the cell grid. */
    double pos[3] = {
      gas_pos[igas * 3 + 0] - c->loc[0],
      gas_pos[igas * 3 + 1] - c->loc[1],
      gas_pos[igas * 3 + 2] - c->loc[2],
    };

    /* Compute the indices of the cell grid. */
    int i = pos[0] > width;
    int j = pos[1] > width;
    int k = pos[2] > width;

    struct cell *cp = &c->progeny[k + 2 * (j + 2 * i)];

    /* Attach a pointer to this particle to the cell. */
    cp->part_pos[cp->part_count * 3] = gas_pos[igas * 3];
    cp->part_pos[cp->part_count * 3 + 1] = gas_pos[igas * 3 + 1];
    cp->part_pos[cp->part_count * 3 + 2] = gas_pos[igas * 3 + 2];
    cp->part_sml[cp->part_count] = gas_sml[igas];
    cp->part_met[cp->part_count] = gas_met[igas];
    cp->part_mass[cp->part_count] = gas_mass[igas];
    cp->part_dtm[cp->part_count++] = gas_dtm[igas];

    /* Have we found a larger smoothing length? */
    if (gas_sml[igas] > c->max_sml_squ)
      cp->max_sml_squ = gas_sml[igas];
    
  }

  /* Recurse... */
  for (int ip = 0; ip < 8; ip++) {
    struct cell *cp = &c->progeny[ip];

    /* Now square the max smooothing length. */
    cp->max_sml_squ = cp->max_sml_squ * cp->max_sml_squ;

    /* Got to the next level */
    populate_cell_tree_recursive(cp, cells, ncells, tot_cells,
                                 maxdepth, centre, depth++,
                                 bounds);
  }
  
}

/**
 * @brief Constructs the cell tree for large numbers of gas particles.
 *
 * @param 
 */
void construct_cell_tree(double *gas_pos, double *gas_sml,
                         double *gas_dtm, double *gas_mass, double *gas_met,
                         int ngas, struct cell *cells, int ncells,
                         int tot_cells, double dim, int maxdepth,
                         double *centre, double *bounds, double width,
                         int cdim) {

  /* Create the top level cells. */
  for (int i = 0; i < cdim; i++) {
    for (int j = 0; j < cdim; j++) {
      for (int k = 0; k < cdim; k++) {

        int index = k + cdim * (j + cdim * i);

        /* Get the cell */
        struct cell *c = &cells[index];

        /* Set the cell properties. */
        c->loc[0] = bounds[0] + (i * width);
        c->loc[1] = bounds[2] + (j * width);
        c->loc[2] = bounds[4] + (k * width);
        c->width = width;
        c->split = 0;
        c->part_count = 0;
        c->max_sml_squ = 0;
        c->depth = 0;

        /* Allocate the particle data with the max possible space we
         * could need. */
        c->part_pos = malloc(ngas * 3 * sizeof(double));
        c->part_sml = malloc(ngas * sizeof(double));
        c->part_met = malloc(ngas * sizeof(double));
        c->part_mass = malloc(ngas * sizeof(double));
        c->part_dtm = malloc(ngas * sizeof(double));

      } 
    }
  }

  /* Loop over gas particles and associate them with thier top level cell. */
  for (int igas = 0; igas < ngas; igas++) {

    /* Get the gas particle position relative to the cell grid. */
    double pos[3] = {
      gas_pos[igas * 3 + 0] - bounds[0],
      gas_pos[igas * 3 + 1] - bounds[2],
      gas_pos[igas * 3 + 2] - bounds[4],
    };

    /* Compute the indices of the cell grid. */
    int i = (int)(pos[0] / width);
    int j = (int)(pos[1] / width);
    int k = (int)(pos[2] / width);

    struct cell *c = &cells[k + cdim * (j + cdim * i)];

    /* Attach a pointer to this particle to the cell. */
    c->part_pos[c->part_count * 3] = gas_pos[igas * 3];
    c->part_pos[c->part_count * 3 + 1] = gas_pos[igas * 3 + 1];
    c->part_pos[c->part_count * 3 + 2] = gas_pos[igas * 3 + 2];
    c->part_sml[c->part_count] = gas_sml[igas];
    c->part_met[c->part_count] = gas_met[igas];
    c->part_mass[c->part_count] = gas_mass[igas];
    c->part_dtm[c->part_count++] = gas_dtm[igas];

    /* Have we found a larger smoothing length? */
    if (gas_sml[igas] > c->max_sml_squ)
      c->max_sml_squ = gas_sml[igas];
  }

  /* Now we have the top level populated we can do the rest of the tree. */
  for (int cid = 0; cid < pow(cdim, 3); cid++) {
    struct cell *c = &cells[cid];

    /* Now square the max smooothinbg length. */
    c->max_sml_squ = c->max_sml_squ * c->max_sml_squ;

    /* And recurse... */
    populate_cell_tree_recursive(c, cells, ncells, tot_cells,
                                 maxdepth, centre, 1,
                                 bounds);
  }
  
}

/**
 * @brief Constructs the cell tree for large numbers of gas particles.
 *
 * @param 
 */
double calculate_los_recursive(struct cell *c, 
                               const double star_x,
                               const double star_y,
                               const double star_z,
                               double threshold, 
                               int kdim, double *kernel) {

  /* Define the line of sight dust surface density. */
  double los_dustsds = 0;

  /* Calculate the separation between the star and this cell. */
  double sep[2] = {
    c->loc[0] + (c->width / 2) - star_x,
    c->loc[1] + (c->width / 2) - star_y,
  };

  /* Calculate distance between star and cell. */
  double dist_squ = (sep[0] * sep[0] + sep[1] * sep[1]);

  /* Early exit if the maximum smoothing lengh is not in range of the star.
   * Here we account for the corner to corner size of the cell with some
   * extra buffer to be safe. */
  if (dist_squ > (c->max_sml_squ + (sqrt(2) * c->width * 1.2)))
    return los_dustsds;

  /* Is the cell split? */
  if (c->split) {

    /* Ok, so we recurse... */
    for (int ip = 0; ip < 8; ip++) {
      struct cell *cp = &c->progeny[ip];
      if (cp->part_count == 0) continue;
      los_dustsds += calculate_los_recursive(cp, star_x, star_y, star_z,
                                             threshold, kdim, kernel);
    }
    
  }

  /* We reached the bottom, do the calculation... */
  else {
    
    /* printf("Calculating at depth %d with count %d\n", c->depth, c->part_count); */

    /* Get the particles in this cell. */
    double *gas_pos = c->part_pos;
    double *gas_sml = c->part_sml;
    double *gas_dtm = c->part_dtm;
    double *gas_mass = c->part_mass;
    double *gas_met = c->part_met;
    int ngas = c->part_count;

    /* Loop over the particles adding their contribution. */
    for (int igas = 0; igas < ngas; igas++) {

      /* Get gas particle data. */
      double gas_x = gas_pos[igas * 3];
      double gas_y = gas_pos[igas * 3 + 1];
      double gas_z = gas_pos[igas * 3 + 2];
      double sml = gas_sml[igas];
      double mass = gas_mass[igas];
      double dtm = gas_dtm[igas];
      double met = gas_met[igas];

      /* Skip straight away if the gas particle is behind the star. */
      if (gas_z > star_z)
        continue;

      /* Calculate the x and y separations. */
      double x = gas_x - star_x;
      double y = gas_y - star_y;

      /* Early skip if the star doesn't fall in the gas particles kernel. */
      if (abs(x) > (threshold * sml) || abs(y) > (threshold * sml))
        continue;

      /* Convert separation to distance. */
      double rsqu = x * x + y * y;

      /* Calculate the impact parameter. */
      double sml_squ = gas_sml[igas] * gas_sml[igas];
      double q = rsqu / sml_squ;

      /* Skip gas particles outside the kernel. */
      if (q > threshold) continue;

      /* Get the value of the kernel at q. */
      int index = kdim * q;
      double kvalue = kernel[index];

      /* Finally, compute the metal surface density itself. */
      los_dustsds += dtm * mass * met / sml_squ * kvalue;
      
    }
  }
  return los_dustsds;
}



/**
 * @brief Computes the line of sight metal surface densities for each of the
 *        stars passed to this function.
 *
 * @param 
 */
PyObject *compute_dust_surface_dens(PyObject *self, PyObject *args) {

  const int nstar, ngas, kdim, force_loop;
  const double threshold, max_sml;
  const PyArrayObject *np_kernel, *np_star_pos, *np_gas_pos, *np_gas_sml;
  const PyArrayObject *np_gas_met, *np_gas_mass, *np_gas_dtm;

  if(!PyArg_ParseTuple(args, "OOOOOOOiiiddi", &np_kernel, &np_star_pos,
                       &np_gas_pos, &np_gas_sml, &np_gas_met, &np_gas_mass,
                       &np_gas_dtm, &nstar, &ngas, &kdim, &threshold,
                       &max_sml, &force_loop))
    return NULL;

  /* Quick check to make sure our inputs are valid. */
  if (nstar == 0) return NULL;
  if (ngas == 0) return NULL;
  if (kdim == 0) return NULL;

  /* Extract a pointers to the actual data in the numpy arrays. */
  const double *kernel = PyArray_DATA(np_kernel);
  const double *star_pos = PyArray_DATA(np_star_pos);
  const double *gas_pos = PyArray_DATA(np_gas_pos);
  const double *gas_sml = PyArray_DATA(np_gas_sml);
  const double *gas_met = PyArray_DATA(np_gas_met);
  const double *gas_mass = PyArray_DATA(np_gas_mass);
  const double *gas_dtm = PyArray_DATA(np_gas_dtm);

  /* Set up arrays to hold the surface densities themselves. */
  double *los_dustsds = malloc(nstar * sizeof(double));
  bzero(los_dustsds, nstar * sizeof(double));

  /* No point constructing cells if there isn't much gas. */
  if (ngas * nstar < 50000 || ngas < 1000 || force_loop) {

    /* Use the simple loop over stars and gas. */
    low_mass_los_loop(star_pos, gas_pos, gas_sml, gas_dtm, gas_mass, gas_met,
                      kernel, los_dustsds, nstar, ngas, kdim, threshold);
    
    /* Reconstruct the python array to return. */
    npy_intp np_dims[1] = {nstar,};
    PyArrayObject *out_los_dustsds =
      (PyArrayObject *) PyArray_SimpleNewFromData(1, np_dims, NPY_FLOAT64,
                                                  los_dustsds);

    return Py_BuildValue("N", out_los_dustsds);
  }

  /* Calculate the width of the gas distribution. */
  double bounds[6] = {
    FLT_MAX, 0, FLT_MAX, 0, FLT_MAX, 0
  };
  for (int igas = 0; igas < ngas; igas++) {

    double x = gas_pos[igas * 3 + 0];
    double y = gas_pos[igas * 3 + 1];
    double z = gas_pos[igas * 3 + 2];
    
    /* Update the boundaries. */
    if (x > bounds[1]) bounds[1] = x;
    if (x < bounds[0]) bounds[0] = x;
    if (y > bounds[3]) bounds[3] = y;
    if (y < bounds[2]) bounds[2] = y;
    if (z > bounds[5]) bounds[5] = z;
    if (z < bounds[4]) bounds[4] = z;
  }
  for (int istar = 0; istar < nstar; istar++) {

    double x = star_pos[istar * 3 + 0];
    double y = star_pos[istar * 3 + 1];
    double z = star_pos[istar * 3 + 2];
    
    /* Update the boundaries. */
    if (x > bounds[1]) bounds[1] = x;
    if (x < bounds[0]) bounds[0] = x;
    if (y > bounds[3]) bounds[3] = y;
    if (y < bounds[2]) bounds[2] = y;
    if (z > bounds[5]) bounds[5] = z;
    if (z < bounds[4]) bounds[4] = z;
  }

  /* Calculate the dim of the cells with some buffer */
  double dims[3] = {
    bounds[1] - bounds[0],
    bounds[3] - bounds[2],
    bounds[5] - bounds[4],
  };
  double centre[3] = {
    bounds[0] + ((bounds[1] - bounds[0]) / 2),
    bounds[2] + ((bounds[3] - bounds[2]) / 2),
    bounds[4] + ((bounds[5] - bounds[4]) / 2),
  };
  double dim = 0;
  for (int i = 0; i < 3; i++)
    if (dims[i] > dim)
      dim = dims[i];
  dim += 0.2 * dim;

  /* Include the buffer region in the bounds. */
  for (int i = 0; i < 3; i++) {
    bounds[2 * i] = centre[i] - (dim / 2);
    bounds[(2 * i) + 1] = centre[i] + (dim / 2);
  }

    /* Allocate cells. */
  int cdim = min((int)fmax(dim / max_sml, 3), 64);
  int maxdepth = 50;
  int ncells = pow(cdim, 3);
  int tot_cells = ncells * pow(8, 3);
  struct cell *cells = malloc(tot_cells * sizeof(struct cell));
  bzero(cells, tot_cells * sizeof(struct cell));
  
  /* Define some cell grid variables we will need. */
  double width = dim / cdim;

  /* Consturct the cell tree. */
  construct_cell_tree(gas_pos, gas_sml, gas_dtm, gas_mass, gas_met,
                      ngas, cells, ncells, tot_cells, dim, maxdepth,
                      centre, bounds, width, cdim);

  /* Loop over stars */
  for (int istar = 0; istar < nstar; istar++) {

    /* Get the star particle z position relative to the cell grid. */
    double star_z = star_pos[istar * 3 + 2] - bounds[4];
    
    /* Compute the cell grid k index. */
    int k = (int)(star_z / width);

    /* Loop over the cell grid skipping cells behind the star. */
    for (int ii = 0; ii < cdim; ii++) {
      for (int jj = 0; jj < cdim; jj++) {
        for (int kk = 0; kk <= k; kk++) {

          /* Get this cell. */
          struct cell *c = &cells[kk + cdim * (jj + cdim * ii)];

          /* Skip empty cells */
          if (c->part_count == 0) continue;
          
          /* Calculate the contribution of particles in this cell. */
          /* printf("Calculating for cell (i j k=[%d %d %d] ii jj kk=[%d %d %d])\n", i, j, k, ii, jj, kk); */
          los_dustsds[istar] += calculate_los_recursive(c,
                                                        star_pos[istar * 3],
                                                        star_pos[istar * 3 + 1],
                                                        star_pos[istar * 3 + 2],
                                                        threshold,
                                                        kdim, kernel);
        } 
      }
    }
  }

  free(cells);

  /* Reconstruct the python array to return. */
  npy_intp np_dims[1] = {nstar,};
  PyArrayObject *out_los_dustsds =
    (PyArrayObject *) PyArray_SimpleNewFromData(1, np_dims, NPY_FLOAT64,
                                                los_dustsds);

  return Py_BuildValue("N", out_los_dustsds);
}

/* Below is all the gubbins needed to make the module importable in Python. */
static PyMethodDef LosMethods[] = {
  {"compute_dust_surface_dens", compute_dust_surface_dens, METH_VARARGS,
   "Method for calculating line of sight metal surface densities."},
  {NULL, NULL, 0, NULL} 
};

/* Make this importable. */
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "los_dust_surface_dens",                                /* m_name */
        "A module to calculate los metal surface densities",   /* m_doc */
        -1,                                                    /* m_size */
        LosMethods,                                            /* m_methods */
        NULL,                                                  /* m_reload */
        NULL,                                                  /* m_traverse */
        NULL,                                                  /* m_clear */
        NULL,                                                  /* m_free */
    };

PyMODINIT_FUNC PyInit_los(void) {
    PyObject *m = PyModule_Create(&moduledef);
    import_array();
    return m;
}
