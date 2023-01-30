/******************************************************************************
 * C functions for quick kernel computations.
 * For now the usage of these is isolated to image creation.
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


float uniform(float r) {
  if (r < 1) 
    return 1. / ((4. / 3.) * M_PI);
  else 
    return 0;
}


float sph_anarchy(float r) {

  if (r <= 1)
    return (21. / (2. * M_PI)) * ((1. - r) * (1. - r) * (1. - r) *
                                  (1. - r) * (1. + 4. * r));
  else
    return 0;
}


float gadget_2(float r) {

  if (r < 0.5)
    return (8. / M_PI) * (1. - 6 * (r * r) + 6 * (r * r * r));
  else if (r < 1.)
    return (8. / M_PI) * 2 * ((1. - r) * (1. - r) * (1. - r));
  else
    return 0;
}


float cubic(float r) {

  if (r < 0.5)
    return (2.546479089470 + 15.278874536822 * (r - 1.0) * r * r);
  else if (r < 1)
    return 5.092958178941 * (1.0 - r) * (1.0 - r) * (1.0 - r);
  else
    return 0;
}


float quintic(float r) {

  if (r < 1 / 3)
    return 27.0 * (6.4457752 * r * r * r * r * (1.0 - r) -
                   1.4323945 * r * r + 0.17507044);
  else if (r < 2 / 3)
    return 27.0 * (3.2228876 * r * r * r * r * (r - 3.0) +
                   10.7429587 * r * r * r - 5.01338071 * r * r +
                   0.5968310366 * r + 0.1352817016);
  else if (r < 1)
    return 27.0 * 0.64457752 * (-r * r * r * r * r + 5.0 * r * r * r * r -
                                10.0 * r * r * r + 10.0 * r * r -
                                5.0 * r + 1.0);
  else
    return 0;
}
