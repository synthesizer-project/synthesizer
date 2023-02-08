

"""
Cython extension to calculate SED weights for star particles.
Calculates weights on an age / metallicity grid given the mass.
Args:
  z - 1 dimensional array of metallicity values (ascending)
  a - 1 dimensional array of age values (ascending)
  particle - 2 dimensional array of particle properties (N, 3)
  first column metallicity, second column age, third column mass
Returns:
  w - 2d array, dimensions (z, a), of weights to apply to SED array
"""

import numpy as np


def calculate_weights(z, a, particle):

    print("Warning: using python weights code, performance may suffer")

    lenz = len(z)
    lena = len(a)

    w_init = np.zeros((lenz, lena))
    w = w_init

    # check particle array right shape
    if particle.shape[1] != 3:
        particle = particle.T

    # if particle.shape[0] == 3:
    #     print("Warning! the particles array may not be transposed correctly. Should be [P,D] where P is number of particles and D = 3.")

    # simple test for sorted z and a arrays
    if z[0] > z[1]:
        raise ValueError('Metallicity array not sorted ascendingly')

    if a[0] > a[1]:
        raise ValueError('Age array not sorted ascendingly')

    for p in range(0, len(particle)):
        #metal, age, mass = particle[p]
        metal = particle[p][0]
        age = particle[p][1]
        # mass = particle[p][2]

        ilow = z.searchsorted(metal)
        if ilow.__eq__(0):  # test if outside array range
            ihigh = ilow  # set upper index to lower
            ifrac = 0  # set fraction to unity
        elif ilow == lenz:
            ilow -= 1  # lower index
            ihigh = ilow  # set upper index to lower
            ifrac = 0
        else:
            ihigh = ilow  # else set upper limit to bisect lower
            ilow -= 1  # and set lower limit to index below
            ifrac = (metal - z[ilow]) / (z[ihigh] - z[ilow])

        jlow = a.searchsorted(age)
        if jlow == 0:
            jhigh = jlow
            jfrac = 0
        elif jlow == lena:
            jlow -= 1
            jhigh = jlow
            jfrac = 0
        else:
            jhigh = jlow
            jlow -= 1
            jfrac = (age - a[jlow]) / (a[jhigh] - a[jlow])

        mfrac = particle[p][2] * (1.-ifrac)
        w[ilow, jlow] = w[ilow, jlow] + mfrac * (1.-jfrac)

        # ensure we're not adding weights more than once when outside range
        if (jlow != jhigh):
            w[ilow, jhigh] = w[ilow, jhigh] + mfrac * jfrac
        if (ilow != ihigh):
            mfrac = particle[p][2] * ifrac
            w[ihigh, jlow] = w[ihigh, jlow] + mfrac * (1.-jfrac)
            if (jlow != jhigh):
                w[ihigh, jhigh] = w[ihigh, jhigh] + mfrac * jfrac

    return np.asarray(w)
