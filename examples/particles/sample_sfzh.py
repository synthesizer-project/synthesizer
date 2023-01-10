

import numpy as np
import matplotlib.pyplot as plt
from unyt import yr, Myr

from synthesizer.parametric.sfzh import SFH, ZH, generate_sfzh
from synthesizer.particle.stars import sample_sfhz

# --- define the grid (normally this would be defined by an SPS grid)


log10ages = np.arange(6., 10.5, 0.1)
metallicities = 10**np.arange(-5., -1.5, 0.1)

# --- define the parameters of the star formation and metal enrichment histories

Z_p = {'Z': 0.01} 
Zh = ZH.deltaConstant(Z_p)

sfh_p = {'duration': 100 * Myr}
sfh = SFH.Constant(sfh_p)  # constant star formation
# sfh.summary()  # print summary of the star formation history
sfzh = generate_sfzh(log10ages, metallicities, sfh, Zh)
sfzh.summary()
# sfzh.plot()

# print(sfzh.sfzh)


# sfzh.sfzh = sfzh.sfzh*0.0
# sfzh.sfzh[0,0] = 1
# sfzh.plot()


stars = sample_sfhz(sfzh, 1000)
print(np.median(stars.ages)/1E6)
print(np.median(stars.metallicities))
# print(np.metallicities(stars.metallicities))
