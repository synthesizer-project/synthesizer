"""
IGM transmission example
========================

Plot IGM transmission curves (Inoue+14 & Madau+96
"""

import matplotlib.pyplot as plt
import numpy as np

import cmasher as cmr

from synthesizer.igm import Madau96, Inoue14


lam = np.arange(0, 20000)


redshifts = [3., 5., 7.]
colors = cmr.take_cmap_colors('cmr.guppy', len(redshifts))

for IGM, ls in zip([Inoue14, Madau96], ['-', ':']):
    igm = IGM()
    for z, color in zip(redshifts, colors):
        plt.plot(lam, igm.T(z, lam), ls=ls, c=color, label=f'{igm.name} z={z}')

plt.legend()
plt.xlabel(r'$\lambda_{obs}/\AA$')
plt.ylabel(r'$T$')
plt.ylim([0, 1.1])
plt.show()
# plt.savefig('../docs/source/images/img.png', bbox_inches='tight', dpi=200)
