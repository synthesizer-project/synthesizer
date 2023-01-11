import matplotlib.pyplot as plt
import numpy as np

from synthesizer.igm import Madau96, Inoue14


lam = np.arange(0, 50000)
z = 7.

for z in [3., 5., 7.]:
    for IGM, ls in zip([Inoue14, Madau96], ['-', ':']):
        igm = IGM()
        plt.plot(lam, igm.T(z, lam), ls=ls)


plt.ylim([0, 1.1])
plt.show()
