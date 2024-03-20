"""
Demonstrate how to create spectra using a collection of line luminosities
instead of the default approach. Mostly for testing purposes.
"""

from synthesizer.grid import Grid
from synthesizer.sed import Sed
import numpy as np
import matplotlib.pyplot as plt
from unyt import c, erg, s, Hz

# define grid
grid_name = "test_grid"
grid_dir = "../../tests/test_grid/"
grid = Grid(grid_name, grid_dir=grid_dir, read_lines=True)

# define grid point
grid_point = grid.get_grid_point((6.5, 0.01))

# get an Sed
sed = grid.get_spectra(grid_point, spectra_id="linecont")


plt.plot(np.log10(sed.lam), np.log10(sed.lnu))
plt.xlim([2.0, 5.0])
plt.ylim([18.0, 23])
plt.xlabel(r"$\rm log_{10}(\lambda/\AA)$")
plt.ylabel(r"$\rm log_{10}(L_{\nu}/erg\ s^{-1}\ Hz^{-1} M_{\odot}^{-1})$")
plt.show()

print(grid.available_lines)


lnu = (np.zeros(len(sed.lam)) + 1) * erg / s / Hz

lc = grid.get_lines(grid_point)

for line in lc:
    idx = (np.abs(sed.lam - line.wavelength)).argmin()
    dl = 0.5 * (sed.lam[idx + 1] - sed.lam[idx - 1])
    n = c / line.wavelength
    llnu = line.wavelength * (line.luminosity / n) / dl
    print(line.id, llnu)
    lnu[idx] += llnu.to("erg/s/Hz")

linecont = Sed(lam=sed.lam, lnu=lnu)

print(np.max(linecont.lnu))
print(np.max(sed.lnu))

plt.plot(np.log10(sed.lam), np.log10(sed.lnu), alpha=0.5, c="r")
# plt.plot(np.log10(linecont.lam), np.log10(linecont.lnu), alpha=0.5, c='b')
plt.scatter(
    np.log10(linecont.lam), np.log10(linecont.lnu), alpha=0.5, color="b", s=1
)
plt.xlim([2.0, 5.0])
plt.ylim([18.0, 23])
plt.xlabel(r"$\rm log_{10}(\lambda/\AA)$")
plt.ylabel(r"$\rm log_{10}(L_{\nu}/erg\ s^{-1}\ Hz^{-1} M_{\odot}^{-1})$")
plt.show()

plt.plot(np.log10(sed.lam), sed.lnu, alpha=0.5, c="r")
# plt.plot(np.log10(linecont.lam), np.log10(linecont.lnu), alpha=0.5, c='b')
plt.scatter(np.log10(linecont.lam), linecont.lnu, alpha=0.5, color="b", s=1)
plt.xlim([2.0, 5.0])
plt.ylim([0, 0.3e23])
plt.xlabel(r"$\rm log_{10}(\lambda/\AA)$")
plt.ylabel(r"$\rm L_{\nu}/erg\ s^{-1}\ Hz^{-1} M_{\odot}^{-1}$")
plt.show()
