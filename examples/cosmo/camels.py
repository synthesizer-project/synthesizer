import sys
import numpy as np
import matplotlib.pyplot as plt

from synthesizer import grid
from synthesizer.sed import Sed
from synthesizer.load_data import load_CAMELS_SIMBA
from synthesizer.filters import UVJ

from synthesizer.galaxy.particle import ParticleGalaxy


if len(sys.argv) > 1:
    grid_dir = str(sys.argv[1])
else:
    grid_dir = None

# first load a spectral grid
_grid = grid.Grid('bc03_chabrier03', grid_dir=grid_dir)

# now load some example CAMELS data using the dedicated data loader
gals = load_CAMELS_SIMBA('data/', snap='033')

""" calculate the spectra for a single galaxy
    here we set the `sed_object` flag to automatically assign
    to an sed object """
_g = gals[0]

_g.generate_intrinsic_spectra(_grid)

_spec = _g.generate_intrinsic_spectra(_grid, sed_object=True)

plt.loglog(_spec.lam, _spec.lnu)
plt.xlabel('$\lambda \,/\, \\AA$')
plt.ylabel('$L_{\\nu} \,/\, \mathrm{erg \; s^{-1} \; Hz^{-1}}$')
# plt.show()
plt.savefig('../../docs/source/images/camels_single_spec.png', dpi=200); plt.close()

""" multiple galaxies
    Here we leave the `sed_object` flag as the default (False), 
    and combine into a single sed object afterwards """
_specs = np.vstack([_g.generate_intrinsic_spectra(_grid)
                    for _g in gals[:10]])

_specs = Sed(lam=_grid.lam, lnu=_specs)

plt.loglog(_grid.lam, _specs.lnu.T)
plt.xlabel('$\lambda \,/\, \\AA$')
plt.ylabel('$L_{\\nu} \,/\, \mathrm{erg \; s^{-1} \; Hz^{-1}}$')
# plt.show()
plt.savefig('../../docs/source/images/camels_multiple_spec.png', dpi=200); plt.close()


""" calculate broadband luminosities """

# first get rest frame 'flux'
_spec.get_fnu0()

# define a filter collection object (UVJ default)
fc = UVJ(new_lam=_grid.lam)

_UVJ = _spec.get_broadband_fluxes(fc)
print(_UVJ)

""" do for multiple, plot UVJ diagram """

# first filter by stellar mass
mstar = np.log10(np.array([np.sum(_g.stars.initial_masses) for _g in gals]) * 1e10)
mask = np.where(mstar > 8)[0]

_specs = np.vstack([gals[_g].generate_intrinsic_spectra(_grid)
                    for _g in mask])

_specs = Sed(lam=_grid.lam, lnu=_specs)
_specs.get_fnu0()
_UVJ = _specs.get_broadband_fluxes(fc)

UV = _UVJ['U'] / _UVJ['V']
VJ = _UVJ['V'] / _UVJ['J']

plt.scatter(VJ, UV, c=mstar[mask], s=4)
plt.xlabel('VJ')
plt.ylabel('UV')
plt.colorbar(label='$\mathrm{log_{10}} \, M_{\star} \,/\, \mathrm{M_{\odot}}$')
# plt.show()
plt.savefig('../../docs/source/images/camels_UVJ.png', dpi=200); plt.close()