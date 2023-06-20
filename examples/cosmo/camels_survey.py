import sys
import numpy as np
import matplotlib.pyplot as plt

from synthesizer import grid
from synthesizer.sed import Sed
from synthesizer.load_data import load_CAMELS_SIMBA
from synthesizer.filters import UVJ
# from synthesizer.imaging.survey import Survey
from synthesizer.survey import Survey

from synthesizer.particle.galaxy import Galaxy


if len(sys.argv) > 1:
    grid_dir = str(sys.argv[1])
else:
    grid_dir = None

# First load a spectral grid
_grid = grid.Grid('bc03_chabrier03-0.1,100', grid_dir=grid_dir)

# now load some example CAMELS data using the dedicated data loader
gals = load_CAMELS_SIMBA('data/', snap='033')

# first filter by stellar mass
mstar = np.log10(np.array([np.sum(_g.stars.initial_masses)
                           for _g in gals]) * 1e10)
mask = np.where(mstar > 8)[0]

# ========================= Using a Survey =========================

# Set up a filter collection object (UVJ default)
fc = UVJ(new_lam=_grid.lam)

# Convert gals to an array
gals = np.array(gals)

# Create an empty Survey object
survey = Survey(super_resolution_factor=1)

# Let's add the filters to an instrument in the survey
survey.add_photometric_instrument(filters=fc, label="UVJ")

# Store the galaxies in the survey
survey.add_galaxies(gals[mask])

# Get the SEDs
survey.get_integrated_stellar_spectra(_grid)

survey.get_integrated_spectra_screen(tauV=0.33)

survey.get_integrated_spectra_charlot_fall_00(_grid, tauV_ISM=0.33, tauV_BC=0.67)

# Compute the photometry in UVJ filters

for spectra_type, c in zip(['stellar', 'attenuated'],
                           [mstar[mask], 'grey']):
    survey.get_photometry(spectra_type=spectra_type)
    _UVJ = survey.photometry

    UV = _UVJ['U'] / _UVJ['V']
    VJ = _UVJ['V'] / _UVJ['J']

    # plt.scatter(VJ, UV, c=c, s=4, label=spectra_type)

    bins = np.linspace(35, 44, 20)
    plt.hist(np.log10(_UVJ['U'].value), label=spectra_type, 
             histtype='step', bins=bins)

plt.legend()
#plt.xlabel('VJ')
#plt.ylabel('UV')
#plt.colorbar(label='$\mathrm{log_{10}} \, M_{\star} \,/\, \mathrm{M_{\odot}}$')
plt.show()
# plt.savefig('../../docs/source/images/camels_UVJ.png', dpi=200)
plt.close()
