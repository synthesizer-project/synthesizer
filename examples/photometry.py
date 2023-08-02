"""
An example demonstrating the observed spectrum for a parametric galaxy including
photometry. This example will:
- build a parametric galaxy (see make_sfzh and make_sed).
- calculate spectral luminosity density (see make_sed).
- calculate observed frame spectra (requires comsology and redshift).
- calculate observed frame fluxes at various redshifts.
- calculate photometry.
- plot the redshift evolution of photometry.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from synthesizer.filters import FilterCollection
from synthesizer.grid import Grid
from synthesizer.parametric.sfzh import SFH, ZH, generate_sfzh
from synthesizer.parametric.galaxy import Galaxy
from synthesizer.plt import single, single_histxy, mlabel
from unyt import yr, Myr, c, angstrom
from synthesizer.igm import Madau96, Inoue14
from astropy.cosmology import Planck18 as cosmo


if __name__ == '__main__':

    # Get the location of this script, __file__ is the absolute path of this
    # script, however we just want to directory
    script_path = os.path.abspath(os.path.dirname(__file__))
    
    # Define the grid
    grid_name = "test_grid"
    grid_dir = script_path + "/../tests/test_grid/"
    grid = Grid(grid_name, grid_dir=grid_dir)

    # define the parameters of the star formation and metal enrichment histories
    sfh_p = {'duration': 10 * Myr}
    Z_p = {'log10Z': -2.0}  # can also use linear metallicity e.g. {'Z': 0.01}
    stellar_mass = 1E9

    # define the functional form of the star formation and metal
    # enrichment histories
    sfh = SFH.Constant(sfh_p)  # constant star formation
    Zh = ZH.deltaConstant(Z_p)  # constant metallicity

    # get the 2D star formation and metal enrichment history for the given
    # SPS grid. This is (age, Z).
    sfzh = generate_sfzh(grid.log10ages, grid.metallicities, sfh, Zh,
                         stellar_mass=stellar_mass)

    # create a galaxy object
    galaxy = Galaxy(sfzh)

    # Define Filters
    filter_codes = [f'JWST/NIRCam.{f}'
                    for f in ['F090W', 'F115W', 'F150W',
                              'F200W', 'F277W', 'F356W', 'F444W']]
    filter_codes += [f'JWST/MIRI.{f}' for f in ['F770W']]
    filters = FilterCollection(filter_codes=filter_codes, new_lam=grid.lam)

    # Get the color for each filter
    colors = {f:plt.rcParams['axes.prop_cycle'].by_key()['color'][ind]
              for ind, f in enumerate(filter_codes)}

    # Now calculate the observed frame spectra for a range of redshifts
    zs = list(range(3, 12, 2))
    seds = {}
    for z in zs:
        
        # Generate spectra using pacman model (complex)
        seds[z] = galaxy.get_spectra_pacman(grid, fesc=0.5, fesc_LyA=0.5,
                                            tau_v=0.1)

        # Generate observed frame spectra
        seds[z].get_fnu(cosmo, z, igm=Madau96())

    # Set up plot
    fig = plt.figure(figsize=(5, 3.5 * len(zs)))
    gs = gridspec.GridSpec(len(zs), 1, hspace=0.0)

    # Loop over redshifts
    for ind, z in enumerate(zs):

        # Set up the axis object
        ax = fig.add_subplot(gs[ind])
        ax.grid(True)
        ax.loglog()
        ax.set_xlim(10**3, 10**5.5)
        ax.set_ylim(10**2, 10**4.5)

        ax.text(0.05, 0.1, f'$z=%.1f$' % z,
                bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1,
                          alpha=0.8),
                transform=ax.transAxes, horizontalalignment='left')
        
        # Plot the SED
        if ind == 0:
            ax.plot(seds[z].obslam, seds[z]._fnu, color="k", linestyle="--",
                    label="SED", zorder=0)
        else:
            ax.plot(seds[z].obslam, seds[z]._fnu, color="k", linestyle="--",
                    zorder=0)
            
        # Make the first legend
        if ind == 0:
            ax.legend()
            
        # Turn off the x axis
        if ind < len(zs) - 1:
            ax.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off
            
        # Loop over the filters
        for f in filters:

            # Calculate the photometry
            phot = f.apply_filter(seds[z]._fnu,
                                  nu=seds[z].nuz)

            # Plot the transmitted portion of the SED
            if ind == len(zs) - 1:
                ax.plot(seds[z].obslam, seds[z]._fnu * f._shifted_t,
                        color=colors[f.filter_code], label=f.filter_code,
                        zorder=1)
            else:
                ax.plot(seds[z].obslam, seds[z]._fnu * f._shifted_t,
                        color=colors[f.filter_code], zorder=1)

            # Plot the photometry
            ax.scatter(f.pivwv(), phot, s=50, color=colors[f.filter_code],
                       marker="D", zorder=2)
            
        # Make the second legened 
        if ind == len(zs) - 1:
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.3),
                fancybox=True,
                shadow=True,
                ncol=3,
            )

    fig.savefig(script_path + "/plots/photometry_from_flux.png",
                bbox_inches="tight", dpi=300)
