"""
Given an SED (from an SPS model, for example), generate a cloudy atmosphere
grid. Can optionally generate an array of input files for selected parameters.
"""

import numpy as np
from scipy import integrate
from pathlib import Path
from unyt import c, h, angstrom, eV, erg, s, Hz, unyt_array
import h5py

from synthesizer.cloudy import calculate_Q_from_U
from synthesizer.abundances import Abundances
from write_submission_script import (apollo_submission_script,
                                     cosma7_submission_script)


def create_cloudy_input(model_name, log10T, abundances, output_dir='./', **kwargs):

    params = {
        'log10U': -2,
        'log10radius': -2,  # radius in log10 parsecs
        'covering_factor': 1.0,
        'stop_T': 4000,  # K
        'stop_efrac': -2,
        'T_floor': 100,  # K
        'log10n_H': 2.0,  # Hydrogen density
        'z': 0.,
        'CMB': False,
        'cosmic_rays': False
    }

    for key, value in list(kwargs.items()):
        params[key] = value

    log10U = params['log10U']

    # ----- start CLOUDY input file (as a list)
    cinput = []

    cinput.append(f'black body t={log10T}\n')

    # --- Define the chemical composition
    for ele in ['He'] + abundances.metals:
        cinput.append((f'element abundance {abundances.name[ele]} '
                       f'{abundances[ele]}\n'))

    """
    add graphite and silicate grains

    This old version does not actually conserve mass
    as the commands `abundances` and `grains` do not
    really talk with each other
    """
    # # graphite, scale by total C abundance relative to ISM
    # scale = 10**a_nodep['C']/2.784000e-04
    # cinput.append(f'grains Orion graphite {scale}'+'\n')
    # # silicate, scale by total Si abundance relative to ISM
    # # NOTE: silicates also rely on other elements.
    # scale = 10**a_nodep['Si']/3.280000e-05
    # cinput.append(f'grains Orion silicate {scale}'+'\n')

    """ specifies graphitic and silicate grains with a size
    distribution and abundance appropriate for those along
    the line of sight to the Trapezium stars in Orion. The
    Orion size distribution is deficient in small particles
    and so produces the relatively grey extinction observed
    in Orion (Baldwin et al., 1991). One problem with the
    grains approach is metals/element abundances do not talk
    to the grains command and hence there is issues with mass
    conservation (see cloudy documentation). To alleviate
    this one needs to make the orion grain abundances
    consistent with the depletion values. Assume 1 per cent of
    C is in PAH's.

    PAHs appear to exist mainly at the interface between the
    H+ region and the molecular clouds. Apparently PAHs are
    destroyed in ionized gas (Sellgren et al., 1990, AGN3
    section 8.5) by ionizing photons and by collisions with
    ions (mainly H+ ) and may be depleted into larger grains
    in molecular regions. Also assume the carbon fraction of
    PAHs from Abel+2008
    (https://iopscience.iop.org/article/10.1086/591505)
    assuming 1 percent of Carbon in PAHs. The factors in the
    denominators are the abundances of the carbon, silicon and
    PAH fractions when setting a value of 1 (fiducial abundance)
    for the orion and PAH grains.

    Another way is to scale the abundance as a function of the
    metallicity using the Z_PAH vs Z_gas relation from
    Galliano+2008
    (https://iopscience.iop.org/article/10.1086/523621,
    y = 4.17*Z_gas_sol - 7.085),
    which will again introduce issues on mass conservation.
    """

    if abundances.d2m > 0:
        delta_C = 10**abundances.a_nodep['C'] - 10**abundances.a['C']
        delta_PAH = 0.01 * (10**abundances.a_nodep['C'])
        delta_graphite = delta_C - delta_PAH
        delta_Si = 10**abundances.a_nodep['Si'] - 10**abundances.a['Si']
        orion_C_abund = -3.6259
        orion_Si_abund = -4.5547
        PAH_abund = -4.446
        f_graphite = delta_graphite/(10**(orion_C_abund))
        f_Si = delta_Si/(10**(orion_Si_abund))
        f_pah = delta_PAH/(10**(PAH_abund))
        command = (f'grains Orion graphite {f_graphite} '
                   f'\ngrains Orion silicate {f_Si} \ngrains '
                   f'PAH {f_pah}')
        cinput.append(command+'\n')
    else:
        f_graphite, f_Si, f_pah = 0, 0, 0

    # cinput.append('element off limit -7') # should speed up the code

    # # --- Define the ionising luminosity
    log10Q = np.log10(calculate_Q_from_U(10**log10U, 10**params["log10n_H"]))
    cinput.append(f'Q(H) = {log10Q}\n')
    # # cinput.append(f'ionization parameter = {log10U} log\n')

    # add background continuum
    if params['cosmic_rays']:
        cinput.append('cosmic rays, background\n')
    if params['CMB']:
        cinput.append(f'CMB {params["z"]}\n')

    # --- Define the geometry
    cinput.append(f'hden {params["log10n_H"]} log constant density\n')
    cinput.append(f'radius {params["log10radius"]} log parsecs\n')
    cinput.append('sphere\n')
    cinput.append(f'covering factor {params["covering_factor"]} linear\n')

    # --- Processing commands
    cinput.append('iterate to convergence\n')
    cinput.append(f'set temperature floor {params["T_floor"]} linear\n')
    cinput.append(f'stop temperature {params["stop_T"]}K\n')
    cinput.append(f'stop efrac {params["stop_efrac"]}\n')

    # --- output commands
    cinput.append(f'print line vacuum\n')  # output vacuum wavelengths
    cinput.append((f'save last continuum "{model_name}.cont" '
                   f'units Angstroms no clobber\n'))
    cinput.append((f'save last lines, array "{model_name}.lines" '
                  'units Angstroms no clobber\n'))
    cinput.append(f'save overview  "{model_name}.ovr" last\n')

    # --- save input file
    open(f'{output_dir}/{model_name}.in', 'w').writelines(cinput)

    return cinput


if __name__ == "__main__":

    grid_name = 'blackbody'

    # replace with arguments
    machine = 'apollo'
    synthesizer_data_dir = "/research/astrodata/highz/synthesizer/"
    output_dir = f"{synthesizer_data_dir}/cloudy/blackbody"
    cloudy = '/its/home/sw376/flare/software/cloudy/c17.03/source/cloudy.exe'

    # create path for cloudy runs
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # for submission system output files
    Path(f'{output_dir}/output').mkdir(parents=True, exist_ok=True)

    # log10T grid
    log10Ts = np.arange(4., 7., 0.2)

    # metallicity grid
    log10Zs = np.arange(-5., -1., 0.5)

    # log10U
    log10Us = np.array([-4., -3, -2, -1., 0.])

    # total number of models
    N = len(log10Ts)*len(log10Zs)*len(log10Us)

    # open the new grid
    with h5py.File(f'{synthesizer_data_dir}/grids/{grid_name}.hdf5', 'w') as hf:

        # add attribute with the grid axes for future when using >2D grid or AGN grids
        hf.attrs['grid_axes'] = ['log10T', 'log10Z', 'log10U']

        hf['log10U'] = log10Us
        hf['log10T'] = log10Ts
        hf['log10Z'] = log10Zs

    # for iT, log10T in enumerate(log10Ts):
    #     for iZ, log10Z in enumerate(log10Zs):
    #         for iU, log10U in enumerate(log10Us):
    #
    #             model_name = f"{iT}_{iZ}_{iU}"
    #
    #             # this will need changing
    #             abundances = Abundances(10**log10Z)
    #
    #             create_cloudy_input(model_name, log10T, abundances,
    #                                 output_dir=output_dir, log10U=log10U)
    #
    #             with open(f"{output_dir}/input_names.txt", "a") as myfile:
    #                 myfile.write(f'{model_name}\n')
    #
    # if machine == 'apollo':
    #     apollo_submission_script(N, output_dir, cloudy)
    # elif machine == 'cosma7':
    #     cosma7_submission_script(N, output_dir, cloudy,
    #                              cosma_project='cosma7',
    #                              cosma_account='dp004')
