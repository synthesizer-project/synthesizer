"""
Run a grid of cloudy models
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path
import yaml

from synthesizer.abundances_sw import Abundances
from synthesizer.grid import SpectralGrid
from synthesizer.cloudy_sw import create_cloudy_input

from write_submission_script import (apollo_submission_script,
                                     cosma7_submission_script)


def load_cloudy_parameters(verbose=False, **cloudy_parameters):
    default_cloudy_parameters = {
        # --- cloudy model
        'cloudy_version': 'v17.03',
        'U_model': 'ref',  # '' for fixed U
        'log10U_ref': -2,
        'log10age_ref': 6.,  # target reference age (only needed if U_model = 'ref')
        'Z_ref': 0.01,  # target reference metallicity (only needed if U_model = 'ref')
    
        # abundance parameters; these are used, alongside the total 
        # metallicity (Z), to define the abundance pattern
        'CO': 0.0,
        'd2m': 0.3,
        'alpha': 0.0,
        'scaling': None,
    
        # --- cloudy parameters
        'log10radius': -2,  # radius in log10 parsecs
        # covering factor. Keep as 1 as it is more efficient to simply combine SEDs 
        # to get != 1.0 values
        'covering_factor': 1.0,
        'stop_T': 4000,  # K
        'stop_efrac': -2,
        'T_floor': 100,  # K
        'log10n_H': 2,  # Hydrogen density
        'z': 0.,
        'CMB': False,
        'cosmic_rays': False
    }
    
    cloudy_params = default_cloudy_parameters

    for k, v in cloudy_parameters.items():
        cloudy_params[k] = v

    if verbose:
        print('-'*40)
        print(p)

    grid_name = (f'cloudy-{cloudy_params["cloudy_version"]}'
                 f'_log10U{cloudy_params["U_model"]}'
                 f'{cloudy_params["log10U_ref"]:.1f}')

    return cloudy_params, grid_name


def make_directories(synthesizer_data_dir, sps_grid, cloudy_name):

    output_dir = f'{synthesizer_data_dir}/cloudy/{sps_grid}_{cloudy_name}'

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(f'{output_dir}/output').mkdir(parents=True, exist_ok=True)  # for apollo output files

    return output_dir


def make_cloudy_input_grid(output_dir, grid, cloudy_params):

    ia_ref = grid.get_nearest_index(cloudy_params['log10age_ref'], grid.log10ages)
    iZ_ref = grid.get_nearest_index(cloudy_params['Z_ref'], grid.metallicities)

    # --- update the parameter file with the actual reference age and metallicity
    cloudy_params['log10age_ref_actual'] = grid.log10ages[ia_ref]
    cloudy_params['Z_ref_actual'] = grid.metallicities[iZ_ref]

    na = len(grid.ages)
    nZ = len(grid.metallicities)
    n = na * nZ

    i = 1

    for iZ in range(nZ):

        # --- get metallicity
        Z = grid.metallicities[iZ]

        # ---- initialise abundances object
        abundances = Abundances().generate_abundances(
            Z, cloudy_params['alpha'], cloudy_params['CO'], cloudy_params['d2m'], 
            scaling=cloudy_params['scaling'])

        for ia in range(na):

            if cloudy_params['U_model'] == 'ref':

                delta_log10Q = grid.log10Q[ia, iZ] - grid.log10Q[ia_ref, iZ_ref]

                log10U = cloudy_params['log10U_ref'] + (1/3) * delta_log10Q

            else:

                log10U = cloudy_params['log10U_ref']

            lam = grid.lam
            lnu = grid.spectra['stellar'][ia, iZ]

            model_name = f'{ia}_{iZ}'

            cloudy_params['log10U'] = log10U

            # TODO: currently this writes out the input script *twice*,
            # once in `create_cloudy_input` as ia_iZ.in, and once again 
            # below as i.in. Unnecessary
            cinput = create_cloudy_input(
                model_name, lam, lnu, abundances, output_dir=output_dir, 
                **cloudy_params)

            # --- write input file
            open(f'{output_dir}/{i}.in', 'w').writelines(cinput)

            i += 1

    yaml.dump(cloudy_params, open(f'{output_dir}/params.yaml', 'w'))

    return n


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=('Write cloudy input files '
                                                  'and submission script for a '
                                                  'given SPS grid.'))

    parser.add_argument("-dir", "--directory", type=str, required=True)
    parser.add_argument("-m", "--machine", type=str, required=True)
    parser.add_argument("-c", "--cloudy", type=str, nargs='?', const='CLOUDY17')
    
    args = parser.parse_args()

    synthesizer_data_dir = args.directory
    cloudy = args.cloudy

    sps_grids = [
        'bc03_chabrier03',
        # 'bpass-v2.2.1-bin_100-100',
        # 'bpass-v2.2.1-bin_100-300',
        # 'bpass-v2.2.1-bin_135-100',
        # 'bpass-v2.2.1-bin_135-300',
        # 'bpass-v2.2.1-bin_135all-100',
        # 'bpass-v2.2.1-bin_170-100',
        # 'bpass-v2.2.1-bin_170-300',
        # 'bpass-v2.2.1-bin_chab-100',
        # 'bpass-v2.2.1-bin_chab-300',
        # 'maraston-rhb_kroupa',
        # 'maraston-rhb_salpeter',
        # 'bc03-2016-Stelib_chabrier03',
        # 'bc03-2016-BaSeL_chabrier03',
        # 'bc03-2016-Miles_chabrier03',
    ]

    # different high-mass slopes
    # sps_grids = [f'fsps-v3.2_imf3:{imf3:.1f}' for imf3 in np.arange(1.5, 3.1, 0.1)]

    # sps_grids = [
    #     f'fsps-v3.2_imfll:{imf_lower_limit:.1f}' for imf_lower_limit in [0.5, 1, 5, 10, 50]]

    for sps_grid in sps_grids:
        # ---- load SPS grid
        grid = SpectralGrid(sps_grid, grid_dir=f'{synthesizer_data_dir}/grids')

        cloudy_params, cloudy_name = load_cloudy_parameters()
        output_dir = make_directories(synthesizer_data_dir, sps_grid, cloudy_name)
        N = make_cloudy_input_grid(output_dir, grid, cloudy_params)

        if args.machine == 'apollo':
            apollo_submission_script(synthesizer_data_dir, cloudy)
        elif args.machine == 'cosma7':
            cosma7_submission_script(N, '', output_dir, cloudy,
                         cosma_project='cosma7', cosma_account='dp004')
        else:
            ValueError(f'Machine {args.machine} not recognised.')

