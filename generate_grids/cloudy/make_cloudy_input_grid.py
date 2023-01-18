"""
Create a grid of cloudy models
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path
import yaml

from synthesizer.abundances_sw import Abundances
from synthesizer.grid import Grid
from synthesizer.cloudy_sw import create_cloudy_input

from write_submission_script import (apollo_submission_script,
                                     cosma7_submission_script)


def load_cloudy_parameters(param_file='default_param.yaml', 
                           verbose=False, **kwarg_parameters):
    """
    A helpful docstring :)
    """

    with open(param_file, "r") as stream:
        try:
            cloudy_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # update any custom parameters
    for k, v in kwarg_parameters.items():
        cloudy_params[k] = v

    if verbose:
        print('-'*40)
        print(p)

    # search for any lists of parameters. 
    # currently exits once it finds the *first* list
    # TODO: adapt to accept multiple varied parameters
    for k, v in cloudy_params.items():
        if type(v) is list:
            output_cloudy_params = []
            output_cloudy_names = []

            for _v in v:
                # update the value in our default dictionary
                cloudy_params[k] = _v

                # save to list of cloudy param dicts
                output_cloudy_params.append(cloudy_params)

                # replace negative '-' with m
                out_str = f'{k}{str(_v).replace("-", "m")}'

                # save to list of output strings
                output_cloudy_names.append(out_str)

            return output_cloudy_params, output_cloudy_names

    return [cloudy_params], ['']


def make_directories(synthesizer_data_dir, sps_grid, cloudy_name):

    output_dir = f'{synthesizer_data_dir}/cloudy/{sps_grid}_{cloudy_name}'

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # for submission system output files
    Path(f'{output_dir}/output').mkdir(parents=True, exist_ok=True)

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

            cinput = create_cloudy_input(
                model_name, lam, lnu, abundances, output_dir=output_dir, 
                **cloudy_params)

            # write filename out
            with open(f"{output_dir}/input_names.txt", "a") as myfile:
                myfile.write(f'{model_name}\n')

    yaml.dump(cloudy_params, open(f'{output_dir}/params.yaml', 'w'))

    return n


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=('Write cloudy input files '
                                                  'and submission script for a '
                                                  'given SPS grid.'))

    parser.add_argument("-dir", "--directory", type=str, required=True)

    parser.add_argument("-m", "--machine", type=str, 
                        choices=['cosma7', 'apollo'], default=None,
                        help=('Write a submission script for the specified machine. '
                              'Default is None - write no submission script.'))
    
    parser.add_argument("-sps", "--sps_grid", type=str, nargs='+', required=True, 
                        help=('The SPS grid(s) to run the cloudy grid on. '
                            'Multiple grids can be listed as: \n '
                            '  --sps_grid grid_1 grid_2'))

    parser.add_argument("-p", "--params", type=str, required=True, 
                        help='YAML parameter file of cloudy parameters')

    parser.add_argument("-c", "--cloudy", type=str, nargs='?', default='$CLOUDY17', 
                        help='CLOUDY executable call')
    
    args = parser.parse_args()

    synthesizer_data_dir = args.directory
    cloudy = args.cloudy

    for sps_grid in args.sps_grid:

        print(f"Loading the SPS grid: {sps_grid}")

        # load the specified SPS grid
        grid = Grid(sps_grid, grid_dir=f'{synthesizer_data_dir}/grids')
    
        print(f"Loading the cloudy parameters from: {args.params}")

        # load the cloudy parameters you are going to run
        c_params, c_name = load_cloudy_parameters(args.params)
        
        for i, (cloudy_params, cloudy_name) in \
                enumerate(zip(c_params, c_name)):
            
            # if no variations, save as 'default' cloudy grid
            if cloudy_name == '':
                cloudy_name = 'cloudy'
    
            output_dir = make_directories(synthesizer_data_dir, sps_grid, cloudy_name)
            
            print(f"Generating cloudy grid for ({i}) {cloudy_name} in {output_dir}")
        
            N = make_cloudy_input_grid(output_dir, grid, cloudy_params)
        
            if args.machine == 'apollo':
                apollo_submission_script(synthesizer_data_dir, cloudy)
            elif args.machine == 'cosma7':
                cosma7_submission_script(N, output_dir, cloudy,
                             cosma_project='cosma7', cosma_account='dp004')
            elif args.machine is None:
                print("No machine specified. Skipping submission script write out") 
            else:
                ValueError(f'Machine {args.machine} not recognised.')
    
