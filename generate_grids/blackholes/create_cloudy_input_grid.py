"""
Given an SED (from an SPS model, for example), generate a cloudy atmosphere
grid. Can optionally generate an array of input files for selected parameters.
"""

import yaml
import numpy as np
from pathlib import Path
import argparse
import h5py
from unyt import Angstrom

# synthesizer modules
from synthesizer.cloudy import create_cloudy_input, ShapeCommands
from synthesizer.abundances import Abundances

# local modules
from emission_models import Feltre16
from utils import (get_grid_properties,
                   apollo_submission_script,
                   cosma7_submission_script)


def load_grid_params(param_file='default.yaml'):
    """
    parameters from a single param_file

    Parameters
    ----------
    param_file : str
        Location of YAML file.

    Returns
    -------
    dict
        Dictionary of cloudy parameters
    """

    # open paramter file
    with open(param_file, "r") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    grid_params = {}
    fixed_params = {}

    for k, v in params.items():
        if isinstance(v, list):
            grid_params[k] = v
        else:
            fixed_params[k] = v

    return fixed_params, grid_params


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Run a grid of Cloudy AGN models')

    # machine (for submission script generation)
    parser.add_argument("-machine", type=str, required=True)
    # path to synthesizer_data_dir
    parser.add_argument("-synthesizer_data_dir", type=str, required=True)
    # grid_name, used to define parameter file
    parser.add_argument("-grid_name", type=str, required=True)
    # path to cloudy directory (not executable; 
    # this is assumed to {cloudy}/{cloudy_version}/source/cloudy.ext)
    parser.add_argument("-cloudy_path", type=str, required=True)
    # boolean for dry run
    parser.add_argument("-dry_run", type=bool, required=False, default=False)

    # get arguments
    args = parser.parse_args()

    # set grid_name
    grid_name = args.grid_name

    # get model family
    family = grid_name.split('_')[0] # e.g. AGN, blackbody, SPS

    # get model
    model = grid_name.split('_')[1]

    machine = args.machine
    output_dir = f"{args.synthesizer_data_dir}/cloudy/{grid_name}"
    cloudy_path = args.cloudy_path

    # load cloudy parameters
    fixed_params, grid_params = load_grid_params(
        param_file=f'params/{grid_name}.yaml')

    cloudy_version = fixed_params['cloudy_version']

    print(machine)
    print(output_dir)
    print(cloudy_path)
    print(model)
    print(cloudy_version)

    for k, v in fixed_params.items():
        print(k, v)

    for k, v in grid_params.items():
        print(k, v)
    
    # make list of models
    grid_axes = list(grid_params.keys())

    # get the properties of the grid
    axes, n_axes, shape, n_models, mesh, model_list, index_list = \
        get_grid_properties(grid_axes, grid_params, verbose=True)

    if not args.dry_run:

        # create path for cloudy runs
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # for submission system output files
        Path(f'{output_dir}/output').mkdir(parents=True, exist_ok=True)
  
        # open the new grid
        with h5py.File(f'{args.synthesizer_data_dir}/grids/{grid_name}.hdf5',
                       'w') as hf:

            # add attribute with the grid axes
            hf.attrs['axes'] = grid_axes

            # add the bin centres for the grid bins
            for axis in axes:
                hf[f'axes/{axis}'] = grid_params[axis]

            # add other parameters as attributes
            for k, v in fixed_params.items():
                hf.attrs[k] = v

    for i, grid_params_ in enumerate(model_list):

        grid_params = dict(zip(grid_axes, grid_params_))

        params = fixed_params | grid_params

        # print(i, params)

        if not args.dry_run:

            # build abundances
            abundances = Abundances(10**params['log10Z'],
                                    d2m=params['d2m'],
                                    alpha=params['alpha'],
                                    C=params['C'],
                                    N=params['N'],)

            if model == 'cloudy':

                # create shape commands
                TBB = 10**params['log10T']
                shape_commands = ShapeCommands.cloudy_agn(TBB,
                                                          aox=params['aox'],
                                                          auv=params['auv'],
                                                          ax=params['ax'])
            
            # call a different model
            # TODO: allow this to call the method from a string
            elif model == 'feltre16':

                # define wavelength grid
                lam = np.arange(1, 20000, 1) * Angstrom

                # determine luminosity
                lnu = Feltre16.intrinsic(lam, alpha=params['aalpha'])

                # create shape commands
                shape_commands = ShapeCommands.table_sed(str(i), lam, lnu,
                                                         output_dir=output_dir)

            else:

                print('ERROR: unrecognised model')


            # create input file
            create_cloudy_input(str(i), shape_commands, abundances,
                                output_dir=output_dir, **params)

            # write out input file
            with open(f"{output_dir}/input_names.txt", "a") as myfile:
                myfile.write(f'{i}\n')

    if machine == 'apollo':
        apollo_submission_script(n_models, output_dir, cloudy_path,
                                 cloudy_version)
    elif machine == 'cosma7':
        cosma7_submission_script(n_models,
                                 output_dir,
                                 cloudy_path,
                                 cloudy_version,
                                 cosma_project='cosma7',
                                 cosma_account='dp004')
