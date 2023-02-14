"""
Create a grid of cloudy models
"""

import argparse
from pathlib import Path
import yaml

from synthesizer.abundances import Abundances
from synthesizer.grid import Grid
from synthesizer.cloudy import create_cloudy_input

from write_submission_script import (apollo_submission_script,
                                     cosma7_submission_script)

from copy import deepcopy


def load_cloudy_parameters(param_file='default.yaml', default_param_file='default.yaml',
                           **kwarg_parameters):
    """
    Load CLOUDY parameters from a YAML file

    Parameters
    ----------
    param_file : str
        Location of YAML file.
    **kwarg_parameters
        Additional parameters you may wish to amend manually

    Returns
    -------
    bool
        True if successful, False otherwise.
    """

    # open paramter file
    with open(param_file, "r") as stream:
        try:
            cloudy_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # open default parameter file
    with open(default_param_file, "r") as stream:
        try:
            default_cloudy_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # copy over defaults where the parameter is not already set
    for k, v in default_cloudy_params.items():
        if k not in cloudy_params.keys():
            cloudy_params[k] = v

    # first loop through and identify changes that are not in lists (e.g. U_model) and add them to the model name
    out_str_ = f"cloudy-{cloudy_params['cloudy_version']}"

    # flag denoting that the parameter file includes at least one list. Used below.
    parameter_list = False
    for k, v in cloudy_params.items():
        if type(v) is not list:
            if v != default_cloudy_params[k]:
                out_str_ += f'-{k}{str(v).replace("-", "m")}'
        else:
            parameter_list = True

    # search for any lists of parameters.
    # currently exits once it finds the *first* list
    # TODO: adapt to accept multiple varied parameters

    if not parameter_list:

        output_cloudy_params = [cloudy_params]
        output_cloudy_names = [out_str_]

    else:

        output_cloudy_params = []
        output_cloudy_names = []

        for k, v in cloudy_params.items():
            if type(v) is list:
                for _v in v:

                    cloudy_params_ = deepcopy(cloudy_params)

                    # update the value in our default dictionary
                    cloudy_params_[k] = _v

                    # save to list of cloudy param dicts
                    output_cloudy_params.append(cloudy_params_)

                    # replace negative '-' with m
                    # out_str = f'-{k}{str(_v).replace("-", "m")}'
                    out_str = f'-{k}{str(_v).replace("-", "m")}'

                    # save to list of output strings
                    output_cloudy_names.append(out_str_+out_str)

    return output_cloudy_params, output_cloudy_names


# def load_cloudy_parameters(param_file='default.yaml', default_param_file='default.yaml',
#                            **kwarg_parameters):
#     """
#     Load CLOUDY parameters from a YAML file
#
#     Parameters
#     ----------
#     param_file : str
#         Location of YAML file.
#     **kwarg_parameters
#         Additional parameters you may wish to amend manually
#
#     Returns
#     -------
#     bool
#         True if successful, False otherwise.
#     """
#
#     # open paramter file
#     with open(param_file, "r") as stream:
#         try:
#             cloudy_params = yaml.safe_load(stream)
#         except yaml.YAMLError as exc:
#             print(exc)
#
#     # open default parameter file
#     with open(default_param_file, "r") as stream:
#         try:
#             default_cloudy_params = yaml.safe_load(stream)
#         except yaml.YAMLError as exc:
#             print(exc)
#
#     # find any differences - this breaks when a list is defined
#     cloudy_params_set = set(cloudy_params.items())
#     default_cloudy_params_set = set(default_cloudy_params.items())
#
#     param_changes = dict(cloudy_params_set - default_cloudy_params_set)
#
#     # update any custom parameters
#     for k, v in kwarg_parameters.items():
#         cloudy_params[k] = v
#         param_changes[k] = v
#
#     out_str = ''
#     for k, v in param_changes.items():
#         out_str += f'-{k}{str(v).replace("-", "m")}'
#
#     return [cloudy_params], [out_str]


def make_directories(synthesizer_data_dir, sps_grid, cloudy_name):
    """
    Create required directories for storing CLOUDY input and output files

    Parameters
    ----------
    synthesizer_data_dir : str
        the top level output directory.
    sps_grid : str
        The name of the SPS grid.
    cloudy_name : str
        The cloudy specific post-fix for the output directory name

    Returns
    -------
    output_dir : str
        Name of the output directory.
    """

    output_dir = f'{synthesizer_data_dir}/cloudy/{sps_grid}_{cloudy_name}'

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # for submission system output files
    Path(f'{output_dir}/output').mkdir(parents=True, exist_ok=True)

    return output_dir


def make_cloudy_input_grid(output_dir, grid, cloudy_params):
    """
    Create a grid of CLOUDY input files for a given SPS grid.

    Parameters
    ----------
    output_dir : str
        Where to write the input files.
    grid : obj
        A synthesizer `Grid` object
    cloudy_params : dict
        Dictionary of cloudy parameters

    Returns
    -------
    N : int
        Total number of models.
    """

    ia_ref = grid.get_nearest_index(cloudy_params['log10age_ref'],
                                    grid.log10ages)
    iZ_ref = grid.get_nearest_index(cloudy_params['Z_ref'], grid.metallicities)

    # add these to the parameter file
    cloudy_params['ia_ref'] = int(ia_ref)
    cloudy_params['iZ_ref'] = int(iZ_ref)

    # update the parameter file with the actual reference age and metallicity
    # converting to float makes the resulting parameter file readable
    cloudy_params['log10age_ref_actual'] = float(grid.log10ages[ia_ref])
    cloudy_params['Z_ref_actual'] = float(grid.metallicities[iZ_ref])

    na = len(grid.ages)
    nZ = len(grid.metallicities)
    n = na * nZ

    for iZ in range(nZ):

        # --- get metallicity
        Z = grid.metallicities[iZ]

        # ---- initialise abundances object
        abundances = Abundances(
            Z=Z, alpha=cloudy_params['alpha'], CO=cloudy_params['CO'], d2m=cloudy_params['d2m'])

        for ia in range(na):

            if cloudy_params['U_model'] == 'ref':

                delta_log10Q = grid.log10Q['HI'][ia, iZ] - \
                    grid.log10Q['HI'][ia_ref, iZ_ref]

                if cloudy_params['geometry'] == 'spherical':

                    # SW: I'm not sure I fully understand this
                    log10U = cloudy_params['log10U_ref'] + (1/3) * delta_log10Q

                elif cloudy_params['geometry'] == 'planeparallel':

                    # log10U just scales with log10Q
                    log10U = cloudy_params['log10U_ref'] + delta_log10Q

                else:

                    print(f"ERROR: do not understand geometry choice: {cloudy_params['geometry']}")

            elif cloudy_params['U_model'] == 'fixed':

                log10U = cloudy_params['log10U_ref']

            else:

                print(f"ERROR: do not understand U model choice: {cloudy_params['U_model']}")

            lam = grid.lam
            lnu = grid.spectra['stellar'][ia, iZ]

            model_name = f'{ia}_{iZ}'

            cloudy_params['log10U'] = float(log10U)

            create_cloudy_input(model_name, lam, lnu, abundances,
                                output_dir=output_dir, **cloudy_params)

            # write filename out
            with open(f"{output_dir}/input_names.txt", "a") as myfile:
                myfile.write(f'{model_name}\n')

    yaml.dump(cloudy_params, open(f'{output_dir}/params.yaml', 'w'))

    return n


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=('Write cloudy input files '
                                                  'and submission script for '
                                                  'a given SPS grid.'))

    parser.add_argument("-dir", "--directory", type=str, required=False, default=None)

    parser.add_argument("-m", "--machine", type=str, choices=['cosma7', 'apollo'], default=None, help=(
        'Write a submission script for the ', 'specified machine. Default is None - ', 'write no submission script.'))

    parser.add_argument("-sps", "--sps_grid", type=str,
                        nargs='+', required=False, default=None,
                        help=('The SPS grid(s) to run the cloudy grid on. '
                              'Multiple grids can be listed as: \n '
                              '  --sps_grid grid_1 grid_2'))

    parser.add_argument("-p", "--params", type=str, required=False, default='default.yaml',
                        help='YAML parameter file of cloudy parameters')

    parser.add_argument("-c", "--cloudy", type=str, nargs='?',
                        default='$CLOUDY17', help='CLOUDY executable call')

    args = parser.parse_args()

    synthesizer_data_dir = args.directory
    cloudy = args.cloudy

    print(f"Loading the cloudy parameters from: {args.params}")

    # load the cloudy parameters you are going to run
    c_params, c_name = load_cloudy_parameters(args.params)

    for i, cloudy_name in enumerate(c_name):
        print(i, cloudy_name)

    if args.sps_grid:

        for sps_grid in args.sps_grid:

            print(f"Loading the SPS grid: {sps_grid}")

            # load the specified SPS grid
            grid = Grid(sps_grid, grid_dir=f'{synthesizer_data_dir}/grids')

            for i, (cloudy_params, cloudy_name) in \
                    enumerate(zip(c_params, c_name)):

                output_dir = make_directories(synthesizer_data_dir, sps_grid,
                                              cloudy_name)

                print((f"Generating cloudy grid for ({i}) "
                       f"{cloudy_name} in {output_dir}"))

                N = make_cloudy_input_grid(output_dir, grid, cloudy_params)

                if args.machine == 'apollo':
                    apollo_submission_script(N, output_dir, cloudy)
                elif args.machine == 'cosma7':
                    cosma7_submission_script(N, output_dir, cloudy,
                                             cosma_project='cosma7',
                                             cosma_account='dp004')
                elif args.machine is None:
                    print(("No machine specified. Skipping "
                           "submission script write out"))
                else:
                    ValueError(f'Machine {args.machine} not recognised.')
