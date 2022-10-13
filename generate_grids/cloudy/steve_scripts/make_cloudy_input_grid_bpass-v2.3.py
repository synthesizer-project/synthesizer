"""
Run a grid of cloudy models
"""

import os
import sys
import numpy as np
from pathlib import Path
import yaml

from synthesizer.abundances_sw import Abundances
from synthesizer.grid import SpectralGrid
from synthesizer.cloudy_sw import create_cloudy_input

synthesizer_data_dir = os.getenv('SYNTHESIZER_DATA')


cloudy = 'CLOUDY17'

apollo_job_script = f"""
######################################################################
# Options for the batch system
# These options are not executed by the script, but are instead read by the
# batch system before submitting the job. Each option is preceeded by '#$' to
# signify that it is for grid engine.
#
# All of these options are the same as flags you can pass to qsub on the
# command line and can be **overriden** on the command line. see man qsub for
# all the details
######################################################################
# -- The shell used to interpret this script
#$ -S /bin/bash
# -- Execute this job from the current working directory.
#$ -cwd
# -- Job output to stderr will be merged into standard out. Remove this line if
# -- you want to have separate stderr and stdout log files
#$ -j y
#$ -o output/
# -- Send email when the job exits, is aborted or suspended
# #$ -m eas
# #$ -M YOUR_USERNAME@sussex.ac.uk

######################################################################
# Job Script
# Here we are writing in bash (as we set bash as our shell above). In here you
# should set up the environment for your program, copy around any data that
# needs to be copied, and then execute the program
######################################################################

${cloudy} -r $SGE_TASK_ID
"""


p = {
        # --- cloudy model
        'cloudy_version' : 'v17.03',
        'U_model' : 'ref', # '' for fixed U
        'log10U_ref' : -2,
        'log10age_ref': 6., # target reference age (only needed if U_model = 'ref')
        'Z_ref': 0.01, # target reference metallicity (only needed if U_model = 'ref')

        # --- abundance parameters,  these are used, alongside the total metallicity (Z), to define the abundance pattern
        'CO' : 0.0,
        'd2m' : 0.3,
        'alpha' : 0.0,
        'scaling' : None,

        # --- cloudy parameters
        'log10radius': -2, # radius in log10 parsecs
        'covering_factor': 1.0, # covering factor. Keep as 1 as it is more efficient to simply combine SEDs to get != 1.0 values
        'stop_T': 4000, # K
        'stop_efrac': -2,
        'T_floor': 100, # K
        'log10n_H': 2, # Hydrogen density
        'z': 0.,
        'CMB': False,
        'cosmic_rays': False
        }


sps_grids = [
    # 'bc03_chabrier03',
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
    'bc03-2016-Stelib_chabrier03',
    'bc03-2016-BaSeL_chabrier03',
    'bc03-2016-Miles_chabrier03',
]

# sps_grids = ['bpass-v2.2.1-bin_chab-100']

cloudy_grid = f'cloudy-{p["cloudy_version"]}_log10U{p["U_model"]}{p["log10U_ref"]}'



for sps_grid in sps_grids:

    print('-'*40)
    print(sps_grid)

    # ---- load SPS grid
    grid = SpectralGrid(sps_grid)

    ia_ref = grid.get_nearest_index(p['log10age_ref'], grid.log10ages)
    iZ_ref = grid.get_nearest_index(p['Z_ref'], grid.metallicities)

    # --- update the parameter file with the actual reference age and metallicity
    p['log10age_ref_actual'] = grid.log10ages[ia_ref]
    p['Z_ref_actual'] = grid.metallicities[iZ_ref]

    output_dir = f'{synthesizer_data_dir}/cloudy/{sps_grid}_{cloudy_grid}'

    Path(output_dir).mkdir(parents = True, exist_ok = True)
    Path(f'{output_dir}/output').mkdir(parents = True, exist_ok = True) # for apollo output files

    na = len(grid.ages)
    nZ = len(grid.metallicities)
    n = na * nZ

    make_input_files = True

    if make_input_files:

        i = 1

        for iZ in range(nZ):

            # --- get metallicity
            Z = grid.metallicities[iZ]

            # ---- initialise abundances object
            abundances = Abundances().generate_abundances(Z, p['alpha'], p['CO'], p['d2m'], scaling = p['scaling']) # abundances object

            for ia in range(na):

                if p['U_model'] == 'ref':

                    delta_log10Q = grid.log10Q[ia, iZ] - grid.log10Q[ia_ref, iZ_ref]

                    log10U = p['log10U_ref'] + (1/3) * delta_log10Q

                else:

                    log10U = p['log10U_ref']


                lam = grid.lam
                lnu = grid.spectra['stellar'][ia, iZ]


                model_name = f'{ia}_{iZ}'


                cinput = create_cloudy_input(model_name, lam, lnu, abundances, log10U, output_dir = output_dir, **p)

                # --- write input file
                open(f'{output_dir}/{i}.in','w').writelines(cinput)

                i += 1


    open(f'{output_dir}/run_grid.job','w').write(apollo_job_script)
    yaml.dump(p, open(f'{output_dir}/params.yaml', 'w'))
    print(output_dir)
    print(f'qsub -t 1:{n} run_grid.job')
