
import numpy as np


def get_grid_properties(axes, axes_values, verbose=True):

    """ 
    Get the properties of the grid including the dimensions etc.

    Args:
        axes (list)
            The axes of the grid
        axes_values (dict)
            Dictionary of axes values

    Returns:
        axes
        n_axes 
        shape
        n_models
        mesh
        model_list
        index_list
    
    """

    # the grid axes
    if verbose:
        print(f'axes: {axes}')

    # number of axes
    n_axes = len(axes)
    if verbose:
        print(f'number of axes: {n_axes}')

    # the shape of the grid (useful for creating outputs)
    shape = list([len(axes_values[axis]) for axis in axes])
    if verbose:
        print(f'shape: {shape}')

    # determine number of models
    n_models = np.prod(shape)
    if verbose: 
        print(f'number of models to run: {n_models}')

    # create the mesh of the grid
    mesh = np.array(np.meshgrid(*[np.array(axes_values[axis]) for axis in axes]))

    # create the list of the models 
    model_list = mesh.T.reshape(n_models, n_axes)
    if verbose: 
        print('model list:')
        print(model_list)

    # create a list of the indices

    index_mesh = np.array(np.meshgrid(*[range(n) for n in shape]))

    index_list =  index_mesh.T.reshape(n_models, n_axes)
    if verbose: 
        print('index list:')
        print(index_list)

    return axes, n_axes, shape, n_models, mesh, model_list, index_list


def apollo_submission_script(n, grid_data_dir, cloudy_path, cloudy_version):

    """
    Create an Apollo SGE submission script.

    Args:
        n (int)
            Number of models to run, sets size of array job.
        synthesizer_data_dir (str)
            where to write the submission script
        cloudy (str)
            bash executable for CLOUDY

    """

    # cloudy executable
    cloudy = f'{cloudy_path}/{cloudy_version}/source/cloudy.exe'

    print(cloudy)

    # cloudy data dir
    cloudy_data_path = f'{cloudy_path}/{cloudy_version}/data/'

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

# increment array task ID so not zero indexed
let index=$SGE_TASK_ID

# set cloudy data path
export CLOUDY_DATA_PATH={cloudy_data_path}

# access line at index from input_names file
id=$(sed "${{index}}q;d" input_names.txt)
{cloudy} -r $id
"""

    open(f'{grid_data_dir}/run_grid.job', 'w').write(apollo_job_script)
    print(grid_data_dir)
    # print(f'qsub -t 1:{n} run_grid.job -v CLOUDY_DATA_PATH={cloudy_data_path}')
    print(f'qsub -t 1:{n} run_grid.job')

    return


def cosma7_submission_script(N, output_dir, cloudy,
                             cosma_project='cosma7', cosma_account='dp004'):
    """
    Create a cosma7 SLURM submission script.

    Args:
        N (int)
            Number of models to run, sets size of array job.
            Warning: if greater than 1000, SLURM may fail.
            May need to set manually.
        output_dir (str)
            where to write the submission script
        cloudy (str)
            bash executable for CLOUDY
        cosma_project (str)
            name of COSMA project / machine to submit to
        cosma_account (str)
            name of COSMA account to charge computing time to

    """

    output = []
    output.append('#!/bin/bash -l\n')
    output.append('#SBATCH --ntasks 1\n')
    output.append('#SBATCH -J job_name\n')
    output.append(f'#SBATCH --array=0-{N}\n')
    # output.append(f'#SBATCH -o standard_output_file.%A.%a.out
    # output.append(f'#SBATCH -e standard_error_file.%A.%a.err
    output.append(f'#SBATCH -p {cosma_project}\n')
    output.append(f'#SBATCH -A {cosma_account}\n')
    output.append('#SBATCH --exclusive\n')
    output.append('#SBATCH -t 00:15:00\n\n')
    # output.append(f'#SBATCH --mail-type=END # notifications for job done &
    # output.append(f'#SBATCH --mail-user=<email address>

    # increment array task ID so not zero indexed
    output.append('let index=$SLURM_ARRAY_TASK_ID+1\n')

    # access line at index from input_names file
    output.append('id=$(sed "${index}q;d" input_names.txt)\n')

    # run CLOUDY for the given model {ia}_{iZ}.in
    output.append(f'${cloudy} -r $id\n')

    open(f'{output_dir}/run.job', 'w').writelines(output)

    return
