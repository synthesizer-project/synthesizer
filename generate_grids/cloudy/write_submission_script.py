
def apollo_submission_script(n, grid_data_dir, cloudy):
    """
    Create an Apollo SGE submission script.

    Parameters
    ----------
    n : int
        Number of models to run, sets size of array job.
    synthesizer_data_dir : str
        where to write the submission script
    cloudy : str
        bash executable for CLOUDY

    Returns
    -------
    None
    """

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
let index=$SGE_TASK_ID+1

# access line at index from input_names file
id=$(sed "${{index}}q;d" input_names.txt)
{cloudy} -r $id
"""

    open(f'{grid_data_dir}/run_grid.job', 'w').write(apollo_job_script)
    print(grid_data_dir)
    print(f'qsub -t 1:{n} run_grid.job')

    return


def cosma7_submission_script(N, output_dir, cloudy,
                             cosma_project='cosma7', cosma_account='dp004'):
    """
    Create a cosma7 SLURM submission script.

    Parameters
    ----------
    N : int
        Number of models to run, sets size of array job.
        Warning: if greater than 1000, SLURM may fail. May need to set manually
    output_dir : str
        where to write the submission script
    cloudy : str
        bash executable for CLOUDY
    cosma_project : str
        name of COSMA project / machine to submit to
    cosma_account : str
        name of COSMA account to charge computing time to

    Returns
    -------
    None
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
