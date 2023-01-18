import argparse
import numpy as np


def apollo_submission_script(synthesizer_data_dir, cloudy):

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

    open(f'{synthesizer_data_dir}/run_grid.job', 'w').write(apollo_job_script)
    print(synthesizer_data_dir)
    print(f'qsub -t 1:{n} run_grid.job')


def cosma7_submission_script(N, output_dir, cloudy,
                             cosma_project='cosma7', cosma_account='dp004'):

    output = []
    output.append(f'#!/bin/bash -l\n')
    output.append(f'#SBATCH --ntasks 1\n')
    output.append(f'#SBATCH -J job_name\n')
    output.append(f'#SBATCH --array=0-{N}\n')
    # output.append(f'#SBATCH -o standard_output_file.%A.%a.out
    # output.append(f'#SBATCH -e standard_error_file.%A.%a.err
    output.append(f'#SBATCH -p {cosma_project}\n')
    output.append(f'#SBATCH -A {cosma_account}\n')
    output.append(f'#SBATCH --exclusive\n')
    output.append(f'#SBATCH -t 00:15:00\n\n')
    # output.append(f'#SBATCH --mail-type=END # notifications for job done &
    # output.append(f'#SBATCH --mail-user=<email address>
    output.append(f'{cloudy} -r $SLURM_ARRAY_TASK_ID\n')

    open(f'{output_dir}/{input_prefix}_run.job','w').writelines(output)

    return


# if __name__ == "__main__":
# 
#     parser = argparse.ArgumentParser(description='Write submission script for various machines.')
#     parser.add_argument("-dir", "--directory", type=str, required=True)
#     parser.add_argument("-m", "--machine", type=str, required=True)
#     parser.add_argument("-c", "--cloudy", type=str, nargs='?', const='CLOUDY17')
#     args = parser.parse_args()
# 
#     synthesizer_data_dir = args.directory
#     cloudy = args.cloudy
# 
#     if args.machine == 'apollo':
#         apollo_submission_script(synthesizer_data_dir, cloudy)
#     elif arg.machine == 'cosma7':
#         cosma7_submission_script()
#     else:
#         ValueError(f'Machine {args.machine} not recognised.')
