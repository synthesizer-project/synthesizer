
"""
This reads in a cloudy grid of models and creates a new SPS grid including the various outputs

"""


from scipy import integrate
import os
import shutil
from synthesizer.cloudy import read_wavelength, read_continuum, read_lines
from synthesizer.sed import calculate_Q
from unyt import eV
import argparse
import numpy as np
import h5py
import yaml
from write_submission_script import (apollo_submission_script,
                                     cosma7_submission_script)


def check_cloudy_runs(grid_name, synthesizer_data_dir, cloudy=None, replace=False, machine='apollo'):
    """
    Check that all the cloudy runs have run properly

    Parameters
    ----------
    grid_name : str
        Name of the grid
    synthesizer_data_dir : str
        Directory where synthesizer data is kept.
    replace : boolean
        If a run has failed simply replace the model with the previous one
    """

    # open the new grid
    with h5py.File(f'{synthesizer_data_dir}/grids/{grid_name}.hdf5', 'r') as hf:

        log10Us = hf['log10U'][:]
        log10Ts = hf['log10T'][:]
        log10Zs = hf['log10Z'][:]

    failed_list = []

    for iT, log10T in enumerate(log10Ts):
        for iZ, log10Z in enumerate(log10Zs):
            for iU, log10U in enumerate(log10Us):

                model_name = f'{iT}_{iZ}_{iU}'
                infile = f'{synthesizer_data_dir}/cloudy/{grid_name}/{model_name}'
                failed = False

                try:
                    spec_dict = read_continuum(infile, return_dict=True)
                except:
                    failed = True

                try:
                    id, blend, wavelength, intrinsic, emergent = read_lines(infile)
                except:
                    failed = True

                # if not os.path.isfile(infile+'.cont'):  # attempt to open run.
                #     failed = True
                # if not os.path.isfile(infile+'.lines'):  # attempt to open run.
                #     failed = True

                if failed:

                    failed_list.append((iT, iZ, iU))

                    with open(f"{synthesizer_data_dir}/cloudy/{grid_name}/reprocess_names.txt", "a") as myfile:
                        myfile.write(f'{model_name}\n')

                    print(f'FAILED: {model_name}')

    N = len(failed_list)

    if N > 0:

        if machine == 'apollo':
            apollo_submission_script(N, f'{synthesizer_data_dir}/cloudy/{grid_name}', cloudy)
        elif machine == 'cosma7':
            cosma7_submission_script(N, f'{synthesizer_data_dir}/cloudy/{grid_name}', cloudy,
                                     cosma_project='cosma7',
                                     cosma_account='dp004')

        failed = True

    return failed


def fix_cloudy_runs(grid_name, synthesizer_data_dir, replace=False):
    """
    If a cloudy run has failed replace it with the previous metallicity grid point
    """

    # open the new grid
    with h5py.File(f'{synthesizer_data_dir}/grids/{grid_name}.hdf5', 'r') as hf:

        log10Us = hf['log10U'][:]
        log10Ts = hf['log10T'][:]
        log10Zs = hf['log10Z'][:]

    for iT, log10T in enumerate(log10Ts):
        for iZ, log10Z in enumerate(log10Zs):
            for iU, log10U in enumerate(log10Us):

                model_name = f'{iT}_{iZ}_{iU}'
                infile = f'{synthesizer_data_dir}/cloudy/{grid_name}/{model_name}'

                try:
                    read_continuum(infile, return_dict=True)

                except:

                    print(model_name)

                    if iZ > 0:
                        nf = f'{iT}_{iZ-1}_{iU}'
                    else:
                        nf = f'{iT}_{iZ}_{iU-1}'

                    if replace:
                        os.system(
                            f'cp {synthesizer_data_dir}/cloudy/{grid_name}/{nf}.cont {synthesizer_data_dir}/cloudy/{grid_name}/{ia}_{iZ}.cont')
                        os.system(
                            f'cp {synthesizer_data_dir}/cloudy/{grid_name}/{nf}.lines {synthesizer_data_dir}/cloudy/{grid_name}/{ia}_{iZ}.lines')


def add_spectra(grid_name, synthesizer_data_dir):
    """
    Open cloudy spectra and add them to the grid

    Parameters
    ----------
    grid_name : str
        Name of the grid
    synthesizer_data_dir : str
        Directory where synthesizer data is kept.
    """

    # spec_names = ['incident','transmitted','nebular','nebular_continuum','total','linecont']
    #  the cloudy spectra to save (others can be generated later)
    spec_names = ['incident', 'transmitted', 'nebular', 'linecont']

    # open the new grid
    with h5py.File(f'{synthesizer_data_dir}/grids/{grid_name}.hdf5', 'a') as hf:

        log10Ts = hf['log10T']
        log10Zs = hf['log10Z']
        log10Us = hf['log10U']

        nTs = len(log10Ts)
        nZs = len(log10Zs)
        nUs = len(log10Us)

        # --- read first spectra from the first grid point to get length and wavelength grid
        lam = read_wavelength(f'{synthesizer_data_dir}/cloudy/{grid_name}/0_0_0')

        spectra = hf.create_group('spectra')  # create a group holding the spectra in the grid file
        spectra.attrs['spec_names'] = spec_names  # save list of spectra as attribute

        spectra['wavelength'] = lam  # save the wavelength
        nlam = len(lam)  # number of wavelength points

        # --- make spectral grids and set them to zero
        for spec_name in spec_names:
            spectra[spec_name] = np.zeros((nTs, nZs, nUs, nlam))

        # --- now loop over meallicity and ages

        for iT, log10T in enumerate(log10Ts):
            for iZ, log10Z in enumerate(log10Zs):
                for iU, log10U in enumerate(log10Us):

                    infile = f'{synthesizer_data_dir}/cloudy/{grid_name}/{iT}_{iZ}_{iU}'

                    spec_dict = read_continuum(infile, return_dict=True)
                    for spec_name in spec_names:

                        # should perhaps normalise to unity and return the factor for use by lines
                        spectra[spec_name][iT, iZ, iU] = spec_dict[spec_name]


def get_default_line_list(interesting=True):

    with open('default_lines.dat') as f:
        line_list = f.read().splitlines()

    if interesting:

        with open('interesting_lines.dat') as f:
            line_list += f.read().splitlines()

    return line_list


# def get_line_list(grid_name, synthesizer_data_dir, threshold_line='H 1 4862.69A', relative_threshold=2.0):
#     """
#     Get a list of lines meeting some threshold at the reference age and metallicity
#
#     NOTE: the updated base grid HDF5 file must have been created first.
#     NOTE: changing the threshold to 2.5 doubles the number of lines and produces repeats that will need to be merged.
#
#     Parameters
#     ----------
#     grid_name : str
#         Name of the grid.
#     synthesizer_data_dir : str
#         Directory where synthesizer data is kept.
#     threshold : float
#         The log threshold relative to Hbeta for which lines should be kept.
#         Default = 2.0 which implies L > 0.01 * L_Hbeta
#
#     Returns
#     ----------
#     list
#         list of the lines meeting the threshold criteria
#     """
#
#     # open the new grid
#     with h5py.File(f'{synthesizer_data_dir}/grids/{grid_name}.hdf5', 'a') as hf:
#
#         # get the reference metallicity and age grid point
#         ia = hf.attrs['ia_ref']
#         iZ = hf.attrs['iZ_ref']
#
#     reference_filename = f'{synthesizer_data_dir}/cloudy/{grid_name}/{ia}_{iZ}'
#
#     cloudy_ids, blends, wavelengths, intrinsic, emergent = read_lines(
#         reference_filename)
#
#     threshold = emergent[cloudy_ids == threshold_line] - relative_threshold
#
#     s = (emergent > threshold) & (blends == False) & (wavelengths < 50000)
#
#     line_list = cloudy_ids[s]
#
#     # print(f'number of lines: {np.sum(s)}')
#     # print(line_list)
#     # print(len(list(set(line_list))))
#
#     return line_list


def add_lines(grid_name, synthesizer_data_dir, lines_to_include, normalisation=1.):
    """
    Open cloudy lines and add them to the HDF5 grid

    Parameters
    ----------
    grid_name: str
        Name of the grid.
    synthesizer_data_dir: str
        Directory where synthesizer data is kept.
    lines_to_include
        List of lines to include
    normalisation
        Optional normalisation
    """

    # open the new grid
    with h5py.File(f'{synthesizer_data_dir}/grids/{grid_name}.hdf5', 'a') as hf:

        # shorthand
        log10Ts = hf['log10T']
        log10Zs = hf['log10Z']
        log10Us = hf['log10U']

        nTs = len(log10Ts)
        nZs = len(log10Zs)
        nUs = len(log10Us)

        lines = hf.create_group('lines')

        for line_id in lines_to_include:
            lines[f'{line_id}/luminosity'] = np.zeros((nTs, nZs, nUs))
            lines[f'{line_id}/stellar_continuum'] = np.zeros((nTs, nZs, nUs))
            lines[f'{line_id}/nebular_continuum'] = np.zeros((nTs, nZs, nUs))
            lines[f'{line_id}/continuum'] = np.zeros((nTs, nZs, nUs))

        for iT, log10T in enumerate(log10Ts):
            for iZ, log10Z in enumerate(log10Zs):
                for iU, log10U in enumerate(log10Us):

                    infile = f'{synthesizer_data_dir}/cloudy/{grid_name}/{iT}_{iZ}_{iU}'

                    # --- get TOTAL continuum spectra
                    nebular_continuum = spectra['nebular'][iT,
                                                           iZ, iU] - spectra['linecont'][iT, iZ, iU]
                    continuum = spectra['transmitted'][iT, iZ, iU] + nebular_continuum

                    # --- get line quantities
                    id, blend, wavelength, intrinsic, emergent = read_lines(infile)

                    s = np.nonzero(np.in1d(id, np.array(lines_to_include)))[0]

                    for id_, wavelength_, emergent_ in zip(id[s], wavelength[s], emergent[s]):

                        line = lines[id_]

                        line.attrs['wavelength'] = wavelength_

                        line['luminosity'][iT, iZ, iU] = normlisation*10**(emergent_)  # erg s^-1
                        line['stellar_continuum'][iT, iZ, iU] = np.interp(
                            wavelength_, lam, spectra['transmitted'][iT, iZ, iU])  # erg s^-1 Hz^-1
                        line['nebular_continuum'][iT, iZ, iU] = np.interp(
                            wavelength_, lam, nebular_continuum)  # erg s^-1 Hz^-1
                        line['continuum'][iT, iZ, iU] = np.interp(
                            wavelength_, lam, continuum)  # erg s^-1 Hz^-1


if __name__ == "__main__":

    grid_name = 'blackbody'
    cloudy = '/its/home/sw376/flare/software/cloudy/c17.03/source/cloudy.exe'

    synthesizer_data_dir = "/research/astrodata/highz/synthesizer/"
    path_to_grids = f'{synthesizer_data_dir}/grids'
    path_to_cloudy_files = f'{synthesizer_data_dir}/cloudy'

    # failed = check_cloudy_runs(grid_name, synthesizer_data_dir, cloudy=cloudy)
    failed = False

    # hacky wayy of fixing consistently failing cloudy runs by copying over nearby grid point
    fix_cloudy_runs(grid_name, synthesizer_data_dir, replace=True)

    if not failed:

        add_spectra(grid_name, synthesizer_data_dir)

        lines_to_include = get_default_line_list()

        add_lines(grid_name, synthesizer_data_dir, lines_to_include)
