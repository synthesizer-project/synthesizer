
""" This reads in a cloudy grid of models and creates a new SPS grid including the various outputs """


from scipy import integrate
import os
import shutil
from synthesizer.utils import read_params
from synthesizer.cloudy import read_wavelength, read_continuum, default_lines, read_lines
from synthesizer.sed import calculate_Q
import argparse
import numpy as np
import h5py

#
# from unyt import c, h

h = 6.626E-34
c = 3.E8


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=('Create synthesizer HDF5 grid '
                                                  'for a given grid.'))

    parser.add_argument("-dir", "--directory", type=str, required=True)

    parser.add_argument("-grid", "--grid", type=str,
                        nargs='+', required=True,
                        help=('The SPS/CLOUDY grid(s) to use. '
                              'Multiple grids can be listed as: \n '
                              '  --sps_grid grid_1 grid_2'))

    args = parser.parse_args()

    synthesizer_data_dir = args.directory
    path_to_grids = f'{synthesizer_data_dir}/grids'
    path_to_cloudy_files = f'{synthesizer_data_dir}/cloudy'

    for grid in args.grid:

        # parse the grid to get the sps model
        sps_grid = grid.split('_cloudy')[0]

        print(grid, sps_grid)

        # spec_names = ['incident','transmitted','nebular','nebular_continuum','total','linecont']
        #  the cloudy spectra to save (others can be generated later)
        spec_names = ['incident', 'transmitted', 'nebular', 'linecont']

        # open the new grid
        with h5py.File(f'{path_to_grids}/{grid}.hdf5', 'w') as hf:

            # open the original SPS model grid
            with h5py.File(f'{path_to_grids}/{sps_grid}.hdf5', 'r') as hf_sps:

                # copy various quantities (all excluding the spectra) from the original sps grid
                for ds in ['metallicities', 'log10ages', 'log10Q']:
                    hf_sps.copy(hf_sps[ds], hf['/'], ds)

                for k, v in hf_sps.attrs.items():
                    print(k, v)
                    hf.attrs[k] = v

            # --- short hand for later
            metallicities = hf['metallicities']
            log10ages = hf['log10ages']

            nZ = len(metallicities)  # number of metallicity grid points
            na = len(log10ages)  # number of age grid points

            #
            # # --- read first spectra from the first grid point to get length and wavelength grid
            # lam = read_wavelength(f'{path_to_cloudy_files}/{sps_model}_{cloudy_model}/0_0')
            #
            # spectra = hf.create_group('spectra')  # create a group holding the spectra in the grid file
            # spectra.attrs['spec_names'] = spec_names  # save list of spectra as attribute
            #
            # spectra['wavelength'] = lam  # save the wavelength
            #
            # nlam = len(lam)  # number of wavelength points
            #
            # # --- make spectral grids and set them to zero
            # for spec_name in spec_names:
            #     spectra[spec_name] = np.zeros((na, nZ, nlam))
            #
            #     # --- now loop over meallicity and ages
            #
            #     dlog10Q = np.zeros((na, nZ))
            #
            #     hf['cloudy_ok'] = np.zeros((na, nZ))
            #
            #     for iZ, Z in enumerate(metallicities):
            #         for ia, log10age in enumerate(log10ages):
            #
            #             infile = f'{path_to_cloudy_files}/{sps_model}_{cloudy_model}/{ia}_{iZ}'
            #
            #             try:
            #
            #                 # if os.path.isfile(infile+'.cont'):  # attempt to open run.
            #                 #     exists = 'exists'
            #
            #                 spec_dict = read_continuum(infile, return_dict=True)
            #                 for spec_name in spec_names:
            #                     spectra[spec_name][ia, iZ] = spec_dict[spec_name]
            #
            #                 # --- we need to rescale the cloudy spectra to the original SPS spectra. This is done here using the ionising photon luminosity, though could in principle by done at a fixed wavelength.
            #
            #                 # calculate log10Q
            #                 log10Q = np.log10(calculate_Q(lam, spectra['incident'][ia, iZ, :]))
            #                 dlog10Q[ia, iZ] = hf['log10Q'][ia, iZ] - log10Q
            #
            #                 hf['cloudy_ok'][ia, iZ] = 1
            #
            #                 for spec_name in spec_names:
            #                     spectra[spec_name][ia, iZ] *= 10**dlog10Q[ia, iZ]
            #
            #             except:
            #
            #                 # if the code fails use the previous metallicity point
            #
            #                 try:
            #                     dlog10Q[ia, iZ] = dlog10Q[ia, iZ-1]
            #                     for spec_name in spec_names:
            #                         spectra[spec_name][ia, iZ] = spectra[spec_name][ia, iZ-1]
            #                 except:
            #
            #                     # if that fails use the previous age point
            #                     dlog10Q[ia, iZ] = dlog10Q[ia-1, iZ]
            #                     for spec_name in spec_names:
            #                         spectra[spec_name][ia, iZ] = spectra[spec_name][ia-1, iZ]
            #
            #                 print('failed for', ia, iZ)
            #
            #             # else:
            #             #     exists = 'does not exist'
            #
            #     # -- get list of lines
            #
            #     lines = hf.create_group('lines')
            #     lines.attrs['lines'] = default_lines  # save list of spectra as attribute
            #
            #     for line_id in default_lines:
            #         lines[f'{line_id}/luminosity'] = np.zeros((na, nZ))
            #         lines[f'{line_id}/stellar_continuum'] = np.zeros((na, nZ))
            #         lines[f'{line_id}/nebular_continuum'] = np.zeros((na, nZ))
            #         lines[f'{line_id}/continuum'] = np.zeros((na, nZ))
            #
            #     for iZ, Z in enumerate(metallicities):
            #         for ia, log10age in enumerate(log10ages):
            #
            #             infile = f'{path_to_cloudy_files}/{sps_model}_{cloudy_model}/{ia}_{iZ}'
            #
            #             try:
            #
            #                 # --- get line quantities
            #                 line_ids, line_wavelengths, _, line_luminosities = read_lines(infile)
            #
            #                 # --- get TOTAL continuum spectra
            #                 nebular_continuum = spectra['nebular'][ia, iZ] - spectra['linecont'][ia, iZ]
            #                 continuum = spectra['transmitted'][ia, iZ] + nebular_continuum
            #
            #                 for line_id, line_wv, line_lum in zip(line_ids, line_wavelengths, line_luminosities):
            #                     lines[line_id].attrs['wavelength'] = line_wv
            #                     lines[f'{line_id}/luminosity'][ia,
            #                                                    iZ] = 10**(line_lum + dlog10Q[ia, iZ])  # erg s^-1
            #                     lines[f'{line_id}/stellar_continuum'][ia,
            #                                                           iZ] = np.interp(line_wv, lam, spectra['transmitted'][ia, iZ])  # erg s^-1 Hz^-1
            #                     lines[f'{line_id}/nebular_continuum'][ia,
            #                                                           iZ] = np.interp(line_wv, lam, nebular_continuum)  # erg s^-1 Hz^-1
            #                     lines[f'{line_id}/continuum'][ia,
            #                                                   iZ] = np.interp(line_wv, lam, continuum)  # erg s^-1 Hz^-1
            #
            #             except:
            #
            #                 hf['cloudy_ok'][ia, iZ] = 0
            #
            #     # hf.visit(print)
            #     hf.flush()
