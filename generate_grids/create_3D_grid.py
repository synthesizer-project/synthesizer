"""
This reads in a series of grids to create a new 3D grid
"""


from scipy import integrate
import os
import shutil
from synthesizer.utils import read_params
from synthesizer.utils import explore_hdf5_grid
from synthesizer.cloudy import read_wavelength, read_continuum, read_lines
from synthesizer.sed import calculate_Q
from unyt import eV
import argparse
import numpy as np
import h5py
import yaml


synthesizer_data_dir = '/Users/stephenwilkins/Dropbox/Research/data/synthesizer'
path_to_grids = f'{synthesizer_data_dir}/grids'


z_name = 'alpha'  # the parameter that is varied
z = [-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # the parameter array

base_grid = 'bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-alpha'
grids = [f'bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-alpha{str(z_).replace("-","m")}' for z_ in z]

print(grids)

# open the new grid
with h5py.File(f'{path_to_grids}/{base_grid}.hdf5', 'w') as hf:

    # open the first grid and copy over attribures and set up arrays
    with h5py.File(f'{path_to_grids}/{grids[0]}.hdf5', 'r') as hf_:

        # copy top-level attributes
        for k, v in hf_.attrs.items():
            # print(k, v)
            hf.attrs[k] = v

        # copy various quantities (all excluding the spectra) from the original sps grid
        for ds in ['metallicities', 'log10ages']:
            hf_.copy(hf_[ds], hf['/'], ds)

        na = len(hf['log10ages'][:])
        nZ = len(hf['metallicities'][:])
        nz = len(grids)

        for k in hf_['log10Q'].keys():
            hf[f'log10Q/{k}'] = np.zeros((na, nZ, nz))

        lines = hf.create_group('lines')
        for line_id in hf_['lines'].keys():
            line = lines.create_group(line_id)
            line.attrs['wavelength'] = hf_['lines'][line_id].attrs['wavelength']
            line['luminosity'] = np.zeros((na, nZ, nz))
            line['continuum'] = np.zeros((na, nZ, nz))
            line['nebular_continuum'] = np.zeros((na, nZ, nz))
            line['stellar_continuum'] = np.zeros((na, nZ, nz))

        spectra = hf.create_group('spectra')

        spectra.attrs['spec_names'] = hf_['spectra'].attrs['spec_names']
        spectra['wavelength'] = hf_['spectra']['wavelength'][:]
        nlam = len(spectra['wavelength'][:])

        for spec_id in spectra.attrs['spec_names']:
            spectra[spec_id] = np.zeros((na, nZ, nz, nlam))

    hf.attrs[z_name] = z

    # add attribute with the grid axes for future when using >2D grid or AGN grids
    hf.attrs['grid_axes'] = ['log10ages', 'metallicities', z_name]

    for iz, grid in enumerate(grids):

        # open the first grid and copy over attribures and set up arrays
        with h5py.File(f'{path_to_grids}/{grids[iz]}.hdf5', 'r') as hf_:

            for k in hf_['log10Q'].keys():
                hf[f'log10Q/{k}'][:, :, iz] = hf_[f'log10Q/{k}'][:]

            for line_id in hf['lines'].keys():
                line = hf['lines'][line_id]
                line_ = hf_['lines'][line_id]
                line['luminosity'][:, :, iz] = line_['luminosity'][:]
                line['continuum'][:, :, iz] = line_['continuum'][:]
                line['nebular_continuum'][:, :, iz] = line_['nebular_continuum'][:]
                line['stellar_continuum'][:, :, iz] = line_['stellar_continuum'][:]

            for spec_id in spectra.attrs['spec_names']:
                spectra[spec_id][:, :, iz, :] = hf_['spectra'][spec_id][:]

    hf.visititems(explore_hdf5_grid)
