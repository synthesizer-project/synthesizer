"""
This reads in a series of grids to create a new 3D grid
"""


from scipy import integrate
import os
import shutil
from synthesizer.utils import read_params
from synthesizer.cloudy import read_wavelength, read_continuum, read_lines
from synthesizer.sed import calculate_Q
from unyt import eV
import argparse
import numpy as np
import h5py
import yaml


path_to_grids = f'{synthesizer_data_dir}/grids'


z_name = 'alpha'  # the parameter that is varied
z = [-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # the parameter array

base_grid = 'bpass-2.2.1-bin_chabrier03-0.1,300.0_cloud-alpha'
grids = [f'bpass-2.2.1-bin_chabrier03-0.1,300.0_cloud-alpha{str(z).replace('-','m')}' for z_ in z]

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

        new_shape = (len(hf['log10ages'][:]), len(hf['metallicities'][:]), len(grids))

        for k in hf_['log10Q'].keys():
            hf['log10Q'][k] = np.zeros(new_shape)





    hf.attrs[z_name] = z

    # add attribute with the grid axes for future when using >2D grid or AGN grids
    hf.attrs['grid_axes'] = ['log10ages', 'metallicities', z_name]
