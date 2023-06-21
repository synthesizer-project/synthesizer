
""" Create a rebinned grid for testing. This test grid should not be used for science """

import argparse
import numpy as np
import h5py
from spectres import spectres

def get_grid_properties_hf(hf, verbose=False):

    """
    A wrapper over get_grid_properties to get the grid properties for a HDF5 grid.
    """
    
    axes = hf.attrs['axes'] # list of axes in the correct order
    axes_values = {axis: hf[f'axes/{axis}'][:] for axis in axes} # dictionary of axis grid points
    
    # Get the properties of the grid including the dimensions etc.
    return axes, *get_grid_properties(axes, axes_values, verbose=verbose)


def get_grid_properties(axes, axes_values, verbose = False):

    """ 
    Get the properties of the grid including the dimensions etc.
    """

    # the grid axes   
    if verbose: print(f'axes: {axes}')

    # number of axes
    n_axes = len(axes)
    if verbose: print(f'number of axes: {n_axes}')

    # the shape of the grid (useful for creating outputs)
    shape = list([len(axes_values[axis]) for axis in axes])
    if verbose: print(f'shape: {shape}')

    # determine number of models
    n_models = np.prod(shape)
    if verbose: print(f'number of models to run: {n_models}')

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

    return n_axes, shape, n_models, mesh, model_list, index_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Re-bin a grid to a different spectral resolution')
   
    parser.add_argument("-input_dir", type=str, required=True) # path to synthesizer_data_dir
    parser.add_argument("-input_grid", type=str, required=True) # grid_name, used to define parameter file
    parser.add_argument("-output_dir", type=str, required=True) # the parameters of the 
    parser.add_argument("-output_grid", type=str, required=True) # the parameters of the 
    args = parser.parse_args()


    # open the original grid
    original_grid = h5py.File(f'{args.input_dir}/{args.input_grid}.hdf5', 'r')

    # open the new grid file
    rebinned_grid = h5py.File(f'{args.output_dir}/{args.output_grid}.hdf5', 'w')

    # copy attributes
    for k, v in original_grid.attrs.items():
        rebinned_grid.attrs[k] = v

    # copy various quantities (all excluding the spectra) from the original grid
    for ds in ['axes', 'log10Q', 'lines']:
        original_grid.copy(original_grid[ds], rebinned_grid['/'], ds)


    # define the new wavelength grid
    lmin, lmax, deltal = 100., 20000., 20.  # min wavelength, max wavelength, resolution
    new_wavs = np.arange(lmin, lmax, deltal)


    # alias
    original_spectra = original_grid['spectra']
    spectra_types = original_spectra.attrs['spec_names']

    # create a group holding the spectra in the grid file
    rebinned_spectra = rebinned_grid.create_group('spectra')
    rebinned_spectra['wavelength'] = new_wavs
    rebinned_spectra.attrs['spec_names'] = original_spectra.attrs['spec_names']


    # get parameters of grid
    axes, n_axes, shape, n_models, mesh, model_list, index_list = get_grid_properties_hf(original_grid)

    for spectra_type in spectra_types:

        rebinned_spectra[spectra_type] =  np.zeros((*shape, len(new_wavs)))

        # loop over all indices

        for i, indices in enumerate(index_list):

            indices = tuple(indices)

            rebinned_spectra[spectra_type][indices] = spectres(
                new_wavs, original_spectra['wavelength'][:], original_spectra[spectra_type][indices][:])
