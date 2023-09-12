
"""
This reads in a cloudy grid of models and creates a synthesizer grid.
"""

import os
import shutil
import argparse
import numpy as np
import h5py

# synthesizer imports
from synthesizer.cloudy import read_wavelength, read_continuum, read_lines, \
    read_linelist

# local imports
from utils import get_grid_properties


def get_grid_properties_hf(hf, verbose=True):

    """
    A wrapper over get_grid_properties to get the grid properties for a HDF5
    grid.
    """
    axes = hf.attrs['axes']  # list of axes

    # dictionary of axis grid points
    axes_values = {axis: hf['axes'][axis][:] for axis in axes}
    
    # Get the properties of the grid including the dimensions etc.
    return get_grid_properties(axes, axes_values, verbose=verbose)


def check_cloudy_runs(grid_name,
                      synthesizer_data_dir,
                      replace=False,
                      files_to_check=['cont', 'elin']):
    """
    Check that all the cloudy runs have run properly.

    Args:
        grid_name (str)
            Name of the grid.
        synthesizer_data_dir (str)
            Directory where synthesizer data is kept.
        replace : boolean
            If a run has failed simply replace the model with the previous one
    """

    # open the new grid
    with h5py.File(f'{synthesizer_data_dir}/grids/{grid_name}.hdf5',
                   'r') as hf:

        # Get the properties of the grid including the dimensions etc.
        axes, n_axes, shape, n_models, mesh, model_list, index_list = \
            get_grid_properties_hf(hf)

        # list of failed models
        failed_list = []

        for i, grid_params_ in enumerate(model_list):

            infile = f"{synthesizer_data_dir}/cloudy/{grid_name}/{i}"

            failed = False

            # check if files exist
            for ext in files_to_check:
                if not os.path.isfile(infile+'.'+ext):  # attempt to open run.
                    print(f'failed existence of {ext} for {i}')
                    failed = True

            # if they exist also check they have size >0
            if not failed:
                for ext in files_to_check:
                    if os.path.getsize(infile+'.'+ext) < 100:
                        print(f'failed size of {ext} for {i}')
                        failed = True

            if failed:

                print(i, model_list[i])
                failed_list.append(i)

                """
                If replace is specified, instead replace the grid point with
                the previous one.
                NOTE: this should be a last resort if the cloudy runs of a
                small number of grid points are consistently failing.
                """
                if replace:
                    for ext in files_to_check:
                        shutil.copyfile(f"{synthesizer_data_dir}/cloudy/\
                                        {grid_name}/{i-1}.{ext}",
                                        infile+'.lines')
                    
        # if the files have been replace set the failed list to empty so the 
        # rest of the code can run     
        if replace:
            failed_list = []

        return failed_list


def add_spectra(grid_name, synthesizer_data_dir):
    """
    Open cloudy spectra and add them to the grid

    Args:
        grid_name (str)
            Name of the grid.
        synthesizer_data_dir (str)
            Directory where synthesizer data is kept.
    """

    #  the cloudy spectra to save (others can be generated later)
    spec_names = ['incident', 'transmitted', 'nebular_continuum', 'nebular',
                  'linecont']

    # open the new grid
    with h5py.File(f'{synthesizer_data_dir}/grids/{grid_name}.hdf5',
                   'a') as hf:

        # Get the properties of the grid including the dimensions etc.
        axes, n_axes, shape, n_models, mesh, model_list, index_list = \
            get_grid_properties_hf(hf)

        # read first spectra from the first grid point to get length and
        # wavelength grid
        lam = read_wavelength(f"{synthesizer_data_dir}/cloudy/{grid_name}/0")

        # delete spectra dataset if it already exists
        if 'spectra' in hf:
            del hf['spectra']

        # create a group holding the spectra in the grid file
        spectra = hf.create_group('spectra')

        # save list of spectra as attribute
        spectra.attrs['spec_names'] = spec_names

        # save the wavelength
        spectra['wavelength'] = lam

        # calcualte the frequency (nu)
        nu = 3E8 / (lam*1E-10)

        # number of wavelength points
        nlam = len(lam)

        # make spectral grids and set them to zero
        for spec_name in spec_names:
            spectra[spec_name] = np.zeros((*shape, nlam))

        # array for holding the normalisation which is calculated below and 
        # used by lines
        spectra['normalisation'] = np.zeros(shape)

        # loop over models
        for i, indices in enumerate(index_list):

            indices = tuple(indices)

            # define the infile
            infile = f"{synthesizer_data_dir}/cloudy/{grid_name}/{i}"

            # read the continuum file containing the spectra
            spec_dict = read_continuum(infile, return_dict=True)

            # for an arbitrary grid, we should normalise by the bolometric
            # luminosity of the incident spectra
            norm = np.trapz(spec_dict['incident'][::-1], x=nu[::-1])

            # save normalisation for later use (rescaling lines)
            spectra['normalisation'][indices] = norm

            # save the normalised spectrum to the correct grid point
            for spec_name in spec_names:
                spectra[spec_name][indices] = spec_dict[spec_name] / norm


def add_lines(grid_name,
              synthesizer_data_dir,
              line_type='linelist',
              lines_to_include=False,
              include_spectra=True):
    """
    Open cloudy lines and add them to the HDF5 grid

    Args:
        grid_name (str)
            Name of the grid.
        synthesizer_data_dir (str)
            Directory where synthesizer data is kept.
        line_type (str)
            The type of line file to use (linelist, lines)
        lines_to_include (bool, list)
            Which lines to include, if at all. If linelist, linelist overwrites
            the lines to include.
        include_spectra (bool)
            Flag whether to include spectra.
    """

    # open the new grid
    with h5py.File(f'{synthesizer_data_dir}/grids/{grid_name}.hdf5',
                   'a') as hf:

        # Get the properties of the grid including the dimensions etc.
        axes, n_axes, shape, n_models, mesh, model_list, index_list = \
            get_grid_properties_hf(hf)

        # delete lines group if it already exists
        if 'lines' in hf:
            del hf['lines']

        # define spectra
        if include_spectra:
            spectra = hf['spectra']
            normalisation = spectra['normalisation'][:]
            lam = spectra['wavelength'][:]

        # create group for holding lines
        lines = hf.create_group('lines')

        # get list of lines to include
        if line_type == 'linelist':
            infile = f"{synthesizer_data_dir}/cloudy/{grid_name}/1"
            lines_to_include, _, _ = read_linelist(infile)

        # set up output arrays
        for line_id in lines_to_include:
            lines[f'{line_id}/luminosity'] = np.zeros(shape)
            lines[f'{line_id}/stellar_continuum'] = np.zeros(shape)
            lines[f'{line_id}/nebular_continuum'] = np.zeros(shape)
            lines[f'{line_id}/continuum'] = np.zeros(shape)

        for i, indices in enumerate(index_list):

            # convert indices array to tuple
            indices = tuple(indices)

            # define the infile
            infile = f"{synthesizer_data_dir}/cloudy/{grid_name}/{i}"

            # get TOTAL continuum spectra
            if include_spectra:
                nebular_continuum = spectra['nebular'][indices] \
                    - spectra['linecont'][indices]
                continuum = spectra['transmitted'][indices] + nebular_continuum

            # get line quantities  <--- THIS NEEDS TO CHANGE
            if line_type == 'lines':

                id, blend, wavelength, intrinsic, luminosity = \
                      read_lines(infile)

                # identify lines we want to keep
                s = np.nonzero(np.in1d(id, np.array(lines_to_include)))[0]

                id = id[s]
                wavelength = wavelength[s]
                luminosity = luminosity[s]

            elif line_type == 'linelist':
                
                id, wavelength, luminosity = read_linelist(infile)

            for id_, wavelength_, luminosity_ in zip(id, wavelength,
                                                     luminosity):

                line = lines[id_]

                # save line wavelength
                line.attrs['wavelength'] = wavelength_

                # if spectra have been calculated pull the normalisation
                if include_spectra:
                    norm = normalisation[indices]
                else:
                    norm = 1.

                # calculate line luminosity and save it. Uses normalisation
                # from spectra. erg s^-1
                line['luminosity'][indices] = 10**(luminosity_)/norm

                if include_spectra:

                    # calculate stellar continuum at the line wavelength and
                    # save it. erg s^-1 Hz^-1
                    line['stellar_continuum'][indices] = np.interp(
                        wavelength_, lam, spectra['transmitted'][indices])

                    # calculate nebular continuum at the line wavelength and
                    # save it. erg s^-1 Hz^-1
                    line['nebular_continuum'][indices] = np.interp(
                        wavelength_, lam, nebular_continuum)

                    # calculate total continuum at the line wavelength and
                    # save it. erg s^-1 Hz^-1
                    line['continuum'][indices] = np.interp(
                        wavelength_, lam, continuum)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=('Create synthesizer HDF5 \
                                                  grid for a given grid.'))

    # path to synthesizer_data_dir
    parser.add_argument("-synthesizer_data_dir", type=str, required=True)

    # grid name
    parser.add_argument("-grid_name", "--grid_name", type=str, required=True)

    # include spectra flag (useful for big grids to explore line emission only)
    parser.add_argument("-include_spectra", "--include_spectra", type=bool,
                        default=True, required=False)

    # replace missing grid points
    parser.add_argument("-replace", "--replace", type=bool, default=False,
                        required=False)

    # line calculation method
    parser.add_argument("-line_calc_method", "--line_calc_method", type=str,
                        default='lines', required=False)

    args = parser.parse_args()

    include_spectra = args.include_spectra

    synthesizer_data_dir = args.synthesizer_data_dir
    grid_name = args.grid_name

    # check cloudy runs
    failed_list = check_cloudy_runs(grid_name,
                                    synthesizer_data_dir,
                                    replace=args.replace
                                    )

    print(failed_list)

    # if failed prompt to re-run
    if len(failed_list) > 0:

        print(f'ERROR: {len(failed_list)} cloudy runs have failed. \
              You should re-run these with command:')
        print(f'  qsub -t 1:{len(failed_list)}  run_grid.job')

        # replace input_names with list of failed runs
        with open(f"{synthesizer_data_dir}/cloudy/{grid_name}/input_names.txt",
                  "w") as myfile:
            myfile.write('\n'.join(map(str, failed_list)))

    # if not failed, go ahead and add spectra and lines
    else:
        
        print('- passed checks')

        # add spectra

        if include_spectra:
            add_spectra(grid_name, synthesizer_data_dir)
            print('- spectra added')

        if args.line_calc_method == 'lines':

            # add lines
            add_lines(grid_name,
                      synthesizer_data_dir,
                      line_type='linelist',
                      include_spectra=include_spectra)
            
