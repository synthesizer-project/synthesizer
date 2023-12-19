""" A module containing general utility functions.

Example usage:

    planck(frequency, temperature=10000 * K)
    rebin_1d(arr, 10, func=np.sum)
"""
import numpy as np
import h5py
import re
from unyt import c, h, um, erg, s, Hz, kb, unyt_array, unyt_quantity

from synthesizer import exceptions


def planck(nu, temperature):
    """
    Planck's law.

    Args:
        nu (unyt_array/array-like, float)
            The frequencies at which to calculate the distribution.
        temperature  (float/array-like, float)
            The dust temperature. Either a single value or the same size
            as nu.

    Returns:
        array-like, float
            The values of the distribution at nu.
    """

    return (2.0 * h * (nu**3) * (c**-2)) * (
        1.0 / (np.exp(h * nu / (kb * temperature)) - 1.0)
    )


def has_units(x):
    """
    Check whether the passed variable has units, i.e. is a unyt_quanity or
    unyt_array.

    Args:
        x (generic variable)
            The variables to check.

    Returns:
        bool
            True if the variable has units, False otherwise.
    """

    # Do the check
    if isinstance(x, (unyt_array, unyt_quantity)):
        return True

    return False


def rebin_1d(arr, resample_factor, func=np.sum):
    """
    A simple function for rebinning a 1D array using a specificed
    function (e.g. sum or mean).

    Args:
        arr (array-like)
            The input 1D array.
        resample_factor (int)
            The integer rebinning factor, i.e. how many bins to rebin by.
        func (func)
            The function to use (e.g. mean or sum).

    Returns:
        array-like
            The input array resampled by i.
    """

    # Ensure the array is 1D
    if arr.ndim != 1:
        raise exceptions.InconsistentArguments(
            f"Input array must be 1D (input was {arr.ndim}D)"
        )

    # Safely handle no integer resamples
    if not isinstance(resample_factor, int):
        print(
            f"resample factor ({resample_factor}) is not an"
            " integer, converting it to ",
            end="\r",
        )
        resample_factor = int(resample_factor)
        print(resample_factor)

    # How many elements in the input?
    n = len(arr)

    # If array is not the right size truncate it
    if n % resample_factor != 0:
        arr = arr[: int(resample_factor * np.floor(n / resample_factor))]

    # Set up the 2D array ready to have func applied
    rows = len(arr) // resample_factor
    brr = arr.reshape(rows, resample_factor)

    return func(brr, axis=1)


def process_dl07_to_hdf5(grid_name='MW3.1', grid_loc='./',
                         data_name='MW3.1',
                         data_loc='./synthesizer_grid_data/DL07'):
    """
    Function to process the text files from dl07 to N-dimensional hdf5 grids
    for use in dust/emission.py
    Args:
        grid_name (str)
            Name of the hdf5 grid to create from input data
        grid_loc (str)
            Location to store the grid (default is current directory)
        data_name (str)
            Dust type of the data file to process (currently only MW type)
        data_loc (str)
            Location of the input data file to process
    """

    spec_names = ['emission']

    # Define the models parameters
    suffix = np.array(['00', '10', '20', '30', '40', '50', '60'])
    qpahs = np.array([0.0047, 0.0112, 0.0177, 0.0250, 0.0319, 0.0390,
                      0.0458])
    umins = np.array(['0.10', '0.15', '0.20', '0.30', '0.40', '0.50', '0.70',
                      '0.80', '1.00', '1.20', '1.50', '2.00', '2.50', '3.00',
                      '4.00', '5.00', '7.00', '8.00', '12.0', '15.0', '20.0',
                      '25.0', '1e2', '3e2', '1e3', '3e3', '1e4', '3e4', '1e5',
                      '3e5'])
    umaxs = np.array(['1e2', '1e3', '1e4', '1e5', '1e6'])

    umins_umaxs = np.array([])
    for ii, umin in enumerate(umins):
        
        umins_umaxs = np.append(umins_umaxs, F'{umin}_{umin}')
        if float(umin)<1e2:
            this_umaxs = umaxs[(umaxs.astype(float)>umin.astype(float))]
            umins_umaxs = np.append(umins_umaxs, np.array([F'{umin}_{jj}' for jj in this_umaxs]))
    
    umins_umaxs = umins_umaxs.astype('S')
    axes_shape = list([len(qpahs), len(umins_umaxs)])
    
    # Read from a file to get wavelength array and its shape
    tmp = np.genfromtxt(F'{data_loc}/U0.10/U0.10_1e6_{data_name}_{suffix[0]}.txt',
                        skip_header=61)
    lam = tmp[:,0][::-1]
    n_lam = len(lam)
    nu = c/(lam*um)
    nu = nu.to(Hz).value

    with h5py.File(F'{grid_loc}/{grid_name}.hdf5', 'w') as hf:
        hf.attrs['axes'] = ['qpah', 'umin_umax']
        axes = hf.create_group('axes')
        axes['qpah'] = qpahs
        axes['qpah'].attrs['Description'] = "Fraction of dust mass in the \
                                             form of PAHs"
        axes['qpah'].attrs['Units'] = "dimensionless"
        axes['umin_umax'] = umins_umaxs
        axes['umin_umax'].attrs['Description'] = "Minimum and Maximum \
                                        radiation field heating the dust"
        axes['umin_umax'].attrs['Units'] = "dimensionless"
        # axes['umin'] = umins
        # axes['umin'].attrs['Description'] = "Radiation field heating majority \
        #                                      of the dust"
        # axes['umin'].attrs['Units'] = "dimensionless"
        # axes['umax'] = umaxs
        # axes['umax'].attrs['Description'] = "Maximum radiation field heating \
        #                                     the dust"
        # axes['umax'].attrs['Units'] = "dimensionless"

        spectra = hf.create_group('spectra')  # create a group holding the spectra in the grid file
        spectra.attrs['spec_names'] = spec_names  # save list of spectra as attribute
        for spec_name in spec_names:
            spectra[spec_name] = np.zeros((*axes_shape, n_lam))

        for ii, qpah in enumerate(qpahs):
            qpah_spectra = np.zeros((len(umins_umaxs), n_lam))
            for jj, umin_umax in enumerate(umins_umaxs):
                    umin_umax = umin_umax.decode('utf-8')
                    umin, umax = re.split('_', umin_umax)
                    
                    with open(F'{data_loc}/U{umin}/U{umin_umax}_{data_name}_{suffix[ii]}.txt') as f:
                        tmp = f.readlines()
                        skip_header=len(tmp)-1001 # Number of wavelength values
                    
                    if skip_header>0:
                        tmp = np.genfromtxt(F'{data_loc}/U{umin}/U{umin_umax}_{data_name}_{suffix[ii]}.txt',
                            skip_header=skip_header)
                        
                        tmp = tmp[:,1][::-1] * erg/s
                        tmp/=(nu*Hz)
                        tmp = tmp.to(erg/s/Hz).value
                        qpah_spectra[jj]=tmp
        
            for spec_name in spec_names: 
                spectra[spec_name][ii] = qpah_spectra

        spectra['wavelength'] = lam * 1e4 # convert to Angstrom
  