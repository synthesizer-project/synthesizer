""" A module containing general utility functions.

Example usage:

    planck(frequency, temperature=10000 * K)
    rebin_1d(arr, 10, func=np.sum)
"""
import numpy as np
import h5py
from unyt import c, h, um, erg, s, Hz, kb, mp, Msun, unyt_array, unyt_quantity

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
    These are the updated 2014 files.
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

    spec_names = ['pdr', 'diffuse', 'fsil_pdr', 'fsil_diffuse']

    # Define the models parameters
    suffix = np.array(['000', '010', '020', '030', '040', '050',
                       '060', '070', '080', '090', '100'])
    qpahs = 0.01 * np.array([0.47, 1.12, 1.77, 2.50, 3.19, 3.90, 4.58,
                             5.26, 5.95, 6.63, 7.32])
    # np.array([0.0047, 0.0112, 0.0177, 0.0250, 0.0319, 0.0390,
    #                   0.0458])
    alphas = np.arange(1.0, 3.1, 0.1)
    umins = np.array(['0.100', '0.120', '0.150', '0.170', '0.200',
                      '0.250', '0.300', '0.350', '0.400', '0.500',
                      '0.600', '0.700', '0.800', '1.000', '1.200',
                      '1.500', '1.700', '2.000', '2.500', '3.000',
                      '3.500', '4.000', '5.000', '6.000', '7.000',
                      '8.000', '10.00', '12.00', '15.00', '17.00',
                      '20.00', '25.00', '30.00', '35.00', '40.00',
                      '50.00'])
    umaxs = np.array(['1e7'])

    axes_shape = list([len(qpahs), len(umins), len(alphas)])

    # Read from a file to get wavelength array and its shape
    test_file = F'{data_loc}/U0.100_0.100_{data_name}_{suffix[0]}/spec_1.0.dat'
    tmp = np.genfromtxt(test_file,
                        skip_header=71)
    lam = tmp[:, 0][::-1]
    n_lam = len(lam)
    nu = c / (lam * um)
    nu = nu.to(Hz).value
    msun_by_mp = (Msun / mp).value

    with h5py.File(F'{grid_loc}/{grid_name}.hdf5', 'w') as hf:
        hf.attrs['axes'] = ['qpah', 'umin', 'alpha']
        axes = hf.create_group('axes')
        axes['qpah'] = qpahs
        axes['qpah'].attrs['Description'] = "Fraction of dust mass in the \
                                             form of PAHs"
        axes['qpah'].attrs['Units'] = "dimensionless"
        axes['umin'] = umins.astype(float)
        axes['umin'].attrs['Description'] = "Radiation field heating majority \
                                             of the dust or diffuse dust"
        axes['umin'].attrs['Units'] = "dimensionless"
        axes['alpha'] = alphas
        axes['alpha'].attrs['Description'] = "Powerlaw slope dU/dM propto \
                                                U^alpha"
        axes['alpha'].attrs['Units'] = "dimensionless"

        # create a group holding the spectra in the grid file
        spectra = hf.create_group('spectra')
        # save list of spectra as attribute
        spectra.attrs['spec_names'] = spec_names
        spectra['pdr'] = np.zeros((*axes_shape, n_lam))
        spectra['diffuse'] = np.zeros((len(qpahs), len(umins), n_lam))

        spectra['pdr_fsil'] = np.zeros((*axes_shape, n_lam))
        spectra['diffuse_fsil'] = np.zeros((len(qpahs), len(umins), n_lam))

        for ii, qpah in enumerate(qpahs):
            diffuse_spectra = np.zeros((len(umins), n_lam))
            pdr_spectra = np.zeros((len(umins), len(alphas), n_lam))
            diffuse_fsil = np.zeros((len(umins), n_lam))
            pdr_fsil = np.zeros((len(umins), len(alphas), n_lam))
            for jj, umin in enumerate(umins):

                # diffuse dust spectrum
                fname = F'{data_loc}/U{umin}_{umin}_{data_name}_{suffix[ii]}'
                with open(F'{fname}/spec_1.0.dat') as f:
                    tmp = f.readlines()
                    # Number of wavelength values
                    skip_header = len(tmp) - 1001

                tmp = np.genfromtxt(F'{fname}', skip_header=skip_header)

                spec = tmp[:, 1][::-1] * erg / s
                spec /= (nu * Hz)
                spec = spec.to(erg / s / Hz).value
                diffuse_spectra[jj] = spec * msun_by_mp  # erg/s/Hz/Mdust

                fsil = tmp[:, 4][::-1] / (tmp[:, 3][::-1] + tmp[:, 4][::-1])
                diffuse_fsil[jj] = fsil

                # pdr dust spectrum
                for kk, alpha in enumerate(alphas):
                    fname = F'{data_loc}/U{umin}_{umaxs[0]}_{data_name}_' \
                            + F'{suffix[ii]}/spec_{np.round(alpha,2)}.dat'
                    with open(F'{fname}') as f:
                        tmp = f.readlines()
                        # Number of wavelength values
                        skip_header = len(tmp) - 1001

                    tmp = np.genfromtxt(F'{fname}', skip_header=skip_header)

                    spec = tmp[:, 1][::-1] * erg / s
                    spec /= (nu * Hz)
                    spec = spec.to(erg / s / Hz).value
                    pdr_spectra[jj, kk] = spec * msun_by_mp  # erg/s/Hz/Mdust

                    fsil = tmp[:, 4][::-1] / (tmp[:, 3][::-1]
                                              + tmp[:, 4][::-1])
                    pdr_fsil[jj, kk] = fsil

            spectra['pdr'][ii] = pdr_spectra  # type: ignore
            spectra['diffuse'][ii] = diffuse_spectra  # type: ignore
            spectra['pdr_fsil'][ii] = pdr_fsil  # type: ignore
            spectra['diffuse_fsil'][ii] = diffuse_fsil  # type: ignore

        spectra['wavelength'] = lam * 1e4  # convert to Angstrom


def value_to_array(value):
    """
    A helper functions for converting a single value to an array holding
    a single value.

    Args:
        value (float/unyt_quantity)
            The value to wrapped into an array.

    Returns:
        array-like/unyt_array
            An array containing the single value

    Raises:
        InconsistentArguments
            If the argument is not a float or unyt_quantity.
    """

    # Just return it if we have been handed an array already or None
    # NOTE: unyt_arrays and quantities are by definition arrays and thus
    # return True for the isinstance below.
    if (isinstance(value, np.ndarray) and value.size > 1) or value is None:
        return value

    if isinstance(value, float):
        arr = np.array(
            [
                value,
            ]
        )

    elif isinstance(value, unyt_quantity):
        arr = (
            np.array(
                [
                    value.value,
                ]
            )
            * value.units
        )
    else:
        raise exceptions.InconsistentArguments(
            "Value to convert to an array wasn't a float or a unyt_quantity:"
            f"type(value) = {type(value)}"
        )

    return arr
