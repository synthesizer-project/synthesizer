import h5py
import numpy as np
import unyt
from unyt import c, h, nJy, erg, s, Hz, pc, kb


def write_data_h5py(filename, name, data, overwrite=False):
    check = check_h5py(filename, name)

    with h5py.File(filename, "a") as h5file:
        if check:
            if overwrite:
                print("Overwriting data in %s" % name)
                del h5file[name]
                h5file[name] = data
            else:
                raise ValueError("Dataset already exists, "
                                 + "and `overwrite` not set")
        else:
            h5file.create_dataset(name, data=data)


def check_h5py(filename, obj_str):
    with h5py.File(filename, "a") as h5file:
        if obj_str not in h5file:
            return False
        else:
            return True


def load_h5py(filename, obj_str):
    with h5py.File(filename, "a") as h5file:
        dat = np.array(h5file.get(obj_str))
    return dat


def write_attribute(filename, obj, key, value):
    """
    Write attribute to an HDF5 file
    Args
    obj (str) group  or dataset to attach attribute to
    key (str) attribute key string
    value (str) content of the attribute
    """
    with h5py.File(filename, "a") as h5file:
        dset = h5file[obj]
        dset.attrs[key] = value


def get_names_h5py(filename, group):
    """
    Return list of the names of objects inside a group
    """
    with h5py.File(filename, "r") as h5file:
        keys = list(h5file[group].keys())

    return keys


def load_arr(name, filename):
    """
    Load Dataset array from file
    """
    with h5py.File(filename, "r") as f:
        if name not in f:
            raise ValueError("'%s' Dataset doesn't exist..." % name)

        arr = np.array(f.get(name))

    return arr


def read_params(param_file):
    """
    Args:
    param_file (str) location of parameter file
    Returns:
    parameters (object)
    """
    return __import__(param_file)


def flux_to_luminosity(flux, cosmo, redshift):
    """
    Converts flux in nJy to luminosity in erg / s / Hz.
    Parameters
    ----------
    flux : array-like (float)/float
        The flux to be converted to luminosity, can either be a singular
        value or array.
    cosmo : obj (astropy.cosmology)
        The cosmology object used to calculate luminosity distance.
    redshift : float
        The redshift of the rest frame.
    """

    # Calculate the luminosity distance
    lum_dist = cosmo.luminosity_distance(redshift).to("cm").value

    # Calculate the luminosity in interim units
    lum = flux * 4 * np.pi * lum_dist**2

    # And convert to erg / s / Hz
    lum *= 1 / (1e9 * 1e23 * (1 + redshift))

    return lum


def fnu_to_m(fnu):
    """
    Converts flux in nJy to apparent magnitude.

    Parameters
    ----------
    flux : array-like (float)/float
        The flux to be converted, can either be a singular value or array.
    """

    # Check whether we have units, if so convert to nJy
    if isinstance(fnu, unyt.array.unyt_quantity):
        fnu_ = fnu.to("nJy").value
    else:
        fnu_ = fnu

    return -2.5 * np.log10(fnu_ / 1e9) + 8.9  # -- assumes flux in nJy


def m_to_fnu(m):
    """
    Converts apparent magnitude to flux in nJy.

    Parameters
    ----------
    m : array-like (float)/float
        The apparent magnitude to be converted, can either be a singular value
        or array.
    """

    return 1e9 * 10 ** (-0.4 * (m - 8.9)) * nJy


def flam_to_fnu(lam, flam):
    """convert f_lam to f_nu

    arguments:
    lam -- wavelength / \\AA
    flam -- spectral luminosity density/erg/s/\\AA
    """

    lam_m = lam * 1e-10

    return flam * lam / (c.value / lam_m)


def fnu_to_flam(lam, fnu):
    """convert f_nu to f_lam

    arguments:
    lam -- wavelength/\\AA
    flam -- spectral luminosity density/erg/s/\\AA
    """

    lam_m = lam * 1e-10

    return fnu * (c.value / lam_m) / lam


class constants:
    tenpc = 10 * pc  # ten parsecs
    # the surface area (in cm) at 10 pc. I HATE the magnitude system
    geo = 4 * np.pi * (tenpc.to("cm").value) ** 2


def M_to_Lnu(M):
    """Convert absolute magnitude (M) to L_nu"""
    return 10 ** (-0.4 * (M + 48.6)) * constants.geo * erg / s / Hz


def Lnu_to_M(Lnu_):
    """Convert L_nu to absolute magnitude (M). If no unit
    provided assumes erg/s/Hz."""
    if type(Lnu_) == unyt.array.unyt_quantity:
        Lnu = Lnu_.to("erg/s/Hz").value
    else:
        Lnu = Lnu_

    return -2.5 * np.log10(Lnu / constants.geo) - 48.6


def planck(nu, T):
    """
    Planck's law.
        
    Args:
        nu (unyt_array/array-like, float)
            The frequencies at which to calculate the distribution.
        T  (float/array-like, float)     
            The dust temperature. Either a single value or the same size
            as nu.

    Returns:
        array-like, float
            The values of the distribution at nu.
    """

    return (2.0 * h * (nu**3) * (c**-2)) * (1.0 / (np.exp(h * nu / (kb * T)) - 1.0))


def rebin_1d(a, i, func=np.sum):

    """
    A simple function for rebinning a 1D array using a specificed
    function (e.g. sum or mean).

    TODO: add exeption to make sure a is an 1D array and that i is an integer.
    
    Args: 
        a (ndarray)
            the input 1D array
        i (int)
            integer rebinning factor, i.e. how many bins to rebin by
        func (func)
            the function to use (e.g. mean or sum)


    """
    
    n = len(a)

    # if array is not the right size truncate it
    if n % i != 0:
        a = a[:int(i * np.floor(n/i))]

    x = len(a) // i
    b = a.reshape(x, i)

    return func(b, axis=1)
