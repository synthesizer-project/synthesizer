""" A module containing functions for conversions.

This module contains helperful conversions for converting between different
observables. This main covers conversions between flux, luminosity and
magnitude systems.

Example usage:



"""
import numpy as np
from unyt import c, nJy, erg, s, Hz


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
