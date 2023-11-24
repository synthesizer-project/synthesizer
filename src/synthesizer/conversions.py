""" A module containing functions for conversions.

This module contains helperful conversions for converting between different
observables. This main covers conversions between flux, luminosity and
magnitude systems.

Example usage:

    lum = flux_to_luminosity(flux, cosmo, redshift)
    fnu = apparent_mag_to_fnu(m)
    ab_mag = absolute_mag_to_lnu(M)

"""
import numpy as np
from unyt import c, nJy, erg, s, Hz, cm, pc

from synthesizer import exceptions
from synthesizer.utils import has_units


def flux_to_luminosity(flux, cosmo, redshift):
    """
    Converts flux to luminosity in erg / s / Hz.

    This can either be flux -> luminosity per wavelength/frequency (intensity)
    or power since all units are handled automatically by unyt.

    Args:
        flux (unyt_quantity/unyt_array)
            The flux to be converted to luminosity, can either be a singular
            value or array.
        cosmo (astropy.cosmology)
            The cosmology object used to calculate luminosity distance.
        redshift (float)
            The redshift of the rest frame.

    Returns:
        unyt_quantity/unyt_array
            The converted luminosity.

    Raises:
        IncorrectUnits
            If units are missing an error is raised.
    """

    # Ensure we have units
    if not has_units(flux):
        raise exceptions.IncorrectUnits("Flux must be given with unyt units.")

    # Calculate the luminosity distance (need to convert from astropy to unyt)
    lum_dist = cosmo.luminosity_distance(redshift).to("cm").value * cm

    # Calculate the luminosity in interim units
    lum = flux * 4 * np.pi * lum_dist**2

    # And convert to erg / s / Hz
    lum /= 1 + redshift

    return lum.to(erg / s / Hz)


def fnu_to_apparent_mag(fnu):
    """
    Converts flux to apparent magnitude.

    Args:
        flux (unyt_quantity/unyt_array)
            The flux to be converted, can either be a singular value or array.

    Returns:
        float
            The apparent magnitude.

    Raises:
        IncorrectUnits
            If units are missing an error is raised.
    """

    # Ensure we have units
    if not has_units(fnu):
        raise exceptions.IncorrectUnits("fnu must be given with unyt units.")

    return -2.5 * np.log10(fnu / (10**9 * nJy)) + 8.9


def apparent_mag_to_fnu(app_mag):
    """
    Converts apparent magnitude to flux.

    Args:
        app_mag (float)
            The apparent magnitude to be converted, can either be a singular
            value or array.

    Returns:
        unyt_quantity/unyt_array
            The flux.

    """

    return 10**9 * 10 ** (-0.4 * (app_mag - 8.9)) * nJy


def flam_to_fnu(lam, flam):
    """
    Converts flux in terms of wavelength (f_lam) to flux in terms of frequency
    (f_nu).

    Args:
        lam (unyt_quantity/unyt_array)
            The wavelength array the flux is defined at.
        flam (unyt_quantity/unyt_array)
            The flux in terms of wavelength.

    Returns:
        unyt_quantity/unyt_array
            The flux in terms of frequency.

    Raises:
        IncorrectUnits
            If units are missing an error is raised.
    """

    # Ensure we have units
    if not has_units(flam):
        raise exceptions.IncorrectUnits("flam must be given with unyt units.")
    if not has_units(lam):
        raise exceptions.IncorrectUnits("lam must be given with unyt units.")

    # Delta lambda
    lam_m = lam * 10**-10

    return flam * lam / (c / lam_m)


def fnu_to_flam(lam, fnu):
    """
    Converts flux in terms of frequency (f_nu) to flux in terms of wavelength
    (flam).

    Args:
        lam (unyt_quantity/unyt_array)
            The wavelength array the flux is defined at.
        fnu (unyt_quantity/unyt_array)
            The flux in terms of frequency.

    Returns:
        unyt_quantity/unyt_array
            The flux in terms of wavlength.

    Raises:
        IncorrectUnits
            If units are missing an error is raised.
    """

    # Ensure we have units
    if not has_units(fnu):
        raise exceptions.IncorrectUnits("fnu must be given with unyt units.")
    if not has_units(lam):
        raise exceptions.IncorrectUnits("lam must be given with unyt units.")

    # Delta lambda
    lam_m = lam * 1e-10

    return fnu * (c / lam_m) / lam


def absolute_mag_to_lnu(ab_mag):
    """Convert absolute magnitude (M) to luminosity.

    Args:
        ab_mag (float)
            The absolute magnitude to convert.

    Returns:
        unyt_quantity/unyt_array
            The luminosity in erg / s / Hz.
    """

    # Define the distance modulus at 10 pcs
    dist_mod = 4 * np.pi * (10 * pc).to("cm").value ** 2

    return 10 ** (-0.4 * (ab_mag + 48.6)) * dist_mod * erg / s / Hz


def lnu_to_absolute_mag(lnu):
    """Convert luminosity to absolute magnitude (M).

    Args:
        unyt_quantity/unyt_array
            The luminosity to convert with units. Unyt

    Returns:
        float
            The absolute magnitude.

    Raises:
        IncorrectUnits
            If units are missing an error is raised.
    """

    # Enusre we have units
    if not has_units(lnu):
        raise exceptions.IncorrectUnits("lnu must be given with unyt units.")

    # Define the distance modulus at 10 pcs
    dist_mod = 4 * np.pi * ((10 * pc).to("cm").value * cm) ** 2

    # Make sure the units are consistent
    lnu = lnu.to(erg / s / Hz)

    return -2.5 * np.log10(lnu / dist_mod / (erg / s / Hz)) - 48.6
