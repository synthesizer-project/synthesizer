"""A module containing functions for conversions.

This module contains helpful conversions for converting between different
observables. This mainly covers conversions between flux, luminosity and
magnitude systems.

Example usage:

    lum = flux_to_luminosity(flux, cosmo, redshift)
    fnu = apparent_mag_to_fnu(m)
    lnu = absolute_mag_to_lnu(M)

"""

import numpy as np
from unyt import Angstrom, Hz, c, cm, erg, nJy, pc, s, unyt_array

from synthesizer import exceptions
from synthesizer.utils import has_units


def flux_to_luminosity(flux, cosmo, redshift):
    """
    Converts flux to luminosity in erg / s.

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

    # And redshift
    lum /= 1 + redshift

    return lum.to(erg / s)


def fnu_to_lnu(fnu, cosmo, redshift):
    """
    Converts spectral flux density to spectral luminosity density
    in erg / s / Hz.

    Args:
        fnu (unyt_quantity/unyt_array)
            The spectral flux dnesity to be converted to luminosity, can
            either be a singular value or array.
        cosmo (astropy.cosmology)
            The cosmology object used to calculate luminosity distance.
        redshift (float)
            The redshift of the rest frame.

    Returns:
        unyt_quantity/unyt_array
            The converted spectral luminosity density.

    Raises:
        IncorrectUnits
            If units are missing an error is raised.
    """

    # Ensure we have units
    if not has_units(fnu):
        raise exceptions.IncorrectUnits("fnu must be given with unyt units.")

    # Calculate the luminosity distance (need to convert from astropy to unyt)
    lum_dist = cosmo.luminosity_distance(redshift).to("cm").value * cm

    # Calculate the luminosity in interim units
    lnu = fnu * 4 * np.pi * lum_dist**2

    # And redshift
    lnu /= 1 + redshift

    return lnu.to(erg / s / Hz)


def fnu_to_apparent_mag(fnu):
    """
    Converts flux to apparent AB magnitude.

    Args:
        flux (unyt_quantity/unyt_array)
            The flux to be converted, can either be a singular value or array.

    Returns:
        float
            The apparent AB magnitude.

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
    Converts apparent AB magnitude to flux.

    Args:
        app_mag (float)
            The apparent AB magnitude to be converted, can either be a
            singular value or array.

    Returns:
        unyt_quantity/unyt_array
            The flux.

    """

    return 10**9 * 10 ** (-0.4 * (app_mag - 8.9)) * nJy


def llam_to_lnu(lam, llam):
    """
    Converts spectral luminosity density in terms of wavelength (llam) to
    spectral luminosity density in terms of frequency (lnu).

    Args:
        lam (unyt_quantity/unyt_array)
            The wavelength array the flux is defined at.
        llam (unyt_quantity/unyt_array)
            The spectral luminoisty density in terms of wavelength.

    Returns:
        unyt_quantity/unyt_array
            The spectral luminosity in terms of frequency, in units of nJy.

    Raises:
        IncorrectUnits
            If units are missing an error is raised.
    """

    # Ensure we have units
    if not has_units(llam):
        raise exceptions.IncorrectUnits("llam must be given with unyt units.")
    if not has_units(lam):
        raise exceptions.IncorrectUnits("lam must be given with unyt units.")

    return (llam * lam**2 / c).to("erg / s / Hz")


def lnu_to_llam(lam, lnu):
    """
    Converts spectral luminoisty density in terms of frequency (lnu)
    to luminoisty in terms of wavelength (llam).

    Args:
        lam (unyt_quantity/unyt_array)
            The wavelength array the luminoisty density is defined at.
        lnu (unyt_quantity/unyt_array)
            The spectral luminoisty density in terms of frequency.

    Returns:
        unyt_quantity/unyt_array
            The spectral luminoisty density in terms of wavelength, in units
            of erg / s / A.

    Raises:
        IncorrectUnits
            If units are missing an error is raised.
    """

    # Ensure we have units
    if not has_units(lnu):
        raise exceptions.IncorrectUnits("lnu must be given with unyt units.")
    if not has_units(lam):
        raise exceptions.IncorrectUnits("lam must be given with unyt units.")

    return ((lnu * c) / lam**2).to("erg / s / angstrom")


def flam_to_fnu(lam, flam):
    """
    Converts spectral flux in terms of wavelength (f_lam) to spectral flux
    in terms of frequency (f_nu).

    Args:
        lam (unyt_quantity/unyt_array)
            The wavelength array the flux is defined at.
        flam (unyt_quantity/unyt_array)
            The spectral flux in terms of wavelength.

    Returns:
        unyt_quantity/unyt_array
            The spectral flux in terms of frequency, in units of nJy.

    Raises:
        IncorrectUnits
            If units are missing an error is raised.
    """

    # Ensure we have units
    if not has_units(flam):
        raise exceptions.IncorrectUnits("flam must be given with unyt units.")
    if not has_units(lam):
        raise exceptions.IncorrectUnits("lam must be given with unyt units.")

    return (flam * lam**2 / c).to("nJy")


def fnu_to_flam(lam, fnu):
    """
    Converts spectral flux density in terms of frequency (f_nu)
    to flux in terms of wavelength (flam).

    Args:
        lam (unyt_quantity/unyt_array)
            The wavelength array the flux density is defined at.
        fnu (unyt_quantity/unyt_array)
            The spectral flux density in terms of frequency.

    Returns:
        unyt_quantity/unyt_array
            The spectral flux density in terms of wavelength, in units
            of erg / s / Hz / cm**2.

    Raises:
        IncorrectUnits
            If units are missing an error is raised.
    """

    # Ensure we have units
    if not has_units(fnu):
        raise exceptions.IncorrectUnits("fnu must be given with unyt units.")
    if not has_units(lam):
        raise exceptions.IncorrectUnits("lam must be given with unyt units.")

    return ((fnu * c) / lam**2).to("erg / s / angstrom / cm**2")


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
    """Convert spectral luminosity density to absolute magnitude (M).

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


def vacuum_to_air(wavelength):
    """
    A function for converting a vacuum wavelength into an air wavelength.

    Arguments:
        wavelength (float or unyt_array)
            A wavelength in air.

    Returns:
        wavelength (unyt_array)
            A wavelength in vacuum.
    """

    # if wavelength is not a unyt_array conver to one assume unit is Anstrom.
    if not isinstance(wavelength, unyt_array):
        wavelength *= Angstrom

    # calculate wavelenegth squared for simplicty
    wave2 = wavelength.to("Angstrom").value ** 2.0

    # calcualte conversion factor
    conversion = (
        1.0 + 2.735182e-4 + 131.4182 / wave2 + 2.76249e8 / (wave2**2.0)
    )

    return wavelength / conversion


def air_to_vacuum(wavelength):
    """
    A function for converting an air wavelength into a vacuum wavelength.

    Arguments
        wavelength (float or unyt_array)
            A standard wavelength.

    Returns
        wavelength (unyt_array)
            A wavelength in vacuum.
    """

    # if wavelength is not a unyt_array conver to one assume unit is Anstrom.
    if not isinstance(wavelength, unyt_array):
        wavelength *= Angstrom

    # Convert to wavenumber squared
    sigma2 = (1.0e4 / wavelength.to("Angstrom").value) ** 2.0

    # Compute conversion factor
    conversion = (
        1.0
        + 6.4328e-5
        + 2.94981e-2 / (146.0 - sigma2)
        + 2.5540e-4 / (41.0 - sigma2)
    )

    return wavelength * conversion


def standard_to_vacuum(wavelength):
    """
    A function for converting a standard wavelength into a vacuum wavelength.

    Standard wavelengths are defined in vacuum at <2000A and air at >= 2000A.

    Arguments
        wavelength (float or unyt_array)
            A standard wavelength.

    Returns
        wavelength (unyt_array)
            A wavelength in vacuum.
    """

    # if wavelength is not a unyt_array conver to one assume unit is Anstrom.
    if not isinstance(wavelength, unyt_array):
        wavelength *= Angstrom

    # if wavelength is < 2000A simply return since no change required.
    if wavelength <= 2000.0 * Angstrom:
        return wavelength

    # otherwise conver to vacuum
    else:
        return air_to_vacuum(wavelength)


def vacuum_to_standard(wavelength):
    """
    A function for converting a vacuum wavelength into a standard wavelength.

    Standard wavelengths are defined in vacuum at <2000A and air at >= 2000A.

    Arguments
        wavelength (float or unyt_array)
            A vacuum wavelength.

    Returns
        wavelength (unyt_array)
            A standard wavelength.
    """

    # if wavelength is not a unyt_array conver to one assume unit is Anstrom.
    if not isinstance(wavelength, unyt_array):
        wavelength *= Angstrom

    # if wavelength is < 2000A simply return since no change required.
    if wavelength <= 2000.0 * Angstrom:
        return wavelength

    # otherwise conver to vacuum
    else:
        return vacuum_to_air(wavelength)
