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
    Convert flux to luminosity.

    The conversion is done using the formula:

            L = F * 4 * pi * D_L**2 / (1 + z)

    The result will be in units of erg / s.

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
    Convert spectral flux density to spectral luminosity density.

    The conversion is done using the formula:

            L_nu = F_nu * 4 * pi * D_L**2 / (1 + z)

    The result will be in units of erg / s / Hz.

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
    Convert flux to apparent AB magnitude.

    The conversion is done using the formula:

                m = -2.5 * log10(fnu / (10**9 * nJy)) + 8.9

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
    Convert apparent AB magnitude to flux.

    The conversion is done using the formula:

                F_nu = 10**9 * 10**(-0.4 * (m - 8.9)) * nJy

    The result will be in units of nJy.

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
    Convert spectral luminosity density per wavelength to per frequency.

    The conversion is done using the formula:

                L_nu = L_lam * lam**2 / c

    The result will be in units of erg / s / Hz.

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
    Convert spectral luminoisty density per frequency to per wavelength.

    The conversion is done using the formula:

                    L_lam = L_nu * c / lam**2

    The result will be in units of erg / s / A.

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
    Convert spectral flux density per wavelength to per frequency.

    The conversion is done using the formula:

                    F_nu = F_lam * lam**2 / c

    The result will be in units of nJy.

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
    Convert spectral flux density per frequency to per wavelength.

    The conversion is done using the formula:

                    F_lam = F_nu * c / lam**2

    The result will be in units of erg / s / A.

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
    """
    Convert absolute magnitude (M) to luminosity.

    The conversion is done using the formula:

                    L_nu = 10**(-0.4 * (M + 48.6)) * dist_mod

    The result will be in units of erg / s / Hz.

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
    """
    Convert spectral luminosity density to absolute magnitude (M).

    The conversion is done using the formula:

                    M = -2.5 * log10(L_nu / dist_mod / (erg / s / Hz)) - 48.6

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
    Convert a vacuum wavelength into an air wavelength.

    Arguments:
        wavelength (float or unyt_array)
            A wavelength in air.

    Returns:
        wavelength (unyt_array)
            A wavelength in vacuum.
    """
    # If wavelength is not a unyt_array conver to one assume unit is Anstrom.
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
    Convert an air wavelength into a vacuum wavelength.

    Arguments
        wavelength (float or unyt_array)
            A standard wavelength.

    Returns
        wavelength (unyt_array)
            A wavelength in vacuum.
    """
    # If wavelength is not a unyt_array conver to one assume unit is Anstrom.
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
    Convert a standard wavelength into a vacuum wavelength.

    Standard wavelengths are defined in vacuum at <2000A and air at >= 2000A.

    Arguments
        wavelength (float or unyt_array)
            A standard wavelength.

    Returns
        wavelength (unyt_array)
            A wavelength in vacuum.
    """
    # If wavelength is not a unyt_array conver to one assume unit is Anstrom.
    if not isinstance(wavelength, unyt_array):
        wavelength *= Angstrom

    # If wavelength is < 2000A simply return since no change required.
    if wavelength <= 2000.0 * Angstrom:
        return wavelength

    # Otherwise, convert to vacuum
    else:
        return air_to_vacuum(wavelength)


def vacuum_to_standard(wavelength):
    """
    Convert a vacuum wavelength into a standard wavelength.

    Standard wavelengths are defined in vacuum at <2000A and air at >= 2000A.

    Arguments
        wavelength (float or unyt_array)
            A vacuum wavelength.

    Returns
        wavelength (unyt_array)
            A standard wavelength.
    """
    # If wavelength is not a unyt_array conver to one assume unit is Anstrom.
    if not isinstance(wavelength, unyt_array):
        wavelength *= Angstrom

    # If wavelength is < 2000A simply return since no change required.
    if wavelength <= 2000.0 * Angstrom:
        return wavelength

    # Otherwise, convert to vacuum
    else:
        return vacuum_to_air(wavelength)


def attenuation_to_optical_depth(attenuation):
    """
    Convert attenuation to optical depth.

    Args:
        attenuation (float):
            The attenuation to convert.

    Return:
        float
            The converted optical depth.
    """
    return attenuation / (2.5 * np.log10(np.e))


def optical_depth_to_attenuation(optical_depth):
    """
    Convert optical depth to attenuation.

    Args:
        optical depth (float):
            The optical depth to convert.

    Return:
        float
            The converted attenuation.
    """
    return 2.5 * np.log10(np.e) * optical_depth


def tau_lam_to_tau_v(dust_curve, tau_lam, lam):
    """
    Convert optical depth in any given band to v-band optical depth.

    Args:
        dust_curve (AttenuationBase):
            The attenutation law to use.
        tau_lam (float):
            The optical depth to convert.
        lam (unyt_quantity):
            The wavelength at which tau_lam was calculated.

    Return:
        float
            The converted optical depth.
    """
    # Ensure we have units
    if not has_units(lam):
        raise exceptions.IncorrectUnits("lam must be given with unyt units.")

    # Convert to angstrom
    lam = lam.to("angstrom")

    tau_norm = dust_curve.get_tau(lam)
    return tau_lam / tau_norm
