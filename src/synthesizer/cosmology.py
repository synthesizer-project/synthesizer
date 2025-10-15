"""A module containing helper functions for cosmology calculations.

This module mainly interfaces with astropy.cosmology and provide helpful
wrappers and importantly cached functions for hot path cosmology calculations.
"""

from functools import lru_cache

from unyt import Mpc


@lru_cache(maxsize=1000)  # Cache up to 1000 different redshifts
def get_luminosity_distance(cosmo, redshift):
    """Get the luminosity distance for a given redshift and cosmology.

    This function is cached to improve performance on repeated calls with
    the same parameters, i.e. the lru_cache will returned the previously
    computed value if the same redshift is requested again with the same
    cosmology.

    Args:
        cosmo (astropy.cosmology.FLRW): An instance of an astropy cosmology.
        redshift (float): The redshift for which to compute the luminosity
            distance.

    Returns:
        unyt_quantity: The luminosity distance in Mpc.
    """
    return cosmo.luminosity_distance(redshift).to("Mpc").value * Mpc


@lru_cache(maxsize=1000)  # Cache up to 1000 different redshifts
def get_angular_diameter_distance(cosmo, redshift):
    """Get the angular diameter distance for a given redshift and cosmology.

    This function is cached to improve performance on repeated calls with
    the same parameters, i.e. the lru_cache will returned the previously
    computed value if the same redshift is requested again with the same
    cosmology.

    Args:
        cosmo (astropy.cosmology.FLRW): An instance of an astropy cosmology.
        redshift (float): The redshift for which to compute the angular
            diameter distance.

    Returns:
        unyt_quantity: The angular diameter distance in Mpc.
    """
    return cosmo.angular_diameter_distance(redshift).to("Mpc").value * Mpc
