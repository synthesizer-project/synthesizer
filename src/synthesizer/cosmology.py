"""A module containing helper functions for cosmology calculations.

This module mainly interfaces with astropy.cosmology and provide helpful
wrappers and importantly cached functions for hot-path cosmology calculations.
"""

import inspect
from functools import lru_cache

import astropy.cosmology as cosmo_module
from astropy import units as u
from unyt import Mpc


def _get_cosmo_key(cosmo):
    """Create a hashable key for a cosmology object.

    We need this because astropy cosmology objects are not hashable by default,
    but we want to use them as keys in a cache. This function extracts the key
    parameters of the cosmology and returns them as a tuple which can be used
    as a cache key and then used to reconstruct the cosmology object.

    Args:
        cosmo (astropy.cosmology.FLRW): An instance of an astropy cosmology.

    Returns:
        tuple: A hashable tuple representing the cosmology parameters.
    """
    # Get the class name for proper reconstruction
    class_name = cosmo.__class__.__name__

    # Extract all the key parameters needed to reconstruct the cosmology
    params = {}

    # Always present parameters
    params["H0"] = cosmo.H0.value
    params["Om0"] = cosmo.Om0
    params["name"] = cosmo.name

    # Optional parameters that may be present
    if hasattr(cosmo, "Ode0") and cosmo.Ode0 is not None:
        params["Ode0"] = cosmo.Ode0

    if hasattr(cosmo, "Tcmb0") and cosmo.Tcmb0 is not None:
        params["Tcmb0"] = cosmo.Tcmb0.value

    if hasattr(cosmo, "Neff") and cosmo.Neff is not None:
        params["Neff"] = cosmo.Neff

    if hasattr(cosmo, "m_nu") and cosmo.m_nu is not None:
        # Convert astropy quantity array to list of values for hashability
        params["m_nu"] = tuple(cosmo.m_nu.value.tolist())

    if hasattr(cosmo, "Ob0") and cosmo.Ob0 is not None:
        params["Ob0"] = cosmo.Ob0

    if hasattr(cosmo, "w0") and cosmo.w0 is not None:
        params["w0"] = cosmo.w0

    if hasattr(cosmo, "wa") and cosmo.wa is not None:
        params["wa"] = cosmo.wa

    # Create a hashable representation
    # Sort the params to ensure consistent ordering
    param_items = tuple(sorted(params.items()))

    return (class_name, param_items)


def _reconstruct_cosmology(cosmo_key):
    """Reconstruct a cosmology object from its key.

    Args:
        cosmo_key (tuple): A tuple containing (class_name, param_items).

    Returns:
        astropy.cosmology.FLRW: The reconstructed cosmology object.
    """
    # Unpack the key (contains everything needed to reconstruct)
    class_name, param_items = cosmo_key
    params = dict(param_items)

    # The name tells us which class to instantiate
    cosmo_class = getattr(cosmo_module, class_name)

    # Get the valid parameters for this cosmology class and filter out the self
    valid_params = set(
        inspect.signature(cosmo_class.__init__).parameters.keys()
    )
    valid_params.discard("self")  # Remove 'self'

    # What are the required parameters for this class?
    required_params = {
        name
        for name, param in inspect.signature(
            cosmo_class.__init__
        ).parameters.items()
        if param.default is param.empty and name != "self"
    }

    # Are we missing any required parameters?
    for param in required_params:
        if param not in params:
            raise ValueError(
                f"Cannot reconstruct {class_name} cosmology: we seem to be "
                f"missing required parameter '{param}'"
            )

    # Prepare parameters with proper units where needed
    kwargs = {}
    for key, value in params.items():
        # Skip parameters not valid for this cosmology class
        if key not in valid_params:
            continue

        # Add units where necessary
        if key == "H0":
            kwargs[key] = value * u.km / u.s / u.Mpc
        elif key == "Tcmb0":
            kwargs[key] = value * u.K
        elif key == "m_nu":
            kwargs[key] = u.Quantity(value, u.eV)
        else:
            kwargs[key] = value

    return cosmo_class(**kwargs)


@lru_cache(maxsize=1000)  # Cache up to 1000 different combinations
def _cached_luminosity_distance(cosmo_key, redshift):
    """Internal cached function for luminosity distance calculation.

    Here we cache the results of luminosity distance calculations based on
    the cosmology parameters and redshift. If we haven't got a cached result
    for the given cosmology and redshift, we reconstruct the cosmology and
    compute the distance, cheap since we only need to do this once per unique
    cosmology and redshift.

    Args:
        cosmo_key (tuple): A hashable representation of the cosmology.
        redshift (float): The redshift for which to compute the distance.

    Returns:
        float: The luminosity distance value in Mpc.
    """
    cosmo = _reconstruct_cosmology(cosmo_key)
    return cosmo.luminosity_distance(redshift).to("Mpc").value


@lru_cache(maxsize=1000)  # Cache up to 1000 different combinations
def _cached_angular_diameter_distance(cosmo_key, redshift):
    """Internal cached function for angular diameter distance calculation.

    Here we cache the results of angular diameter distance calculations based
    on the cosmology parameters and redshift. If we haven't got a cached result
    for the given cosmology and redshift, we reconstruct the cosmology and
    compute the distance, cheap since we only need to do this once per unique
    cosmology and redshift.

    Args:
        cosmo_key (tuple): A hashable representation of the cosmology.
        redshift (float): The redshift for which to compute the distance.

    Returns:
        float: The angular diameter distance value in Mpc.
    """
    cosmo = _reconstruct_cosmology(cosmo_key)
    return cosmo.angular_diameter_distance(redshift).to("Mpc").value


def get_luminosity_distance(cosmo, redshift):
    """Get the luminosity distance for a given redshift and cosmology.

    This function is cached to improve performance on repeated calls with
    the same parameters, i.e. the lru_cache will return the previously
    computed value if the same redshift is requested again with the same
    cosmology.

    Args:
        cosmo (astropy.cosmology.FLRW): An instance of an astropy cosmology.
        redshift (float): The redshift for which to compute the luminosity
            distance.

    Returns:
        unyt_quantity: The luminosity distance in Mpc.
    """
    cosmo_key = _get_cosmo_key(cosmo)
    result_value = _cached_luminosity_distance(cosmo_key, redshift)
    return result_value * Mpc


def get_angular_diameter_distance(cosmo, redshift):
    """Get the angular diameter distance for a given redshift and cosmology.

    This function is cached to improve performance on repeated calls with
    the same parameters, i.e. the lru_cache will return the previously
    computed value if the same redshift is requested again with the same
    cosmology.

    Args:
        cosmo (astropy.cosmology.FLRW): An instance of an astropy cosmology.
        redshift (float): The redshift for which to compute the angular
            diameter distance.

    Returns:
        unyt_quantity: The angular diameter distance in Mpc.
    """
    cosmo_key = _get_cosmo_key(cosmo)
    result_value = _cached_angular_diameter_distance(cosmo_key, redshift)
    return result_value * Mpc
