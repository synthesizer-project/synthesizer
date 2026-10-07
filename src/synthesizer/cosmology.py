"""A module containing helper functions for cosmology calculations.

This module mainly interfaces with astropy.cosmology and provides helpful
wrappers and importantly cached functions for hot path cosmology calculations.

The module is explicitly tested against, and supports, the following
astropy cosmology models:
    - FLRW (base class)
    - FlatLambdaCDM (flat ΛCDM model)
    - wCDM (constant dark energy equation of state)
    - FlatwCDM (flat version of wCDM)
    - w0waCDM (time-varying dark energy equation of state)
    - Flatw0waCDM (flat version of w0waCDM)
    - wpwaCDM (pivot redshift parameterization)
    - FlatwpwaCDM (flat version of wpwaCDM)
    - w0wzCDM (redshift derivative parameterization)
    - Flatw0wzCDM (flat version of w0wzCDM)
"""

import inspect
from functools import lru_cache

import astropy.cosmology as cosmo_module
import numpy as np
from astropy import units as u
from unyt import Mpc

from synthesizer import exceptions

# NOTE: Constructing an astropy cosmology leaves reference cycles (from unit
# composition) that pin the calling frames, e.g. a whole galaxy, until the
# garbage collector runs. Cache misses therefore use the caller's cosmology,
# stored here by key, rather than reconstructing one.
_cosmologies = {}

# The most cosmologies to keep, matching the distance caches' size
_MAX_COSMOLOGIES = 1024


def _remember_cosmology(cosmo_key, cosmo):
    """Store a cosmology so distance cache misses can use it.

    The distance caches are keyed by a hashable representation of the
    cosmology (see _get_cosmo_key), since astropy cosmologies aren't hashable.
    On a cache miss they need an actual cosmology object to compute the
    distance with. Rather than reconstructing one from the key, which leaves
    reference cycles that pin the caller's frames (see the note on
    _cosmologies), they look up the caller's own cosmology stored here.

    The store keeps at most _MAX_COSMOLOGIES entries, so it can't grow without
    bound if a run sees many distinct cosmologies. When it is full the oldest
    entry is dropped. A cache miss for a dropped cosmology falls back to
    reconstructing it, which is still correct, just without avoiding the
    reference cycles.

    Args:
        cosmo_key (tuple):
            The hashable representation of the cosmology, as used to key
            the distance caches.
        cosmo (astropy.cosmology.FLRW):
            The cosmology the key was made from.
    """
    # Keep the first cosmology stored for a key, they are all equivalent
    if cosmo_key in _cosmologies:
        return

    # Make room by dropping the oldest entry (dicts keep insertion order)
    if len(_cosmologies) >= _MAX_COSMOLOGIES:
        del _cosmologies[next(iter(_cosmologies))]

    _cosmologies[cosmo_key] = cosmo


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

    # Get the valid parameters for this cosmology class from its constructor
    valid_params = set(
        inspect.signature(cosmo.__class__.__init__).parameters.keys()
    )
    valid_params.discard("self")  # Remove 'self'

    # Dynamically extract all parameters that the constructor accepts
    for param_name in valid_params:
        if hasattr(cosmo, param_name):
            value = getattr(cosmo, param_name)

            # Only include non-None values that are hashable or convertible
            if value is not None:
                # Handle astropy Quantities explicitly
                if isinstance(value, u.Quantity):
                    # Arrays must be tuples, otherwise extract values
                    if isinstance(value.value, np.ndarray):
                        params[param_name] = tuple(value.value.tolist())
                    else:
                        params[param_name] = value.value
                elif isinstance(value, (int, float, str)):
                    # Simple hashable types
                    params[param_name] = value
                else:
                    # Skip unhashable types, not needed for reconstruction
                    pass

    # Create a sorted hashable representation
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
        elif isinstance(value, tuple):
            # Reconstruct Quantity arrays
            kwargs[key] = u.Quantity(value, u.eV)
        else:
            # All other parameters are plain floats/ints/strings
            kwargs[key] = value

    return cosmo_class(**kwargs)


@lru_cache(maxsize=_MAX_COSMOLOGIES)  # Cache up to 1000 combinations
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
    cosmo = _cosmologies.get(cosmo_key)
    if cosmo is None:
        cosmo = _reconstruct_cosmology(cosmo_key)
    return cosmo.luminosity_distance(redshift).to("Mpc").value


@lru_cache(maxsize=_MAX_COSMOLOGIES)  # Cache up to 1000 combinations
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
    cosmo = _cosmologies.get(cosmo_key)
    if cosmo is None:
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
    if cosmo is None:
        raise exceptions.InconsistentArguments(
            "Cosmology object cannot be None"
        )

    cosmo_key = _get_cosmo_key(cosmo)
    _remember_cosmology(cosmo_key, cosmo)
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
    _remember_cosmology(cosmo_key, cosmo)
    result_value = _cached_angular_diameter_distance(cosmo_key, redshift)
    return result_value * Mpc
