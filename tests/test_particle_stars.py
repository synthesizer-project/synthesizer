import numpy as np
import pytest
from unyt import Myr, kpc

from synthesizer import exceptions
from synthesizer.exceptions import InconsistentAddition
from synthesizer.parametric.stars import Stars as ParaStars
from synthesizer.particle.stars import Stars


def test_cant_add_different_types(particle_stars_A, particle_gas_A):
    """
    Test we can't add different types of particles together.
    """
    with pytest.raises(InconsistentAddition):
        particle_stars_A + particle_gas_A


def test_add_stars(particle_stars_A, particle_stars_B):
    """
    Test we can add two Stars objects together.
    """
    assert isinstance(particle_stars_A + particle_stars_B, Stars)


def test_cant_add_stars_different_redshifts(
    particle_stars_A, particle_stars_B
):
    """
    Test we can't add two Stars objects with different redshifts.
    """
    particle_stars_B.redshift = 2.0

    with pytest.raises(InconsistentAddition):
        particle_stars_A + particle_stars_B


def test_add_stars_with_different_attributes(
    particle_stars_A, particle_stars_B
):
    """
    Test we can add two Stars objects with different attributes.
    """
    particle_stars_B.dummy_attr = None

    assert isinstance(particle_stars_A + particle_stars_B, Stars)


def test_parametric_young_stars(particle_stars_A, test_grid):
    """
    Test we can use parametric_young_stars to replace
    young star particles.
    """
    particle_stars_A.parametric_young_stars(
        age=10 * Myr,
        parametric_sfh="constant",
        grid=test_grid,
    )

    assert isinstance(particle_stars_A, Stars)
    assert isinstance(particle_stars_A._parametric_young_stars, ParaStars)
    assert isinstance(particle_stars_A._old_stars, Stars)
    assert isinstance(particle_stars_A.young_stars_parametrisation, dict)
    assert particle_stars_A.young_stars_parametrisation["age"] == 10 * Myr
    assert (
        particle_stars_A.young_stars_parametrisation["parametrisation"]
        == "constant"
    )


def test_calculate_radii(particle_stars_A):
    with pytest.raises(
        exceptions.InconsistentArguments,
        match="Can't calculate radii without a centre.",
    ):
        particle_stars_A.get_radii()

    particle_stars_A.centre = np.array([1.0, 0.0, 2.0]) * kpc

    assert isinstance(particle_stars_A.get_radii(), np.ndarray)
    assert (particle_stars_A.radii <= 6 * kpc).all()
