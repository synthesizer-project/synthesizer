"""A suie of tests for testing emission models."""

import numpy as np
import pytest

from synthesizer import exceptions
from synthesizer.emission_models import (
    NebularEmission,
    TransmittedEmission,
)


def test_nebular_emissions(test_grid, particle_stars_A):
    """Test the nebular emission makes sense."""
    # Define a model with a reduced lyman lapha escape fraction
    model = NebularEmission(test_grid, fesc_ly_alpha=0.9)

    # Get the spectra
    particle_stars_A.get_spectra(model)

    # Check that the nebular spectra is the same as sum of the individual
    # line contribution and the continuum
    neb_spec = particle_stars_A.spectra["nebular"]._lnu
    line_spec = particle_stars_A.spectra["_nebular_line"]._lnu
    neb_continuum = particle_stars_A.spectra["_nebular_continuum"]._lnu

    assert np.allclose(neb_spec, line_spec + neb_continuum)

    # Make sure lyman alpha is actually included in the wavelength array
    assert np.any(particle_stars_A.spectra["nebular"]._lam.min() < 1216)
    assert np.any(particle_stars_A.spectra["nebular"]._lam.max() > 1216)

    # Lets also make sure the lyman alpha escape fraction is actually doing
    # something (covers a previous issue where it wasn't)
    model = NebularEmission(test_grid, fesc_ly_alpha=0.001)

    particle_stars_A.clear_all_emissions()

    # Get the spectra
    particle_stars_A.get_spectra(model)

    # Ensure the lyman alpha escape fraction is doing something
    assert not np.allclose(
        neb_spec, particle_stars_A.spectra["nebular"]._lnu
    ), "The lyman alpha escape fraction is not doing anything"

    # Also check that not setting the escape fraction results in a higher
    # luminosity than setting it < 1
    model = NebularEmission(test_grid)

    particle_stars_A.clear_all_emissions()

    # Get the spectra
    particle_stars_A.get_spectra(model)

    # Ensure the luminosity is higher than when the escape fraction is set
    assert np.sum(particle_stars_A.spectra["nebular"]._lnu) > np.sum(neb_spec)


def test_escape_fraction(
    transmitted_emission_model, particle_stars_A, test_grid
):
    """Test the escape fraction is applied correctly."""
    # Get the spectra with the escape fraction set to 0.1 on the emitter
    particle_stars_A.fesc = 0.1
    particle_stars_A.get_spectra(transmitted_emission_model)
    spectra_with_escape = particle_stars_A.spectra["transmitted"]._lnu

    # Clear the spectra
    particle_stars_A.clear_all_emissions()

    # Get the spectra with the escape fraction set to 0 on the emitter
    particle_stars_A.fesc = 0.0
    particle_stars_A.get_spectra(transmitted_emission_model)
    spectra_without_escape = particle_stars_A.spectra["transmitted"]._lnu

    # Ensure the escape fraction is applied correctly
    assert not np.allclose(spectra_with_escape, spectra_without_escape), (
        "The escape fraction is not being applied correctly when "
        "set on the emitter."
    )
    assert np.sum(spectra_with_escape) < np.sum(spectra_without_escape), (
        "The escape fraction is not being applied correctly when "
        "set on the emitter."
    )

    # Do the same check but for transmitted emission models with fesc set on
    # them
    model_with_escape = TransmittedEmission(grid=test_grid, fesc=0.1)
    model_without_escape = TransmittedEmission(grid=test_grid, fesc=0.0)

    # Get the spectra with the escape fraction set to 0.1
    particle_stars_A.get_spectra(model_with_escape)
    spectra_with_escape = particle_stars_A.spectra["transmitted"]._lnu

    # Clear the spectra
    particle_stars_A.clear_all_emissions()

    # Get the spectra with the escape fraction set to 0
    particle_stars_A.get_spectra(model_without_escape)
    spectra_without_escape = particle_stars_A.spectra["transmitted"]._lnu

    # And test...
    assert not np.allclose(spectra_with_escape, spectra_without_escape), (
        "The escape fraction is not being applied correctly when "
        "set on a EmissionModel."
    )
    assert np.sum(spectra_with_escape) < np.sum(spectra_without_escape), (
        "The escape fraction is not being applied correctly when "
        "set on a EmissionModel."
    )


@pytest.mark.parametrize(
    "override",
    [
        {"dust_curves": {}},
        {"tau_v": 0.5},
        {"fesc": 0.1},
        {"covering_fraction": 0.9},
        {"mask": {}},
        {"vel_shift": True},
        {"tau_v_ism": 0.3},
    ],
)
def test_removed_spectra_overrides_raise(
    override, nebular_emission_model, random_particle_galaxy
):
    """Test the removed get_spectra overrides raise a clear error."""
    # Passing any of them to a component should raise
    with pytest.raises(exceptions.InconsistentArguments, match="no longer"):
        random_particle_galaxy.stars.get_spectra(
            nebular_emission_model, **override
        )

    # As should passing them to a galaxy
    with pytest.raises(exceptions.InconsistentArguments, match="no longer"):
        random_particle_galaxy.get_spectra(nebular_emission_model, **override)


def test_removed_lines_overrides_raise(
    nebular_emission_model, random_particle_galaxy
):
    """Test the removed get_lines overrides raise a clear error."""
    # Passing one to a component should raise
    with pytest.raises(exceptions.InconsistentArguments, match="tau_v"):
        random_particle_galaxy.stars.get_lines(
            ["H 1 6562.80A"], nebular_emission_model, tau_v=0.5
        )

    # As should passing one to a galaxy
    with pytest.raises(exceptions.InconsistentArguments, match="fesc"):
        random_particle_galaxy.get_lines(
            ["H 1 6562.80A"], nebular_emission_model, fesc=0.1
        )

    # Keywords get_lines never took (e.g. vel_shift) should raise the same
    # error rather than a bare TypeError
    with pytest.raises(exceptions.InconsistentArguments, match="vel_shift"):
        random_particle_galaxy.get_lines(
            ["H 1 6562.80A"], nebular_emission_model, vel_shift=True
        )
    with pytest.raises(exceptions.InconsistentArguments, match="tau_v_ism"):
        random_particle_galaxy.stars.get_lines(
            ["H 1 6562.80A"], nebular_emission_model, tau_v_ism=0.3
        )
