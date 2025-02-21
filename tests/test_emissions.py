"""A suite of tests for testing emission models."""

import numpy as np

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
    line_spec = particle_stars_A.spectra["nebular_line"]._lnu
    neb_continuum = particle_stars_A.spectra["nebular_continuum"]._lnu

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
    # Get the spectra with the escape fraction set to 0.1
    particle_stars_A.get_spectra(transmitted_emission_model, fesc=0.1)
    spectra_with_escape = particle_stars_A.spectra["transmitted"]._lnu

    # Clear the spectra
    particle_stars_A.clear_all_emissions()

    # Get the spectra with the escape fraction set to 0
    particle_stars_A.get_spectra(transmitted_emission_model, fesc=0.0)
    spectra_without_escape = particle_stars_A.spectra["transmitted"]._lnu

    # Ensure the escape fraction is applied correctly
    assert not np.allclose(spectra_with_escape, spectra_without_escape), (
        "The escape fraction is not being applied correctly when "
        "set via a get_spectra call."
    )
    assert np.sum(spectra_with_escape) < np.sum(spectra_without_escape), (
        "The escape fraction is not being applied correctly when "
        "set via a get_spectra call."
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
