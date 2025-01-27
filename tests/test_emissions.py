"""A suite of tests for testing emission models."""

import numpy as np

from synthesizer.emission_models import NebularEmission


def test_nebular_emissions(test_grid, particle_stars_A):
    """Test the nebular emission makes sense."""
    # Define a model with a reduced lyman lapha escape fraction
    model = NebularEmission(test_grid, fesc_ly_alpha=0.9)

    # Get the spectra
    particle_stars_A.get_spectra(model)

    # Check that the nebular spectra is the same as sum of the individual
    # line contribution and the continuum
    neb_spec = particle_stars_A.spectra["nebular"]._lnu
    line_spec = particle_stars_A.spectra["linecont"]._lnu
    neb_continuum = particle_stars_A.spectra["nebular_continuum"]._lnu

    assert np.allclose(neb_spec, line_spec + neb_continuum)

    # Lets also make sure the lyman alpha escape fraction is actually doing
    # something (covers a previous issue where it wasn't)
    model = NebularEmission(test_grid, fesc_ly_alpha=0.5)

    # Get the spectra
    particle_stars_A.get_spectra(model)

    # Ensure the lyman alpha escape fraction is doing something
    assert np.sum(particle_stars_A.spectra["nebular"]._lnu) < np.sum(neb_spec)

    # Also check that not setting the escape fraction results in a higher
    # luminosity than setting it < 1
    model = NebularEmission(test_grid)

    # Get the spectra
    particle_stars_A.get_spectra(model)

    # Ensure the luminosity is higher than when the escape fraction is set
    assert np.sum(particle_stars_A.spectra["nebular"]._lnu) > np.sum(neb_spec)
