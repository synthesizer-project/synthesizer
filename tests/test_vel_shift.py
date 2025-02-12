"""Test the velocity shifting of particle spectra works."""

import time

import numpy as np


def test_velocity_shift(random_part_stars, nebular_emission_model):
    """Test the velocity shift of particle spectra."""
    # Compute the spectra with and without velocity shift
    with_shift_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        vel_shift=True,
    )
    without_shift_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        vel_shift=False,
    )

    # Get and print a seed for reproducibility
    seed = int(time.time())
    np.random.seed(seed)

    # Ensure that the spectra are different
    assert not np.allclose(with_shift_spec._lnu, without_shift_spec._lnu)

    # Ensure that the overall flux is conserved, since we know it won't
    # be exactly the same at the edges due to the boundaries of the wavelength
    # array we can instead compare only the central part of the spectra
    # where the spectra overlap
    assert np.allclose(
        np.sum(with_shift_spec._lnu[100:-100]),
        np.sum(without_shift_spec._lnu[100:-100]),
        rtol=1e-6,
    ), f"The total flux of the spectra is not conserved (seed: {seed})"
