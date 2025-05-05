"""Test the velocity shifting of particle spectra works."""

import time

import numpy as np


def test_velocity_shift_applys_cic(random_part_stars, nebular_emission_model):
    """Test the velocity shift of particle spectra."""
    # Compute the spectra with and without velocity shift
    with_shift_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        vel_shift=True,
        grid_assignment_method="cic",
    )
    random_part_stars.clear_all_emissions()
    without_shift_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        vel_shift=False,
        grid_assignment_method="cic",
    )
    random_part_stars.clear_all_emissions()

    # Get and print a seed for reproducibility
    seed = int(time.time())
    print(f"Seed for reproducibility: {seed} (use this to reproduce the test)")
    np.random.seed(seed)

    # Ensure that the spectra are different
    assert not np.allclose(with_shift_spec._lnu, without_shift_spec._lnu)


def test_velocity_shift_applys_cic_on_model(
    random_part_stars,
    nebular_emission_model,
):
    """Test the velocity shift of particle spectra."""
    # Tell the model to use the velocity shift
    nebular_emission_model.set_vel_shift(True, set_all=True)

    # Compute the spectra with and without velocity shift
    with_shift_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="cic",
    )
    random_part_stars.clear_all_emissions()
    nebular_emission_model.set_vel_shift(False, set_all=True)
    without_shift_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="cic",
    )
    random_part_stars.clear_all_emissions()

    # Get and print a seed for reproducibility
    seed = int(time.time())
    print(f"Seed for reproducibility: {seed} (use this to reproduce the test)")
    np.random.seed(seed)

    # Ensure that the spectra are different
    assert not np.allclose(with_shift_spec._lnu, without_shift_spec._lnu)


def test_velocity_shift_conservation_cic(
    random_part_stars,
    nebular_emission_model,
):
    """Test the velocity shift of particle spectra."""
    # Compute the spectra with and without velocity shift
    with_shift_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        vel_shift=True,
        grid_assignment_method="cic",
    )
    random_part_stars.clear_all_emissions()
    without_shift_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        vel_shift=False,
        grid_assignment_method="cic",
    )
    random_part_stars.clear_all_emissions()

    # Get and print a seed for reproducibility
    seed = int(time.time())
    print(f"Seed for reproducibility: {seed} (use this to reproduce the test)")
    np.random.seed(seed)

    # Ensure that the overall flux is conserved, since we know it won't
    # be exactly the same at the edges due to the boundaries of the wavelength
    # array we can instead compare only the central part of the spectra
    # where the spectra overlap
    with_shift_sum = np.sum(with_shift_spec._lnu[100:-100])
    without_shift_sum = np.sum(without_shift_spec._lnu[100:-100])
    assert np.isclose(
        with_shift_sum,
        without_shift_sum,
        rtol=1e-6,
    ), (
        f"The total flux of the spectra is not conserved (seed: {seed}, "
        "with-without/without: "
        f"{(with_shift_sum - without_shift_sum) / without_shift_sum}, "
        f"with: {with_shift_sum}, without: {without_shift_sum})"
    )


def test_velocity_shift_applys_ngp(random_part_stars, nebular_emission_model):
    """Test the velocity shift of particle spectra."""
    # Compute the spectra with and without velocity shift
    with_shift_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        vel_shift=True,
        grid_assignment_method="ngp",
    )
    without_shift_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        vel_shift=False,
        grid_assignment_method="ngp",
    )

    # Get and print a seed for reproducibility
    seed = int(time.time())
    print(f"Seed for reproducibility: {seed} (use this to reproduce the test)")
    np.random.seed(seed)

    # Ensure that the spectra are different
    assert not np.allclose(with_shift_spec._lnu, without_shift_spec._lnu)


def test_velocity_shift_applys_ngp_on_model(
    random_part_stars,
    nebular_emission_model,
):
    """Test the velocity shift of particle spectra."""
    # Tell the model to use the velocity shift
    nebular_emission_model.set_vel_shift(True, set_all=True)

    # Compute the spectra with and without velocity shift
    with_shift_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="ngp",
    )
    random_part_stars.clear_all_emissions()
    nebular_emission_model.set_vel_shift(False, set_all=True)
    without_shift_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="ngp",
    )

    # Get and print a seed for reproducibility
    seed = int(time.time())
    print(f"Seed for reproducibility: {seed} (use this to reproduce the test)")
    np.random.seed(seed)

    # Ensure that the spectra are different
    assert not np.allclose(with_shift_spec._lnu, without_shift_spec._lnu)


def test_velocity_shift_conservation_ngp(
    random_part_stars, nebular_emission_model
):
    """Test the velocity shift of particle spectra."""
    # Compute the spectra with and without velocity shift
    with_shift_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        vel_shift=True,
        grid_assignment_method="ngp",
    )
    without_shift_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        vel_shift=False,
        grid_assignment_method="ngp",
    )

    # Get and print a seed for reproducibility
    seed = int(time.time())
    print(f"Seed for reproducibility: {seed} (use this to reproduce the test)")
    np.random.seed(seed)

    # Ensure that the overall flux is conserved, since we know it won't
    # be exactly the same at the edges due to the boundaries of the wavelength
    # array we can instead compare only the central part of the spectra
    # where the spectra overlap
    with_shift_sum = np.sum(with_shift_spec._lnu[100:-100])
    without_shift_sum = np.sum(without_shift_spec._lnu[100:-100])
    assert np.isclose(
        with_shift_sum,
        without_shift_sum,
        rtol=1e-6,
    ), (
        f"The total flux of the spectra is not conserved (seed: {seed}, "
        "with-without/without: "
        f"{(with_shift_sum - without_shift_sum) / without_shift_sum}, "
        f"with: {with_shift_sum}, without: {without_shift_sum})"
    )
