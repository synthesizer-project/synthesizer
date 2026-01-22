"""A suite of tests to ensure that the Extractor classes work as expected."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from unyt import Hz, erg, s

from synthesizer.emission_models.extractors.extractor import (
    DopplerShiftedParticleExtractor,
    Extractor,
    IntegratedDopplerShiftedParticleExtractor,
    IntegratedParametricExtractor,
    IntegratedParticleExtractor,
    ParticleExtractor,
)
from synthesizer.emissions import Sed
from synthesizer.exceptions import InconsistentArguments
from synthesizer.parametric import Stars as ParametricStars


# Mock Extractor implementation for testing abstract base class methods
class MockExtractor(Extractor):
    """Mock implementation of the Extractor class for testing purposes."""

    def generate_lnu(self, *args, **kwargs):
        """Mock implementation of generate_lnu method."""
        return "mock_lnu"

    def generate_line(self, *args, **kwargs):
        """Mock implementation of generate_line method."""
        return "mock_line"


# Tests for base Extractor class
def test_extractor_initialization(test_grid):
    """Test that the Extractor base class initializes correctly."""
    # Create a mock extractor
    extractor = MockExtractor(test_grid, "incident")

    # Check that the attributes are correctly set
    assert hasattr(extractor, "_emitter_attributes")
    assert hasattr(extractor, "_grid_axes")
    assert hasattr(extractor, "_axes_units")
    assert hasattr(extractor, "_weight_var")
    assert hasattr(extractor, "_spectra_grid")
    assert hasattr(extractor, "_grid_dims")
    assert hasattr(extractor, "_grid_naxes")
    assert hasattr(extractor, "_grid_nlam")
    assert hasattr(extractor, "_log_emitter_attr")
    assert extractor._grid == test_grid


def test_get_emitter_attrs(
    test_grid, particle_stars_A, nebular_emission_model
):
    """Test that get_emitter_attrs correctly extracts attributes."""
    # Create a mock extractor
    extractor = MockExtractor(test_grid, "incident")

    # Mock the _emitter_attributes, _axes_units, and
    # _log_emitter_attr attributes
    extractor._emitter_attributes = ["age", "metallicity"]
    extractor._axes_units = ["Myr", "dimensionless"]
    extractor._log_emitter_attr = [False, False]
    extractor._weight_var = "initial_mass"

    # Call get_emitter_attrs
    extracted, weight = extractor.get_emitter_attrs(
        particle_stars_A, nebular_emission_model, False
    )

    # Check that the extracted values and weight are correct
    assert len(extracted) == 2, (
        "Incorrect number of extracted attributes "
        f"(expected 2, got {len(extracted)})"
    )
    assert np.array_equal(
        extracted[0], particle_stars_A.ages.to("Myr").value
    ), (
        f"Ages not equal (extracted: {extracted[0]}, "
        f"expected: {particle_stars_A.ages.to('Myr').value})"
    )
    assert np.array_equal(extracted[1], particle_stars_A.metallicities), (
        f"Metallicities not equal (extracted: {extracted[1]}, "
        f"expected: {particle_stars_A.metallicities})"
    )
    assert np.array_equal(weight, particle_stars_A.initial_masses.value), (
        f"Weights not equal (extracted: {weight}, "
        f"expected: {particle_stars_A.initial_masses.value})"
    )


def test_check_emitter_attrs(test_grid, particle_stars_A):
    """Test that check_emitter_attrs identifies out-of-bounds attributes."""
    # Create a mock extractor
    extractor = MockExtractor(test_grid, "incident")

    # Create mock grid axes with known bounds
    mock_axis1 = np.array(
        [0.5, 1.5]
    )  # Some values of extracted_attrs[0] will be outside bounds
    mock_axis2 = np.array(
        [0.01, 0.02]
    )  # Some values of extracted_attrs[1] will be outside bounds

    extractor._grid_axes = (mock_axis1, mock_axis2)

    # Create mock extracted attributes
    extracted_attrs = (
        np.array([1.0, 2.0, 3.0]),  # Last two outside bounds of mock_axis1
        np.array(
            [0.01, 0.02, 0.03]
        ),  # Last value outside bounds of mock_axis2
    )

    # Patch warn function to check it's called
    with patch(
        "synthesizer.emission_models.extractors.extractor.warn"
    ) as mock_warn:
        extractor.check_emitter_attrs(particle_stars_A, extracted_attrs)

        # Check that warn was called with a message about attributes
        # outside bounds
        mock_warn.assert_called_once()
        assert "outside the grid axes" in mock_warn.call_args[0][0]


# Tests for IntegratedParticleExtractor
@patch(
    "synthesizer.emission_models.extractors.extractor.compute_integrated_sed"
)
def test_integrated_particle_generate_lnu(
    mock_compute_integrated_sed,
    test_grid,
    particle_stars_A,
    nebular_emission_model,
):
    """Test that IntegratedParticleExtractor.generate_lnu works correctly."""
    # Create an IntegratedParticleExtractor
    extractor = IntegratedParticleExtractor(test_grid, "incident")

    # Mock the get_emitter_attrs method to return known values
    extractor.get_emitter_attrs = MagicMock(
        return_value=(("mock_extracted",), "mock_weight")
    )

    # Set up the compute_integrated_sed mock to return a known spectrum
    mock_spectrum = np.ones(test_grid.nlam)
    mock_compute_integrated_sed.return_value = (mock_spectrum, None)

    # Call generate_lnu
    result = extractor.generate_lnu(
        particle_stars_A, nebular_emission_model, None, None, "cic", 1, False
    )

    # Check that compute_integrated_sed was called with the right parameters
    mock_compute_integrated_sed.assert_called_once()
    args = mock_compute_integrated_sed.call_args[0]
    assert args[0] is extractor._spectra_grid  # spectra_grid
    assert args[1] is extractor._grid_axes  # grid_axes
    assert args[2] == ("mock_extracted",)  # extracted
    assert args[3] == "mock_weight"  # weight

    # Ensure the output sed is "integrated" (has ndim == 1)
    assert result.lnu.ndim == 1, f"Expected 1D lnu, got {result.lnu.ndim}"

    # Check that the result is a Sed object with the right values
    assert isinstance(result, Sed)
    assert np.array_equal(result.lnu, mock_spectrum * erg / s / Hz)


def test_integrated_particle_empty_case(test_grid, nebular_emission_model):
    """Test that generate_lnu handles empty emitters correctly."""
    # Create an IntegratedParticleExtractor
    extractor = IntegratedParticleExtractor(test_grid, "incident")

    # Create a mock emitter with no particles
    mock_emitter = MagicMock()
    mock_emitter.nparticles = 0

    # Patch warn function to check it's called
    with patch(
        "synthesizer.emission_models.extractors.extractor.warn"
    ) as mock_warn:
        # Call generate_lnu
        result = extractor.generate_lnu(
            mock_emitter,
            nebular_emission_model,
            None,
            None,
            "cic",
            1,
            False,
        )

        # Check that warn was called with a message about no particles
        mock_warn.assert_called_once()
        assert "no particles" in mock_warn.call_args[0][0]

        # Check that an empty Sed is returned
        assert isinstance(result, Sed)
        assert np.array_equal(
            result.lnu, np.zeros(test_grid.nlam) * erg / s / Hz
        )


def test_integrated_particle_masked_empty_case(
    test_grid, particle_stars_A, nebular_emission_model
):
    """Test that generate_lnu handles masked empty emitters correctly."""
    # Create an IntegratedParticleExtractor
    extractor = IntegratedParticleExtractor(test_grid, "incident")

    # Create a mask that filters out all particles
    mask = np.zeros(len(particle_stars_A.ages), dtype=bool)

    # Patch warn function to check it's called
    with patch(
        "synthesizer.emission_models.extractors.extractor.warn"
    ) as mock_warn:
        # Call generate_lnu
        result = extractor.generate_lnu(
            particle_stars_A,
            nebular_emission_model,
            mask,
            None,
            "cic",
            1,
            False,
        )

        # Check that warn was called with a message about filtered particles
        mock_warn.assert_called_once()
        assert "filtered out all particles" in mock_warn.call_args[0][0]

        # Check that an empty Sed is returned
        assert isinstance(result, Sed)
        assert np.array_equal(
            result.lnu, np.zeros(test_grid.nlam) * erg / s / Hz
        )


@patch(
    "synthesizer.emission_models.extractors."
    "extractor.compute_part_seds_with_vel_shift",
)
def test_doppler_shifted_generate_lnu(
    mock_compute_part_seds_with_vel_shift,
    test_grid,
    random_part_stars,
    nebular_emission_model,
):
    """Test that doppler shifted generate_lnu works correctly."""
    # Create a DopplerShiftedParticleExtractor
    extractor = DopplerShiftedParticleExtractor(test_grid, "incident")

    # Mock the get_emitter_attrs method to return known values
    extractor.get_emitter_attrs = MagicMock(
        return_value=(("mock_extracted",), "mock_weight")
    )

    # Set up the compute_part_seds_with_vel_shift mock to return
    # a known spectrum
    n_particles = len(random_part_stars.ages)
    mock_spectrum = np.ones((n_particles, test_grid.nlam))
    mock_compute_part_seds_with_vel_shift.return_value = (
        mock_spectrum,
        mock_spectrum,
    )

    # Call generate_lnu
    result, _ = extractor.generate_lnu(
        random_part_stars, nebular_emission_model, None, None, "cic", 1, False
    )

    # Check that compute_part_seds_with_vel_shift was called with the
    # right parameters
    mock_compute_part_seds_with_vel_shift.assert_called_once()
    args = mock_compute_part_seds_with_vel_shift.call_args[0]
    assert args[0] is extractor._spectra_grid  # spectra_grid
    assert args[2] is extractor._grid_axes  # grid_axes
    assert args[3] == ("mock_extracted",)  # extracted
    assert args[4] == "mock_weight"  # weight
    assert args[5] is random_part_stars._velocities  # velocities

    # Ensure the output sed is "per-particle" (has ndim == 2)
    assert result.lnu.ndim == 2, f"Expected 2D lnu, got {result.lnu.ndim}"

    # Check that the result is a Sed object with the right values
    assert isinstance(result, Sed)
    assert np.array_equal(result.lnu, mock_spectrum * erg / s / Hz)


def test_doppler_shifted_no_velocities(
    test_grid, particle_stars_A, nebular_emission_model
):
    """Test that generate_lnu raises handles missing velocities."""
    # Create a DopplerShiftedParticleExtractor
    extractor = DopplerShiftedParticleExtractor(test_grid, "incident")

    # Ensure particle_stars_A has no velocities
    particle_stars_A._velocities = None

    # Call generate_lnu and check that it raises an
    # InconsistentArguments exception
    with pytest.raises(InconsistentArguments) as excinfo:
        extractor.generate_lnu(
            particle_stars_A,
            nebular_emission_model,
            None,
            None,
            "cic",
            1,
            False,
        )

    # Check the exception message
    assert "no star velocities provided" in str(excinfo.value)


@patch(
    "synthesizer.emission_models.extractors."
    "extractor.compute_part_seds_with_vel_shift"
)
def test_integrated_doppler_shifted_generate_lnu(
    mock_compute_part_seds_with_vel_shift,
    test_grid,
    random_part_stars,
    nebular_emission_model,
):
    """Test that doppler shifted generate_lnu works correctly."""
    # Create an IntegratedDopplerShiftedParticleExtractor
    extractor = IntegratedDopplerShiftedParticleExtractor(
        test_grid, "incident"
    )

    # Mock the get_emitter_attrs method to return known values
    extractor.get_emitter_attrs = MagicMock(
        return_value=(("mock_extracted",), "mock_weight")
    )

    # Set up the compute_part_seds_with_vel_shift mock to return a
    # known spectrum
    mock_spectrum = np.ones(test_grid.nlam)
    mock_compute_part_seds_with_vel_shift.return_value = (
        mock_spectrum,
        mock_spectrum,
    )

    # Call generate_lnu
    result = extractor.generate_lnu(
        random_part_stars,
        nebular_emission_model,
        None,
        None,
        "cic",
        1,
        False,
    )

    # Check that compute_part_seds_with_vel_shift was called with the
    # right parameters
    mock_compute_part_seds_with_vel_shift.assert_called_once()
    args = mock_compute_part_seds_with_vel_shift.call_args[0]
    assert args[0] is extractor._spectra_grid  # spectra_grid
    assert args[2] is extractor._grid_axes  # grid_axes
    assert args[3] == ("mock_extracted",)  # extracted
    assert args[4] == "mock_weight"  # weight
    assert args[5] is random_part_stars._velocities  # velocities

    # Ensure the output sed is "integrated" (has ndim == 1)
    assert result.lnu.ndim == 1, f"Expected 1D lnu, got {result.lnu.ndim}"

    # Check that the result is a Sed object with the right values
    assert isinstance(result, Sed), f"Expected Sed, got {type(result)}"


@patch(
    "synthesizer.emission_models.extractors.extractor.compute_particle_seds"
)
def test_particle_generate_lnu(
    mock_compute_particle_seds,
    test_grid,
    particle_stars_A,
    nebular_emission_model,
):
    """Test that ParticleExtractor.generate_lnu works correctly."""
    # Create a ParticleExtractor
    extractor = ParticleExtractor(test_grid, "incident")

    # Mock the get_emitter_attrs method to return known values
    extractor.get_emitter_attrs = MagicMock(
        return_value=(("mock_extracted",), "mock_weight")
    )

    # Set up the compute_particle_seds mock to return a known spectrum
    n_particles = len(particle_stars_A.ages)
    mock_spectrum = np.ones((n_particles, test_grid.nlam))
    mock_int_spectrum = np.ones(test_grid.nlam)
    mock_compute_particle_seds.return_value = (
        mock_spectrum,
        mock_int_spectrum,
    )

    # Call generate_lnu
    part_spec, spec = extractor.generate_lnu(
        particle_stars_A, nebular_emission_model, None, None, "cic", 1, False
    )

    # Check that compute_particle_seds was called with the right parameters
    mock_compute_particle_seds.assert_called_once()
    args = mock_compute_particle_seds.call_args[0]
    assert args[0] is extractor._spectra_grid  # spectra_grid
    assert args[1] is extractor._grid_axes  # grid_axes
    assert args[2] == ("mock_extracted",)  # extracted
    assert args[3] == "mock_weight"  # weight

    # Ensure the output sed is "per-particle" (has ndim == 2)
    assert part_spec.lnu.ndim == 2, (
        f"Expected 2D lnu, got {part_spec.lnu.ndim}"
    )
    assert spec.lnu.ndim == 1, f"Expected 1D lnu, got {spec.lnu.ndim}"

    # Check that the result is a Sed object with the right values
    assert isinstance(part_spec, Sed)
    assert isinstance(spec, Sed)


def test_integrated_parametric_generate_lnu(test_grid, nebular_emission_model):
    """Test that IntegratedParametricExtractor.generate_lnu works correctly."""
    # Create an IntegratedParametricExtractor
    extractor = IntegratedParametricExtractor(test_grid, "incident")

    # Create a mock parametric stars object
    mock_parametric_stars = MagicMock(spec=ParametricStars)
    mock_parametric_stars.get_mask.return_value = np.array(
        [[True, False], [True, True], [True, False]]
    )

    # Create mock SFZH and spectra grid
    mock_sfzh = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    mock_parametric_stars.sfzh = mock_sfzh

    # Set a mock spectra grid with known values
    mock_spectra = np.ones((*mock_sfzh.shape, 1))
    extractor._spectra_grid = mock_spectra
    extractor._grid_naxes = 2

    # Call generate_lnu
    with patch.object(nebular_emission_model, "_lam", np.array([1.0])):
        result = extractor.generate_lnu(
            mock_parametric_stars,
            nebular_emission_model,
            None,
            None,
            "cic",
            1,
            False,
        )

    # Check that get_mask was called with the right parameters
    mock_parametric_stars.get_mask.assert_called_once_with(
        "sfzh", 0, ">", mask=None
    )

    # Check that the result is a Sed object with the right values
    assert isinstance(result, Sed), f"Expected Sed, got {type(result)}"

    # Ensure the output sed is "integrated" (has ndim == 1)
    assert result.lnu.ndim == 1, f"Expected 1D lnu, got {result.lnu.ndim}"

    # Ensure we have the correct "spectrum" for the parametric stars, this
    # can be check by summing the sfzh values for the particles that are
    # included in the mask
    expected = mock_sfzh.sum()
    assert np.array_equal(result.lnu.sum(), expected * erg / s / Hz), (
        f"Expected {expected * erg / s / Hz}, got {result.lnu.sum()}"
        f"{result.lnu} {mock_sfzh}"
    )
