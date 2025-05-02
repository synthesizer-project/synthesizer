"""Test suite for premade emission models."""

import numpy as np
import pytest
from unyt import K

from synthesizer import exceptions
from synthesizer.emission_models import (
    BimodalPacmanEmission,
    CharlotFall2000,
    PacmanEmission,
    ScreenEmission,
)
from synthesizer.emission_models.dust.emission import Greybody
from synthesizer.emission_models.transformers import Calzetti2000, PowerLaw
from synthesizer.emission_models.utils import get_param


class TestPacmanEmission:
    """Test suite for PacmanEmission."""

    def test_init(self, test_grid):
        """Test the initialization of the PacmanEmission object."""
        # Define the emission model
        model = PacmanEmission(
            test_grid,
            tau_v=0.33,
            dust_curve=PowerLaw(slope=-1),
            fesc=0.1,
            fesc_ly_alpha=0.5,
        )

        assert model.dust_curve.slope == -1
        assert model._fesc == 0.1
        assert model._fesc_ly_alpha == 0.5
        assert model._tau_v == 0.33

    def test_missing_optical_depth(self, test_grid, random_part_stars):
        """Test the initialization of the PacmanEmission object."""
        # Define the emission model
        model = PacmanEmission(
            test_grid,
            dust_curve=PowerLaw(slope=-1),
            fesc=0.1,
            fesc_ly_alpha=0.5,
        )

        # Try to generate spectra without setting tau_v on the model or
        # stars
        random_part_stars.tau_v = None
        with pytest.raises(exceptions.MissingAttribute):
            random_part_stars.get_spectra(model)

    def test_pacman_with_no_fesc(self, test_grid, random_part_stars):
        """Test the initialization of the PacmanEmission object."""
        # Define the emission model
        model = PacmanEmission(
            test_grid,
            tau_v=0.33,
            dust_curve=PowerLaw(slope=-1),
            fesc_ly_alpha=0.5,
        )

        # Generate spectra
        spec = random_part_stars.get_spectra(model)

        # Now check that passing an override to get_spectra works
        with_fesc_spec = random_part_stars.get_spectra(model, fesc=0.1)

        assert spec.lnu.sum() > with_fesc_spec.lnu.sum(), (
            "fesc override had no effect"
        )

        expected_keys = [
            "full_transmitted",
            "nebular_continuum",
            "incident",
            "escaped",
            "nebular_line",
            "nebular",
            "transmitted",
            "reprocessed",
            "intrinsic",
            "attenuated",
            "emergent",
        ]

        # Ensure the keys appear in the spectra
        for key in expected_keys:
            assert key in random_part_stars.spectra.keys()

    def test_pacman_with_no_fesc_ly_alpha(self, test_grid, random_part_stars):
        """Test the initialization of the PacmanEmission object."""
        # Define the emission model
        model = PacmanEmission(
            test_grid,
            tau_v=0.33,
            dust_curve=PowerLaw(slope=-1),
            fesc=0.1,
        )

        # Generate spectra
        spec = random_part_stars.get_spectra(model)

        # Ensure the result is a finite number
        assert not np.isnan(spec.lnu.sum())

        expected_keys = [
            "full_transmitted",
            "nebular_continuum",
            "incident",
            "escaped",
            "nebular_line",
            "nebular",
            "transmitted",
            "reprocessed",
            "intrinsic",
            "attenuated",
            "emergent",
        ]

        # Ensure the keys appear in the spectra
        for key in expected_keys:
            assert key in random_part_stars.spectra.keys()

    def test_pacman_with_dust_emission(self, test_grid, random_part_stars):
        """Test the initialization of the PacmanEmission object."""
        # Define the emission model
        model = PacmanEmission(
            test_grid,
            tau_v=0.33,
            dust_curve=PowerLaw(slope=-1),
            fesc=0.1,
            fesc_ly_alpha=0.5,
            dust_emission=Greybody(temperature=10**4 * K, emissivity=2),
        )

        # Generate get_spectra
        spec = random_part_stars.get_spectra(model)

        # Ensure the result is a finite number
        assert not np.isnan(spec.lnu.sum())

        expected_keys = [
            "full_transmitted",
            "nebular_continuum",
            "incident",
            "escaped",
            "transmitted",
            "nebular_line",
            "nebular",
            "reprocessed",
            "intrinsic",
            "attenuated",
            "dust_emission",
            "emergent",
            "total",
        ]

        # Ensure the keys appear in the spectra
        for key in expected_keys:
            assert key in random_part_stars.spectra.keys()


class TestScreenEmission:
    """Test suite for ScreenEmission."""

    def test_init(self, test_grid):
        """Test the initialization of the ScreenEmission object."""
        # Define the emission model
        model = ScreenEmission(
            test_grid,
            tau_v=0.33,
            dust_curve=PowerLaw(slope=-1),
        )

        assert model.dust_curve.slope == -1
        assert model._tau_v == 0.33

    def test_missing_optical_depth(self, test_grid, random_part_stars):
        """Test the initialization of the ScreenEmission object."""
        # Define the emission model without tau_v (this must be provided on
        # a ScreenEmission object)
        with pytest.raises(TypeError):
            model = ScreenEmission(
                test_grid,
                dust_curve=PowerLaw(slope=-1),
            )

        # Set tau_v on the model, the stars has it already in the fixture, make
        # sure the version returned by get_param is from the model
        model = ScreenEmission(
            test_grid,
            tau_v=0.33,
            dust_curve=PowerLaw(slope=-1),
        )

        # Extract the tau_v and check it's the right one
        assert get_param("tau_v", model, None, random_part_stars) == 0.33, (
            "get_param returned the wrong tau_v "
            f"(model.tau_v = {model.tau_v}, "
            f"stars.tau_v = {random_part_stars.tau_v})"
        )

    def test_screen(self, test_grid, random_part_stars):
        """Test the initialization of the ScreenEmission object."""
        # Define the emission model
        model = ScreenEmission(
            test_grid,
            tau_v=0.33,
            dust_curve=PowerLaw(slope=-1),
            fesc=0.1,
            fesc_ly_alpha=0.5,
        )

        # Generate spectra
        spec = random_part_stars.get_spectra(model)

        # Now check that passing an override to get_spectra works
        with_fesc_spec = random_part_stars.get_spectra(model, fesc=0.1)

        assert spec.lnu.sum() > with_fesc_spec.lnu.sum(), (
            "fesc override had no effect"
        )

        expected_keys = [
            "full_transmitted",
            "nebular_continuum",
            "incident",
            "escaped",
            "nebular_line",
            "nebular",
            "transmitted",
            "reprocessed",
            "intrinsic",
            "attenuated",
            "emergent",
        ]

        # Ensure the keys appear in the spectra
        for key in expected_keys:
            assert key in random_part_stars.spectra.keys()


class TestCharlotFallEmission:
    """Test suite for CharlotFall2000."""

    def test_init(self, test_grid):
        """Test the initialization of the CharlotFall2000 object."""
        # Define the emission model
        model = CharlotFall2000(
            test_grid,
            tau_v_ism=0.33,
            tau_v_birth=0.33,
        )

        assert model.tau_v_ism == 0.33
        assert model.tau_v_birth == 0.33
        assert model._dust_curve_ism.__class__ == Calzetti2000
        assert model._dust_curve_birth.__class__ == Calzetti2000

    def test_missing_optical_depth(self, test_grid, random_part_stars):
        """Test the initialization of the CharlotFall2000 object."""
        with pytest.raises(TypeError):
            _ = CharlotFall2000(
                test_grid,
            )
        with pytest.raises(TypeError):
            _ = CharlotFall2000(
                test_grid,
                tau_v_ism=0.33,
            )

    def test_charlot_fall_with_no_dust_emission(
        self,
        test_grid,
        random_part_stars,
    ):
        """Test the initialization of the CharlotFall2000 object."""
        # Define the emission model
        model = CharlotFall2000(
            test_grid,
            tau_v_ism=0.33,
            tau_v_birth=0.33,
        )

        # Generate spectra
        spec = random_part_stars.get_spectra(model)

        # Ensure the result is a finite number
        assert not np.isnan(spec.lnu.sum())

        expected_keys = [
            "full_old_transmitted",
            "old_nebular_continuum",
            "young_nebular_continuum",
            "full_young_transmitted",
            "old_incident",
            "young_incident",
            "old_escaped",
            "old_linecont",
            "old_nebular",
            "old_transmitted",
            "old_reprocessed",
            "old_attenuated",
            "old_emergent",
            "young_linecont",
            "young_nebular",
            "young_transmitted",
            "young_reprocessed",
            "young_attenuated_nebular",
            "young_attenuated",
            "young_escaped",
            "young_emergent",
            "escaped",
            "young_intrinsic",
            "old_intrinsic",
            "intrinsic",
            "nebular",
            "young_attenuated_ism",
            "attenuated",
            "transmitted",
            "reprocessed",
            "incident",
            "emergent",
        ]

        # Ensure the keys appear in the spectra
        for key in expected_keys:
            assert key in random_part_stars.spectra.keys()

    def test_charlot_fall_with_dust_emission(
        self, test_grid, random_part_stars
    ):
        """Test the initialization of the CharlotFall2000 object."""
        # Define the emission model
        model = CharlotFall2000(
            test_grid,
            tau_v_ism=0.33,
            tau_v_birth=0.33,
            dust_emission_ism=Greybody(temperature=10**4 * K, emissivity=2),
            dust_emission_birth=Greybody(temperature=10**4 * K, emissivity=2),
        )

        # Generate get_spectra
        spec = random_part_stars.get_spectra(model)

        # Ensure the result is a finite number
        assert not np.isnan(spec.lnu.sum())

        expected_keys = [
            "full_old_transmitted",
            "old_nebular_continuum",
            "young_nebular_continuum",
            "full_young_transmitted",
            "young_incident",
            "old_incident",
            "old_escaped",
            "old_linecont",
            "old_nebular",
            "old_transmitted",
            "old_reprocessed",
            "old_attenuated",
            "old_emergent",
            "old_intrinsic",
            "old_dust_emission",
            "old_total",
            "young_linecont",
            "young_nebular",
            "young_transmitted",
            "young_reprocessed",
            "young_attenuated_nebular",
            "young_escaped",
            "young_intrinsic",
            "young_dust_emission_birth",
            "young_attenuated_ism",
            "young_dust_emission_ism",
            "young_dust_emission",
            "young_attenuated",
            "young_emergent",
            "young_total",
            "emergent",
            "attenuated",
            "dust_emission",
            "incident",
            "transmitted",
            "escaped",
            "intrinsic",
            "reprocessed",
            "nebular",
            "total",
        ]

        # Ensure the keys appear in the spectra
        for key in expected_keys:
            assert key in random_part_stars.spectra.keys()


class TestBimodalPacmanEmission:
    """Test suite for BimodalPacmanEmission."""

    def test_init(self, test_grid):
        """Test the initialization of the BimodalPacmanEmission object."""
        # Define the emission model
        model = BimodalPacmanEmission(
            test_grid,
            dust_curve_ism=PowerLaw(slope=-0.7),
            dust_curve_birth=PowerLaw(slope=-1.3),
            fesc=0.1,
            fesc_ly_alpha=0.5,
        )

        assert model._dust_curve_ism.slope == -0.7
        assert model._dust_curve_birth.slope == -1.3
        assert model._fesc == 0.1
        assert model._fesc_ly_alpha == 0.5

    def test_missing_optical_depth(self, test_grid, random_part_stars):
        """Test the initialization of the BimodalPacmanEmission object."""
        # Define the emission model
        model = BimodalPacmanEmission(
            test_grid,
            dust_curve_ism=PowerLaw(slope=-0.7),
            dust_curve_birth=PowerLaw(slope=-1.3),
            fesc=0.1,
            fesc_ly_alpha=0.5,
        )

        # Try to generate spectra without setting tau_v on the model or
        # stars
        with pytest.raises(exceptions.MissingAttribute):
            random_part_stars.get_spectra(model)

    def test_bimodal_pacman_with_no_fesc(self, test_grid, random_part_stars):
        """Test the initialization of the BimodalPacmanEmission object."""
        # Define the emission model
        model = BimodalPacmanEmission(
            test_grid,
            dust_curve_ism=PowerLaw(slope=-0.7),
            dust_curve_birth=PowerLaw(slope=-1.3),
            fesc_ly_alpha=0.5,
            tau_v_ism=0.33,
            tau_v_birth=0.33,
        )

        # Generate spectra
        spec = random_part_stars.get_spectra(model)

        # Now check that passing an override to get_spectra works
        with_fesc_spec = random_part_stars.get_spectra(model, fesc=0.1)

        assert spec.lnu.sum() > with_fesc_spec.lnu.sum(), (
            "fesc override had no effect"
        )

        expected_keys = [
            "full_old_transmitted",
            "old_nebular_continuum",
            "young_nebular_continuum",
            "full_young_transmitted",
            "old_incident",
            "young_incident",
            "old_transmitted",
            "old_linecont",
            "old_nebular",
            "old_reprocessed",
            "old_attenuated",
            "old_escaped",
            "old_emergent",
            "young_linecont",
            "young_nebular",
            "young_transmitted",
            "young_reprocessed",
            "young_attenuated_nebular",
            "young_attenuated",
            "young_escaped",
            "young_emergent",
            "old_intrinsic",
            "transmitted",
            "nebular",
            "young_intrinsic",
            "intrinsic",
            "reprocessed",
            "attenuated",
            "escaped",
            "young_attenuated_ism",
            "incident",
            "emergent",
        ]

        # Ensure the keys appear in the spectra
        for key in expected_keys:
            assert key in random_part_stars.spectra.keys()

    def test_bimodal_pacman_with_no_fesc_ly_alpha(
        self, test_grid, random_part_stars
    ):
        """Test the initialization of the BimodalPacmanEmission object."""
        # Define the emission model
        model = BimodalPacmanEmission(
            test_grid,
            dust_curve_ism=PowerLaw(slope=-0.7),
            dust_curve_birth=PowerLaw(slope=-1.3),
            fesc=0.1,
            tau_v_ism=0.33,
            tau_v_birth=0.33,
        )

        # Generate spectra
        spec = random_part_stars.get_spectra(model)

        # Ensure the result is a finite number
        assert not np.isnan(spec.lnu.sum())

        expected_keys = [
            "full_old_transmitted",
            "old_nebular_continuum",
            "young_nebular_continuum",
            "full_young_transmitted",
            "old_incident",
            "young_incident",
            "old_transmitted",
            "old_linecont",
            "old_nebular",
            "old_reprocessed",
            "old_attenuated",
            "old_escaped",
            "old_emergent",
            "young_linecont",
            "young_nebular",
            "young_transmitted",
            "young_reprocessed",
            "young_attenuated_nebular",
            "young_attenuated",
            "young_escaped",
            "young_emergent",
            "old_intrinsic",
            "transmitted",
            "nebular",
            "young_intrinsic",
            "intrinsic",
            "reprocessed",
            "attenuated",
            "escaped",
            "young_attenuated_ism",
            "incident",
            "emergent",
        ]

        # Ensure the keys appear in the spectra
        for key in expected_keys:
            assert key in random_part_stars.spectra.keys()

    def test_bimodal_pacman_with_dust_emission(
        self, test_grid, random_part_stars
    ):
        """Test the initialization of the BimodalPacmanEmission object."""
        # Define the emission model
        model = BimodalPacmanEmission(
            test_grid,
            dust_curve_ism=PowerLaw(slope=-0.7),
            dust_curve_birth=PowerLaw(slope=-1.3),
            fesc=0.1,
            fesc_ly_alpha=0.5,
            tau_v_ism=0.33,
            tau_v_birth=0.33,
            dust_emission_ism=Greybody(temperature=10**4 * K, emissivity=2),
            dust_emission_birth=Greybody(temperature=10**4 * K, emissivity=2),
        )

        # Generate get_spectra
        spec = random_part_stars.get_spectra(model)

        # Ensure the result is a finite number
        assert not np.isnan(spec.lnu.sum())

        expected_keys = [
            "full_old_transmitted",
            "old_nebular_continuum",
            "young_nebular_continuum",
            "full_young_transmitted",
            "old_incident",
            "young_incident",
            "old_transmitted",
            "old_linecont",
            "old_nebular",
            "old_reprocessed",
            "old_attenuated",
            "old_escaped",
            "old_emergent",
            "young_linecont",
            "young_nebular",
            "young_transmitted",
            "young_reprocessed",
            "young_attenuated_nebular",
            "young_attenuated",
            "young_escaped",
            "young_emergent",
            "old_intrinsic",
            "transmitted",
            "nebular",
            "young_intrinsic",
            "intrinsic",
            "reprocessed",
            "attenuated",
            "escaped",
            "young_attenuated_ism",
            "incident",
            "emergent",
            "dust_emission",
        ]

        # Ensure the keys appear in the spectra
        for key in expected_keys:
            assert key in random_part_stars.spectra.keys()
