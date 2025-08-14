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

        assert model["attenuated"].dust_curve.slope == -1
        assert model["attenuated"].fixed_parameters["tau_v"] == 0.33

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
            fesc=0.0,
        )
        model_with_fesc = PacmanEmission(
            test_grid,
            tau_v=0.33,
            dust_curve=PowerLaw(slope=-1),
            fesc=0.2,
            fesc_ly_alpha=0.5,
        )

        # Generate spectra
        spec = random_part_stars.get_spectra(model)
        with_fesc_spec = random_part_stars.get_spectra(model_with_fesc)

        assert (
            spec.bolometric_luminosity < with_fesc_spec.bolometric_luminosity
        ), "fesc override had no effect"

        expected_keys = []
        for key in model._models.keys():
            if model._models[key].save:
                expected_keys.append(key)

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

        expected_keys = []
        for key in model._models.keys():
            if model._models[key].save:
                expected_keys.append(key)

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

        expected_keys = []
        for key in model._models.keys():
            if model._models[key].save:
                expected_keys.append(key)

        # Ensure the keys appear in the spectra
        for key in expected_keys:
            assert key in random_part_stars.spectra.keys()

    def test_pacman_spectra(self, particle_stars_B, pacman_emission_model):
        """Test the generation of PACMAN spectra."""
        # Set the fixed tau_vs
        particle_stars_B.tau_v = 0.1
        particle_stars_B.tau_v_birth = 0.3
        particle_stars_B.tau_v_ism = 0.1

        # Generate the PACMAN spectra
        pacman_spec = particle_stars_B.get_spectra(
            pacman_emission_model,
            grid_assignment_method="ngp",
        )
        particle_stars_B.clear_all_emissions()

        # Ensure that the PACMAN spectra is not empty
        assert np.sum(pacman_spec._lnu) > 0, (
            "The PACMAN spectra has no emission."
        )

    def test_emergent_is_combo(self, particle_stars_B, pacman_emission_model):
        """Test the generation of PACMAN spectra."""
        # Set the fixed tau_vs
        particle_stars_B.tau_v = 0.1
        particle_stars_B.tau_v_birth = 0.3
        particle_stars_B.tau_v_ism = 0.1
        particle_stars_B.fesc = 0.5

        # Generate the PACMAN spectra
        _ = particle_stars_B.get_spectra(
            pacman_emission_model,
            grid_assignment_method="ngp",
        )

        # Ensure emergent exists
        assert "emergent" in particle_stars_B.spectra, (
            "The emergent spectra is not in the particle stars."
        )

        # Ensure emergent is nonzero
        assert np.sum(particle_stars_B.spectra["emergent"]._lnu) > 0, (
            "The emergent spectra has no emission."
        )

        # Ensure escape is nonzero
        assert np.sum(particle_stars_B.spectra["escaped"]._lnu) > 0, (
            "The escaped spectra has no emission."
        )

        # Ensure emergent is equal to attenuated + escaped
        explicit_emergent = (
            particle_stars_B.spectra["attenuated"]
            + particle_stars_B.spectra["escaped"]
        )
        assert np.allclose(
            particle_stars_B.spectra["emergent"]._lnu,
            explicit_emergent._lnu,
        ), (
            "The emergent spectra is not equal to the sum of the "
            "attenuated and escaped spectra."
        )

    def test_intrinsic_is_combo(
        self,
        particle_stars_B,
        pacman_emission_model,
    ):
        """Test the generation of PACMAN spectra."""
        # Set the fixed tau_vs
        particle_stars_B.tau_v = 0.1
        particle_stars_B.tau_v_birth = 0.3
        particle_stars_B.tau_v_ism = 0.1
        particle_stars_B.fesc = 0.5

        # Generate the PACMAN spectra
        _ = particle_stars_B.get_spectra(
            pacman_emission_model,
            grid_assignment_method="ngp",
        )

        # Ensure intrinsic exists
        assert "intrinsic" in particle_stars_B.spectra, (
            "The intrinsic spectra is not in the particle stars."
        )

        # Ensure intrinsic is nonzero
        assert np.sum(particle_stars_B.spectra["intrinsic"]._lnu) > 0, (
            "The intrinsic spectra has no emission."
        )

        # Ensure escape is nonzero
        assert np.sum(particle_stars_B.spectra["escaped"]._lnu) > 0, (
            "The escaped spectra has no emission."
        )

        # Ensure intrinsic is equal to reprocessed + escaped
        explicit_intrinsic = (
            particle_stars_B.spectra["reprocessed"]
            + particle_stars_B.spectra["escaped"]
        )
        assert np.allclose(
            particle_stars_B.spectra["intrinsic"]._lnu,
            explicit_intrinsic._lnu,
        ), (
            "The intrinsic spectra is not equal to the sum of the "
            "reprocessed and escaped spectra."
        )

    def test_transmitted_escaped(
        self,
        particle_stars_B,
        pacman_emission_model,
    ):
        """Test the generation of PACMAN spectra."""
        # Set the fixed tau_vs
        particle_stars_B.tau_v = 0.1
        particle_stars_B.tau_v_birth = 0.3
        particle_stars_B.tau_v_ism = 0.1
        particle_stars_B.fesc = 0.5

        # Generate the PACMAN spectra
        _ = particle_stars_B.get_spectra(
            pacman_emission_model,
            grid_assignment_method="ngp",
        )

        # Ensure transmitted exists
        assert "transmitted" in particle_stars_B.spectra, (
            "The transmitted spectra is not in the particle stars."
        )

        # Ensure transmitted is nonzero
        assert np.sum(particle_stars_B.spectra["transmitted"]._lnu) > 0, (
            "The transmitted spectra has no emission."
        )

        # Ensure escaped exists
        assert "escaped" in particle_stars_B.spectra, (
            "The escaped spectra is not in the particle stars."
        )

        # Ensure escaped is nonzero
        assert np.sum(particle_stars_B.spectra["escaped"]._lnu) > 0, (
            "The escaped spectra has no emission."
        )

        # Ensure incident is nonzero
        assert np.sum(particle_stars_B.spectra["incident"]._lnu) > 0, (
            "The incident spectra has no emission."
        )

        # Ensure escaped equals 0.5 * incident
        assert np.allclose(
            particle_stars_B.spectra["escaped"]._lnu,
            0.5 * particle_stars_B.spectra["incident"]._lnu,
        ), (
            "The escaped spectra is not equal to 0.5 times the incident "
            "spectra."
        )

    def test_reprocessed_is_combo(
        self,
        particle_stars_B,
        pacman_emission_model,
    ):
        """Test the generation of PACMAN spectra."""
        # Set the fixed tau_vs
        particle_stars_B.tau_v = 0.1
        particle_stars_B.tau_v_birth = 0.3
        particle_stars_B.tau_v_ism = 0.1
        particle_stars_B.fesc = 0.5

        # Generate the PACMAN spectra
        _ = particle_stars_B.get_spectra(
            pacman_emission_model,
            grid_assignment_method="ngp",
        )

        # Ensure reprocessed exists
        assert "reprocessed" in particle_stars_B.spectra, (
            "The reprocessed spectra is not in the particle stars."
        )

        # Ensure reprocessed is nonzero
        assert np.sum(particle_stars_B.spectra["reprocessed"]._lnu) > 0, (
            "The reprocessed spectra has no emission."
        )

        # Ensure nebular exists
        assert "nebular" in particle_stars_B.spectra, (
            "The nebular spectra is not in the particle stars."
        )

        # Ensure nebular is nonzero
        assert np.sum(particle_stars_B.spectra["nebular"]._lnu) > 0, (
            "The nebular spectra has no emission."
        )

        # Ensure reprocessed is equal to nebular + transmitted
        explicit_reprocessed = (
            particle_stars_B.spectra["nebular"]
            + particle_stars_B.spectra["transmitted"]
        )
        assert np.allclose(
            particle_stars_B.spectra["reprocessed"]._lnu,
            explicit_reprocessed._lnu,
        ), (
            "The reprocessed spectra is not equal to the sum of the "
            "nebular and transmitted spectra."
        )


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

        assert model["attenuated"].dust_curve.slope == -1

    def test_missing_optical_depth(self, test_grid, random_part_stars):
        """Test the initialization of the ScreenEmission object."""
        # Set tau_v on the model, the stars has it already in the fixture, make
        # sure the version returned by get_param is from the model
        model = ScreenEmission(
            test_grid,
            tau_v=0.33,
            dust_curve=PowerLaw(slope=-1),
        )

        # Extract the tau_v and check it's the right one
        assert (
            get_param("tau_v", model["attenuated"], None, random_part_stars)
            == 0.33
        ), (
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
        _ = random_part_stars.get_spectra(model)

        expected_keys = []
        for key in model._models.keys():
            if model._models[key].save:
                expected_keys.append(key)

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

        assert model["old_attenuated"].dust_curve.__class__ == Calzetti2000
        assert (
            model["young_attenuated_ism"].dust_curve.__class__ == Calzetti2000
        )
        assert model["old_attenuated"].fixed_parameters["tau_v"] == 0.33
        assert model["young_attenuated_ism"].fixed_parameters["tau_v"] == 0.33

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

        expected_keys = []
        for key in model._models.keys():
            if model._models[key].save:
                expected_keys.append(key)

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

        expected_keys = []
        for key in model._models.keys():
            if model._models[key].save:
                expected_keys.append(key)

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

        assert model["young_attenuated_nebular"].dust_curve.slope == -1.3
        assert model["young_attenuated_ism"].dust_curve.slope == -0.7
        assert model["old_attenuated"].dust_curve.slope == -0.7
        assert model["young_transmitted"].fixed_parameters["fesc"] == 0.1
        assert model["old_transmitted"].fixed_parameters["fesc"] == 0.1

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

        assert spec.lnu.sum() < with_fesc_spec.lnu.sum(), (
            "fesc override had no effect"
        )

        expected_keys = []
        for key in model._models.keys():
            if model._models[key].save:
                expected_keys.append(key)

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

        expected_keys = []
        for key in model._models.keys():
            if model._models[key].save:
                expected_keys.append(key)

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

        expected_keys = []
        for key in model._models.keys():
            if model._models[key].save:
                expected_keys.append(key)

        # Ensure the keys appear in the spectra
        for key in expected_keys:
            assert key in random_part_stars.spectra.keys()

    @pytest.fixture(autouse=True)
    def setup(self, particle_stars_B, bimodal_pacman_emission_model):
        """Setup for all tests in this class."""
        # common setup for all tests
        self.ps = particle_stars_B
        self.model = bimodal_pacman_emission_model

        # fix optical depths & escape fractions
        self.ps.tau_v = 0.1
        self.ps.tau_v_birth = 0.3
        self.ps.tau_v_ism = 0.1
        self.ps.fesc = 0.5
        self.ps.fesc_ly_alpha = 0.2

        # generate spectra
        self.spec = self.ps.get_spectra(
            self.model,
            grid_assignment_method="ngp",
        )

    def test_incident_components(self):
        """Test the incident components of the spectra."""
        # incident spectra should include young, old, combined
        for key in ("young_incident", "old_incident", "incident"):
            assert key in self.ps.spectra, f"{key} missing"
            arr = self.ps.spectra[key]._lnu
            assert np.sum(arr) > 0, f"{key} has no emission"

    def test_transmitted_and_escaped_components(self):
        """Test the transmitted and escaped components of the spectra."""
        # transmitted + escaped for young, old, combined
        for kind in ("transmitted", "escaped"):
            # combined
            assert kind in self.ps.spectra, f"{kind} missing"
            assert np.sum(self.ps.spectra[kind]._lnu) > 0

            # young & old
            for pop in ("young", "old"):
                key = f"{pop}_{kind}"
                assert key in self.ps.spectra, f"{key} missing"
                assert np.sum(self.ps.spectra[key]._lnu) > 0

        # check escaped = fesc * incident for each pop & combined
        for pop in ("young", "old", ""):
            inc = self.ps.spectra[f"{pop + '_' if pop else ''}incident"]._lnu
            esc = self.ps.spectra[f"{pop + '_' if pop else ''}escaped"]._lnu
            assert np.allclose(esc, 0.5 * inc), (
                f"{pop or 'combined'} escaped != fesc * incident"
            )

    def test_nebular_and_reprocessed_components(self):
        """Test the nebular and reprocessed components of the spectra."""
        # nebular
        for key in ("young_nebular", "old_nebular", "nebular"):
            assert key in self.ps.spectra, f"{key} missing"
            assert np.sum(self.ps.spectra[key]._lnu) > 0

        # reprocessed = nebular + transmitted
        combined_reproc = (
            self.ps.spectra["nebular"] + self.ps.spectra["transmitted"]
        )
        assert np.allclose(
            self.ps.spectra["reprocessed"]._lnu,
            combined_reproc._lnu,
        ), "reprocessed != nebular + transmitted"

        # per-population
        for pop in ("young", "old"):
            neb = self.ps.spectra[f"{pop}_nebular"]
            trn = self.ps.spectra[f"{pop}_transmitted"]
            rep = self.ps.spectra[f"{pop}_reprocessed"]
            assert np.allclose(
                rep._lnu,
                (neb + trn)._lnu,
            ), f"{pop}_reprocessed != {pop}_nebular + {pop}_transmitted"

    def test_intrinsic_components(self):
        """Test the intrinsic components of the spectra."""
        # intrinsic exists and nonzero
        for key in ("young_intrinsic", "old_intrinsic", "intrinsic"):
            assert key in self.ps.spectra, f"{key} missing"
            assert np.sum(self.ps.spectra[key]._lnu) > 0

        # intrinsic = reprocessed + escaped
        combined_intr = (
            self.ps.spectra["reprocessed"] + self.ps.spectra["escaped"]
        )
        assert np.allclose(
            self.ps.spectra["intrinsic"]._lnu,
            combined_intr._lnu,
        ), "intrinsic != reprocessed + escaped"

    def test_attenuated_components(self):
        """Test the attenuated components of the spectra."""
        # attenuated combined
        assert "attenuated" in self.ps.spectra
        assert np.sum(self.ps.spectra["attenuated"]._lnu) > 0

        # young & old attenuated nebular/ism totals exist
        for pop in ("young",):
            for sub in ("attenuated_nebular", "attenuated_ism"):
                key = f"{pop}_{sub}"
                assert key in self.ps.spectra, f"{key} missing"
                assert np.sum(self.ps.spectra[key]._lnu) > 0

        # old_attenuated
        assert "old_attenuated" in self.ps.spectra
        assert np.sum(self.ps.spectra["old_attenuated"]._lnu) > 0

    def test_emergent_components(self):
        """Test the emergent components of the spectra."""
        # emergent = attenuated + escaped
        for key in ("young_emergent", "old_emergent", "emergent"):
            assert key in self.ps.spectra, f"{key} missing"
            assert np.sum(self.ps.spectra[key]._lnu) > 0

        combined_emergent = (
            self.ps.spectra["attenuated"] + self.ps.spectra["escaped"]
        )
        assert np.allclose(
            self.ps.spectra["emergent"]._lnu,
            combined_emergent._lnu,
        ), "emergent != attenuated + escaped"
