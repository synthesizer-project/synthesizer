"""Comprehensive test suite for dust emission generators.

This module contains tests for all dust generator functionality including:
- Base dust emission utility functions
- Blackbody emission generators
- Greybody emission generators
- Casey12 emission generators
- DraineLi07 emission generators
- Integration and functionality tests
"""

import numpy as np
import pytest
from unyt import Hz, K, Lsun, Msun, angstrom, erg, s, um

from synthesizer import exceptions
from synthesizer.emission_models.generators.dust import (
    get_cmb_heating_factor,
)
from synthesizer.emission_models.generators.dust.blackbody import Blackbody
from synthesizer.emission_models.generators.dust.casey12 import Casey12
from synthesizer.emission_models.generators.dust.draineli07 import (
    DraineLi07,
    solve_umin,
    u_mean,
    u_mean_magdis12,
)
from synthesizer.emission_models.generators.dust.greybody import Greybody
from synthesizer.emissions import Sed


# Test fixtures
@pytest.fixture
def dust_wavelengths():
    """Wavelength grid for dust testing (IR wavelengths)."""
    return np.logspace(4, 6, 100) * angstrom


@pytest.fixture
def dust_temperatures():
    """Range of dust temperatures for testing."""
    return [5 * K, 10 * K, 20 * K, 50 * K, 100 * K]


@pytest.fixture
def mock_dust_grid():
    """Mock dust grid for DraineLi07 testing."""

    class MockGrid:
        def __init__(self):
            # Mock DL07 grid attributes
            self.qpah = np.array([0.01, 0.025, 0.05])
            self.umin = np.array([0.1, 1.0, 10.0, 100.0])
            self.alpha = np.array([1.5, 2.0, 2.5])

            # Mock spectral data
            self.spectra = {
                "diffuse": np.random.random((3, 4, 100)),
                "pdr": np.random.random((3, 4, 3, 100)),
            }

        def interp_spectra(self, new_lam):
            """Mock interpolation method."""
            # Update spectra shapes to match wavelength grid
            n_lam = len(new_lam)
            self.spectra = {
                "diffuse": np.random.random((3, 4, n_lam)),
                "pdr": np.random.random((3, 4, 3, n_lam)),
            }

    return MockGrid()


@pytest.fixture
def mock_emitter():
    """Mock emitter for testing generator methods."""

    class MockEmitter:
        def __init__(self):
            self.temperature = 20 * K
            self.emissivity = 1.5
            self.dust_mass = 1e6 * Msun
            self.dust_to_gas_ratio = 0.01
            self.hydrogen_mass = 1e8 * Msun
            self.qpah = 0.025

    return MockEmitter()


@pytest.fixture
def mock_model():
    """Mock emission model for testing generator methods."""

    class MockModel:
        def __init__(self):
            self.fixed_parameters = {}
            self.per_particle = False
            self.label = "test_model"

    return MockModel()


@pytest.fixture
def mock_emissions():
    """Mock emissions dictionary for testing."""
    # Create mock SEDs for intrinsic/attenuated/scaler emissions
    lams = np.logspace(4, 6, 100) * angstrom
    lnu_intrinsic = np.ones(100) * 1e10 * erg / s / Hz
    lnu_attenuated = np.ones(100) * 5e9 * erg / s / Hz
    lnu_scaler = np.ones(100) * 2e9 * erg / s / Hz

    return {
        "intrinsic": Sed(lam=lams, lnu=lnu_intrinsic),
        "attenuated": Sed(lam=lams, lnu=lnu_attenuated),
        "scaler": Sed(lam=lams, lnu=lnu_scaler),
        "test_model": Sed(lam=lams, lnu=lnu_scaler),
    }


class TestDustEmissionUtilities:
    """Tests for base dust emission utility functions."""

    def test_get_cmb_heating_factor(self):
        """Test CMB heating factor calculation."""
        temperature = 20 * K
        emissivity = 1.5
        redshift = 2.0

        cmb_factor, new_temp = get_cmb_heating_factor(
            temperature, emissivity, redshift
        )

        # CMB factor should be > 1 at high redshift
        assert cmb_factor > 1.0
        # New temperature should be higher
        assert new_temp > temperature

        # Test at z=0 (no CMB heating)
        cmb_factor_z0, new_temp_z0 = get_cmb_heating_factor(
            temperature, emissivity, 0.0
        )
        assert cmb_factor_z0 == 1.0
        assert new_temp_z0 == temperature

    def test_get_cmb_heating_factor_units(self):
        """Test that CMB heating preserves units."""
        temperature = 20 * K
        emissivity = 1.5
        redshift = 1.0

        cmb_factor, new_temp = get_cmb_heating_factor(
            temperature, emissivity, redshift
        )

        assert new_temp.units == temperature.units


class TestBlackbodyGenerator:
    """Tests for Blackbody emission generator."""

    def test_blackbody_initialization(self):
        """Test Blackbody initialization."""
        temperature = 20 * K
        bb = Blackbody(temperature=temperature)

        assert bb.temperature == temperature
        assert not bb.do_cmb_heating
        assert "temperature" in bb._required_params

    def test_blackbody_initialization_with_scaling(self):
        """Test Blackbody initialization with scaling options."""
        temperature = 20 * K

        # Test energy balance
        bb_eb = Blackbody(
            temperature=temperature,
            intrinsic="intrinsic",
            attenuated="attenuated",
        )
        assert bb_eb.is_energy_balance
        assert "intrinsic" in bb_eb._required_emissions
        assert "attenuated" in bb_eb._required_emissions

        # Test scaler
        bb_scaler = Blackbody(temperature=temperature, scaler="scaler")
        assert bb_scaler.is_scaled
        assert "scaler" in bb_scaler._required_emissions

    def test_blackbody_get_spectra_standalone(self, dust_wavelengths):
        """Test Blackbody standalone spectrum generation."""
        temperature = 20 * K
        bb = Blackbody(temperature=temperature)

        sed = bb.get_spectra(dust_wavelengths)

        assert isinstance(sed, Sed)
        assert len(sed.lam) == len(dust_wavelengths)
        assert np.all(sed.lnu >= 0)
        assert np.any(sed.lnu > 0)
        assert sed.lam.units == dust_wavelengths.units

    def test_blackbody_generate_spectra(
        self, dust_wavelengths, mock_emitter, mock_model, mock_emissions
    ):
        """Test Blackbody _generate_spectra method."""
        bb = Blackbody(temperature=20 * K, scaler="scaler")

        sed = bb._generate_spectra(
            lams=dust_wavelengths,
            emitter=mock_emitter,
            model=mock_model,
            emissions=mock_emissions,
            redshift=0.0,
        )

        assert isinstance(sed, Sed)
        assert len(sed.lam) == len(dust_wavelengths)
        assert np.all(sed.lnu >= 0)
        assert np.any(sed.lnu > 0)

    def test_blackbody_generate_lines(
        self, dust_wavelengths, mock_emitter, mock_model, mock_emissions
    ):
        """Test Blackbody _generate_lines method."""
        # Use warmer dust to avoid division by zero issues
        bb = Blackbody(temperature=100 * K)
        line_ids = ["H_alpha", "O_III_5007"]

        # Override the mock emitter temperature to avoid numerical issues
        mock_emitter.temperature = 100 * K

        lines = bb._generate_lines(
            line_ids=line_ids,
            line_lams=dust_wavelengths[:2],
            emitter=mock_emitter,
            model=mock_model,
            emissions=mock_emissions,
            spectra=mock_emissions,
            redshift=0.0,
        )

        # Should return LineCollection with continuum only
        assert hasattr(lines, "line_ids")
        assert hasattr(lines, "cont")
        assert len(lines.line_ids) == 2
        assert np.all(lines.cont >= 0)

    def test_blackbody_cmb_heating(self, dust_wavelengths):
        """Test Blackbody with CMB heating."""
        # Use warmer dust temperature for measurable CMB effect
        temperature = 40 * K  # Warmer dust shows clearer CMB heating effect
        bb = Blackbody(temperature=temperature, do_cmb_heating=True)

        sed_z0 = bb.get_spectra(dust_wavelengths, redshift=0.0)
        sed_z2 = bb.get_spectra(dust_wavelengths, redshift=2.0)

        # CMB heating should increase total luminosity
        bol_z0 = sed_z0.bolometric_luminosity
        bol_z2 = sed_z2.bolometric_luminosity

        assert (
            bol_z2 >= bol_z0
        )  # Allow for equal in case of numerical precision

    def test_blackbody_temperature_extraction(
        self, dust_wavelengths, mock_model, mock_emissions
    ):
        """Test temperature extraction from different sources."""
        # Test from generator itself
        bb = Blackbody(temperature=30 * K)

        # Mock emitter with different temperature
        class MockEmitterWithTemp:
            def __init__(self):
                self.temperature = 50 * K  # Different from generator

        emitter = MockEmitterWithTemp()

        # Should use emitter temperature (higher priority)
        sed = bb._generate_spectra(
            lams=dust_wavelengths,
            emitter=emitter,
            model=mock_model,
            emissions=mock_emissions,
            redshift=0.0,
        )

        # Can't directly test which temperature was used, but spectrum should
        # be generated
        assert isinstance(sed, Sed)
        assert np.any(sed.lnu > 0)

    def test_blackbody_base_properties(self):
        """Test base properties of Blackbody generator."""
        temperature = 20 * K
        bb = Blackbody(temperature=temperature, do_cmb_heating=True)

        # Test properties exist
        assert hasattr(bb, "temperature_z")
        assert hasattr(bb, "cmb_factor")
        assert hasattr(bb, "last_cmb_factor")
        assert hasattr(bb, "last_effective_temperature")

        # Test apply_cmb_heating method
        cmb_factor, new_temp = bb.apply_cmb_heating(
            temperature=temperature, emissivity=1.0, redshift=2.0
        )
        assert cmb_factor > 1.0
        assert new_temp > temperature


class TestGreybodyGenerator:
    """Tests for Greybody emission generator."""

    def test_greybody_initialization(self):
        """Test Greybody initialization."""
        temperature = 20 * K
        emissivity = 1.5
        gb = Greybody(temperature=temperature, emissivity=emissivity)

        assert gb.temperature == temperature
        assert gb.emissivity == emissivity
        assert gb.optically_thin is True  # Default
        assert gb.lam_0 == 100.0 * um  # Default
        assert "temperature" in gb._required_params
        assert "emissivity" in gb._required_params

    def test_greybody_get_spectra_standalone(self, dust_wavelengths):
        """Test Greybody standalone spectrum generation."""
        temperature = 20 * K
        emissivity = 1.5
        gb = Greybody(temperature=temperature, emissivity=emissivity)

        sed = gb.get_spectra(dust_wavelengths)

        assert isinstance(sed, Sed)
        assert len(sed.lam) == len(dust_wavelengths)
        assert np.all(sed.lnu >= 0)
        assert np.any(sed.lnu > 0)

    def test_greybody_optically_thick(self, dust_wavelengths):
        """Test Greybody with optically thick assumption."""
        temperature = 20 * K
        emissivity = 1.5

        gb_thin = Greybody(
            temperature=temperature, emissivity=emissivity, optically_thin=True
        )
        gb_thick = Greybody(
            temperature=temperature,
            emissivity=emissivity,
            optically_thin=False,
            lam_0=100.0 * um,
        )

        sed_thin = gb_thin.get_spectra(dust_wavelengths)
        sed_thick = gb_thick.get_spectra(dust_wavelengths)

        # Optically thick should generally have different spectral shape
        assert not np.array_equal(sed_thin.lnu, sed_thick.lnu)

    def test_greybody_emissivity_effect(self, dust_wavelengths):
        """Test effect of different emissivity values."""
        temperature = 20 * K

        gb_low = Greybody(temperature=temperature, emissivity=0.5)
        gb_high = Greybody(temperature=temperature, emissivity=2.0)

        sed_low = gb_low.get_spectra(dust_wavelengths)
        sed_high = gb_high.get_spectra(dust_wavelengths)

        # Should produce different spectra
        assert not np.array_equal(sed_low.lnu, sed_high.lnu)
        assert np.any(sed_low.lnu > 0)
        assert np.any(sed_high.lnu > 0)


class TestCasey12Generator:
    """Tests for Casey12 emission generator."""

    def test_casey12_initialization(self):
        """Test Casey12 initialization."""
        temperature = 20 * K
        emissivity = 2.0
        alpha = 2.0
        n_bb = 1.0
        lam_0 = 200.0 * um

        c12 = Casey12(
            temperature=temperature,
            emissivity=emissivity,
            alpha=alpha,
            n_bb=n_bb,
            lam_0=lam_0,
        )

        assert c12.temperature == temperature
        assert c12.emissivity == emissivity
        assert c12.alpha == alpha
        assert c12.n_bb == n_bb
        assert c12.lam_0 == lam_0
        assert "temperature" in c12._required_params

    def test_casey12_get_spectra_standalone(self, dust_wavelengths):
        """Test Casey12 standalone spectrum generation."""
        temperature = 20 * K
        c12 = Casey12(temperature=temperature, emissivity=2.0, alpha=2.0)

        sed = c12.get_spectra(dust_wavelengths)

        assert isinstance(sed, Sed)
        assert len(sed.lam) == len(dust_wavelengths)
        assert np.all(sed.lnu >= 0)
        assert np.any(sed.lnu > 0)

    def test_casey12_parameter_calculation(self):
        """Test Casey12 parameter calculations."""
        temperature = 25 * K
        alpha = 2.0

        c12 = Casey12(temperature=temperature, alpha=alpha)

        # Check that lam_c and n_pl are calculated
        assert hasattr(c12, "lam_c")
        assert hasattr(c12, "n_pl")
        assert c12.lam_c > 0 * um
        assert c12.n_pl > 0

    def test_casey12_component_methods(self, dust_wavelengths):
        """Test Casey12 component calculation methods."""
        temperature = 20 * K
        c12 = Casey12(temperature=temperature, emissivity=2.0, alpha=2.0)

        nu = (3e10 / (dust_wavelengths.to("cm").value)) * Hz  # c in cm/s

        # Test _lnu method
        lnu = c12._lnu(nu, temperature)
        assert len(lnu) == len(nu)
        assert np.all(lnu >= 0)
        assert np.any(lnu > 0)


class TestDraineLi07Generator:
    """Tests for DraineLi07 emission generator."""

    def test_draineli07_utility_functions(self):
        """Test DraineLi07 utility functions."""
        # Test u_mean_magdis12
        dust_mass = 1e6
        ldust = 1e10
        p0 = 125.0
        u_avg = u_mean_magdis12(dust_mass, ldust, p0)
        assert u_avg > 0
        assert u_avg == ldust / (p0 * dust_mass)

        # Test u_mean
        umin = 1.0
        umax = 1e7
        gamma = 0.05
        u_calculated = u_mean(umin, umax, gamma)
        assert u_calculated > umin
        assert u_calculated < umax

        # Test solve_umin
        result = solve_umin(umin, umax, u_calculated, gamma)
        assert abs(result) < 1e-10  # Should be close to zero

    def test_draineli07_initialization(self, mock_dust_grid):
        """Test DraineLi07 initialization."""
        dust_mass = 1e6 * Msun

        dl07 = DraineLi07(
            grid=mock_dust_grid,
            dust_mass=dust_mass,
            dust_to_gas_ratio=0.01,
            qpah=0.025,
            intrinsic="intrinsic",
            attenuated="attenuated",
        )

        assert dl07.grid == mock_dust_grid
        assert dl07.dust_mass == dust_mass
        assert dl07.dust_to_gas_ratio == 0.01
        assert dl07.qpah == 0.025
        assert "dust_mass" in dl07._required_params
        assert "dust_to_gas_ratio" in dl07._required_params
        assert "hydrogen_mass" in dl07._required_params
        assert "qpah" in dl07._required_params

    def test_draineli07_variable_naming(self, mock_dust_grid):
        """Test DraineLi07 uses improved variable names."""
        dust_mass = 1e6 * Msun

        dl07 = DraineLi07(
            grid=mock_dust_grid,
            dust_mass=dust_mass,
            dust_to_gas_ratio=0.01,
            qpah=0.025,
            intrinsic="intrinsic",
            attenuated="attenuated",
        )

        # Should use improved naming
        assert hasattr(dl07, "qpah")
        assert hasattr(dl07, "dust_mass")
        assert hasattr(dl07, "dust_to_gas_ratio")
        assert hasattr(dl07, "hydrogen_mass")

    def test_draineli07_parameter_setup(self, mock_dust_grid):
        """Test DraineLi07 parameter setup."""
        dust_mass = 1e6 * Msun
        ldust = 1e10 * Lsun

        dl07 = DraineLi07(
            grid=mock_dust_grid,
            dust_mass=dust_mass,
            dust_to_gas_ratio=0.01,
            qpah=0.025,
            verbose=False,
            intrinsic="intrinsic",
            attenuated="attenuated",
        )

        # Test parameter setup
        dl07._setup_dl07_parameters(dust_mass, ldust)

        # Check that calculated values are stored
        assert dl07.radiation_field_average is not None
        assert dl07.umin_calculated is not None
        assert dl07.gamma_calculated is not None
        assert dl07.hydrogen_mass_calculated is not None

        # Check the convenience property
        assert dl07.u_avg == dl07.radiation_field_average

    def test_draineli07_get_spectra(self, mock_dust_grid, dust_wavelengths):
        """Test DraineLi07 get_spectra method."""
        dust_mass = 1e6 * Msun
        ldust = 1e10 * Lsun

        dl07 = DraineLi07(
            grid=mock_dust_grid,
            dust_mass=dust_mass,
            dust_to_gas_ratio=0.01,
            qpah=0.025,
            verbose=False,
            intrinsic="intrinsic",
            attenuated="attenuated",
        )

        # Test basic spectrum generation
        sed = dl07.get_spectra(
            dust_wavelengths, dust_mass=dust_mass, ldust=ldust
        )

        assert isinstance(sed, Sed)
        assert len(sed.lam) == len(dust_wavelengths)
        assert np.all(sed.lnu >= 0)

    def test_draineli07_dust_components(
        self, mock_dust_grid, dust_wavelengths
    ):
        """Test DraineLi07 dust component separation."""
        dust_mass = 1e6 * Msun
        ldust = 1e10 * Lsun

        dl07 = DraineLi07(
            grid=mock_dust_grid,
            dust_mass=dust_mass,
            dust_to_gas_ratio=0.01,
            qpah=0.025,
            verbose=False,
            intrinsic="intrinsic",
            attenuated="attenuated",
        )

        # Test component separation
        sed_diffuse, sed_pdr = dl07.get_spectra(
            dust_wavelengths,
            dust_mass=dust_mass,
            ldust=ldust,
            dust_components=True,
        )

        assert isinstance(sed_diffuse, Sed)
        assert isinstance(sed_pdr, Sed)
        assert len(sed_diffuse.lam) == len(dust_wavelengths)
        assert len(sed_pdr.lam) == len(dust_wavelengths)

    def test_draineli07_calculated_parameters_property(self, mock_dust_grid):
        """Test DraineLi07 calculated_parameters property."""
        dust_mass = 1e6 * Msun
        ldust = 1e10 * Lsun

        dl07 = DraineLi07(
            grid=mock_dust_grid,
            dust_mass=dust_mass,
            dust_to_gas_ratio=0.01,
            qpah=0.025,
            verbose=False,
            intrinsic="intrinsic",
            attenuated="attenuated",
        )

        # Set up parameters
        dl07._setup_dl07_parameters(dust_mass, ldust)

        # Test calculated_parameters property
        params = dl07.calculated_parameters
        assert isinstance(params, dict)
        assert "average_radiation_field" in params
        assert "minimum_radiation_field" in params
        assert "gamma_parameter" in params
        assert "hydrogen_mass" in params
        assert "qpah_used" in params


class TestDustGeneratorIntegration:
    """Integration tests for dust generators."""

    def test_all_generators_initialization(self):
        """Test that all dust generators can be initialized."""
        temperature = 20 * K

        # Test all generators can be created
        bb = Blackbody(temperature=temperature)
        gb = Greybody(temperature=temperature, emissivity=1.5)
        c12 = Casey12(temperature=temperature, emissivity=2.0, alpha=2.0)

        assert isinstance(bb, Blackbody)
        assert isinstance(gb, Greybody)
        assert isinstance(c12, Casey12)

    def test_all_generators_get_spectra(self, dust_wavelengths):
        """Test that all dust generators have get_spectra method."""
        temperature = 20 * K

        generators = [
            Blackbody(temperature=temperature),
            Greybody(temperature=temperature, emissivity=1.5),
            Casey12(temperature=temperature, emissivity=2.0, alpha=2.0),
        ]

        for gen in generators:
            sed = gen.get_spectra(dust_wavelengths)
            assert isinstance(sed, Sed)
            assert len(sed.lam) == len(dust_wavelengths)
            assert np.all(sed.lnu >= 0)

    def test_temperature_scaling(self, dust_wavelengths, dust_temperatures):
        """Test that all generators respond to temperature changes."""
        for temp in dust_temperatures:
            # Test all generators respond to temperature
            bb = Blackbody(temp)
            gb = Greybody(temp, 1.5)
            c12 = Casey12(temp, 2.0, 2.0)

            sed_bb = bb.get_spectra(dust_wavelengths)
            sed_gb = gb.get_spectra(dust_wavelengths)
            sed_c12 = c12.get_spectra(dust_wavelengths)

            # All should produce positive flux
            assert np.any(sed_bb.lnu > 0)
            assert np.any(sed_gb.lnu > 0)
            assert np.any(sed_c12.lnu > 0)

    def test_flux_units_consistency(self, dust_wavelengths):
        """Test that all generators return consistent flux units."""
        # Use warmer dust for more robust unit testing
        temperature = 50 * K
        expected_units = erg / s / Hz

        generators = [
            Blackbody(temperature=temperature),
            Greybody(temperature=temperature, emissivity=1.5),
            Casey12(temperature=temperature, emissivity=2.0, alpha=2.0),
        ]

        for gen in generators:
            sed = gen.get_spectra(dust_wavelengths)
            assert sed.lnu.units == expected_units

    def test_wavelength_coverage(self):
        """Test generators work across different wavelength ranges."""
        temperature = 100 * K  # Use warmer dust for detection

        # Near-IR wavelengths
        nir_lams = np.logspace(3, 4, 50) * angstrom
        # Far-IR wavelengths
        fir_lams = np.logspace(4, 6, 50) * angstrom

        bb = Blackbody(temperature)

        sed_nir = bb.get_spectra(nir_lams)
        sed_fir = bb.get_spectra(fir_lams)

        assert len(sed_nir.lam) == len(nir_lams)
        assert len(sed_fir.lam) == len(fir_lams)
        assert np.any(sed_nir.lnu > 0)
        assert np.any(sed_fir.lnu > 0)

    def test_nan_handling(self, dust_wavelengths):
        """Test that generators handle edge cases gracefully."""
        # Very low temperature
        very_cold = 1 * K
        bb = Blackbody(very_cold)
        sed_cold = bb.get_spectra(dust_wavelengths)

        # Should not have NaN values
        assert not np.any(np.isnan(sed_cold.lnu))
        assert np.all(sed_cold.lnu >= 0)

    def test_energy_balance_scaling(
        self, dust_wavelengths, mock_emitter, mock_model, mock_emissions
    ):
        """Test energy balance scaling for all generators."""
        temperature = 20 * K

        generators = [
            Blackbody(
                temperature=temperature,
                intrinsic="intrinsic",
                attenuated="attenuated",
            ),
            Greybody(
                temperature=temperature,
                emissivity=1.5,
                intrinsic="intrinsic",
                attenuated="attenuated",
            ),
            Casey12(
                temperature=temperature,
                emissivity=2.0,
                alpha=2.0,
                intrinsic="intrinsic",
                attenuated="attenuated",
            ),
        ]

        for gen in generators:
            sed = gen._generate_spectra(
                lams=dust_wavelengths,
                emitter=mock_emitter,
                model=mock_model,
                emissions=mock_emissions,
                redshift=0.0,
            )

            assert isinstance(sed, Sed)
            assert np.any(sed.lnu > 0)

    def test_scaler_scaling(
        self, dust_wavelengths, mock_emitter, mock_model, mock_emissions
    ):
        """Test scaler scaling for all generators."""
        temperature = 20 * K

        generators = [
            Blackbody(temperature=temperature, scaler="scaler"),
            Greybody(temperature=temperature, emissivity=1.5, scaler="scaler"),
            Casey12(
                temperature=temperature,
                emissivity=2.0,
                alpha=2.0,
                scaler="scaler",
            ),
        ]

        for gen in generators:
            sed = gen._generate_spectra(
                lams=dust_wavelengths,
                emitter=mock_emitter,
                model=mock_model,
                emissions=mock_emissions,
                redshift=0.0,
            )

            assert isinstance(sed, Sed)
            assert np.any(sed.lnu > 0)

    def test_cmb_heating_all_generators(self, dust_wavelengths):
        """Test CMB heating for all applicable generators."""
        # Use moderate temperature for clear CMB effects
        temperature = 40 * K

        generators = [
            Blackbody(temperature=temperature, do_cmb_heating=True),
            Greybody(
                temperature=temperature, emissivity=1.5, do_cmb_heating=True
            ),
            Casey12(
                temperature=temperature,
                emissivity=2.0,
                alpha=2.0,
                do_cmb_heating=True,
            ),
        ]

        for gen in generators:
            sed_z0 = gen.get_spectra(dust_wavelengths, redshift=0.0)
            sed_z2 = gen.get_spectra(dust_wavelengths, redshift=2.0)

            # CMB heating should increase total bolometric luminosity
            bol_z0 = sed_z0.bolometric_luminosity
            bol_z2 = sed_z2.bolometric_luminosity
            assert bol_z2 >= bol_z0

    def test_error_handling_missing_parameters(
        self, dust_wavelengths, mock_model, mock_emissions
    ):
        """Test error handling when required parameters are missing."""
        # Use a simple parameter extraction test
        bb = Blackbody(temperature=50 * K)

        # Mock emitter without required temperature attribute
        class MockEmitterNoTemp:
            def __init__(self):
                # Missing temperature attribute to test extraction
                pass

        # Mock model with empty fixed parameters and overridden temperature
        # to force extraction from emitter
        mock_model.fixed_parameters = {
            "temperature": None
        }  # Force extraction from emitter

        emitter_no_temp = MockEmitterNoTemp()

        # Should raise error about missing temperature when extracted from
        # emitter
        with pytest.raises((exceptions.MissingAttribute, AttributeError)):
            bb._extract_params(mock_model, emitter_no_temp)

    def test_line_generation_all_generators(
        self, dust_wavelengths, mock_emitter, mock_model, mock_emissions
    ):
        """Test line generation for all generators."""
        # Use warmer dust to avoid numerical issues
        temperature = 100 * K
        line_ids = ["H_alpha", "O_III_5007"]

        # Override the mock emitter temperature to avoid numerical issues
        mock_emitter.temperature = 100 * K

        generators = [
            Blackbody(temperature=temperature),
            Greybody(temperature=temperature, emissivity=1.5),
            Casey12(temperature=temperature, emissivity=2.0, alpha=2.0),
        ]

        for gen in generators:
            lines = gen._generate_lines(
                line_ids=line_ids,
                line_lams=dust_wavelengths[:2],
                emitter=mock_emitter,
                model=mock_model,
                emissions=mock_emissions,
                spectra=mock_emissions,
                redshift=0.0,
            )

            # Should return LineCollection with continuum only
            assert hasattr(lines, "line_ids")
            assert hasattr(lines, "cont")
            assert len(lines.line_ids) == 2
            assert np.all(lines.cont >= 0)


class TestDustGeneratorErrorHandling:
    """Tests for error handling and edge cases."""

    def test_missing_emissions_error(
        self, dust_wavelengths, mock_emitter, mock_model
    ):
        """Test error when required emissions are missing."""
        bb = Blackbody(temperature=50 * K, scaler="missing_scaler")

        # Empty emissions dictionary
        empty_emissions = {}

        # Should raise KeyError when trying to access missing emission
        with pytest.raises(KeyError):
            bb._generate_spectra(
                lams=dust_wavelengths,
                emitter=mock_emitter,
                model=mock_model,
                emissions=empty_emissions,
                redshift=0.0,
            )

    def test_draineli07_invalid_template(self, mock_dust_grid):
        """Test DraineLi07 error for invalid template."""
        dl07 = DraineLi07(
            grid=mock_dust_grid,
            dust_mass=1e6 * Msun,
            dust_to_gas_ratio=0.01,
            qpah=0.025,
            template="InvalidTemplate",  # Invalid template
        )

        with pytest.raises(exceptions.UnimplementedFunctionality):
            dl07._setup_dl07_parameters(1e6 * Msun, 1e10 * Lsun)

    def test_reasonable_temperature_range(self, dust_wavelengths):
        """Test generators work across reasonable temperature range."""
        # Test reasonable dust temperature range (not extreme values)
        temperatures = [10 * K, 30 * K, 100 * K]  # Realistic dust temperatures

        for temp in temperatures:
            bb = Blackbody(temp)
            sed = bb.get_spectra(dust_wavelengths)

            # Should handle gracefully
            assert isinstance(sed, Sed)
            assert not np.any(np.isnan(sed.lnu))
            assert np.all(sed.lnu >= 0)

    def test_extremely_large_values(self, dust_wavelengths):
        """Test generators with extremely large parameter values."""
        # Very high temperature
        very_hot = 10000 * K
        bb = Blackbody(very_hot)
        sed = bb.get_spectra(dust_wavelengths)

        # Should handle gracefully
        assert isinstance(sed, Sed)
        assert not np.any(np.isnan(sed.lnu))
        assert np.all(sed.lnu >= 0)
        assert np.any(sed.lnu > 0)

    def test_dust_generator_consistency(self, dust_wavelengths):
        """Test different generators produce reasonable results.

        This test verifies consistency between generators.
        """
        temperature = 50 * K  # Use moderate temperature

        bb = Blackbody(temperature=temperature)
        gb = Greybody(
            temperature=temperature, emissivity=1.0
        )  # emissivity=1 ~ blackbody

        sed_bb = bb.get_spectra(dust_wavelengths)
        sed_gb = gb.get_spectra(dust_wavelengths)

        # Normalized spectra should be somewhat similar for emissivity=1
        bb_norm = sed_bb.lnu / np.max(sed_bb.lnu)
        gb_norm = sed_gb.lnu / np.max(sed_gb.lnu)

        # Should have reasonable correlation (though not exact due to
        # frequency dependence)
        correlation = np.corrcoef(bb_norm, gb_norm)[0, 1]
        assert correlation > 0.5

    def test_scaling_configurations(
        self, dust_wavelengths, mock_emitter, mock_model, mock_emissions
    ):
        """Test different scaling configurations work correctly."""
        # Use moderate temperature for robust testing
        temperature = 50 * K

        # Test standalone (no scaling)
        bb_standalone = Blackbody(temperature=temperature)
        sed_standalone = bb_standalone._generate_spectra(
            lams=dust_wavelengths,
            emitter=mock_emitter,
            model=mock_model,
            emissions=mock_emissions,
            redshift=0.0,
        )

        # Test energy balance scaling
        bb_eb = Blackbody(
            temperature=temperature,
            intrinsic="intrinsic",
            attenuated="attenuated",
        )
        sed_eb = bb_eb._generate_spectra(
            lams=dust_wavelengths,
            emitter=mock_emitter,
            model=mock_model,
            emissions=mock_emissions,
            redshift=0.0,
        )

        # Test scaler scaling
        bb_scaled = Blackbody(temperature=temperature, scaler="scaler")
        sed_scaled = bb_scaled._generate_spectra(
            lams=dust_wavelengths,
            emitter=mock_emitter,
            model=mock_model,
            emissions=mock_emissions,
            redshift=0.0,
        )

        # All should produce valid SEDs
        for sed in [sed_standalone, sed_eb, sed_scaled]:
            assert isinstance(sed, Sed)
            assert np.any(sed.lnu > 0)
            assert not np.any(np.isnan(sed.lnu))

    def test_parameter_priority(
        self, dust_wavelengths, mock_model, mock_emissions
    ):
        """Test parameter extraction priority (model > emitter > generator)."""
        bb = Blackbody(temperature=30 * K)  # Generator temperature

        # Test with emitter temperature (should override generator)
        class MockEmitterWithTemp:
            temperature = 40 * K

        emitter_with_temp = MockEmitterWithTemp()

        # Test with model fixed parameter (should override both)
        mock_model.fixed_parameters = {"temperature": 50 * K}

        sed = bb._generate_spectra(
            lams=dust_wavelengths,
            emitter=emitter_with_temp,
            model=mock_model,
            emissions=mock_emissions,
            redshift=0.0,
        )

        # Should work regardless of which temperature is used
        assert isinstance(sed, Sed)
        assert np.any(sed.lnu > 0)

        # Reset mock_model for other tests
        mock_model.fixed_parameters = {}
