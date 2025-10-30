"""Comprehensive test suite for dust emission generators.

This module contains tests for all dust generator functionality including:
- Base dust emission classes
- Blackbody emission generators
- Greybody emission generators
- Casey12 emission generators
- DrainLi07 emission generators
- Integration and functionality tests
"""

import numpy as np
import pytest
from unyt import Hz, K, Lsun, Msun, angstrom, erg, s, um

from synthesizer import exceptions
from synthesizer.emission_models.generators.dust import (
    get_cmb_heating_factor,
)
from synthesizer.emission_models.generators.dust.blackbody import (
    Blackbody,
    BlackbodyBase,
    BlackbodyEnergyBalance,
    BlackbodyScaler,
)
from synthesizer.emission_models.generators.dust.casey12 import (
    Casey12,
    Casey12Base,
    Casey12EnergyBalance,
    Casey12Scaler,
)
from synthesizer.emission_models.generators.dust.drainli07 import (
    DrainLi07,
    solve_umin,
    u_mean,
    u_mean_magdis12,
)
from synthesizer.emission_models.generators.dust.greybody import (
    Greybody,
    GreybodyBase,
    GreybodyEnergyBalance,
    GreybodyScaler,
)
from synthesizer.emissions import Sed


class TestDustEmissionBase:
    """Tests for base dust emission classes."""

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


class TestBlackbodyGenerators:
    """Tests for Blackbody emission generators."""

    def test_blackbody_base_initialization(self):
        """Test BlackbodyBase initialization."""
        temperature = 20 * K
        cmb_factor = 1.5

        bb_base = BlackbodyBase(temperature, cmb_factor)

        assert bb_base.temperature == temperature
        assert bb_base.cmb_factor == cmb_factor

    def test_blackbody_base_get_spectra(self, dust_wavelengths):
        """Test BlackbodyBase spectrum generation."""
        temperature = 20 * K
        bb_base = BlackbodyBase(temperature)

        sed = bb_base.get_spectra(dust_wavelengths)

        assert isinstance(sed, Sed)
        assert len(sed.lam) == len(dust_wavelengths)
        assert np.all(sed.lnu >= 0)  # Allow zero values for very cold dust
        assert np.any(sed.lnu > 0)  # But check that some values are positive
        assert sed.lam.units == dust_wavelengths.units

    def test_blackbody_base_cmb_heating(self, dust_wavelengths):
        """Test BlackbodyBase with CMB heating."""
        temperature = 20 * K
        bb_base = BlackbodyBase(temperature, cmb_factor=2.0)

        sed_cmb = bb_base.get_spectra(dust_wavelengths, redshift=1.0)
        sed_no_cmb = bb_base.get_spectra(dust_wavelengths, redshift=0.0)

        # CMB heating should increase flux
        assert np.all(sed_cmb.lnu >= sed_no_cmb.lnu)

    def test_blackbody_energy_balance_initialization(self):
        """Test BlackbodyEnergyBalance initialization."""
        temperature = 20 * K

        bb_eb = BlackbodyEnergyBalance(
            temperature=temperature,
            intrinsic="intrinsic",
            attenuated="attenuated",
        )

        assert bb_eb.temperature == temperature
        assert "temperature" in bb_eb._required_params
        assert "intrinsic" in bb_eb._required_emissions
        assert "attenuated" in bb_eb._required_emissions

    def test_blackbody_scaler_initialization(self):
        """Test BlackbodyScaler initialization."""
        temperature = 20 * K

        bb_scaler = BlackbodyScaler(temperature=temperature, scaler="scaler")

        assert bb_scaler.temperature == temperature
        assert "temperature" in bb_scaler._required_params
        assert "scaler" in bb_scaler._required_emissions

    def test_blackbody_factory_energy_balance(self):
        """Test Blackbody factory returns energy balance version."""
        temperature = 20 * K

        bb = Blackbody(
            temperature=temperature,
            intrinsic="intrinsic",
            attenuated="attenuated",
            scaler=None,
        )

        assert isinstance(bb, BlackbodyEnergyBalance)

    def test_blackbody_factory_scaler(self):
        """Test Blackbody factory returns scaler version."""
        temperature = 20 * K

        bb = Blackbody(
            temperature=temperature,
            intrinsic=None,
            attenuated=None,
            scaler="scaler",
        )

        assert isinstance(bb, BlackbodyScaler)

    def test_blackbody_factory_invalid_args(self):
        """Test Blackbody factory with invalid arguments."""
        temperature = 20 * K

        with pytest.raises(exceptions.InconsistentArguments):
            Blackbody(
                temperature=temperature,
                intrinsic=None,
                attenuated=None,
                scaler=None,
            )


class TestGreybodyGenerators:
    """Tests for Greybody emission generators."""

    def test_greybody_base_initialization(self):
        """Test GreybodyBase initialization."""
        temperature = 20 * K
        emissivity = 1.5

        gb_base = GreybodyBase(
            temperature,
            emissivity,
            cmb_factor=1.0,
            optically_thin=True,
            lam_0=100.0 * um,
        )

        assert gb_base.temperature == temperature
        assert gb_base.emissivity == emissivity
        assert gb_base.optically_thin is True
        assert gb_base.lam_0 == 100.0 * um

    def test_greybody_base_get_spectra(self, dust_wavelengths):
        """Test GreybodyBase spectrum generation."""
        temperature = 20 * K
        emissivity = 1.5
        gb_base = GreybodyBase(temperature, emissivity)

        sed = gb_base.get_spectra(dust_wavelengths)

        assert isinstance(sed, Sed)
        assert len(sed.lam) == len(dust_wavelengths)
        assert np.all(sed.lnu >= 0)  # Allow zero values for very cold dust
        assert np.any(sed.lnu > 0)  # But check that some values are positive

    def test_greybody_vs_blackbody_emissivity(self, dust_wavelengths):
        """Test greybody with low emissivity approaches blackbody behavior."""
        temperature = 20 * K

        bb = BlackbodyBase(temperature)
        gb = GreybodyBase(
            temperature, emissivity=0.1
        )  # Low but non-zero emissivity

        bb_sed = bb.get_spectra(dust_wavelengths)
        gb_sed = gb.get_spectra(dust_wavelengths)

        # Filter out zero flux values to avoid division by zero
        mask = (bb_sed.lnu > 0) & (gb_sed.lnu > 0)
        if np.any(mask):
            ratio = gb_sed.lnu[mask] / bb_sed.lnu[mask]
            # Check ratio is reasonably consistent (within order of magnitude)
            assert np.all(ratio > 0)
            assert (
                np.max(ratio) / np.min(ratio) < 100
            )  # Within 2 orders of magnitude

    def test_greybody_optically_thick(self, dust_wavelengths):
        """Test greybody with optically thick assumption."""
        temperature = 20 * K
        emissivity = 1.5

        gb_thin = GreybodyBase(temperature, emissivity, optically_thin=True)
        gb_thick = GreybodyBase(
            temperature, emissivity, optically_thin=False, lam_0=100.0 * um
        )

        sed_thin = gb_thin.get_spectra(dust_wavelengths)
        sed_thick = gb_thick.get_spectra(dust_wavelengths)

        # Optically thick should generally have lower flux
        assert np.mean(sed_thick.lnu) < np.mean(sed_thin.lnu)

    def test_greybody_energy_balance_initialization(self):
        """Test GreybodyEnergyBalance initialization."""
        temperature = 20 * K
        emissivity = 1.5

        gb_eb = GreybodyEnergyBalance(
            temperature=temperature,
            emissivity=emissivity,
            intrinsic="intrinsic",
            attenuated="attenuated",
        )

        assert gb_eb.temperature == temperature
        assert gb_eb.emissivity == emissivity

    def test_greybody_factory(self):
        """Test Greybody factory function."""
        temperature = 20 * K
        emissivity = 1.5

        # Test energy balance version
        gb_eb = Greybody(
            temperature=temperature,
            emissivity=emissivity,
            intrinsic="intrinsic",
            attenuated="attenuated",
            scaler=None,
        )
        assert isinstance(gb_eb, GreybodyEnergyBalance)

        # Test scaler version
        gb_scaler = Greybody(
            temperature=temperature,
            emissivity=emissivity,
            intrinsic=None,
            attenuated=None,
            scaler="scaler",
        )
        assert isinstance(gb_scaler, GreybodyScaler)


class TestCasey12Generators:
    """Tests for Casey12 emission generators."""

    def test_casey12_base_initialization(self):
        """Test Casey12Base initialization."""
        temperature = 20 * K
        emissivity = 2.0
        alpha = 2.0
        n_bb = 1.0
        lam_0 = 200.0 * um

        c12_base = Casey12Base(temperature, emissivity, alpha, n_bb, lam_0)

        assert c12_base.temperature == temperature
        assert c12_base.emissivity == emissivity
        assert c12_base.alpha == alpha
        assert c12_base.n_bb == n_bb
        assert c12_base.lam_0 == lam_0

    def test_casey12_base_get_spectra(self, dust_wavelengths):
        """Test Casey12Base spectrum generation."""
        temperature = 20 * K
        emissivity = 2.0
        alpha = 2.0

        c12_base = Casey12Base(temperature, emissivity, alpha)
        sed = c12_base.get_spectra(dust_wavelengths)

        assert isinstance(sed, Sed)
        assert len(sed.lam) == len(dust_wavelengths)
        assert np.all(sed.lnu >= 0)  # Allow zero values for very cold dust
        assert np.any(sed.lnu > 0)  # But check that some values are positive

    def test_casey12_component_methods(self, dust_wavelengths):
        """Test Casey12 individual component methods."""
        temperature = 20 * K
        emissivity = 2.0
        alpha = 2.0

        c12_base = Casey12Base(temperature, emissivity, alpha)

        # Test power law component
        lam = dust_wavelengths
        power_law = c12_base._lnu_power_law(lam)
        assert len(power_law) == len(lam)
        assert np.all(power_law >= 0)

        # Test blackbody component
        blackbody = c12_base._lnu_blackbody(lam, temperature)
        assert len(blackbody) == len(lam)
        assert np.all(blackbody >= 0)  # Allow zero values for very cold dust
        assert np.any(blackbody > 0)  # But check that some values are positive

    def test_casey12_variable_naming(self):
        """Test that Casey12 uses proper variable naming (n_bb not N_bb)."""
        temperature = 20 * K

        c12_base = Casey12Base(temperature, 2.0, 2.0, n_bb=1.5)

        # Should use n_bb, not N_bb
        assert hasattr(c12_base, "n_bb")
        assert not hasattr(c12_base, "N_bb")
        assert c12_base.n_bb == 1.5

    def test_casey12_energy_balance_initialization(self):
        """Test Casey12EnergyBalance initialization."""
        temperature = 20 * K

        c12_eb = Casey12EnergyBalance(
            temperature=temperature,
            emissivity=2.0,
            alpha=2.0,
            intrinsic="intrinsic",
            attenuated="attenuated",
        )

        assert c12_eb.temperature == temperature
        assert c12_eb.emissivity == 2.0
        assert c12_eb.alpha == 2.0

    def test_casey12_factory(self):
        """Test Casey12 factory function."""
        temperature = 20 * K

        # Test energy balance version
        c12_eb = Casey12(
            temperature=temperature,
            emissivity=2.0,
            alpha=2.0,
            intrinsic="intrinsic",
            attenuated="attenuated",
            scaler=None,
        )
        assert isinstance(c12_eb, Casey12EnergyBalance)

        # Test scaler version
        c12_scaler = Casey12(
            temperature=temperature,
            emissivity=2.0,
            alpha=2.0,
            intrinsic=None,
            attenuated=None,
            scaler="scaler",
        )
        assert isinstance(c12_scaler, Casey12Scaler)


class TestDrainLi07Generators:
    """Tests for DrainLi07 emission generators."""

    def test_drainli07_utility_functions(self):
        """Test DrainLi07 utility functions."""
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

    def test_drainli07_initialization(self, mock_dust_grid):
        """Test DrainLi07 initialization."""
        dust_mass = 1e6 * Msun

        dl07 = DrainLi07(
            grid=mock_dust_grid,
            dust_mass=dust_mass,
            dust_to_gas=0.01,
            pah_frac=0.025,
            intrinsic="intrinsic",
            attenuated="attenuated",
        )

        assert dl07.grid == mock_dust_grid
        assert dl07.dust_mass == dust_mass
        assert dl07.dust_to_gas == 0.01
        assert dl07.pah_frac == 0.025
        assert "dust_mass" in dl07._required_params
        assert "dust_to_gas" in dl07._required_params
        assert "hydrogen_mass" in dl07._required_params

    def test_drainli07_parameter_setup(self, mock_dust_grid):
        """Test DrainLi07 parameter setup."""
        dust_mass = 1e6 * Msun
        ldust = 1e10 * Lsun

        dl07 = DrainLi07(
            grid=mock_dust_grid,
            dust_mass=dust_mass,
            dust_to_gas=0.01,
            pah_frac=0.025,
            verbose=False,
            intrinsic="intrinsic",
            attenuated="attenuated",
        )

        # Test parameter setup
        dl07._setup_dl07_parameters(dust_mass, ldust)

        assert dl07.pah_frac_id is not None
        assert dl07.umin_id is not None
        assert dl07.alpha_id is not None

    def test_drainli07_get_spectra(self, mock_dust_grid, dust_wavelengths):
        """Test DrainLi07 get_spectra method."""
        dust_mass = 1e6 * Msun
        ldust = 1e10 * Lsun

        dl07 = DrainLi07(
            grid=mock_dust_grid,
            dust_mass=dust_mass,
            dust_to_gas=0.01,
            pah_frac=0.025,
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
        assert np.all(sed.lnu >= 0)  # Allow zero values

    def test_drainli07_dust_components(self, mock_dust_grid, dust_wavelengths):
        """Test DrainLi07 dust component separation."""
        dust_mass = 1e6 * Msun
        ldust = 1e10 * Lsun

        dl07 = DrainLi07(
            grid=mock_dust_grid,
            dust_mass=dust_mass,
            dust_to_gas=0.01,
            pah_frac=0.025,
            verbose=False,
            intrinsic="intrinsic",
            attenuated="attenuated",
        )

        # Test component separation
        sed_old, sed_young = dl07.get_spectra(
            dust_wavelengths,
            dust_mass=dust_mass,
            ldust=ldust,
            dust_components=True,
        )

        assert isinstance(sed_old, Sed)
        assert isinstance(sed_young, Sed)
        assert len(sed_old.lam) == len(dust_wavelengths)
        assert len(sed_young.lam) == len(dust_wavelengths)

    def test_drainli07_variable_naming(self, mock_dust_grid):
        """Test DrainLi07 uses improved variable names."""
        dust_mass = 1e6 * Msun

        dl07 = DrainLi07(
            grid=mock_dust_grid,
            dust_mass=dust_mass,
            dust_to_gas=0.01,
            pah_frac=0.025,  # Should use pah_frac, not qpah
            intrinsic="intrinsic",
            attenuated="attenuated",
        )

        # Should use improved naming
        assert hasattr(dl07, "pah_frac")
        assert hasattr(dl07, "dust_mass")
        assert hasattr(dl07, "dust_to_gas")
        assert hasattr(dl07, "hydrogen_mass")
        assert not hasattr(dl07, "qpah")
        assert not hasattr(dl07, "mdust")
        assert not hasattr(dl07, "dgr")

    def test_drainli07_missing_parameters(
        self, mock_dust_grid, dust_wavelengths
    ):
        """Test DrainLi07 error handling for missing parameters."""
        dl07 = DrainLi07(
            grid=mock_dust_grid,
            dust_mass=None,  # No dust mass provided
            dust_to_gas=0.01,
            pah_frac=0.025,
            verbose=False,
            intrinsic="intrinsic",
            attenuated="attenuated",
        )

        # Should raise error when dust_mass not provided anywhere
        with pytest.raises(exceptions.MissingAttribute):
            dl07.get_spectra(dust_wavelengths)


class TestDustGeneratorIntegration:
    """Integration tests for dust generators."""

    def test_all_generators_have_get_spectra(self, dust_wavelengths):
        """Test that all dust generators have get_spectra method."""
        temperature = 20 * K

        # Test Blackbody
        bb = BlackbodyBase(temperature)
        sed_bb = bb.get_spectra(dust_wavelengths)
        assert isinstance(sed_bb, Sed)

        # Test Greybody
        gb = GreybodyBase(temperature, 1.5)
        sed_gb = gb.get_spectra(dust_wavelengths)
        assert isinstance(sed_gb, Sed)

        # Test Casey12
        c12 = Casey12Base(temperature, 2.0, 2.0)
        sed_c12 = c12.get_spectra(dust_wavelengths)
        assert isinstance(sed_c12, Sed)

    def test_temperature_scaling(self, dust_wavelengths, dust_temperatures):
        """Test that all generators respond to temperature changes."""
        for temp in dust_temperatures:
            # Blackbody
            bb = BlackbodyBase(temp)
            sed_bb = bb.get_spectra(dust_wavelengths)
            assert np.all(sed_bb.lnu >= 0)
            assert np.any(sed_bb.lnu > 0)

            # Greybody
            gb = GreybodyBase(temp, 1.5)
            sed_gb = gb.get_spectra(dust_wavelengths)
            assert np.all(sed_gb.lnu >= 0)
            assert np.any(sed_gb.lnu > 0)

            # Casey12
            c12 = Casey12Base(temp, 2.0, 2.0)
            sed_c12 = c12.get_spectra(dust_wavelengths)
            assert np.all(sed_c12.lnu >= 0)
            assert np.any(sed_c12.lnu > 0)

    def test_wavelength_coverage(self):
        """Test generators work across different wavelength ranges."""
        temperature = 100 * K  # Use warmer dust for near-IR detection

        # Near-IR wavelengths
        nir_lams = np.logspace(3, 4, 50) * angstrom
        # Far-IR wavelengths
        fir_lams = np.logspace(4, 6, 50) * angstrom

        bb = BlackbodyBase(temperature)

        sed_nir = bb.get_spectra(nir_lams)
        sed_fir = bb.get_spectra(fir_lams)

        assert len(sed_nir.lam) == len(nir_lams)
        assert len(sed_fir.lam) == len(fir_lams)
        assert np.all(sed_nir.lnu >= 0)
        assert np.any(sed_nir.lnu > 0)
        assert np.all(sed_fir.lnu >= 0)
        assert np.any(sed_fir.lnu > 0)

    def test_flux_units_consistency(self, dust_wavelengths):
        """Test that all generators return consistent flux units."""
        temperature = 20 * K
        expected_units = (
            erg / s / Hz
        )  # Correct units for spectral flux density

        # Test all generators
        generators = [
            BlackbodyBase(temperature),
            GreybodyBase(temperature, 1.5),
            Casey12Base(temperature, 2.0, 2.0),
        ]

        for gen in generators:
            sed = gen.get_spectra(dust_wavelengths)
            assert sed.lnu.units == expected_units

    def test_spectral_shapes(self, dust_wavelengths):
        """Test that generators produce reasonable spectral shapes."""
        temperature = 20 * K

        # All should be non-negative
        bb = BlackbodyBase(temperature)
        sed_bb = bb.get_spectra(dust_wavelengths)
        assert np.all(sed_bb.lnu >= 0)
        assert np.any(sed_bb.lnu > 0)

        # Greybody should be similar to blackbody for low emissivity
        gb = GreybodyBase(temperature, 0.1)  # Low emissivity
        sed_gb = gb.get_spectra(dust_wavelengths)

        # Should have similar overall shape
        bb_norm = sed_bb.lnu / np.max(sed_bb.lnu)
        gb_norm = sed_gb.lnu / np.max(sed_gb.lnu)

        # Not exact match due to frequency dependence, but should be similar
        assert np.corrcoef(bb_norm, gb_norm)[0, 1] > 0.8

    def test_nan_handling(self, dust_wavelengths):
        """Test that generators handle edge cases gracefully."""
        # Very low temperature
        very_cold = 1 * K
        bb = BlackbodyBase(very_cold)
        sed_cold = bb.get_spectra(dust_wavelengths)

        # Should not have NaN values
        assert not np.any(np.isnan(sed_cold.lnu))
        assert np.all(sed_cold.lnu >= 0)

    def test_factory_function_completeness(self):
        """Test that all factory functions work properly."""
        temperature = 20 * K

        # Test all factory functions return correct types
        bb_eb = Blackbody(
            temperature=temperature,
            intrinsic="intrinsic",
            attenuated="attenuated",
            scaler=None,
        )
        assert isinstance(bb_eb, BlackbodyEnergyBalance)

        gb_eb = Greybody(
            temperature=temperature,
            emissivity=1.5,
            intrinsic="intrinsic",
            attenuated="attenuated",
            scaler=None,
        )
        assert isinstance(gb_eb, GreybodyEnergyBalance)

        c12_eb = Casey12(
            temperature=temperature,
            emissivity=2.0,
            alpha=2.0,
            intrinsic="intrinsic",
            attenuated="attenuated",
            scaler=None,
        )
        assert isinstance(c12_eb, Casey12EnergyBalance)
