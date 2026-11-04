"""Comprehensive test suite for dust emission generators.

This module contains tests for all dust generator functionality including:
- Base dust emission utility functions
- Blackbody emission generators
- Greybody emission generators
- Casey12 emission generators
- DraineLi07 emission generators
- Integration and functionality tests
"""

import warnings
from pathlib import Path

import h5py
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


def _draineli07_templates(qpah, umin, alpha, nlam):
    """Return analytic DL07 templates, linear in every grid axis."""
    # qpah gets its own wavelength dependence so it changes the spectral
    # shape (and not just the normalisation)
    shape = np.linspace(1.0, 2.0, nlam)
    ramp = np.linspace(0.0, 1.0, nlam)
    diffuse = (1.0 + umin)[..., np.newaxis] * shape + (10.0 * qpah)[
        ..., np.newaxis
    ] * ramp
    pdr = (1.0 + umin + alpha)[..., np.newaxis] * shape + (10.0 * qpah)[
        ..., np.newaxis
    ] * ramp
    return diffuse, pdr


def _write_draineli07_grid(path: Path, log_on_read=False):
    """Write a minimal DL07 dust emission grid in the standard Grid format."""
    qpah = np.array([0.01, 0.025, 0.05])
    umin = np.array([0.1, 1.0, 10.0, 100.0])
    alpha = np.array([1.5, 2.0, 2.5])
    nlam = 100
    diffuse, pdr = _draineli07_templates(
        qpah[:, None, None], umin[None, :, None], alpha[None, None, :], nlam
    )

    with h5py.File(path, "w") as hdf:
        hdf.attrs["axes"] = np.array(["qpah", "umin", "alpha"], dtype=object)
        hdf.attrs["WeightVariable"] = "None"

        axes = hdf.create_group("axes")
        for name, values in (("qpah", qpah), ("umin", umin), ("alpha", alpha)):
            axis = axes.create_dataset(name, data=values)
            axis.attrs["Units"] = "dimensionless"
            axis.attrs["log_on_read"] = log_on_read and name == "umin"

        spectra = hdf.create_group("spectra")
        wavelength = spectra.create_dataset(
            "wavelength", data=np.logspace(3, 7, nlam)
        )
        wavelength.attrs["Units"] = "Å"
        spectra.create_dataset("diffuse", data=diffuse[:, :, 0, :])
        spectra.create_dataset("pdr", data=pdr)


@pytest.fixture
def draineli07_grid(tmp_path):
    """Create a temporary DL07 dust emission grid."""
    grid_path = tmp_path / "draineli07_test_grid.hdf5"
    _write_draineli07_grid(grid_path)
    return grid_path


@pytest.fixture
def draineli07_log_grid(tmp_path):
    """Create a temporary DL07 dust emission grid with log umin."""
    grid_path = tmp_path / "draineli07_log_test_grid.hdf5"
    _write_draineli07_grid(grid_path, log_on_read=True)
    return grid_path


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
            self.model_param_cache = {}

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

        class MockEmitterWithTemp:
            def __init__(self, **kwargs):
                self.model_param_cache = {}
                self.__dict__.update(kwargs)

        def generate(gen, emitter):
            return gen._generate_spectra(
                lams=dust_wavelengths,
                emitter=emitter,
                model=mock_model,
                emissions=mock_emissions,
                redshift=0.0,
            ).lnu

        reference = generate(
            Blackbody(temperature=50 * K), MockEmitterWithTemp()
        )

        # With no temperature on the generator, the emitter's is used
        from_emitter = generate(
            Blackbody(), MockEmitterWithTemp(temperature=50 * K)
        )
        np.testing.assert_allclose(from_emitter, reference, rtol=1e-12)

        # A temperature set on the generator overrides the emitter's
        overridden = generate(
            Blackbody(temperature=50 * K),
            MockEmitterWithTemp(temperature=30 * K),
        )
        np.testing.assert_allclose(overridden, reference, rtol=1e-12)

    def test_blackbody_base_properties(self):
        """Test base properties of Blackbody generator."""
        temperature = 20 * K
        bb = Blackbody(temperature=temperature, do_cmb_heating=True)

        # Test apply_cmb_heating method
        cmb_factor, new_temp = bb.apply_cmb_heating(
            temperature=temperature, emissivity=1.0, redshift=2.0
        )
        assert cmb_factor > 1.0
        assert new_temp > temperature

    def test_cmb_heating_not_stored_on_shared_generator(self):
        """Test calls with different redshifts can't see each other's result.

        One generator can be shared by several models, so the effective
        temperature of each call must come from its return value rather than
        from the generator.
        """
        temperature = 20 * K
        bb = Blackbody(temperature=temperature, do_cmb_heating=True)

        factor_z2, temp_z2 = bb.apply_cmb_heating(
            temperature=temperature, emissivity=1.0, redshift=2.0
        )
        factor_z8, temp_z8 = bb.apply_cmb_heating(
            temperature=temperature, emissivity=1.0, redshift=8.0
        )

        # The first call's results are unaffected by the second
        assert temp_z2 < temp_z8
        assert factor_z2 != factor_z8
        assert bb.apply_cmb_heating(
            temperature=temperature, emissivity=1.0, redshift=2.0
        ) == (factor_z2, temp_z2)

        # And nothing per-call is left on the generator
        assert not hasattr(bb, "temperature_z")
        assert not hasattr(bb, "cmb_factor")


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
        """Test Casey12 power-law parameter calculations."""
        from unyt import c, h, kb

        c12 = Casey12()
        temperature, alpha = 25 * K, 2.0

        # lam_c and n_pl are derived on demand from the parameters
        lam_c, n_pl = c12._power_law_params(temperature, 2.0, alpha)
        assert lam_c > 0 * um
        assert n_pl > 0

        # lam_c (Casey+12, Table 1) depends on temperature and alpha only
        expected_lam_c = (
            0.75
            / (
                (26.68 + 6.246 * alpha) ** -2
                + (0.0001905 + 0.00007243 * alpha) * 25.0
            )
            * um
        )
        np.testing.assert_allclose(lam_c.to("um").value, expected_lam_c.value)

        # n_pl depends on the emissivity through the optical depth at lam_c
        def expected_n_pl(emissivity):
            return (
                c12.n_bb
                * (1 - np.exp(-((c12.lam_0 / lam_c) ** emissivity)))
                * (c / lam_c) ** 3
                / (np.exp(h * c / (lam_c * kb * temperature)) - 1)
            )

        n_pls = []
        for emissivity in (1.2, 2.0, 2.5):
            lam_c_e, n_pl_e = c12._power_law_params(
                temperature, emissivity, alpha
            )
            np.testing.assert_allclose(lam_c_e.value, lam_c.value)
            np.testing.assert_allclose(
                n_pl_e.value, expected_n_pl(emissivity).to(n_pl_e.units).value
            )
            n_pls.append(n_pl_e.value)

        # ...and different emissivities give different normalisations
        assert len(set(np.round(n_pls, 12))) == 3

    def test_casey12_component_methods(self, dust_wavelengths):
        """Test Casey12 component calculation methods."""
        temperature = 20 * K
        c12 = Casey12(temperature=temperature, emissivity=2.0, alpha=2.0)

        nu = (3e10 / (dust_wavelengths.to("cm").value)) * Hz  # c in cm/s

        # Test _lnu method
        lnu = c12._lnu(nu, temperature, 2.0, 2.0)
        assert len(lnu) == len(nu)
        assert np.all(lnu >= 0)
        assert np.any(lnu > 0)

    def test_casey12_required_params(self):
        """Emissivity and alpha can be supplied by the emitter."""
        c12 = Casey12(temperature=20 * K)
        assert set(c12._required_params) >= {
            "temperature",
            "emissivity",
            "alpha",
        }

    def test_casey12_per_emitter_emissivity_and_alpha(
        self, dust_wavelengths, mock_model, mock_emissions
    ):
        """Unset generator params come from the emitter; set ones override."""

        class Emitter:
            def __init__(self, **kwargs):
                self.model_param_cache = {}
                self.__dict__.update(kwargs)

        def generate(gen, emitter):
            return gen._generate_spectra(
                lams=dust_wavelengths,
                emitter=emitter,
                model=mock_model,
                emissions=mock_emissions,
                redshift=0.0,
            ).lnu

        reference = generate(
            Casey12(temperature=45 * K, emissivity=2.1, alpha=1.3),
            Emitter(),
        )

        # Parameters left as None on the generator are read from the emitter
        from_emitter = generate(
            Casey12(),
            Emitter(temperature=45 * K, emissivity=2.1, alpha=1.3),
        )
        np.testing.assert_allclose(from_emitter, reference, rtol=1e-12)

        # Parameters set on the generator override those on the emitter
        overridden = generate(
            Casey12(temperature=45 * K, emissivity=2.1, alpha=1.3),
            Emitter(temperature=30 * K, emissivity=1.5, alpha=2.0),
        )
        np.testing.assert_allclose(overridden, reference, rtol=1e-12)

        # A mix: alpha fixed on the generator, the rest from the emitter
        mixed = generate(
            Casey12(alpha=1.3),
            Emitter(temperature=45 * K, emissivity=2.1, alpha=2.0),
        )
        np.testing.assert_allclose(mixed, reference, rtol=1e-12)

        # Changing alpha changes the spectrum
        other_alpha = generate(
            Casey12(alpha=2.0),
            Emitter(temperature=45 * K, emissivity=2.1),
        )
        assert not np.allclose(other_alpha, reference, atol=0)

    def test_casey12_model_fixed_params_override_generator(
        self, dust_wavelengths, mock_model, mock_emissions
    ):
        """EmissionModel fixed parameters still beat generator values."""

        class Emitter:
            def __init__(self, **kwargs):
                self.model_param_cache = {}
                self.__dict__.update(kwargs)

        reference = (
            Casey12(temperature=45 * K, emissivity=2.1, alpha=1.3)
            ._generate_spectra(
                lams=dust_wavelengths,
                emitter=Emitter(),
                model=mock_model,
                emissions=mock_emissions,
                redshift=0.0,
            )
            .lnu
        )

        mock_model.fixed_parameters = {"alpha": 1.3}
        overridden = (
            Casey12(temperature=45 * K, emissivity=2.1, alpha=2.0)
            ._generate_spectra(
                lams=dust_wavelengths,
                emitter=Emitter(alpha=2.0),
                model=mock_model,
                emissions=mock_emissions,
                redshift=0.0,
            )
            .lnu
        )
        np.testing.assert_allclose(overridden, reference, rtol=1e-12)

    def test_casey12_missing_params_raise(
        self, dust_wavelengths, mock_model, mock_emissions
    ):
        """Params unset on both the generator and emitter raise."""

        class Emitter:
            def __init__(self, **kwargs):
                self.model_param_cache = {}
                self.__dict__.update(kwargs)

        with pytest.raises(exceptions.MissingAttribute, match="alpha"):
            Casey12(temperature=45 * K, emissivity=2.1)._generate_spectra(
                lams=dust_wavelengths,
                emitter=Emitter(),
                model=mock_model,
                emissions=mock_emissions,
                redshift=0.0,
            )

    def test_casey12_no_stored_power_law_params(self):
        """lam_c and n_pl are derived on demand, never stored."""
        c12 = Casey12(temperature=45 * K, emissivity=2.1, alpha=1.3)
        assert not hasattr(c12, "lam_c")
        assert not hasattr(c12, "n_pl")

    def test_casey12_lines_use_emitter_params(
        self, mock_model, mock_emissions
    ):
        """Line generation uses the same per-emitter parameters."""

        class Emitter:
            def __init__(self, **kwargs):
                self.model_param_cache = {}
                self.__dict__.update(kwargs)

        line_lams = np.array([1e5, 5e5]) * angstrom

        def lines(gen, emitter):
            return gen._generate_lines(
                line_ids=["a", "b"],
                line_lams=line_lams,
                emitter=emitter,
                model=mock_model,
                emissions=mock_emissions,
                spectra=mock_emissions,
                redshift=0.0,
            ).continuum

        from_emitter = lines(
            Casey12(),
            Emitter(temperature=45 * K, emissivity=2.1, alpha=1.3),
        )
        from_instance = lines(
            Casey12(temperature=45 * K, emissivity=2.1, alpha=1.3),
            Emitter(),
        )
        np.testing.assert_allclose(from_emitter, from_instance, rtol=1e-12)

    def test_casey12_cmb_heating_uses_emitter_emissivity(
        self, dust_wavelengths, mock_model, mock_emissions
    ):
        """CMB heating uses the emitter's emissivity in spectra and lines."""

        class Emitter:
            def __init__(self, **kwargs):
                self.model_param_cache = {}
                self.__dict__.update(kwargs)

        redshift = 6.0
        line_lams = np.array([1e5, 5e5]) * angstrom

        def spectra(gen, emitter):
            return gen._generate_spectra(
                lams=dust_wavelengths,
                emitter=emitter,
                model=mock_model,
                emissions=mock_emissions,
                redshift=redshift,
            ).lnu

        def lines(gen, emitter):
            return gen._generate_lines(
                line_ids=["a", "b"],
                line_lams=line_lams,
                emitter=emitter,
                model=mock_model,
                emissions=mock_emissions,
                spectra=mock_emissions,
                redshift=redshift,
            ).continuum

        for generate in (spectra, lines):
            from_instance = generate(
                Casey12(
                    temperature=30 * K,
                    emissivity=1.2,
                    alpha=2.0,
                    do_cmb_heating=True,
                ),
                Emitter(),
            )
            from_emitter = generate(
                Casey12(temperature=30 * K, alpha=2.0, do_cmb_heating=True),
                Emitter(emissivity=1.2),
            )
            np.testing.assert_allclose(from_emitter, from_instance, rtol=1e-12)

            # A different emissivity gives a different heated result, so
            # the comparison above is sensitive to which value is used
            other = generate(
                Casey12(
                    temperature=30 * K,
                    emissivity=2.2,
                    alpha=2.0,
                    do_cmb_heating=True,
                ),
                Emitter(),
            )
            assert not np.allclose(other, from_instance, atol=0)


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

    def test_draineli07_initialization(self, draineli07_grid):
        """Test DraineLi07 initialization."""
        dust_mass = 1e6 * Msun

        dl07 = DraineLi07(
            grid_name=draineli07_grid.name,
            grid_dir=draineli07_grid.parent,
            dust_mass=dust_mass,
            dust_to_gas_ratio=0.01,
            qpah=0.025,
            intrinsic="intrinsic",
            attenuated="attenuated",
        )

        assert dl07.grid_name == draineli07_grid.name
        assert dl07.dust_mass == dust_mass
        assert dl07.dust_to_gas_ratio == 0.01
        assert dl07.qpah == 0.025
        assert dl07._required_params == ("dust_mass",)

    def test_draineli07_variable_naming(self, draineli07_grid):
        """Test DraineLi07 uses improved variable names."""
        dust_mass = 1e6 * Msun

        dl07 = DraineLi07(
            grid_name=draineli07_grid.name,
            grid_dir=draineli07_grid.parent,
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

    def test_draineli07_has_no_cmb_heating_option(self, draineli07_grid):
        """Test CMB heating can't be requested, rather than being ignored."""
        with pytest.raises(TypeError):
            DraineLi07(
                grid_name=draineli07_grid.name,
                grid_dir=draineli07_grid.parent,
                dust_mass=1e6 * Msun,
                do_cmb_heating=True,
            )

    def test_draineli07_parameter_setup(self, draineli07_grid):
        """Test DraineLi07 parameter setup."""
        dust_mass = 1e6 * Msun
        ldust = 1e10 * Lsun

        dl07 = DraineLi07(
            grid_name=draineli07_grid.name,
            grid_dir=draineli07_grid.parent,
            dust_mass=dust_mass,
            dust_to_gas_ratio=0.01,
            qpah=0.025,
            verbose=False,
            intrinsic="intrinsic",
            attenuated="attenuated",
        )

        # Test parameter setup
        setup = dl07._setup_dl07_parameters(dust_mass, ldust)

        # Check that calculated values are returned
        assert setup["radiation_field_average"] is not None
        assert setup["umin_calculated"] is not None
        assert setup["gamma_calculated"] is not None
        assert setup["hydrogen_mass_calculated"] is not None

        # Check nothing per-call is left on the shared generator
        assert not hasattr(dl07, "radiation_field_average")
        assert not hasattr(dl07, "u_avg")

    def test_draineli07_get_spectra(self, draineli07_grid, dust_wavelengths):
        """Test DraineLi07 get_spectra method."""
        dust_mass = 1e6 * Msun
        ldust = 1e10 * Lsun

        dl07 = DraineLi07(
            grid_name=draineli07_grid.name,
            grid_dir=draineli07_grid.parent,
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
        self, draineli07_grid, dust_wavelengths
    ):
        """Test DraineLi07 dust component separation."""
        dust_mass = 1e6 * Msun
        ldust = 1e10 * Lsun

        dl07 = DraineLi07(
            grid_name=draineli07_grid.name,
            grid_dir=draineli07_grid.parent,
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
        assert sed_diffuse.lnu.dtype == dl07.grid._dtype
        assert sed_pdr.lnu.dtype == dl07.grid._dtype

    def test_draineli07_leaves_grid_unchanged(
        self, draineli07_grid, dust_wavelengths
    ):
        """Test generating spectra doesn't resample the shared grid."""
        dl07 = DraineLi07(
            grid_name=draineli07_grid.name,
            grid_dir=draineli07_grid.parent,
            dust_mass=1e6 * Msun,
            dust_to_gas_ratio=0.01,
            qpah=0.025,
            verbose=False,
        )
        lam = dl07.grid._lam.copy()
        diffuse = dl07.grid.spectra["diffuse"].copy()
        pdr = dl07.grid.spectra["pdr"].copy()

        dl07.get_spectra(dust_wavelengths)

        np.testing.assert_array_equal(dl07.grid._lam, lam)
        np.testing.assert_array_equal(dl07.grid.spectra["diffuse"], diffuse)
        np.testing.assert_array_equal(dl07.grid.spectra["pdr"], pdr)


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

    def test_energy_balance_conserves_absorbed_luminosity(
        self, dust_wavelengths, mock_emitter, mock_model, mock_emissions
    ):
        """Energy-balance dust emission must conserve absorbed energy.

        The physical invariant "energy balance" mode is named for: the
        dust-emitted bolometric luminosity should equal the energy
        removed from the stellar continuum by attenuation, i.e.
        intrinsic.bolometric_luminosity - attenuated.bolometric_luminosity.
        The existing test_energy_balance_scaling only checks the result is
        a positive, non-empty Sed; it does not check this quantitatively.
        """
        temperature = 20 * K
        expected_ldust = (
            mock_emissions["intrinsic"].bolometric_luminosity
            - mock_emissions["attenuated"].bolometric_luminosity
        )

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

            assert np.isclose(
                sed.bolometric_luminosity.to("erg/s").value,
                expected_ldust.to("erg/s").value,
                rtol=1e-6,
            ), (
                f"{gen.__class__.__name__} energy-balance dust emission "
                f"({sed.bolometric_luminosity}) does not match the "
                f"absorbed stellar luminosity ({expected_ldust})"
            )

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

        # Should raise MissingAttribute when the emission is missing
        with pytest.raises(exceptions.MissingAttribute):
            bb._generate_spectra(
                lams=dust_wavelengths,
                emitter=mock_emitter,
                model=mock_model,
                emissions=empty_emissions,
                redshift=0.0,
            )

    def test_draineli07_invalid_template(self, draineli07_grid):
        """Test DraineLi07 error for invalid template."""
        dl07 = DraineLi07(
            grid_name=draineli07_grid.name,
            grid_dir=draineli07_grid.parent,
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
        """Test parameter extraction priority (model > generator > emitter)."""

        class MockEmitterWithTemp:
            temperature = 40 * K

            def __init__(self):
                self.model_param_cache = {}

        def temperature_used():
            return Blackbody(temperature=30 * K)._extract_params(
                mock_model, MockEmitterWithTemp()
            )["temperature"]

        # The generator temperature overrides the emitter's
        assert temperature_used() == 30 * K

        # A model fixed parameter overrides both
        mock_model.fixed_parameters = {"temperature": 50 * K}
        assert temperature_used() == 50 * K

        # Reset mock_model for other tests
        mock_model.fixed_parameters = {}


class _ParamEmitter:
    """A minimal emitter carrying arbitrary parameters."""

    def __init__(self, **kwargs):
        self.model_param_cache = {}
        self.__dict__.update(kwargs)


# Generator classes using the generator > emitter priority, with a full set
# of parameter values, an alternative set differing in every parameter, and
# any fixed (non-overridable) constructor arguments
OVERRIDE_GENERATORS = [
    pytest.param(
        Blackbody,
        {"temperature": 45 * K},
        {"temperature": 30 * K},
        {},
        id="Blackbody",
    ),
    pytest.param(
        Greybody,
        {"temperature": 45 * K, "emissivity": 2.1},
        {"temperature": 30 * K, "emissivity": 1.2},
        {},
        id="Greybody",
    ),
    pytest.param(
        Greybody,
        {"temperature": 45 * K, "emissivity": 2.1},
        {"temperature": 30 * K, "emissivity": 1.2},
        {"optically_thin": False},
        id="Greybody-optically-thick",
    ),
    pytest.param(
        Casey12,
        {"temperature": 45 * K, "emissivity": 2.1, "alpha": 1.3},
        {"temperature": 30 * K, "emissivity": 1.2, "alpha": 2.0},
        {},
        id="Casey12",
    ),
]


@pytest.mark.parametrize("gen_cls,params,other,fixed", OVERRIDE_GENERATORS)
class TestGeneratorOverridesEmitter:
    """Generator values override the emitter (model fixed params win)."""

    @staticmethod
    def _spectra(gen, emitter, lams, model, emissions, redshift=0.0):
        return gen._generate_spectra(
            lams=lams,
            emitter=emitter,
            model=model,
            emissions=emissions,
            redshift=redshift,
        ).lnu

    @staticmethod
    def _lines(gen, emitter, model, emissions, redshift=0.0):
        return gen._generate_lines(
            line_ids=["a", "b"],
            line_lams=np.array([1e5, 5e5]) * angstrom,
            emitter=emitter,
            model=model,
            emissions=emissions,
            spectra=emissions,
            redshift=redshift,
        ).continuum

    def test_params_default_to_none(self, gen_cls, params, other, fixed):
        """Overridable parameters are None unless given."""
        gen = gen_cls(**fixed)
        for param in params:
            assert getattr(gen, param) is None

    @pytest.mark.parametrize("redshift", [0.0, 6.0])
    def test_emitter_and_generator_priority(
        self,
        gen_cls,
        params,
        other,
        fixed,
        dust_wavelengths,
        mock_model,
        mock_emissions,
        redshift,
    ):
        """Unset params come from the emitter; set ones override it.

        At nonzero redshift with CMB heating this also checks that the
        heating uses the resolved (e.g. emitter) emissivity.
        """
        kw = {"do_cmb_heating": redshift > 0}
        for generate in ("spectra", "lines"):

            def run(gen, emitter, generate=generate):
                if generate == "spectra":
                    return self._spectra(
                        gen,
                        emitter,
                        dust_wavelengths,
                        mock_model,
                        mock_emissions,
                        redshift,
                    )
                return self._lines(
                    gen, emitter, mock_model, mock_emissions, redshift
                )

            reference = run(gen_cls(**fixed, **params, **kw), _ParamEmitter())

            # All parameters read from the emitter
            from_emitter = run(gen_cls(**fixed, **kw), _ParamEmitter(**params))
            np.testing.assert_allclose(from_emitter, reference, rtol=1e-12)

            # All parameters set on the generator override the emitter
            overridden = run(
                gen_cls(**fixed, **params, **kw), _ParamEmitter(**other)
            )
            np.testing.assert_allclose(overridden, reference, rtol=1e-12)

            # Each parameter individually: set on the generator, with the
            # emitter holding a different value for it and the rest
            for param in params:
                mixed = run(
                    gen_cls(**fixed, **{param: params[param]}, **kw),
                    _ParamEmitter(**{**params, param: other[param]}),
                )
                np.testing.assert_allclose(mixed, reference, rtol=1e-12)

                # ...and the comparison is sensitive to that parameter
                changed = run(
                    gen_cls(**fixed, **kw),
                    _ParamEmitter(**{**params, param: other[param]}),
                )
                assert not np.allclose(changed, reference, atol=0), param

    def test_model_fixed_params_beat_generator(
        self, gen_cls, params, other, fixed, mock_model
    ):
        """EmissionModel fixed parameters still beat generator values."""
        gen = gen_cls(**fixed, **other)
        for param in params:
            mock_model.fixed_parameters = {param: params[param]}
            extracted = gen._extract_params(mock_model, _ParamEmitter(**other))
            assert extracted[param] == params[param]
            for p in params:
                if p != param:
                    assert extracted[p] == other[p]
        mock_model.fixed_parameters = {}

    def test_missing_params_raise(
        self, gen_cls, params, other, fixed, mock_model
    ):
        """Params unset on the generator, model and emitter raise."""
        for param in params:
            given = {p: v for p, v in params.items() if p != param}
            with pytest.raises(exceptions.MissingAttribute, match=param):
                gen_cls(**fixed, **given)._extract_params(
                    mock_model, _ParamEmitter()
                )

    def test_get_spectra_requires_generator_params(
        self, gen_cls, params, other, fixed, dust_wavelengths
    ):
        """Standalone get_spectra has no emitter, so params must be set."""
        assert np.any(
            gen_cls(**fixed, **params).get_spectra(dust_wavelengths).lnu > 0
        )
        with pytest.raises(exceptions.MissingAttribute):
            gen_cls(**fixed).get_spectra(dust_wavelengths)

    def test_cmb_heating_uses_resolved_emissivity(
        self,
        gen_cls,
        params,
        other,
        fixed,
        dust_wavelengths,
        mock_model,
        mock_emissions,
    ):
        """CMB heating gets the emissivity resolved for this emitter.

        Comparing outputs can't isolate this, since the emissivity also
        enters the spectral shape, so record what apply_cmb_heating receives.
        Blackbody always heats with an emissivity of 1.
        """
        expected = params.get("emissivity", 1.0)
        for gen, emitter in (
            # Emissivity only on the emitter
            (
                gen_cls(**fixed, do_cmb_heating=True),
                _ParamEmitter(**params),
            ),
            # Generator value overriding a different emitter value
            (
                gen_cls(**fixed, **params, do_cmb_heating=True),
                _ParamEmitter(**other),
            ),
        ):
            seen = []
            real = gen.apply_cmb_heating

            def spy(temperature, emissivity, redshift, real=real, seen=seen):
                seen.append(emissivity)
                return real(temperature, emissivity, redshift)

            gen.apply_cmb_heating = spy
            self._spectra(
                gen,
                emitter,
                dust_wavelengths,
                mock_model,
                mock_emissions,
                6.0,
            )
            self._lines(gen, emitter, mock_model, mock_emissions, 6.0)
            assert seen == [expected, expected]

        # The emissivity changes the heating, so the check above matters
        factors = {
            emissivity: get_cmb_heating_factor(45 * K, emissivity, 6.0)[0]
            for emissivity in (1.0, 1.2, 2.1)
        }
        assert len(set(np.round(list(factors.values()), 12))) == 3


def test_draineli07_generator_overrides_emitter(draineli07_grid, mock_model):
    """DraineLi07 values override the emitter, unset ones come from it."""
    gen = DraineLi07(
        grid_name=draineli07_grid.name,
        grid_dir=draineli07_grid.parent,
        dust_mass=1e6 * Msun,
        qpah=0.025,
    )
    emitter = _ParamEmitter(
        qpah=0.04, dust_to_gas_ratio=0.02, hydrogen_mass=1e8 * Msun
    )
    params = gen._extract_optional_params(mock_model, emitter)

    # The generator's values win
    assert params["qpah"] == 0.025

    # Values not set on the generator come from the emitter, rather than
    # being overridden by a generator default
    assert params["dust_to_gas_ratio"] == 0.02
    assert params["hydrogen_mass"] == 1e8 * Msun

    # Values set nowhere are left for the setup to derive or default
    assert params["gamma"] is None
    assert params["umin"] is None
    assert params["alpha"] is None


class FakeEmissionModel:
    """A minimal stand in for an EmissionModel carrying only a label."""

    def __init__(self, label):
        """Initialise the fake model.

        Args:
            label (str):
                The label of the fake model.
        """
        self.label = label


class TestRequiredEmissions:
    """Tests for the required emissions bookkeeping on dust generators."""

    def test_setters_update_required_emissions(self):
        """Test the setters keep _required_emissions up to date."""
        intrinsic = FakeEmissionModel("intrinsic")
        attenuated = FakeEmissionModel("attenuated")
        scaler = FakeEmissionModel("scaler")

        bb = Blackbody(temperature=20 * K, scaler=scaler)
        assert bb._required_emissions == (scaler,)

        # Switching to energy balance must repoint the required emissions
        bb.set_energy_balance(intrinsic, attenuated)
        assert bb._required_emissions == (intrinsic, attenuated)

        # And switching back to a scaler must too
        bb.set_scaler(scaler)
        assert bb._required_emissions == (scaler,)

    def test_required_emissions_resolve_to_labels(self, mock_emissions):
        """Test model objects are resolved into emission dictionary keys."""
        intrinsic = FakeEmissionModel("intrinsic")
        attenuated = FakeEmissionModel("attenuated")

        bb = Blackbody(
            temperature=20 * K,
            intrinsic=intrinsic,
            attenuated=attenuated,
        )

        extracted = bb._extract_emissions(mock_emissions)
        assert extracted["intrinsic"] is mock_emissions["intrinsic"]
        assert extracted["attenuated"] is mock_emissions["attenuated"]

        # Labels are resolved lazily so relabelling a model is picked up
        attenuated.label = "test_model"
        extracted = bb._extract_emissions(mock_emissions)
        assert extracted["test_model"] is mock_emissions["test_model"]

    def test_missing_emission_raises_missing_attribute(self, mock_emissions):
        """Test a missing dependency raises MissingAttribute, not KeyError."""
        bb = Blackbody(
            temperature=20 * K,
            intrinsic=FakeEmissionModel("intrinsic"),
            attenuated=FakeEmissionModel("missing"),
        )

        with pytest.raises(exceptions.MissingAttribute):
            bb.get_scaling(None, None, mock_emissions)

        bb.set_scaler(FakeEmissionModel("missing"))
        with pytest.raises(exceptions.MissingAttribute):
            bb.get_scaling(None, None, mock_emissions)

    def test_scaling_after_switching_type(self, mock_emissions):
        """Test the scaling is computed from the emissions set last."""
        bb = Blackbody(
            temperature=20 * K,
            intrinsic=FakeEmissionModel("intrinsic"),
            attenuated=FakeEmissionModel("attenuated"),
        )

        ldust = bb.get_scaling(None, None, mock_emissions)
        assert ldust == (
            mock_emissions["intrinsic"].bolometric_luminosity
            - mock_emissions["attenuated"].bolometric_luminosity
        )

        # Switching to a scaler must use the scaler's luminosity
        bb.set_scaler(FakeEmissionModel("scaler"))
        lscale = bb.get_scaling(None, None, mock_emissions)
        assert lscale == mock_emissions["scaler"].bolometric_luminosity


def _draineli07_setup(dl07, **params):
    """Set up DL07 parameters, with gamma and umin given unless overridden."""
    values = dict(
        qpah=0.025,
        gamma=0.1,
        umin=1.0,
        alpha=2.0,
        hydrogen_mass=1e8 * Msun,
    )
    values.update(params)
    dust_mass = params.pop("dust_mass", 1e6 * Msun)
    values.pop("dust_mass", None)
    return dl07._setup_dl07_parameters(dust_mass, 1e10 * Lsun, **values)


def _expected_templates(qpah, umin, alpha, nlam=100):
    """Return the analytic templates of the test grid at some parameters."""
    return _draineli07_templates(
        np.asarray(qpah, dtype=float),
        np.asarray(umin, dtype=float),
        np.asarray(alpha, dtype=float),
        nlam,
    )


def test_draineli07_uses_grid_extraction(draineli07_grid):
    """Particle extraction reproduces the expected templates."""
    dl07 = DraineLi07(
        grid_name=draineli07_grid.name,
        grid_dir=draineli07_grid.parent,
    )

    qpah = np.array([0.025, 0.0375])
    umin = np.array([1.0, 5.5])
    alpha = np.array([2.0, 2.25])
    setup = _draineli07_setup(
        dl07,
        dust_mass=np.full(2, 1e6) * Msun,
        qpah=qpah,
        umin=umin,
        alpha=alpha,
    )

    diffuse, pdr = dl07._get_templates_at_lam(dl07.grid.lam, setup)
    expected_diffuse, expected_pdr = _expected_templates(qpah, umin, alpha)

    np.testing.assert_allclose(diffuse, expected_diffuse)
    np.testing.assert_allclose(pdr, expected_pdr)


def test_draineli07_masks_zero_and_nan_inputs(draineli07_grid):
    """Zero and NaN dust masses and parameters should give no emission."""
    dl07 = DraineLi07(
        grid_name=draineli07_grid.name,
        grid_dir=draineli07_grid.parent,
    )

    setup = _draineli07_setup(
        dl07,
        dust_mass=np.array([1e6, 0.0, np.nan, 1e6]) * Msun,
        qpah=np.array([0.025, 0.025, 0.025, np.nan]),
    )

    diffuse, pdr = dl07._get_templates_at_lam(dl07.grid.lam, setup)
    expected_diffuse, expected_pdr = _expected_templates(0.025, 1.0, 2.0)

    np.testing.assert_allclose(diffuse[0], expected_diffuse)
    np.testing.assert_allclose(pdr[0], expected_pdr)
    np.testing.assert_allclose(diffuse[1:], 0.0)
    np.testing.assert_allclose(pdr[1:], 0.0)
    assert np.all(np.isfinite(diffuse)) and np.all(np.isfinite(pdr))


def test_draineli07_resamples_non_native_wavelengths(draineli07_grid):
    """Non-native wavelength requests should use grid resampling."""
    lam = np.logspace(4, 6, 20) * angstrom
    dl07 = DraineLi07(
        lam=lam,
        grid_name=draineli07_grid.name,
        grid_dir=draineli07_grid.parent,
    )

    setup = _draineli07_setup(dl07, dust_mass=np.full(2, 1e6) * Msun)
    diffuse, pdr = dl07._get_templates_at_lam(lam, setup)

    resampled = dl07.grid.reduce_rest_frame_lam(lam)
    np.testing.assert_allclose(
        diffuse, np.vstack([resampled.spectra["diffuse"][1, 1]] * 2)
    )
    np.testing.assert_allclose(
        pdr, np.vstack([resampled.spectra["pdr"][1, 1, 1]] * 2)
    )


def test_draineli07_resamples_scalar_wavelengths(draineli07_grid):
    """Scalar wavelength requests should also use grid resampling."""
    lam = np.array([2.0e4]) * angstrom
    dl07 = DraineLi07(
        lam=lam,
        grid_name=draineli07_grid.name,
        grid_dir=draineli07_grid.parent,
    )

    setup = _draineli07_setup(dl07, dust_mass=np.full(2, 1e6) * Msun)
    diffuse, pdr = dl07._get_templates_at_lam(lam, setup)

    at_lam = dl07.grid.get_spectra_at_lam(lam)
    np.testing.assert_allclose(
        diffuse, np.vstack([at_lam["diffuse"][1, 1]] * 2).reshape(2, 1)
    )
    np.testing.assert_allclose(
        pdr, np.vstack([at_lam["pdr"][1, 1, 1]] * 2).reshape(2, 1)
    )


def test_draineli07_supports_log10_umin_grids(draineli07_log_grid):
    """Logarithmic umin extraction grids should work correctly."""
    dl07 = DraineLi07(
        grid_name=draineli07_log_grid.name,
        grid_dir=draineli07_log_grid.parent,
    )

    # Grid points, and the point half way between 1 and 10 in log10 umin
    umin = np.array([1.0, 10.0, 10**0.5])
    setup = _draineli07_setup(
        dl07, dust_mass=np.full(3, 1e6) * Msun, umin=umin
    )
    diffuse, _ = dl07._get_templates_at_lam(dl07.grid.lam, setup)
    expected, _ = _expected_templates(0.025, umin[:2], 2.0)

    np.testing.assert_allclose(diffuse[:2], expected)
    np.testing.assert_allclose(diffuse[2], expected.mean(axis=0))


def test_draineli07_ignores_negative_dust_masses(draineli07_grid):
    """Negative dust masses are treated as non-contributing inputs."""
    dl07 = DraineLi07(
        grid_name=draineli07_grid.name,
        grid_dir=draineli07_grid.parent,
    )

    setup = _draineli07_setup(dl07, dust_mass=np.array([-1e6, -1e6]) * Msun)
    diffuse, pdr = dl07._get_templates_at_lam(dl07.grid.lam, setup)

    np.testing.assert_allclose(diffuse, np.zeros((2, 100)))
    np.testing.assert_allclose(pdr, np.zeros((2, 100)))


def test_draineli07_clamps_out_of_range_parameters(draineli07_grid):
    """Parameters outside the grid use the grid edge."""
    dl07 = DraineLi07(
        grid_name=draineli07_grid.name,
        grid_dir=draineli07_grid.parent,
    )

    setup = _draineli07_setup(dl07, umin=1e3)
    diffuse, _ = dl07._get_templates_at_lam(dl07.grid.lam, setup)
    expected, _ = _expected_templates(0.025, 100.0, 2.0)

    np.testing.assert_allclose(diffuse, expected)


def test_draineli07_rejects_invalid_grid_axis(tmp_path):
    """The dust emission grid must expose the qpah, umin and alpha axes."""
    grid_path = tmp_path / "bad_draineli07_grid.hdf5"
    with h5py.File(grid_path, "w") as hdf:
        hdf.attrs["axes"] = np.array(["qpah", "size", "alpha"], dtype=object)
        hdf.attrs["WeightVariable"] = "None"

        axes = hdf.create_group("axes")
        for name in ("qpah", "size", "alpha"):
            axis = axes.create_dataset(name, data=np.array([0.1, 0.2]))
            axis.attrs["Units"] = "dimensionless"
            axis.attrs["log_on_read"] = False

        spectra = hdf.create_group("spectra")
        wavelength = spectra.create_dataset(
            "wavelength", data=np.array([1.0e4, 2.0e4, 3.0e4])
        )
        wavelength.attrs["Units"] = "Å"
        spectra.create_dataset("diffuse", data=np.ones((2, 2, 3)))
        spectra.create_dataset("pdr", data=np.ones((2, 2, 2, 3)))

    with pytest.raises(exceptions.UnimplementedFunctionality):
        DraineLi07(grid_name=grid_path.name, grid_dir=grid_path.parent)


def test_draineli07_rejects_missing_template_grid(tmp_path):
    """The dust emission grid must contain both DL07 templates."""
    grid_path = tmp_path / "missing_template_draineli07_grid.hdf5"
    with h5py.File(grid_path, "w") as hdf:
        hdf.attrs["axes"] = np.array(["qpah", "umin", "alpha"], dtype=object)
        hdf.attrs["WeightVariable"] = "None"

        axes = hdf.create_group("axes")
        for name in ("qpah", "umin", "alpha"):
            axis = axes.create_dataset(name, data=np.array([0.1, 0.2]))
            axis.attrs["Units"] = "dimensionless"
            axis.attrs["log_on_read"] = False

        spectra = hdf.create_group("spectra")
        wavelength = spectra.create_dataset(
            "wavelength", data=np.array([1.0e4, 2.0e4, 3.0e4])
        )
        wavelength.attrs["Units"] = "Å"
        spectra.create_dataset("diffuse", data=np.ones((2, 2, 3)))

    with pytest.raises(exceptions.InconsistentArguments):
        DraineLi07(grid_name=grid_path.name, grid_dir=grid_path.parent)


def test_draineli07_rejects_mismatched_parameter_lengths(draineli07_grid):
    """Every parameter must have one value or one per particle."""
    dl07 = DraineLi07(
        grid_name=draineli07_grid.name,
        grid_dir=draineli07_grid.parent,
    )

    with pytest.raises(exceptions.InconsistentArguments):
        _draineli07_setup(
            dl07,
            dust_mass=np.full(3, 1e6) * Msun,
            qpah=np.array([0.01, 0.05]),
        )


def test_draineli07_per_particle_parameters_need_per_particle_model(
    draineli07_grid, mock_model, mock_emissions, dust_wavelengths
):
    """Arrays of parameters can't make one integrated spectrum."""
    dl07 = DraineLi07(
        grid_name=draineli07_grid.name,
        grid_dir=draineli07_grid.parent,
        intrinsic="intrinsic",
        attenuated="attenuated",
    )
    emitter = _ParamEmitter(
        dust_mass=np.full(2, 1e6) * Msun,
        hydrogen_mass=1e8 * Msun,
        qpah=np.array([0.01, 0.05]),
        gamma=0.1,
        umin=1.0,
        alpha=2.0,
    )

    with pytest.raises(exceptions.InconsistentArguments):
        dl07._generate_spectra(
            dust_wavelengths, emitter, mock_model, mock_emissions
        )


def test_draineli07_resampled_grid_cached(draineli07_grid, dust_wavelengths):
    """The grid is resampled once per wavelength array, never in place."""
    dl07 = DraineLi07(
        grid_name=draineli07_grid.name,
        grid_dir=draineli07_grid.parent,
    )

    first = dl07._get_resampled_grid(dust_wavelengths)
    assert dl07._get_resampled_grid(dust_wavelengths) is first
    assert dl07._get_resampled_grid(dl07.grid.lam) is dl07.grid
    assert dl07.grid.nlam == 100


def test_draineli07_magdis_only_without_gamma_or_umin(draineli07_grid):
    """Giving gamma and umin with alpha = 2 doesn't use Magdis+12."""
    dl07 = DraineLi07(
        grid_name=draineli07_grid.name,
        grid_dir=draineli07_grid.parent,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        setup = _draineli07_setup(dl07)

    assert not any("Magdis" in str(w.message) for w in caught)
    np.testing.assert_array_equal(setup["umin_calculated"], [1.0])
    np.testing.assert_array_equal(setup["gamma_calculated"], [0.1])


@pytest.mark.parametrize("alpha", (1.5, 2.0, 2.5))
def test_draineli07_magdis_umin_uses_alpha(draineli07_grid, alpha):
    """The derived umin gives the Magdis+12 <U> for the given alpha."""
    dl07 = DraineLi07(
        grid_name=draineli07_grid.name,
        grid_dir=draineli07_grid.parent,
    )

    setup = _draineli07_setup(dl07, umin=None, alpha=alpha, gamma=0.05)
    np.testing.assert_allclose(
        u_mean(setup["umin_calculated"], 1e7, 0.05, alpha),
        setup["radiation_field_average"],
        rtol=1e-8,
    )


def test_draineli07_magdis_umin_without_solution_raises(draineli07_grid):
    """A power law alone above the target <U> can't be solved."""
    dl07 = DraineLi07(
        grid_name=draineli07_grid.name,
        grid_dir=draineli07_grid.parent,
    )

    with pytest.raises(exceptions.InconsistentArguments):
        dl07._setup_dl07_parameters(
            1e8 * Msun, 1e10 * Lsun, gamma=0.05, alpha=1.0
        )


@pytest.mark.parametrize("alpha", (1.0, 1.5, 2.0, 2.5))
def test_draineli07_u_mean_any_alpha(alpha):
    """The mean radiation field matches the integral for any alpha."""
    umin, umax, gamma = 1.0, 1e7, 0.1
    log_u = np.linspace(np.log(umin), np.log(umax), 200001)
    u = np.exp(log_u)
    dm_du = u ** (-alpha)
    norm = np.trapezoid(dm_du * u, log_u)
    u_power_law = np.trapezoid(dm_du * u * u, log_u) / norm
    np.testing.assert_allclose(
        u_mean(umin, umax, gamma, alpha),
        (1 - gamma) * umin + gamma * u_power_law,
        rtol=1e-6,
    )


def test_draineli07_parameters_vary_per_emitter(
    draineli07_grid, mock_model, mock_emissions, dust_wavelengths
):
    """Each emitter's qpah is used."""
    dl07 = DraineLi07(
        grid_name=draineli07_grid.name,
        grid_dir=draineli07_grid.parent,
        intrinsic="intrinsic",
        attenuated="attenuated",
    )
    params = dict(
        dust_mass=1e6 * Msun,
        hydrogen_mass=1e8 * Msun,
        gamma=0.1,
        umin=1.0,
        alpha=2.0,
    )

    seds = [
        dl07._generate_spectra(
            dust_wavelengths,
            _ParamEmitter(qpah=qpah, **params),
            mock_model,
            mock_emissions,
        )
        for qpah in (0.01, 0.03)
    ]
    assert not np.allclose(seds[0].lnu, seds[1].lnu)


def test_draineli07_normalised_over_native_wavelengths(
    draineli07_grid, mock_model, mock_emissions
):
    """A truncated wavelength grid gives the same spectrum as a wider one."""
    dl07 = DraineLi07(
        grid_name=draineli07_grid.name,
        grid_dir=draineli07_grid.parent,
        intrinsic="intrinsic",
        attenuated="attenuated",
    )
    emitter = _ParamEmitter(
        dust_mass=1e6 * Msun,
        hydrogen_mass=1e8 * Msun,
        qpah=0.025,
        gamma=0.1,
        umin=1.0,
        alpha=2.0,
    )

    lam = dl07.grid.lam
    wide, truncated = (
        dl07._generate_spectra(lams, emitter, mock_model, mock_emissions)
        for lams in (lam[1:-1], lam[40:60])
    )
    np.testing.assert_allclose(truncated.lnu[1:-1], wide.lnu[40:58], rtol=1e-5)
