"""Test suite for the UnifiedAGN emission model."""

import numpy as np
import pytest
from unyt import K, Mpc, Msun, deg, yr

from synthesizer.emission_models.agn.unified_agn import UnifiedAGN
from synthesizer.emission_models.generators.dust.greybody import Greybody
from synthesizer.emission_models.transformers.dust_attenuation import PowerLaw
from synthesizer.emission_models.utils import get_param
from synthesizer.exceptions import InconsistentParameter
from synthesizer.particle.blackholes import BlackHoles


def make_black_holes(
    covering_fraction_blr,
    covering_fraction_nlr,
):
    """Create a black hole population with controlled covering fractions."""
    nblackholes = len(np.atleast_1d(covering_fraction_blr))
    return BlackHoles(
        masses=np.ones(nblackholes) * 1e6 * Msun,
        accretion_rates=np.ones(nblackholes) * Msun / yr,
        inclinations=np.zeros(nblackholes) * deg,
        coordinates=np.zeros((nblackholes, 3)) * Mpc,
        covering_fraction_blr=covering_fraction_blr,
        covering_fraction_nlr=covering_fraction_nlr,
    )


def make_unified_agn(
    test_grid,
    disc_transmission="weighted_combination",
    variant="intrinsic",
):
    """Create a UnifiedAGN model for testing."""
    kwargs = {}
    if variant in {"attenuated", "total"}:
        kwargs["diffuse_dust_curve"] = PowerLaw(slope=-1)
    if variant == "total":
        kwargs["diffuse_dust_emission_model"] = Greybody(
            temperature=10**4 * K,
            emissivity=2,
        )

    return UnifiedAGN(
        nlr_grid=test_grid,
        blr_grid=test_grid,
        torus_emission_model=Greybody(temperature=10**4 * K, emissivity=2),
        disc_transmission=disc_transmission,
        **kwargs,
    )


def get_intrinsic_model(model):
    """Return the intrinsic UnifiedAGN model carrying transmission branches."""
    return model if hasattr(model, "disc_escaped") else model.intrinsic


class TestUnifiedAGN:
    """Test suite for UnifiedAGN transmission behavior."""

    @pytest.mark.parametrize(
        ("variant", "expected_label"),
        [
            ("intrinsic", "intrinsic"),
            ("attenuated", "attenuated"),
            ("total", "total"),
        ],
    )
    def test_default_disc_transmission_is_weighted_combination(
        self,
        test_grid,
        variant,
        expected_label,
    ):
        """Test the default disc transmission branch."""
        model = make_unified_agn(test_grid, variant=variant)

        assert model.label == expected_label
        assert model["disc_transmitted"].combine[0].label == (
            "disc_transmitted_weighted_combination"
        )

    @pytest.mark.parametrize("variant", ["intrinsic", "attenuated", "total"])
    def test_weighted_combination_uses_transmission_fraction_attrs(
        self,
        test_grid,
        variant,
    ):
        """Test weighted transmission uses deterministic emitter attrs."""
        black_holes = make_black_holes(
            covering_fraction_blr=np.array([0.2, 0.3]),
            covering_fraction_nlr=np.array([0.1, 0.4]),
        )
        model = make_unified_agn(test_grid, variant=variant)

        assert model["disc_escaped"].transformer._required_params == (
            "transmission_fraction_escape",
        )
        assert model["disc_transmitted_nlr"].transformer._required_params == (
            "transmission_fraction_nlr",
        )
        assert model["disc_transmitted_blr"].transformer._required_params == (
            "transmission_fraction_blr",
        )

        escape = get_param(
            "transmission_fraction_escape",
            model["disc_escaped"],
            None,
            black_holes,
        )
        nlr = get_param(
            "transmission_fraction_nlr",
            model["disc_transmitted_nlr"],
            None,
            black_holes,
        )
        blr = get_param(
            "transmission_fraction_blr",
            model["disc_transmitted_blr"],
            None,
            black_holes,
        )

        np.testing.assert_allclose(escape, [0.7, 0.3])
        np.testing.assert_allclose(nlr, [0.1, 0.4])
        np.testing.assert_allclose(blr, [0.2, 0.3])

    @pytest.mark.parametrize("variant", ["intrinsic", "attenuated", "total"])
    def test_random_disc_transmission_uses_random_fraction_attrs(
        self,
        test_grid,
        variant,
    ):
        """Test random transmission resolves the random one-hot attrs."""
        np.random.seed(0)
        black_holes = make_black_holes(
            covering_fraction_blr=np.array([0.2, 0.3, 0.0]),
            covering_fraction_nlr=np.array([0.1, 0.4, 0.5]),
        )
        model = make_unified_agn(
            test_grid,
            disc_transmission="random",
            variant=variant,
        )
        intrinsic_model = get_intrinsic_model(model)

        assert (
            intrinsic_model.disc_escaped.fixed_parameters[
                "transmission_fraction_escape"
            ]
            == "random_transmission_fraction_escape"
        )
        assert (
            intrinsic_model.disc_transmitted_nlr.fixed_parameters[
                "transmission_fraction_nlr"
            ]
            == "random_transmission_fraction_nlr"
        )
        assert (
            intrinsic_model.disc_transmitted_blr.fixed_parameters[
                "transmission_fraction_blr"
            ]
            == "random_transmission_fraction_blr"
        )

        escape = get_param(
            "transmission_fraction_escape",
            intrinsic_model.disc_escaped,
            None,
            black_holes,
        )
        nlr = get_param(
            "transmission_fraction_nlr",
            intrinsic_model.disc_transmitted_nlr,
            None,
            black_holes,
        )
        blr = get_param(
            "transmission_fraction_blr",
            intrinsic_model.disc_transmitted_blr,
            None,
            black_holes,
        )

        np.testing.assert_allclose(
            escape,
            black_holes.random_transmission_fraction_escape,
        )
        np.testing.assert_allclose(
            nlr,
            black_holes.random_transmission_fraction_nlr,
        )
        np.testing.assert_allclose(
            blr,
            black_holes.random_transmission_fraction_blr,
        )

    @pytest.mark.parametrize(
        ("disc_transmission", "expected"),
        [
            ("none", (1.0, 0.0, 0.0)),
            ("escaped", (1.0, 0.0, 0.0)),
            ("nlr", (0.0, 1.0, 0.0)),
            ("blr", (0.0, 0.0, 1.0)),
        ],
    )
    @pytest.mark.parametrize("variant", ["intrinsic", "attenuated", "total"])
    def test_forced_disc_transmission_modes(
        self,
        test_grid,
        disc_transmission,
        expected,
        variant,
    ):
        """Test forced transmission modes use fixed scalars."""
        black_holes = make_black_holes(
            covering_fraction_blr=np.array([0.2]),
            covering_fraction_nlr=np.array([0.1]),
        )
        model = make_unified_agn(
            test_grid,
            disc_transmission=disc_transmission,
            variant=variant,
        )
        intrinsic_model = get_intrinsic_model(model)

        assert (
            intrinsic_model.disc_escaped.fixed_parameters[
                "transmission_fraction_escape"
            ]
            == expected[0]
        )
        assert (
            intrinsic_model.disc_transmitted_nlr.fixed_parameters[
                "transmission_fraction_nlr"
            ]
            == expected[1]
        )
        assert (
            intrinsic_model.disc_transmitted_blr.fixed_parameters[
                "transmission_fraction_blr"
            ]
            == expected[2]
        )

        escape = get_param(
            "transmission_fraction_escape",
            intrinsic_model.disc_escaped,
            None,
            black_holes,
        )
        nlr = get_param(
            "transmission_fraction_nlr",
            intrinsic_model.disc_transmitted_nlr,
            None,
            black_holes,
        )
        blr = get_param(
            "transmission_fraction_blr",
            intrinsic_model.disc_transmitted_blr,
            None,
            black_holes,
        )

        assert escape == expected[0]
        assert nlr == expected[1]
        assert blr == expected[2]

    @pytest.mark.parametrize("variant", ["intrinsic", "attenuated", "total"])
    def test_covering_fraction_edge_case_sum_to_unity(
        self, test_grid, variant
    ):
        """Test full covering leaves no deterministic escape fraction."""
        black_holes = make_black_holes(
            covering_fraction_blr=np.array([0.25]),
            covering_fraction_nlr=np.array([0.75]),
        )
        model = make_unified_agn(test_grid, variant=variant)

        escape = get_param(
            "transmission_fraction_escape",
            model["disc_escaped"],
            None,
            black_holes,
        )
        nlr = get_param(
            "transmission_fraction_nlr",
            model["disc_transmitted_nlr"],
            None,
            black_holes,
        )
        blr = get_param(
            "transmission_fraction_blr",
            model["disc_transmitted_blr"],
            None,
            black_holes,
        )

        assert escape == 0.0
        assert nlr == 0.75
        assert blr == 0.25

    def test_invalid_disc_transmission_raises(self, test_grid):
        """Test invalid transmission options are rejected."""
        with pytest.raises(InconsistentParameter):
            make_unified_agn(test_grid, disc_transmission="definitely_wrong")
