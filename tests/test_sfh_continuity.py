"""Tests for the Continuity family of non-parametric SFHs."""

import numpy as np
import pytest
from unyt import yr

from synthesizer import exceptions
from synthesizer.parametric import SFH

AGEBINS = np.array([[1.0, 1e8], [1e8, 1e9], [1e9, 1e10]]) * yr


def test_continuity_sfr_ratios():
    """Adjacent bin SFRs should follow the requested log SFR ratios."""
    ratios = np.array([0.3, -0.2])
    sfh = SFH.Continuity(logsfr_ratios=ratios, agebins=AGEBINS)

    np.testing.assert_allclose(sfh.masses.sum(), 1.0)
    np.testing.assert_allclose(
        np.log10(sfh.sfrs[:-1] / sfh.sfrs[1:]), ratios, rtol=1e-12
    )


def test_continuity_rejects_out_of_range_ratios():
    """Ratios beyond +/-10 should raise rather than be clipped."""
    with pytest.raises(exceptions.InconsistentArguments):
        SFH.Continuity(logsfr_ratios=[11.0, 0.0], agebins=AGEBINS)


def test_continuity_flex_rejects_out_of_range_ratios():
    """ContinuityFlex should also raise for ratios beyond +/-10."""
    with pytest.raises(exceptions.InconsistentArguments):
        SFH.ContinuityFlex(logsfr_ratio_young=-12.0)


@pytest.mark.parametrize(
    "draw",
    [
        lambda seed: SFH.Continuity.init_from_prior(
            agebins=AGEBINS, random_seed=seed
        ),
        lambda seed: SFH.ContinuityFlex.init_from_prior(
            N_flex_bins=3,
            fixed_young_bin=[0.0, 7.5],
            fixed_old_bin=[9.7, 10.136],
            random_seed=seed,
        ),
        lambda seed: SFH.ContinuityPSB.init_from_prior(random_seed=seed),
    ],
)
def test_init_from_prior_seeded(draw):
    """The same seed should give the same SFH."""
    np.testing.assert_array_equal(draw(42).masses, draw(42).masses)
    assert not np.array_equal(draw(42).masses, draw(43).masses)
