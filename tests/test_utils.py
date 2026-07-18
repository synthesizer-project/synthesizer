"""A test suite for the utils module."""

import numpy as np
import pytest


class TestPluralization:
    """Test suite for pluralize and depluralize functions."""

    def test_pluralize_gas(self):
        """Test pluralize with 'gas' (ends in s but is singular)."""
        from synthesizer.utils.util_funcs import pluralize

        assert pluralize("gas") == "gases"

    def test_pluralize_blackhole(self):
        """Test pluralize with 'blackhole' (codebase component)."""
        from synthesizer.utils.util_funcs import pluralize

        assert pluralize("blackhole") == "blackholes"

    def test_depluralize_blackholes(self):
        """Test depluralize with 'blackholes' (codebase component)."""
        from synthesizer.utils.util_funcs import depluralize

        assert depluralize("blackholes") == "blackhole"

    def test_pluralize_star(self):
        """Test pluralize with 'star' (codebase component)."""
        from synthesizer.utils.util_funcs import pluralize

        assert pluralize("star") == "stars"

    def test_depluralize_stars(self):
        """Test depluralize with 'stars' (codebase component)."""
        from synthesizer.utils.util_funcs import depluralize

        assert depluralize("stars") == "star"

    def test_depluralize_ages(self):
        """Test depluralize with 'ages'."""
        from synthesizer.utils.util_funcs import depluralize

        assert depluralize("ages") == "age"

    def test_depluralize_mass(self):
        """Test depluralize with 'mass' (should not depluralize)."""
        from synthesizer.utils.util_funcs import depluralize

        assert depluralize("mass") == "mass"

    def test_depluralize_gas(self):
        """Test depluralize with 'gas' (should not depluralize)."""
        from synthesizer.utils.util_funcs import depluralize

        assert depluralize("gas") == "gas"

    def test_pluralize_mass(self):
        """Test pluralize with 'mass' (common codebase attribute)."""
        from synthesizer.utils.util_funcs import pluralize

        assert pluralize("mass") == "masses"

    def test_depluralize_masses(self):
        """Test depluralize with 'masses' (common codebase attribute)."""
        from synthesizer.utils.util_funcs import depluralize

        assert depluralize("masses") == "mass"

    def test_pluralize_axis(self):
        """Test pluralize with 'axis' (common grid attribute)."""
        from synthesizer.utils.util_funcs import pluralize

        assert pluralize("axis") == "axes"

    def test_depluralize_axes(self):
        """Test depluralize with 'axes' (common grid attribute)."""
        from synthesizer.utils.util_funcs import depluralize

        assert depluralize("axes") == "axis"

    def test_pluralize_age(self):
        """Test pluralize with 'age' (common attribute)."""
        from synthesizer.utils.util_funcs import pluralize

        assert pluralize("age") == "ages"

    def test_depluralize_ages_real(self):
        """Test depluralize ages (common attribute check)."""
        from synthesizer.utils.util_funcs import depluralize

        assert depluralize("ages") == "age"

    def test_pluralize_metallicity(self):
        """Test pluralize with 'metallicity' (common codebase attribute)."""
        from synthesizer.utils.util_funcs import pluralize

        assert pluralize("metallicity") == "metallicities"

    def test_depluralize_metallicities(self):
        """Test depluralize with 'metallicities'."""
        from synthesizer.utils.util_funcs import depluralize

        assert depluralize("metallicities") == "metallicity"


class TestPrecisionConfig:
    """Test suite for the global output precision configuration."""

    def test_default_is_float64(self):
        """The out-of-the-box default output dtype should be float64."""
        from synthesizer.utils.precision import (
            get_default_out_dtype,
            resolve_out_dtype,
        )

        assert get_default_out_dtype() == np.dtype(np.float64)
        assert resolve_out_dtype(None) == np.dtype(np.float64)

    def test_explicit_dtype_overrides_default(self):
        """An explicit out_dtype should win over the global default."""
        from synthesizer.utils.precision import resolve_out_dtype

        assert resolve_out_dtype(np.float32) == np.dtype(np.float32)

    def test_set_default_out_dtype(self):
        """Changing the global default should flow through resolution."""
        from synthesizer.utils.precision import (
            resolve_out_dtype,
            set_default_out_dtype,
        )

        set_default_out_dtype(np.float32)
        try:
            assert resolve_out_dtype(None) == np.dtype(np.float32)
        finally:
            set_default_out_dtype(np.float64)

    def test_invalid_dtype_raises(self):
        """Non-float32/float64 dtypes should be rejected."""
        from synthesizer.utils.precision import (
            resolve_out_dtype,
            set_default_out_dtype,
        )

        with pytest.raises(ValueError):
            set_default_out_dtype(np.int32)
        with pytest.raises(ValueError):
            resolve_out_dtype(np.float16)

    def test_global_default_controls_integration_output(self):
        """Integration outputs should follow the global default dtype."""
        from synthesizer.utils.integrate import integrate_last_axis
        from synthesizer.utils.precision import set_default_out_dtype

        xs = np.linspace(0.0, 1.0, 32)
        ys = np.ones((4, 32))

        assert integrate_last_axis(xs, ys).dtype == np.float64

        set_default_out_dtype(np.float32)
        try:
            assert integrate_last_axis(xs, ys).dtype == np.float32
        finally:
            set_default_out_dtype(np.float64)
