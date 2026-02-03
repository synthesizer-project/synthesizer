"""A test suite for the utils module."""


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


class TestPipelineProfilingHelpers:
    """Test profiling helper utilities."""

    def test_build_random_galaxies(self):
        """Ensure random galaxy builder returns expected sizes."""
        from profiling.pipeline_profile import build_random_galaxies

        galaxies = build_random_galaxies(
            nparticles=5,
            ngalaxies=2,
            seed=123,
            redshift=0.1,
        )

        assert len(galaxies) == 2
        assert galaxies[0].stars.nstars == 5
