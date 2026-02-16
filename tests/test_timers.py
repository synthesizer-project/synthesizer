"""Unit tests for timer functionality and OperationTimers class."""

import time
import unittest

from synthesizer import check_atomic_timing
from synthesizer.extensions.timers import (
    _test_toc_from_c,
    get_operation_names,
    get_operation_timings,
    reset_timings,
    tic,
    toc,
)
from synthesizer.utils.operation_timers import OperationTimers


def setUpModule():
    """Check if atomic timing is available before running any tests.

    All tests in this module require ATOMIC_TIMING to be enabled during
    compilation. If not available, skip the entire module with a helpful
    message.
    """
    if not check_atomic_timing():
        raise unittest.SkipTest(
            "ATOMIC_TIMING not enabled. These tests require atomic timing "
            "accumulation. Recompile with: ATOMIC_TIMING=1 pip install -e ."
        )


class TestTimerBasics(unittest.TestCase):
    """Test basic timer functionality (tic/toc)."""

    def setUp(self):
        """Reset timers before each test."""
        reset_timings()

    def test_tic_returns_time(self):
        """Test that tic returns a time value."""
        start = tic()
        self.assertIsInstance(start, float)
        self.assertGreater(start, 0)

    def test_toc_accumulates_single_operation(self):
        """Test that toc accumulates timing for a single operation."""
        start = tic()
        time.sleep(0.001)  # Sleep 1ms
        toc("Test operation", start)

        # Check that operation was recorded
        names = get_operation_names()
        self.assertIn("Test operation", names)

        # Check timing data
        cumtime, count, source = get_operation_timings("Test operation")
        self.assertGreater(cumtime, 0)
        self.assertEqual(count, 1)
        self.assertEqual(source, "Python")

    def test_toc_accumulates_multiple_calls(self):
        """Test that toc accumulates multiple calls to same operation."""
        # Call operation twice
        for _ in range(2):
            start = tic()
            time.sleep(0.001)
            toc("Repeated operation", start)

        # Check accumulation
        cumtime, count, source = get_operation_timings("Repeated operation")
        self.assertEqual(count, 2)
        self.assertGreater(cumtime, 0)

    def test_toc_multiple_operations(self):
        """Test that toc tracks multiple different operations."""
        # Time operation A
        start = tic()
        toc("Operation A", start)

        # Time operation B
        start = tic()
        toc("Operation B", start)

        # Check both are recorded
        names = get_operation_names()
        self.assertIn("Operation A", names)
        self.assertIn("Operation B", names)
        self.assertEqual(len(names), 2)


class TestTimerReset(unittest.TestCase):
    """Test timer reset functionality."""

    def setUp(self):
        """Reset timers before each test."""
        reset_timings()

    def test_reset_clears_all_data(self):
        """Test that reset clears all accumulated timing data."""
        # Accumulate some data
        start = tic()
        toc("Operation 1", start)
        start = tic()
        toc("Operation 2", start)

        # Verify data exists
        names = get_operation_names()
        self.assertEqual(len(names), 2)

        # Reset
        reset_timings()

        # Verify data is cleared
        names = get_operation_names()
        self.assertEqual(len(names), 0)

    def test_reset_allows_fresh_accumulation(self):
        """Test that reset allows fresh accumulation."""
        # First accumulation
        start = tic()
        toc("Test op", start)
        _, count1, _ = get_operation_timings("Test op")
        self.assertEqual(count1, 1)

        # Reset and accumulate again
        reset_timings()
        start = tic()
        toc("Test op", start)

        # Should have count of 1, not 2
        _, count2, _ = get_operation_timings("Test op")
        self.assertEqual(count2, 1)


class TestOperationTimersClass(unittest.TestCase):
    """Test OperationTimers dict-like interface."""

    def setUp(self):
        """Create fresh timers for each test."""
        self.timers = OperationTimers()
        self.timers.reset()

    def test_keys_returns_list(self):
        """Test that keys() returns list of operation names."""
        # Start with empty
        self.assertEqual(len(self.timers.keys()), 0)

        # Add some operations
        start = tic()
        toc("Op1", start)
        start = tic()
        toc("Op2", start)

        # Check keys
        keys = self.timers.keys()
        self.assertIsInstance(keys, list)
        self.assertEqual(len(keys), 2)
        self.assertIn("Op1", keys)
        self.assertIn("Op2", keys)

    def test_getitem_returns_tuple(self):
        """Test that __getitem__ returns (cumtime, count, source) tuple."""
        start = tic()
        toc("Test op", start)

        result = self.timers["Test op"]
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

        cumtime, count, source = result
        self.assertIsInstance(cumtime, float)
        self.assertIsInstance(count, int)
        self.assertIsInstance(source, str)

    def test_getitem_raises_keyerror(self):
        """Test that __getitem__ raises KeyError for missing operation."""
        with self.assertRaises(KeyError):
            _ = self.timers["Nonexistent operation"]

    def test_contains_operator(self):
        """Test that 'in' operator works correctly."""
        start = tic()
        toc("Existing op", start)

        self.assertIn("Existing op", self.timers)
        self.assertNotIn("Missing op", self.timers)

    def test_get_source(self):
        """Test get_source method."""
        start = tic()
        toc("Python op", start)

        source = self.timers.get_source("Python op")
        self.assertEqual(source, "Python")

    def test_get_source_raises_keyerror(self):
        """Test that get_source raises KeyError for missing operation."""
        with self.assertRaises(KeyError):
            self.timers.get_source("Missing op")

    def test_items_iteration(self):
        """Test that items() allows iteration over (name, data) pairs."""
        # Add operations
        start = tic()
        toc("Op1", start)
        start = tic()
        toc("Op2", start)

        # Iterate
        items_list = list(self.timers.items())
        self.assertEqual(len(items_list), 2)

        # Check structure
        for name, data in items_list:
            self.assertIsInstance(name, str)
            self.assertIsInstance(data, tuple)
            self.assertEqual(len(data), 3)

    def test_len(self):
        """Test __len__ method."""
        self.assertEqual(len(self.timers), 0)

        start = tic()
        toc("Op1", start)
        self.assertEqual(len(self.timers), 1)

        start = tic()
        toc("Op2", start)
        self.assertEqual(len(self.timers), 2)

    def test_repr(self):
        """Test __repr__ method."""
        start = tic()
        toc("Op1", start)

        repr_str = repr(self.timers)
        self.assertIn("OperationTimers", repr_str)
        self.assertIn("1", repr_str)  # Should show count

    def test_reset_method(self):
        """Test reset method on OperationTimers instance."""
        # Add data
        start = tic()
        toc("Op1", start)
        self.assertEqual(len(self.timers), 1)

        # Reset via timers instance
        self.timers.reset()
        self.assertEqual(len(self.timers), 0)


class TestTimerAccumulation(unittest.TestCase):
    """Test timing accumulation behavior."""

    def setUp(self):
        """Reset timers before each test."""
        reset_timings()

    def test_cumulative_time_increases(self):
        """Test that cumulative time increases with repeated calls."""
        # First call
        start = tic()
        time.sleep(0.001)
        toc("Accumulating op", start)
        cumtime1, count1, _ = get_operation_timings("Accumulating op")

        # Second call
        start = tic()
        time.sleep(0.001)
        toc("Accumulating op", start)
        cumtime2, count2, _ = get_operation_timings("Accumulating op")

        # Cumulative time should increase
        self.assertGreater(cumtime2, cumtime1)
        self.assertEqual(count2, count1 + 1)

    def test_call_count_increments(self):
        """Test that call count increments correctly."""
        for i in range(5):
            start = tic()
            toc("Counted op", start)

        _, count, _ = get_operation_timings("Counted op")
        self.assertEqual(count, 5)

    def test_source_preserved(self):
        """Test that source is set correctly and preserved."""
        # Python source (via toc wrapper)
        start = tic()
        toc("Python op", start)
        _, _, source = get_operation_timings("Python op")
        self.assertEqual(source, "Python")


class TestTimerSourceTracking(unittest.TestCase):
    """Test source tracking (C vs Python)."""

    def setUp(self):
        """Reset timers before each test."""
        self.timers = OperationTimers()
        self.timers.reset()

    def test_python_source_via_toc(self):
        """Test that Python toc() sets source to 'Python'."""
        start = tic()
        toc("Python operation", start)

        source = self.timers.get_source("Python operation")
        self.assertEqual(source, "Python")

    def test_linestyle_determination(self):
        """Test that source can be used to determine linestyle."""
        # Add Python operation
        start = tic()
        toc("Python op", start)

        # Check we can determine linestyle from source
        source = self.timers.get_source("Python op")
        linestyle = "--" if source == "Python" else "-"
        self.assertEqual(linestyle, "--")


class TestPyCapsuleArchitecture(unittest.TestCase):
    """Test that the PyCapsule architecture correctly wires C++ extensions."""

    def test_all_consuming_extensions_import(self):
        """Test that all C++ extensions import without error.

        Each consuming extension calls import_toc_capsule() in its PyInit_*
        function. If the PyCapsule retrieval fails, the import would raise
        an ImportError.
        """
        import synthesizer.extensions.column_density  # noqa: F401
        import synthesizer.extensions.doppler_particle_spectra  # noqa: F401
        import synthesizer.extensions.integrated_spectra  # noqa: F401
        import synthesizer.extensions.integration  # noqa: F401
        import synthesizer.extensions.particle_spectra  # noqa: F401
        import synthesizer.extensions.sfzh  # noqa: F401
        import synthesizer.extensions.weights  # noqa: F401
        import synthesizer.imaging.extensions.circular_aperture  # noqa: F401
        import synthesizer.imaging.extensions.image  # noqa: F401

    def test_capsule_exists_on_timers_module(self):
        """Test that the _toc_accumulate PyCapsule is exposed."""
        import synthesizer.extensions.timers as tm

        self.assertTrue(hasattr(tm, "_toc_accumulate"))
        # PyCapsule objects have a specific type string
        self.assertIn("PyCapsule", type(tm._toc_accumulate).__name__)


class TestCppTimingEndToEnd(unittest.TestCase):
    """Test that C++ toc() calls reach global_timings via the PyCapsule."""

    def setUp(self):
        """Reset timers before each test."""
        reset_timings()

    def test_c_source_accumulation(self):
        """Test that toc_accumulate with source='C' is recorded correctly."""
        _test_toc_from_c("C++ test operation", 0.042)

        names = get_operation_names()
        self.assertIn("C++ test operation", names)

        cumtime, count, source = get_operation_timings("C++ test operation")
        self.assertAlmostEqual(cumtime, 0.042, places=6)
        self.assertEqual(count, 1)
        self.assertEqual(source, "C")

    def test_c_source_accumulates_multiple(self):
        """Test that repeated C-source calls accumulate correctly."""
        _test_toc_from_c("Repeated C op", 0.01)
        _test_toc_from_c("Repeated C op", 0.02)
        _test_toc_from_c("Repeated C op", 0.03)

        cumtime, count, source = get_operation_timings("Repeated C op")
        self.assertAlmostEqual(cumtime, 0.06, places=6)
        self.assertEqual(count, 3)
        self.assertEqual(source, "C")

    def test_mixed_c_and_python_sources(self):
        """Test that C and Python operations coexist with correct sources."""
        # C operation
        _test_toc_from_c("C operation", 0.1)

        # Python operation
        start = tic()
        toc("Python operation", start)

        # Both should be recorded
        names = get_operation_names()
        self.assertIn("C operation", names)
        self.assertIn("Python operation", names)

        # Sources should be distinct
        _, _, c_source = get_operation_timings("C operation")
        _, _, py_source = get_operation_timings("Python operation")
        self.assertEqual(c_source, "C")
        self.assertEqual(py_source, "Python")

    def test_c_source_survives_reset(self):
        """Test that C operations are cleared by reset and re-accumulable."""
        _test_toc_from_c("Transient op", 0.5)
        self.assertIn("Transient op", get_operation_names())

        reset_timings()
        self.assertNotIn("Transient op", get_operation_names())

        # Re-accumulate after reset
        _test_toc_from_c("Transient op", 0.25)
        cumtime, count, _ = get_operation_timings("Transient op")
        self.assertAlmostEqual(cumtime, 0.25, places=6)
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
