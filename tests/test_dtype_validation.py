"""Tests that unsupported dtypes are rejected, not silently reinterpreted.

The C++ extensions dispatch on dtype via a `dispatch_float` helper that, by
design, treats anything that isn't float64 as float32 (see
python_to_cpp.h). Every call site is expected to validate its input arrays'
dtypes *before* calling into that dispatch, so an unsupported dtype (e.g.
float16, int32) raises a clear TypeError rather than being silently
reinterpreted as float32 -- which would corrupt every downstream
computation. These tests lock in that validation for the entry points that
matter most for scientific correctness: particle spectra weighting, star
formation history weighting, line-of-sight column density, and numerical
integration.
"""

import numpy as np
import pytest


class TestUnsupportedDtypesAreRejected:
    """Unsupported floating-point dtypes must raise, not misdispatch."""

    def test_integration_rejects_float16(self):
        """trapz/simps extensions should reject float16 inputs."""
        from synthesizer.extensions.integration import trapz_last_axis

        xs = np.linspace(0, 1, 8, dtype=np.float16)
        ys = np.ones((2, 8), dtype=np.float16)

        with pytest.raises(TypeError):
            trapz_last_axis(xs, ys, 1, None)

    def test_compute_particle_seds_rejects_float16_particle_property(self):
        """compute_particle_seds should reject a float16 particle property."""
        from synthesizer.extensions.particle_spectra import (
            compute_particle_seds,
        )

        axis = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        grid_spectra = np.arange(15, dtype=np.float64).reshape(3, 5) + 1.0
        grid_dims = np.array([3], dtype=np.int32)
        part_props = (np.array([0.25, 1.5, -1.0, 3.0], dtype=np.float16),)
        weights = np.array([2.0, 3.0, 5.0, 7.0], dtype=np.float64)

        with pytest.raises(TypeError):
            compute_particle_seds(
                grid_spectra,
                (axis,),
                part_props,
                weights,
                grid_dims,
                1,
                weights.size,
                5,
                "ngp",
                1,
                None,
                None,
                False,
                np.float64,
                ("x",),
            )

    def test_compute_sfzh_rejects_float16_grid_axis(self):
        """compute_sfzh should reject a float16 grid axis."""
        from synthesizer.extensions.sfzh import compute_sfzh

        axis16 = np.array([0.0, 1.0, 2.0], dtype=np.float16)
        part_props = (np.array([0.25, 1.5, -1.0, 3.0], dtype=np.float64),)
        weights = np.array([2.0, 3.0, 5.0, 7.0], dtype=np.float64)
        grid_dims = np.array([3], dtype=np.int32)

        with pytest.raises(TypeError):
            compute_sfzh(
                (axis16,),
                part_props,
                weights,
                grid_dims,
                1,
                weights.size,
                "ngp",
                1,
                None,
                ("x",),
                np.float64,
            )

    def test_compute_column_density_rejects_int32_positions(self):
        """compute_column_density should reject non-floating positions."""
        from synthesizer.extensions.column_density import (
            compute_column_density,
        )

        kernel = np.ones(8, dtype=np.float64)
        truncated_kernel = np.ones((4, 4), dtype=np.float64)
        pos_i = np.zeros((1, 3), dtype=np.int32)
        pos_j = np.zeros((1, 3), dtype=np.float64)
        smls = np.ones(1, dtype=np.float64)
        surf_den_vals = np.ones(1, dtype=np.float64)

        with pytest.raises(TypeError):
            compute_column_density(
                kernel,
                truncated_kernel,
                pos_i,
                pos_j,
                smls,
                surf_den_vals,
                1,
                1,
                8,
                4,
                4,
                1.0,
                1,
                8,
                1,
            )
