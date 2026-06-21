"""SPH field-evaluation helpers for deterministic spatial resampling.

Field-mode resampling always uses the shared C++ SPH evaluator so there is one
well-tested implementation of the field interpolation path.
"""

import numpy as np
from unyt import Mpc

from synthesizer.extensions.sph_density import evaluate_sph_density
from synthesizer.units import accepts
from synthesizer.utils.operation_timers import timed


def _as_float64_c_contiguous(arr):
    """Return a float64, C-contiguous numpy view of *arr* without units."""
    raw = arr.ndview if hasattr(arr, "ndview") else arr
    return np.ascontiguousarray(np.asarray(raw, dtype=np.float64))


@timed("particle.evaluate_sph_field")
@accepts(
    query_positions=Mpc,
    particle_positions=Mpc,
    smoothing_lengths=Mpc,
)
def evaluate_sph_field(
    query_positions,
    particle_positions,
    smoothing_lengths,
    masses,
    kernel,
    attributes,
):
    """Evaluate the SPH density field and local attribute means.

    Args:
        query_positions (unyt_array, (N_query, 3)):
            Query positions.
        particle_positions (unyt_array, (N_part, 3)):
            Source particle coordinates.
        smoothing_lengths (unyt_array, (N_part,)):
            Source smoothing lengths.
        masses (unyt_array, (N_part,)):
            Source masses.
        kernel (Kernel):
            SPH kernel definition.
        attributes (dict[str, np.ndarray or unyt_array]):
            Per-particle attributes to interpolate as SPH-weighted means.

    Returns:
        tuple[unyt_array, dict]:
            ``(density, interpolated_attrs)``.
    """
    attr_names = []
    attr_arrays = []
    attr_units = {}

    for name, arr in attributes.items():
        if arr is None:
            continue
        attr_names.append(name)
        # The extension expects raw float64 buffers. We keep the units on the
        # Python side and reattach them once the weighted sums come back.
        attr_arrays.append(_as_float64_c_contiguous(arr))
        attr_units[name] = getattr(arr, "units", None)

    density_raw, weighted_arrays = evaluate_sph_density(
        _as_float64_c_contiguous(query_positions),
        _as_float64_c_contiguous(particle_positions),
        _as_float64_c_contiguous(smoothing_lengths),
        _as_float64_c_contiguous(masses),
        tuple(attr_arrays),
        kernel.name,
    )

    density_units = masses.units / (particle_positions.units**3)
    density = density_raw * density_units

    # The extension returns weighted sums. Turn those back into local field
    # means one attribute at a time, guarding zero-density pixels defensively.
    non_zero = density_raw > 0.0
    interpolated = {}
    for name, weighted in zip(attr_names, weighted_arrays):
        mean_raw = np.zeros_like(weighted)
        if weighted.ndim == 1:
            mean_raw[non_zero] = weighted[non_zero] / density_raw[non_zero]
        else:
            mean_raw[non_zero] = (
                weighted[non_zero] / density_raw[non_zero, np.newaxis]
            )

        if attr_units[name] is None:
            interpolated[name] = mean_raw
        else:
            interpolated[name] = mean_raw * attr_units[name]

    return density, interpolated
