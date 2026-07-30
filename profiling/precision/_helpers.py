"""Shared helpers for photometry precision/memory profiling scripts."""

from __future__ import annotations

import numpy as np
from unyt import angstrom

from synthesizer.instruments import FilterCollection


def make_synthetic_spectra(nparticles, nlam, dtype, rng):
    """Create a contiguous synthetic 2D spectra array."""
    lam_axis = np.linspace(-3.0, 3.0, nlam, dtype=np.float64)
    base = np.exp(-0.5 * lam_axis**2) + 0.15 * np.sin(4.0 * lam_axis)
    base = base[None, :]

    amplitudes = rng.uniform(0.5, 2.0, size=(nparticles, 1))
    slopes = rng.uniform(0.0, 0.2, size=(nparticles, 1))
    continuum = np.linspace(0.8, 1.2, nlam, dtype=np.float64)[None, :]
    spectra = amplitudes * base + slopes * continuum

    return np.array(spectra, dtype=dtype, order="C", copy=True)


def make_test_filters(lam, nfilters):
    """Create a deterministic set of top-hat filters for profiling.

    Using synthetic filters keeps the profiling scripts self-contained and
    avoids depending on an external cached instrument file.
    """
    lam_values = lam.to("angstrom").value
    lam_min = lam_values.min()
    lam_max = lam_values.max()
    centres = np.linspace(lam_min * 1.1, lam_max * 0.9, nfilters)
    widths = np.full(nfilters, max((lam_max - lam_min) / (2 * nfilters), 50.0))

    tophat_dict = {}
    for i, (centre, width) in enumerate(zip(centres, widths, strict=True)):
        tophat_dict[f"f{i + 1}"] = {
            "lam_eff": centre * angstrom,
            "lam_fwhm": width * angstrom,
        }

    return FilterCollection(tophat_dict=tophat_dict, new_lam=lam)
