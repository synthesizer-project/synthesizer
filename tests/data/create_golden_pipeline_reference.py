"""Regenerate the golden full-pipeline regression reference.

This script computes the reference output used by
``tests/test_pipeline_regression.py`` and writes it to
``golden_pipeline_reference.hdf5``.

The golden test exists to catch *accidental* numeric drift in the full
particle -> grid-weighted spectra -> combination -> attenuation ->
integration -> observed-flux pipeline (the machinery most heavily
rewritten by the adaptive-precision templating work). It is not a test of
absolute physical correctness -- that is covered by the analytic/quadrature
tests elsewhere in the suite -- it is a tripwire for unintended changes.

Only re-run this script when a pipeline change is expected to alter these
numbers. Publish the regenerated file as the current
``golden-pipeline-reference`` Syndex release and note why in the commit
message; the file itself is ignored by Git.
"""

from pathlib import Path

import h5py
import numpy as np
from astropy.cosmology import Planck18
from unyt import Mpc, Msun, Myr

from synthesizer.emission_models import PacmanEmission
from synthesizer.emission_models.transformers import PowerLaw
from synthesizer.grid import Grid
from synthesizer.particle.stars import Stars

if __name__ == "__main__":
    grid = Grid("test_grid.hdf5")

    stars = Stars(
        initial_masses=np.array([1.0e6, 2.5e6, 1.2e6, 3.0e6, 0.8e6]) * Msun,
        ages=np.array([3.0e6, 1.0e7, 5.0e7, 2.0e8, 1.0e9]) * Myr,
        metallicities=np.array([0.001, 0.004, 0.01, 0.02, 0.03]),
        redshift=5.0,
        coordinates=np.zeros((5, 3)) * Mpc,
    )

    model = PacmanEmission(
        grid,
        tau_v=0.5,
        dust_curve=PowerLaw(slope=-1.0),
        fesc=0.1,
        fesc_ly_alpha=0.5,
    )

    sed = stars.get_spectra(model, out_dtype=np.float64)
    fnu = sed.get_fnu(Planck18, z=5.0, out_dtype=np.float64)

    path = Path(__file__).parent / "golden_pipeline_reference.hdf5"
    datasets = {
        "lam": (sed.lam.to("angstrom").value, "angstrom"),
        "lnu": (sed.lnu.to("erg/s/Hz").value, "erg/s/Hz"),
        "obslam": (sed._obslam, "angstrom"),
        "fnu": (fnu.to("nJy").value, "nJy"),
        "bolometric_luminosity": (
            sed.bolometric_luminosity.to("erg/s").value,
            "erg/s",
        ),
    }
    with h5py.File(path, "w") as hdf:
        for name, (data, units) in datasets.items():
            dataset = hdf.create_dataset(
                name,
                data=data,
                compression="gzip" if np.ndim(data) else None,
            )
            dataset.attrs["Units"] = units

    print(f"Wrote {path.name}")
