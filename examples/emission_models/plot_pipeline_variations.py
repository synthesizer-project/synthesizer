"""
Pipeline with model variations example
=====================================

This example runs a ``Pipeline`` over an emission model carrying two parameter
variations, and computes photometry for every variant in a single pass.

Photometry is where variations earn their keep: a colour-colour diagram wants
one point per set of parameters, and a ``ParameterList`` gives you the whole
family from one model, sharing the grid extractions between the variants
instead of repeating them once per value.

The expanded labels flow straight through the pipeline and into the output
file, so this example also checks that: the photometry written out is compared
against the variants the model expanded into before anything is plotted.
"""

import argparse
import tempfile
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from unyt import angstrom

from synthesizer import TEST_DATA_DIR
from synthesizer.emission_models import PacmanEmission
from synthesizer.emission_models.parameters import ParameterList
from synthesizer.grid import Grid
from synthesizer.instruments import UVJ, PhotometricInstrument
from synthesizer.load_data.load_camels import load_CAMELS_IllustrisTNG
from synthesizer.pipeline import Pipeline

# Where the photometry ends up in the output file
PHOTOMETRY_GROUP = "Galaxies/Stars/Photometry/Luminosities"


def magnitude_colour(luminosities, blue, red):
    """Return the colour between two bands.

    Args:
        luminosities (dict):
            The luminosity in each band, keyed by filter code.
        blue (str):
            The filter code of the bluer band.
        red (str):
            The filter code of the redder band.

    Returns:
        np.ndarray:
            The colour of each galaxy, in magnitudes.
    """
    return -2.5 * np.log10(luminosities[blue] / luminosities[red])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run a pipeline over an emission model with varied parameters "
            "and plot the photometry of every variant"
        )
    )
    parser.add_argument(
        "-grid_name",
        type=str,
        default="test_grid",
        help="the SPS grid to use",
    )
    args = parser.parse_args()

    grid = Grid(args.grid_name)

    # Declare the variations. The optical depth and the escape fraction are
    # independent, so the three optical depths and two escape fractions
    # combine into six variants of the emergent emission.
    varied = PacmanEmission(
        grid,
        tau_v=ParameterList([0.1, 0.5, 1.0], label_modifier="tauv%.1f"),
        fesc=ParameterList([0.0, 0.5], label_modifier="fesc%.1f"),
    )
    unexpanded_count = len(varied._models)
    model = varied.expand_models()

    # Only the emergent emission is wanted, so nothing else is written out.
    # The models feeding it are still computed, they are just not saved, and
    # the ones which no variation can reach are computed once for all six
    # variants rather than once each.
    model.save_spectra("emergent*")
    variants = sorted(model.select("emergent*"), key=lambda m: m.label)

    print(f"{unexpanded_count} models became {len(model._models)}")
    print(
        f"one model per value would have needed "
        f"{unexpanded_count * len(variants)} models"
    )
    print(f"{len(variants)} variants of the emergent emission will be saved")

    # A UVJ instrument, which is three filters and therefore cheap
    lam = np.linspace(1e3, 1e5, 500) * angstrom
    instrument = PhotometricInstrument(label="UVJ", filters=UVJ(new_lam=lam))

    galaxies = load_CAMELS_IllustrisTNG(
        TEST_DATA_DIR,
        snap_name="camels_snap.hdf5",
        group_name="camels_subhalo.hdf5",
        physical=True,
    )[:3]
    n_galaxies = len(galaxies)

    # Run the pipeline. Photometry is the only operation asked for, so the
    # spectra are generated and used without ever being written out.
    pipeline = Pipeline(model, nthreads=1, verbose=0)
    pipeline.add_galaxies(galaxies)
    pipeline.get_photometry_luminosities(instrument)
    pipeline.run()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "pipeline_variations.hdf5"
        pipeline.write(str(path), verbose=0)

        # Read the photometry back, keyed by the variant which produced it,
        # along with what the file says each variant was made of. The
        # parameters are written alongside the models, so the file can be read
        # without picking the generated labels apart.
        with h5py.File(path, "r") as hdf:
            written = set(hdf[PHOTOMETRY_GROUP].keys())
            photometry = {
                model.label: {
                    filter_code: hdf[
                        f"{PHOTOMETRY_GROUP}/{model.label}/{filter_code}"
                    ][:]
                    for filter_code in instrument.filters.filter_codes
                }
                for model in variants
            }
            recorded = {
                model.label: dict(
                    hdf[f"EmissionModel/{model.label}/VariantParameters"].attrs
                )
                for model in variants
            }

    # Every variant should have made it through the pipeline and into the
    # file under its own label, with a luminosity for every galaxy
    expected = {model.label for model in variants}
    assert written == expected, (
        f"the pipeline wrote {sorted(written)}, expected {sorted(expected)}"
    )
    for label, bands in photometry.items():
        for filter_code, lums in bands.items():
            assert lums.shape == (n_galaxies,), (
                f"{label}/{filter_code} has shape {lums.shape}, "
                f"expected {(n_galaxies,)}"
            )
            assert np.all(lums > 0.0), (
                f"{label}/{filter_code} holds non-positive luminosities"
            )

    # The file records the parameters behind every variant, not just its name
    for model in variants:
        assert recorded[model.label] == model.variant_params, (
            f"{model.label} was written as {recorded[model.label]}, "
            f"expected {model.variant_params}"
        )

    print(f"\nevery variant was written to the output file: {len(written)}")
    print("and the file says what each one was made of, e.g.")
    for model in variants[:2]:
        print(f"    {model.label}: {recorded[model.label]}")

    # Style each variant by the parameters which produced it rather than by
    # picking its label apart
    tau_vs = sorted({model.variant_params["tau_v"] for model in variants})
    fescs = sorted({model.variant_params["fesc"] for model in variants})
    colours = {fescs[0]: "#2f6fb8", fescs[-1]: "#c1440e"}
    markers = {fescs[0]: "o", fescs[-1]: "s"}

    # Gather the colours into a track per galaxy, so the effect of the dust is
    # a line rather than a scatter
    tracks = {
        (fesc, blue, red): np.array(
            [
                magnitude_colour(photometry[model.label], blue, red)
                for model in sorted(
                    (
                        model
                        for model in variants
                        if model.variant_params["fesc"] == fesc
                    ),
                    key=lambda model: model.variant_params["tau_v"],
                )
            ]
        )
        for fesc in fescs
        for blue, red in [("U", "V"), ("V", "J")]
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)

    for ax, (blue, red) in zip(axes, [("U", "V"), ("V", "J")]):
        for fesc in fescs:
            track = tracks[(fesc, blue, red)]
            for galaxy_index in range(n_galaxies):
                ax.plot(
                    tau_vs,
                    track[:, galaxy_index],
                    color=colours[fesc],
                    marker=markers[fesc],
                    markersize=5,
                    lw=1.4,
                    alpha=0.85,
                    label=(
                        rf"$f_{{esc}} = {fesc:.1f}$"
                        if galaxy_index == 0 and ax is axes[0]
                        else None
                    ),
                )

        ax.set_xlabel(r"$\tau_V$")
        ax.set_ylabel(f"{blue} - {red}")
        ax.set_xticks(tau_vs)
        ax.grid(True, alpha=0.2)

    axes[0].legend(frameon=False)
    fig.suptitle(
        f"UVJ colours of {n_galaxies} galaxies across "
        f"{len(variants)} model variants"
    )
    plt.show()
