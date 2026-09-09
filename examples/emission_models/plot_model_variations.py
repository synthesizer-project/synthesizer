"""
Plot model variations example
============================

This example demonstrates how to vary the parameters of an emission model and
generate a spectrum for every value in one go, using a ``ParameterList``.

Declaring the variation inside a single model, rather than building one model
per value, means the parts of the model the parameter cannot affect are only
computed once. Here we first vary the V band optical depth of a Pacman model,
which leaves the grid extractions shared between every variant, and then vary
the escape fraction alongside it to show two parameters combining into a grid
of spectra from a single model.
"""

import argparse

import matplotlib.pyplot as plt
from unyt import Msun, Myr, kelvin

from synthesizer.emission_models import Greybody, PacmanEmission
from synthesizer.emission_models.attenuation import PowerLaw
from synthesizer.emission_models.parameters import ParameterList
from synthesizer.emissions import plot_spectra
from synthesizer.grid import Grid
from synthesizer.parametric import SFH, Stars, ZDist
from synthesizer.parametric.galaxy import Galaxy

if __name__ == "__main__":
    # initialise argument parser
    parser = argparse.ArgumentParser(
        description=(
            "Create a plot of the emergent spectrum at a range of optical "
            "depths, using a single emission model"
        )
    )

    # The name of the grid. Defaults to the test grid.
    parser.add_argument(
        "-grid_name",
        "--grid_name",
        type=str,
        required=False,
        default="test_grid",
    )

    # Get dictionary of arguments
    args, _ = parser.parse_known_args()

    # initialise grid
    grid = Grid(args.grid_name)

    # define the parameters of the star formation and metal
    # enrichment histories
    sfh = SFH.Constant(max_age=100 * Myr)
    metal_dist = ZDist.DeltaConstant(log10metallicity=-2.0)

    # get the 2D star formation and metal enrichment history for the given
    # SPS grid. This is (age, Z).
    stars = Stars(
        grid.log10ages,
        grid.metallicities,
        sf_hist=sfh,
        metal_dist=metal_dist,
        initial_mass=1e9 * Msun,
    )

    # Create a galaxy object
    galaxy = Galaxy(stars)

    # Define the emission model, declaring that the optical depth should be
    # varied over a set of values rather than fixed to one
    model = PacmanEmission(
        grid,
        tau_v=ParameterList(
            [0.1, 0.3, 0.6, 1.0],
            label_modifier="tauv_%.1f",
        ),
        fesc=0.1,
        dust_curve=PowerLaw(slope=-1),
        dust_emission=Greybody(temperature=30 * kelvin, emissivity=2.0),
    )

    # Turn the declaration into models. Everything downstream of the optical
    # depth is duplicated, everything upstream is shared.
    expanded = model.expand_models()

    print(f"{len(model._models)} models became {len(expanded._models)}")
    print(
        f"one model per value would have needed "
        f"{4 * len(model._models)} models"
    )

    # Generate every variant in a single pass over the model
    galaxy.stars.get_spectra(expanded)

    # Collect the total spectrum from each variant, labelled by the optical
    # depth which produced it rather than by the model label
    spectra = {}
    for variant in expanded.select("total*"):
        tau_v = variant.variant_params["tau_v"]
        spectra[rf"$\tau_V = {tau_v:.1f}$"] = galaxy.stars.spectra[
            variant.label
        ]

    # Plot them together
    plot_spectra(
        spectra,
        show=False,
        figsize=(8, 5),
        quantity_to_plot="lnu",
    )
    plt.title("Total emission with varying optical depth")
    plt.show()

    # The dust emission is the other side of the same story: the light the dust
    # absorbs has to come back out in the infrared
    dust = {}
    for variant in expanded.select("dust_emission*"):
        tau_v = variant.variant_params["tau_v"]
        dust[rf"$\tau_V = {tau_v:.1f}$"] = galaxy.stars.spectra[variant.label]

    plot_spectra(
        dust,
        show=False,
        figsize=(8, 5),
        quantity_to_plot="lnu",
    )
    plt.title("Dust emission with varying optical depth")
    plt.show()

    # Two parameters at once. The optical depth sets how much of the
    # reprocessed light the dust takes, while the escape fraction sets how much
    # light never reaches the dust in the first place, so the two combine.
    coupled = PacmanEmission(
        grid,
        tau_v=ParameterList(
            [0.1, 0.3, 0.6, 1.0],
            label_modifier="tauv_%.1f",
        ),
        fesc=ParameterList([0.0, 0.5], label_modifier="fesc_%.1f"),
        dust_curve=PowerLaw(slope=-1),
        dust_emission=Greybody(temperature=30 * kelvin, emissivity=2.0),
    ).expand_models()

    galaxy.stars.get_spectra(coupled)

    print(
        f"\ntwo parameters gave {len(coupled.select('total*'))} spectra "
        f"from {len(coupled._models)} models"
    )

    # Draw the family, using the recorded parameters to style each curve rather
    # than picking its label apart: colour for the optical depth, line style
    # for the escape fraction
    tau_vs = sorted(
        {model.variant_params["tau_v"] for model in coupled.select("total*")}
    )
    fescs = sorted(
        {model.variant_params["fesc"] for model in coupled.select("total*")}
    )
    colours = plt.cm.viridis(
        [i / max(len(tau_vs) - 1, 1) for i in range(len(tau_vs))]
    )
    styles = ["-", "--", ":", "-."]

    fig, ax = plt.subplots(figsize=(8, 5))
    for model in coupled.select("total*"):
        spectra = galaxy.stars.spectra[model.label]
        tau_v = model.variant_params["tau_v"]
        fesc = model.variant_params["fesc"]
        ax.loglog(
            spectra.lam,
            spectra.lnu,
            color=colours[tau_vs.index(tau_v)],
            linestyle=styles[fescs.index(fesc) % len(styles)],
            label=rf"$\tau_V = {tau_v:.1f}$, $f_{{esc}} = {fesc:.1f}$",
            lw=1.2,
        )

    ax.set_xlim(1e2, 1e7)
    ax.set_ylim(1e26, 1e32)
    ax.set_xlabel(r"$\lambda / \AA$")
    ax.set_ylabel(r"$L_{\nu} / \mathrm{erg\ s^{-1}\ Hz^{-1}}$")
    ax.set_title("Total emission varying optical depth and escape fraction")
    ax.legend(fontsize=8, ncol=2, labelspacing=0.2)
    plt.show()
