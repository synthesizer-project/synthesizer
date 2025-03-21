"""
Generating Lines from a Parametric Galaxy
=========================================

In this example we're going to generate emission line predictions for a
parametric galaxy.
"""

from unyt import Msun, Myr

from synthesizer.emission_models import AttenuatedEmission, NebularEmission
from synthesizer.emission_models.attenuation import PowerLaw
from synthesizer.emissions import O3, Hb, O3b, O3r
from synthesizer.grid import Grid
from synthesizer.parametric import SFH, Stars, ZDist
from synthesizer.parametric.galaxy import Galaxy

if __name__ == "__main__":
    # Begin by defining and initialising the grid.
    grid_name = "test_grid"
    grid_dir = "../../tests/test_grid/"
    grid = Grid(grid_name, grid_dir=grid_dir)

    # Define the emission model
    nebular = NebularEmission(grid)

    # Let's now build a galaxy following the other tutorials:
    # Define the functional form of the star formation and metal
    # enrichment histories

    # Constant star formation
    sfh = SFH.Constant(max_age=100 * Myr)

    # Constant metallicity
    metal_dist = ZDist.DeltaConstant(log10metallicity=-2.0)

    # Get the 2D star formation and metal enrichment history
    # for the given SPS grid. This is (age, Z).
    stars = Stars(
        grid.log10age,
        grid.metallicity,
        sf_hist=sfh,
        metal_dist=metal_dist,
        initial_mass=10**8.5 * Msun,
    )

    # Create the Galaxy object
    galaxy = Galaxy(stars)

    # Print a summary of the Galaxy object
    print(galaxy)

    # Let's define a list of lines that we're interested in. To do this we'll
    # first extract some line ratios from the `line_ratios` module, and then
    # Extract the individual lines from each ratio
    line_ids = [
        Hb,  # "H 1 4861.32A"
        O3b,  # "O 3 4958.91A"
        O3r,  # "O 3 5006.84A"
        O3,  # ["O 3 4958.91A", "O 3 5006.84A"]
    ]
    line_ids = [lid.strip() for lids in line_ids for lid in lids.split(",")]

    # Next, let's get the intrinsic line properties:
    lines = galaxy.stars.get_lines(line_ids, nebular)

    # This produces a LineCollection object which has some useful methods and
    # information.
    print(lines)

    # Those lines are now associated with the `Galaxy` object, revealed by
    # using the print function:
    print(galaxy)

    # Next, lets get the attenuated line properties:
    model = AttenuatedEmission(
        emitter="stellar",
        tau_v=1.0,
        dust_curve=PowerLaw(slope=-1),
        apply_to=nebular,
    )

    lines_att = galaxy.stars.get_lines(
        line_ids,
        model,
    )
    print(lines_att)
