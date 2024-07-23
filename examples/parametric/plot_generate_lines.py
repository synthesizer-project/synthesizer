"""
Generating Lines from a Parametric Galaxy
=========================================

In this example we're going to generate emission line predictions for a
parametric galaxy.
"""

import synthesizer.line_ratios as line_ratios
from synthesizer.emission_models import AttenuatedEmission, IncidentEmission
from synthesizer.emission_models.attenuation import PowerLaw
from synthesizer.grid import Grid
from synthesizer.parametric import SFH, Stars, ZDist
from synthesizer.parametric.galaxy import Galaxy
from unyt import Myr

if __name__ == "__main__":
    # Begin by defining and initialising the grid. By setting `read_spectra`
    # to `False` we can avoid reading in the spectra,
    # reducing the memory footprint.
    grid_name = "test_grid"
    grid_dir = "../../tests/test_grid/"
    grid = Grid(grid_name, grid_dir=grid_dir)  # , read_spectra=False)

    # Define the emission model
    incident = IncidentEmission(grid)

    # Let's now build a galaxy following the other tutorials:
    # Define the functional form of the star formation and metal
    # enrichment histories

    # Constant star formation
    sfh = SFH.Constant(duration=100 * Myr)

    # Constant metallicity
    metal_dist = ZDist.DeltaConstant(log10metallicity=-2.0)

    # Get the 2D star formation and metal enrichment history
    # for the given SPS grid. This is (age, Z).
    stars = Stars(
        grid.log10age,
        grid.metallicity,
        sf_hist=sfh,
        metal_dist=metal_dist,
        initial_mass=10**8.5,
    )

    # Create the Galaxy object
    galaxy = Galaxy(stars)

    # Print a summary of the Galaxy object
    print(galaxy)

    # Let's define a list of lines that we're interested in. Note that we can
    # provide multiples which are automatically summed as if they were blended.
    line_ids = [
        line_ratios.Hb,  # "H 1 4861.32A"
        line_ratios.O3b,  # "O 3 4958.91A"
        line_ratios.O3r,  # "O 3 5006.84A"
        line_ratios.O3,  # ["O 3 4958.91A", "O 3 5006.84A"]
        line_ratios.O3
        + ","
        + line_ratios.Hb,  # ["O 3 4958.91A", "O 3 5006.84A", "H 1 4861.32A"]
    ]

    # Next, let's get the intrinsic line properties:
    lines = galaxy.stars.get_lines(line_ids, incident)

    # This produces a LineCollection object which has some useful methods and
    # information.
    print(lines)

    # Let's now examine individual lines (or doublets):
    for line in lines:
        print(line)

    # Those lines are now associated with the `Galaxy` object, revealed by
    # using the print function:
    print(galaxy)

    # Next, lets get the attenuated line properties:
    model = AttenuatedEmission(
        emitter='stellar',
        tau_v=1.0,
        dust_curve=PowerLaw(slope=-1),
        apply_dust_to=incident,
    )

    lines_att = galaxy.stars.get_lines(
        line_ids, model,
    )
    print(lines_att)
