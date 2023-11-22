"""
Generate lines from parametric galaxy
=====================================

Example for generating a emission lines for a parametric galaxy. This example 
will:
- show the available lines to a grid
- build a parametric galaxy (see make_stars and make_sed)
- calculate intrinsic line properties
- calculate dust-attenuated line properties
"""


from synthesizer.grid import Grid
from synthesizer.parametric import SFH, ZDist, Stars
from synthesizer.parametric.galaxy import Galaxy
from unyt import Myr


if __name__ == "__main__":
    # Get the location of this script, __file__ is the absolute path of this
    # script, however we just want to directory
    # script_path = os.path.abspath(os.path.dirname(__file__))

    # Define the grid
    grid_name = "test_grid"
    grid_dir = "../../tests/test_grid/"

    # open the grid
    grid = Grid(grid_name, grid_dir=grid_dir, read_spectra=False, read_lines=True)

    # define the functional form of the star formation and metal enrichment
    # histories
    sfh = SFH.Constant(duration=100 * Myr)
    metal_dist = ZDist.DeltaConstant(log10metallicity=-2.0)

    # get the 2D star formation and metal enrichment history for the given SPS
    # grid and print summary.
    stars = Stars(grid.log10age, grid.metallicity, sf_hist=sfh, metal_dist=metal_dist)
    print(stars)

    # create the Galaxy object and print a summary
    galaxy = Galaxy(stars)
    print(galaxy)

    # define list of lines that we're interested in. Note that we can provide
    # multiples which are automatically summed
    line_ids = [
        "H 1 4862.69A",
        "O 3 4958.91A",
        "O 3 5006.84A",
        ["O 3 4958.91A", "O 3 5006.84A"],
    ]

    # create the Lines dictionary which contains line objects
    lines = galaxy.get_line_intrinsic(grid, line_ids)
    print("-" * 50)
    print("INTRINSIC")
    for line in lines:
        print(line)

    # calculate attenuated line properties assuming uniform dust (should
    # leave EW unchanged)
    lines = galaxy.get_line_screen(grid, line_ids, tau_v=0.5)
    print(lines)
    print("-" * 50)
    print("SCREEN")
    for line in lines:
        print(line)

    # calculate attenuated line properties assuming different dust affecting
    # stellar and nebular components
    lines = galaxy.get_line_attenuated(
        grid, line_ids, tau_v_stellar=0.1, tau_v_nebular=0.5
    )

    print("-" * 50)
    print("ATTENUATED")
    for line in lines:
        print(line)
