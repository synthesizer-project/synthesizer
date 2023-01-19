from synthesizer.grid import Grid
from synthesizer.parametric.sfzh import SFH, ZH, generate_sfzh
from synthesizer.galaxy.parametric import ParametricGalaxy as Galaxy
from unyt import yr, Myr


if __name__ == '__main__':

    """Example for generating a line object containing the luminosities, equivalent widths and other properties of lines.

    """

    # list of lines. Lines in nested lists (or tuples) denote doublets for which the combined line properties are calculated
    # line_ids = ['HI4861', 'OIII4959', 'OIII5007', ['OIII4959', 'OIII5007']]
    # should result in the same behaviour as above
    line_ids = ['HI4861', 'OIII4959', 'OIII5007', 'OIII4959,OIII5007']

    # open test grid though without reading spectra BUT reading the required lines
    grid_dir = '../../tests/test_grid'
    grid_name = 'test_grid'
    grid = Grid(grid_name, grid_dir=grid_dir, read_spectra=False, read_lines=line_ids)

    # --- define the parameters of the star formation and metal enrichment histories
    sfh_p = {'duration': 100 * Myr}
    Z_p = {'log10Z': -2.0}  # can also use linear metallicity e.g. {'Z': 0.01}

    # --- define the functional form of the star formation and metal enrichment histories
    sfh = SFH.Constant(sfh_p)  # constant star formation
    Zh = ZH.deltaConstant(Z_p)  # constant metallicity

    # --- get the 2D star formation and metal enrichment history for the given SPS grid. This is (age, Z).
    sfzh = generate_sfzh(grid.log10ages, grid.metallicities, sfh, Zh)

    # --- create the Galaxy object
    galaxy = Galaxy(sfzh)

    # --- create the Lines dictionary which contains line objects
    lines = galaxy.get_intrinsic_line(grid, line_ids)

    # --- print a summary of the Galaxy object
    print(galaxy)

    # --- print summaries of each line
    for line_id, line in lines.items():
        print(line)
