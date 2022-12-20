from synthesizer.grid import LineGrid
from synthesizer.parametric.sfzh import SFH, ZH, generate_sfzh
from synthesizer.parametric.galaxy import LineGenerator
from unyt import yr, Myr


if __name__ == '__main__':

    # -------------------------------------------------
    # --- calcualte the EW for a given line as a function of age

    grid_dir = '/example/grid_directory/synthesizer_data/grids/'
    model = 'bpass-v2.2.1-bin_chab-100_cloudy-v17.03_log10Uref-2'
    # model = 'bpass-v2.2.1-bin_chab-300_cloudy-v17.03_log10Uref-2'
    
    grid = LineGrid(model, grid_dir=grid_dir)

    target_Z = 0.01  # target metallicity

    line_id = 'HI6563'
    line_id = ('HI6563')
    line_id = ('HI4861', 'OIII4959', 'OIII5007')

    # --- define the parameters of the star formation and metal enrichment histories
    sfh_p = {'duration': 100 * Myr}
    Z_p = {'log10Z': -2.0}  # can also use linear metallicity e.g. {'Z': 0.01}

    # --- define the functional form of the star formation and metal enrichment histories
    sfh = SFH.Constant(sfh_p)  # constant star formation
    Zh = ZH.deltaConstant(Z_p)  # constant metallicity

    # --- get the 2D star formation and metal enrichment history for the given SPS grid. This is (age, Z).
    sfzh = generate_sfzh(grid.log10ages, grid.metallicities, sfh, Zh)

    galaxy = LineGenerator(grid, sfzh)

    print(galaxy.get_intrinsinc_quantities(line_id))
