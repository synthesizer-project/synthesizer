from synthesizer.grid import SpectralGrid

if __name__ == '__main__':

    # -------------------------------------------------
    # --- calcualte the EW for a given line as a function of age

    grid_dir = '/example/grid/directory/synthesizer_data/grids/'
    model = 'bpass-v2.2.1-bin_chab-100_cloudy-v17.03_log10Uref-2'
    target_Z = 0.01 # target metallicity

    line_id = 'HI6563'
    line_id = ('HI6563')
    line_id = ('HI4861', 'OIII4959', 'OIII5007')

    # grid = LineGrid(model, verbose = True)
    grid = SpectralGrid(model, verbose = True)
    
    # we can perform all the suaul grid operations
    iZ = grid.get_nearest_index(target_Z, grid.metallicities)

    # we can grab a single line 
    _line = grid.fetch_line('MgII2803')
    print(_line.keys())

    # by default this saves it to the grid object, however we can 
    # also just load it on the fly
    _line = grid.fetch_line('MgII2803', save=False)

    # we can also calculate equivalent widths for a combination of lines
    grid.get_line_info(line_id, 5, 6)

    for ia, log10age in enumerate(grid.log10ages):
        print(log10age, grid.get_line_info(line_id, ia, iZ))
