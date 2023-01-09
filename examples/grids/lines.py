from synthesizer.grid import Grid

if __name__ == '__main__':

    # -------------------------------------------------
    # --- calcualte the EW for a given line as a function of age

    model = 'bpass-v2.2.1-bin_chab-100_cloudy-v17.03_log10Uref-2'

    lines = ['HI4861', 'OIII4959', 'OIII5007']

    # read in just some specific lines (excluding spectra)
    grid = Grid(model, read_spectra = False, read_lines = lines)

    # alternatively we could read in all lines by simply setting read_lines to be True
    # grid = Grid(model, read_spectra = False, read_lines = True)

    # we can also calculate luminosities and equivalent widths for a single line ...
    line_info = grid.get_line_info('HI4861', 5, 6) # 5,6 denote ia, iZ the age and metallicity grid point
    print(line_info)

    # or a combination combination of lines, e.g. a doublet
    line_info = grid.get_line_info(['HI4861', 'OIII4959', 'OIII5007'], 5, 6)
    print(line_info)


    # we can grab a different line that wasn't previously read in single line
    _line = grid.fetch_line('MgII2803')
    print(_line.keys())

    # by default this saves it to the grid object, however we can
    # also just load it on the fly
    _line = grid.fetch_line('MgII2803', save=False)
