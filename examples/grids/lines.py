"""
This example demonstrates how to:
- get a list of lines associated with a grid
- initialise a grid object with lines
- get line quantities for a single grid point
- ad hoc load an additional line
"""

from synthesizer.grid import Grid, get_available_lines

if __name__ == '__main__':

    grid_dir = '../../tests/test_grid'
    grid_name = 'test_grid'

    # get list of available lines for the grid
    lines = get_available_lines(grid_name, grid_dir=grid_dir)

    # print this list of lines
    for line in lines:
        print(line)

    # read in just some specific lines (excluding spectra), note any line inside the nested brackets is interpreted as a doublet
    lines = ['H 1 4862.69A', 'O 3 4960.29A', 'O 3 5008.24A', ['O 3 4960.29A', 'O 3 5008.24A']]
    grid = Grid(grid_name, grid_dir=grid_dir, read_spectra=False, read_lines=lines)

    # alternatively we could read in all lines by simply setting read_lines to be True
    # grid = Grid(grid_name, grid_dir=grid_dir, read_spectra = False, read_lines = True)

    # we can also calculate luminosities and equivalent widths for a single line ...
    # 5,6 denote ia, iZ the age and metallicity grid point
    line = grid.get_line_info('H 1 4862.69A', 5, 6)
    print(line)

    # or a combination combination of lines, e.g. a doublet
    line = grid.get_line_info(['H 1 4862.69A', 'O 3 4960.29A', 'O 3 5008.24A'], 5, 6)
    print(line)

    # we can grab a different line that wasn't previously read in single line
    line = grid.fetch_line('Si 2 1533.43A')
    print(line)

    # by default this saves it to the grid object, however we can
    # also just load it on the fly
    line = grid.fetch_line('Si 2 1533.43A', save=False)
    print(line)
