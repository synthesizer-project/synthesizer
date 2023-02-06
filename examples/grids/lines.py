"""
This example demonstrates how to:
- get a list of lines associated with a grid
- initialise a grid object with lines
- get line quantities for a single grid point
- ad hoc load an additional line
"""
import os
from synthesizer.grid import Grid

import matplotlib.pyplot as plt
import numpy as np
from synthesizer.grid import Grid, get_available_lines

if __name__ == '__main__':

    # Get the location of this script, __file__ is the absolute path of this
    # script, however we just want to directory
    script_path = os.path.abspath(os.path.dirname(__file__))

    # Define the grid
    grid_name = "test_grid"
    grid_dir = script_path + "/../../tests/test_grid/"

    # get list of available lines for the grid
    line_ids = get_available_lines(grid_name, grid_dir=grid_dir)

    # print this list of lines
    for line_id in line_ids:
        print(line_id)

    # read in just some specific lines (excluding spectra), note any line
    # inside the nested brackets is interpreted as a doublet
    # lines = ['H 1 4862.69A', 'O 3 4960.29A', 'O 3 5008.24A',
    #          ['O 3 4960.29A', 'O 3 5008.24A'], 'H 1 6564.62A']
    # grid = Grid(grid_name, grid_dir=grid_dir, read_spectra=False,
    #             read_lines=lines)

    # Alternatively we could read in all lines by simply setting read_lines
    # to be True.
    grid = Grid(grid_name, grid_dir=grid_dir,
                read_spectra=False, read_lines=True)

    print(grid.line_list)

    # we can grab a different line that wasn't previously read in single line
    line = grid.fetch_line('Si 2 1533.43A')
    print(line)

    # by default this saves it to the grid object, however we can
    # also just load it on the fly
    line = grid.fetch_line('Si 2 1533.43A', save=False)
    print(line)

    # we can also calculate luminosities and equivalent widths for a single
    # line 5,6 denote ia, iZ the age and metallicity grid point
    line = grid.get_line_info('H 1 4862.69A', 5, 6)
    print(line)

    # or a combination combination of lines, e.g. a doublet
    line = grid.get_line_info(['O 3 4960.29A', 'O 3 5008.24A'], 5, 6)
    print(line)

    # if we want multiple lines we can create a line collection instead
    lines = grid.get_lines_info(['H 1 4862.69A', 'O 3 4960.29A',
                                 'O 3 5008.24A', 'H 1 6564.62A'], 5, 6)
    print(lines)

    # we could get all lines available
    lines = grid.get_lines_info(line_ids, 5, 6)
    print(lines)

    # lines can be access like this:
    line = lines.lines['H 1 4862.69A']
    print(line)

    # ... or like this (uses __getitem__):
    line = lines['H 1 4862.69A']
    print(line)

    # we can measure line ratios
    ratio = lines.get_ratio('BalmerDecrement')  # R23, R2, R3, ...

    # or loop over availalable ratios
    for ratio_id in lines.available_ratios:
        ratio = lines.get_ratio(ratio_id)
        print(f'{ratio_id}: {ratio:.2f}')

    # we can plot a ratio against metallicity by looping over the metallicity grid
    ratio_id = 'R23'
    ia = 0  # 1 Myr old for test grid
    ratios = []
    for iZ, Z in enumerate(grid.metallicities):
        lines = grid.get_lines_info(line_ids, ia, iZ)
        ratios.append(lines.get_ratio(ratio_id))

    Zsun = grid.metallicities/0.0124
    plt.plot(Zsun, ratios)
    plt.xlim([0.01, 1])
    plt.ylim([1, 20])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$Z/Z_{\odot}$')
    plt.ylabel(lines.get_ratio_label(ratio_id))
    plt.show()

    # we can also generate "diagrams" pairs of line ratios like the BPT diagram
    diagram_id = 'BPT-NII'
    ia = 0  # 1 Myr old for test grid
    x = []
    y = []
    for iZ, Z in enumerate(grid.metallicities):
        lines = grid.get_lines_info(line_ids, ia, iZ)
        x_, y_ = lines.get_diagram(diagram_id)
        x.append(x_)
        y.append(y_)

    plt.plot(x, y)
    plt.xlim([0.01, 10])
    plt.ylim([0.05, 20])
    plt.xscale('log')
    plt.yscale('log')

    # grab x and y labels, this time use "fancy" label ids
    xlabel, ylabel = lines.get_diagram_label(diagram_id, fancy=True)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
