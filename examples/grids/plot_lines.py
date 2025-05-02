"""
Get lines example
=================

This example demonstrates how to:
- get a list of lines associated with a grid
- initialise a grid object with lines
- get line quantities for a single grid point
- ad hoc load an additional line
"""

import matplotlib.pyplot as plt

from synthesizer.emissions.line_ratios import (
    get_diagram_labels,
    get_ratio_label,
)
from synthesizer.grid import Grid

if __name__ == "__main__":
    # Get the location of this script, __file__ is the absolute path of this
    # script, however we just want to directory
    # script_path = os.path.abspath(os.path.dirname(__file__))

    # Define the grid
    grid_name = "test_grid"
    grid_dir = "../../tests/test_grid/"

    # initialise grid
    grid = Grid(grid_name, grid_dir=grid_dir, read_lines=True)

    # get list of lines
    print(grid.available_lines)

    # choose age and metallicity
    log10age = 6.0  # log10(age/yr)
    metallicity = 0.01  # metallicity

    # get the grid point for this log10age and metallicity
    grid_point = grid.get_grid_point(
        log10ages=log10age,
        metallicity=metallicity,
    )

    # get information on one line
    line = grid.get_lines_at_grid_point(grid_point, "H 1 4861.32A")
    print(line)

    # or a combination of lines, e.g. a doublet
    line = grid.get_lines_at_grid_point(
        grid_point, ["H 1 4861.32A", "O 3 4958.91A", "O 3 5006.84A"]
    )
    print(line)

    # create a line collection from all lines
    lines = grid.get_lines_at_grid_point(grid_point, grid.available_lines)
    print(lines)

    # we can measure line ratios
    ratio_id = "BalmerDecrement"
    ratio = lines.get_ratio(ratio_id)  # R23, R2, R3, ...
    print(f"{ratio_id}: {ratio:.2f}")

    # or loop over availalable ratios
    for ratio_id in lines.available_ratios:
        ratio = lines.get_ratio(ratio_id)
        print(f"{ratio_id}: {ratio:.2f}")

    # we can plot a ratio against metallicity by looping over the metallicity
    # grid
    ratio_id = "R23"
    ia = 0  # 1 Myr old for test grid
    ratios = []
    for iZ, Z in enumerate(grid.metallicity):
        grid_point = (ia, iZ)
        lines = grid.get_lines_at_grid_point(grid_point, grid.available_lines)
        ratios.append(lines.get_ratio(ratio_id))

    Zsun = grid.metallicity / 0.0124
    plt.plot(Zsun, ratios)
    plt.xlim([0.01, 1])
    plt.ylim([1, 20])
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$Z/Z_{\odot}$")
    plt.ylabel(rf"${get_ratio_label(ratio_id)}$")
    plt.show()

    # we can also generate "diagrams" pairs of line ratios like the BPT diagram
    diagram_id = "BPT-NII"
    ia = 0  # 1 Myr old for test grid
    x = []
    y = []
    for iZ, Z in enumerate(grid.metallicity):
        grid_point = (ia, iZ)
        lines = grid.get_lines_at_grid_point(grid_point, grid.available_lines)
        x_, y_ = lines.get_diagram(diagram_id)
        x.append(x_)
        y.append(y_)

    plt.plot(x, y)
    plt.xlim([0.01, 10])
    plt.ylim([0.05, 20])
    plt.xscale("log")
    plt.yscale("log")

    # grab x and y labels, this time use "fancy" label ids
    xlabel, ylabel = get_diagram_labels(diagram_id)

    plt.xlabel(rf"${xlabel}$")
    plt.ylabel(rf"${ylabel}$")
    plt.show()
