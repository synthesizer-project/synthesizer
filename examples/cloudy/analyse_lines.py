import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cmasher as cmr
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

from synthesizer.line import get_roman_numeral
from synthesizer.abundances import Elements
from synthesizer.cloudy import read_linelist, read_lines

def plot_lines(line_ids, wavelengths, luminosities, show = True, xlim = [900, 10000], ylim = [-3., 1.0], reference_line = 'H 1 4862.69A'):


    # get relative luminosities
    reference_line_luminosity = luminosities[line_ids==reference_line][0]
    relative_luminosities = luminosities - reference_line_luminosity # this is logged

    # get element (e.g. C) of each line
    element = np.array([''.join(id.split(' ')[0]) for id in line_ids])

    # get ion (e.g. II) of each line
    ion = np.array([''.join(id.split(' ')[1]) for id in line_ids])

    # get set of elements
    elements = list(set(element))

    # get masses of each element
    masses = np.array([Elements().A[element_] for element_ in elements])
    elements = [x for _, x in sorted(zip(masses, elements))]

    # create an array of element colours
    elements_colours = cmr.take_cmap_colors('cmr.pride_r', len(elements), cmap_range=(0.15, 0.85))

    # create an array containing the colour of every line
    colours = np.empty((len(ion), 3))

    for ele, colour in zip(elements, elements_colours):
        colours[element == ele] = colour

    # initialise figure
    fig = plt.figure(figsize=(7.0, 3.5))

    left = 0.1
    height = 0.8
    bottom = 0.15
    width = 0.75

    # add exes
    ax = fig.add_axes((left, bottom, width, height))

    # set xlimits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    handles = []

    for ion_, marker in zip(['1', '2', '3', '4', '5'], ['o', 'd', 's', '^', 'p', 'h']):

        s = ion == ion_

        ax.scatter(wavelengths[s], relative_luminosities[s], c=colours[s],  marker=marker)

        handles.append(Line2D([0], [0], marker=marker, lw=0, color='0.5', label=get_roman_numeral(int(ion_)),
                            markerfacecolor='0.5', markersize=5))


    # optional labelling

    for e, i, wv_, c, l, in zip(element, ion, wavelengths, colours, relative_luminosities):

        try:
            label = e+get_roman_numeral(int(i))
        except:
            label = e+i

        # ax.text(wv_, l+0.1, label, c=c, ha='center', va='bottom', fontsize=3,
        #         rotation=90, path_effects=[pe.withStroke(linewidth=1, foreground="white")])

        if wv_ < xlim[1] and wv_ > xlim[0]:
            ax.text(wv_, l, label, c=c, ha='center', va='center', fontsize=3,
                    path_effects=[pe.withStroke(linewidth=1, foreground="white")])


    for ele, c in zip(elements, elements_colours):

        handles.append(Line2D([0], [0], marker='o', lw=0, color=c, label=ele,
                            markerfacecolor=c, markersize=5))

    ax.set_xlabel(rf'$\rm \lambda/\regular{{\AA}} $')
    ax.set_ylabel(rf'$\rm L/{reference_line}$')

    ax.legend(handles=handles, fontsize=8, labelspacing=0.0,
            loc='upper left', bbox_to_anchor=(1., 1.0))

    if show:
        plt.show()

    return fig, ax






if __name__ == '__main__':

    """ 
    Demonstrate how to read in lines from cloudy outputs.
    """

    # output directory
    output_dir = 'output'

    # model name
    model_name = 'default'


    # read lines from a linelist file (.elin)
    line_ids, wavelengths, luminosities = read_linelist(f'{output_dir}/{model_name}')

    print(line_ids)
    print(wavelengths)
    print(luminosities)


    # read lines from a full .lines file

    # reference_line = 'H 1 4862.69A' # Hbeta (vacuum)
    reference_line = 'H 1 4861.33A' # Hbeta (air)
    relative_threshold = 2.0 # log

    line_ids, blends, wavelengths, intrinsic, emergent = read_lines(
        f'{output_dir}/{model_name}')

    # select emergent lines to use 
    luminosities = emergent

    threshold = luminosities[line_ids == reference_line] - relative_threshold

    print(threshold)

    # select line meeting various conditions
    s = (luminosities > threshold) & (blends == False) & (wavelengths < 50000)


    print(f'number of lines: {len(line_ids[s])}')

    # save a list of all lines
    with open('alllines.dat', 'w') as f:
        f.writelines('\n'.join(line_ids) + '\n')

    # save a list of only lines meeting the conditions above
    with open('defaultlines.dat', 'w') as f:
        f.writelines('\n'.join(line_ids[s]) + '\n')




    # plot filtered lines
    fig, ax = plot_lines(line_ids[s], wavelengths[s], luminosities[s], reference_line = reference_line)