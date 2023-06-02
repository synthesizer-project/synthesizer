

import numpy as np
import matplotlib.pyplot as plt
from synthesizer.cloudy import read_lines
from synthesizer.abundances import Elements
from synthesizer.line import get_roman_numeral
from matplotlib.colors import Normalize
import cmasher as cmr
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

plt.style.use('http://stephenwilkins.co.uk/matplotlibrc.txt')

synthesizer_data_dir = '/Users/stephenwilkins/Dropbox/Research/data/synthesizer/cloudy/'
model = 'bpass-2.2.1-bin_chabrier03-0.1,100.0_cloudy'
filename = '0_8'
threshold_line = 'H 1 4862.69A'
relative_threshold = 2.5

cloudy_ids, blends, wavelengths, intrinsic, emergent = read_lines(
    f'{synthesizer_data_dir}/{model}/{filename}')

threshold = emergent[cloudy_ids == threshold_line] - relative_threshold

relative_emergent = emergent - emergent[cloudy_ids == threshold_line]


s = (emergent > threshold) & (blends == False) & (wavelengths < 50000)

# element
element = np.array([''.join(id.split(' ')[0]) for id in cloudy_ids])[s]

ion = np.array([''.join(id.split(' ')[1]) for id in cloudy_ids])[s]


elements = list(set(element))
masses = np.array([Elements().A[element_] for element_ in elements])
elements = [x for _, x in sorted(zip(masses, elements))]

elements_colours = cmr.take_cmap_colors('cmr.pride_r', len(elements), cmap_range=(0.15, 0.85))

colours = np.empty((len(ion), 3))

for ele, colour in zip(elements, elements_colours):
    colours[element == ele] = colour


fig = plt.figure(figsize=(7.0, 3.5))

left = 0.1
height = 0.8
bottom = 0.15
width = 0.75

ax = fig.add_axes((left, bottom, width, height))

xlim = [900, 10000]

ax.set_xlim(xlim)
ax.set_ylim([-3., 1.0])

handles = []

for ion_, marker in zip(['1', '2', '3', '4', '5'], ['o', 'd', 's', '^', 'p', 'h']):

    sion = ion == ion_

    ax.scatter(wavelengths[s][sion], relative_emergent[s][sion], c=colours[sion],  marker=marker)

    handles.append(Line2D([0], [0], marker=marker, lw=0, color='0.5', label=get_roman_numeral(int(ion_)),
                          markerfacecolor='0.5', markersize=5))


# optional labelling

for e, i, wv_, c, l, in zip(element, ion, wavelengths[s], colours, relative_emergent[s]):

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

ax.set_xlabel(r'$\rm \lambda/\regular{\AA} $')
ax.set_ylabel(r'$\rm L/L(H\beta)$')

ax.legend(handles=handles, fontsize=8, labelspacing=0.0,
          loc='upper left', bbox_to_anchor=(1., 1.0))

fig.savefig(f'lines.pdf')

fig.clf()
