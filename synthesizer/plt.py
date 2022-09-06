

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

rcParams = {}
rcParams['savefig.dpi'] = 300
rcParams['path.simplify'] = True

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'stixsans'
rcParams['text.usetex'] = False
rcParams['font.size'] = 9
rcParams['mathtext.fontset'] = 'stixsans'

rcParams['axes.linewidth'] = 0.5

rcParams['xtick.major.size'] = 3
rcParams['ytick.major.size'] = 3
rcParams['xtick.minor.size'] = 1.5
rcParams['ytick.minor.size'] = 1.5
rcParams['xtick.labelsize'] = 7
rcParams['ytick.labelsize'] = 7
rcParams['ytick.direction'] = 'in'
rcParams['xtick.direction'] = 'in'
rcParams['ytick.minor.visible'] = True
rcParams['xtick.minor.visible'] = True
rcParams['xtick.major.width'] = 0.25
rcParams['ytick.major.width'] = 0.25
rcParams['xtick.minor.width'] = 0.25
rcParams['ytick.minor.width'] = 0.25

rcParams['grid.alpha'] = 0.1
rcParams['grid.color'] = 'k'
rcParams['grid.linestyle'] = '-'
rcParams['grid.linewidth'] = 0.8

rcParams['legend.frameon'] = False

mpl.rcParams.update(rcParams)


def mlabel(l):
    return rf'$\rm {l}$'



def single(size = 3.5):

    if type(size) == float or type(size) == int:
        size = (size, size)

    fig = plt.figure(figsize = size)

    left  = 0.15
    height = 0.8
    bottom = 0.15
    width = 0.8

    ax = fig.add_axes((left, bottom, width, height))

    return fig, ax



def single_wcbar_right(hsize = 3.5):

    left  = 0.15
    height = 0.8
    bottom = 0.15
    width = 0.65

    fig = plt.figure(figsize = (hsize, hsize*width/height))

    ax = fig.add_axes((left, bottom, width, height))
    cax = fig.add_axes([left+width, bottom, 0.03, height])

    return fig, ax, cax



def single_histxy(hsize = 3.5, set_axis_off = True):

    fig = plt.figure(figsize = (hsize, hsize))

    left  = 0.15
    height = 0.65
    bottom = 0.15
    width = 0.65

    ax = fig.add_axes((left, bottom, width, height)) # main panel
    haxx = fig.add_axes((left, bottom + height, width, 0.15))  # x-axis hist panel
    haxy = fig.add_axes((left + width, bottom, 0.15, height))  # y-axis hist panel  # y-axis hist panel

    if set_axis_off:
        haxx.axis('off')
        haxy.axis('off')

    return fig, ax, haxx, haxy
