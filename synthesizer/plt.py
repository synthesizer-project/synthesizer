

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

plt.style.use('http://stephenwilkins.co.uk/matplotlibrc.txt')


def mlabel(l):
    return rf'$\rm {l}$'


def single(size=3.5):

    if type(size) == float or type(size) == int:
        size = (size, size)

    fig = plt.figure(figsize=size)

    left = 0.15
    height = 0.8
    bottom = 0.15
    width = 0.8

    ax = fig.add_axes((left, bottom, width, height))

    return fig, ax


def single_wcbar_right(hsize=3.5):

    left = 0.15
    height = 0.8
    bottom = 0.15
    width = 0.65

    fig = plt.figure(figsize=(hsize, hsize*width/height))

    ax = fig.add_axes((left, bottom, width, height))
    cax = fig.add_axes([left+width, bottom, 0.03, height])

    return fig, ax, cax

# def single_wcbar_top(hsize = 3.5):
#
#     left  = 0.15
#     height = 0.8
#     bottom = 0.15
#     width = 0.65
#
#     fig = plt.figure(figsize = (hsize, hsize*width/height))
#
#     ax = fig.add_axes((left, bottom, width, height))
#     cax = fig.add_axes([left+width, bottom, 0.03, height])
#
#     return fig, ax, cax


def single_histxy(hsize=3.5, set_axis_off=True):

    fig = plt.figure(figsize=(hsize, hsize))

    left = 0.15
    height = 0.65
    bottom = 0.15
    width = 0.65

    ax = fig.add_axes((left, bottom, width, height))  # main panel
    haxx = fig.add_axes((left, bottom + height, width, 0.15))  # x-axis hist panel
    # y-axis hist panel  # y-axis hist panel
    haxy = fig.add_axes((left + width, bottom, 0.15, height))

    if set_axis_off:
        haxx.axis('off')
        haxy.axis('off')

    return fig, ax, haxx, haxy
