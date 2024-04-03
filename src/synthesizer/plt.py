"""A module containg helper functions for working with matplotlib.

Example usage:

    fig, ax = single(3.5)
    fig, ax, haxx, haxy = single_histxy(size=3.5, set_axis_off=True)
"""

from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def single(size: float = 3.5) -> Tuple[Figure, Axes]:
    """
    Set up a matplotlib figure containing a single axis.

    Args:
        size: The size of the plot along both axes in inches.

    Returns:
        The created figure containing the axes.
        The axis on which to plot.
    """
    # Create the figure
    fig: Figure = plt.figure(figsize=(size, size))

    # Define the coordinates of the axis
    left: float = 0.15
    height: float = 0.8
    bottom: float = 0.15
    width: float = 0.8

    # Create the single set of axes
    ax: Axes = fig.add_axes((left, bottom, width, height))

    return fig, ax


def single_wcbar_right(hsize: float = 3.5) -> Tuple[Figure, Axes, Axes]:
    """
    Set up a matplotlib figure containing a single axis and a colorbar.

    Args:
        hsize: The horizontal size of the plot in inches.

    Returns:
        The created figure containing the axes.
        The axis on which to plot.
        The axis for the colorbar.
    """
    # Define the coordinates of the axis
    left: float = 0.15
    height: float = 0.8
    bottom: float = 0.15
    width: float = 0.65

    # Create the figure
    fig: Figure = plt.figure(figsize=(hsize, hsize * width / height))

    # Create the single set of axes and colorbar axes
    ax: Axes = fig.add_axes((left, bottom, width, height))
    cax: Axes = fig.add_axes((left + width, bottom, 0.03, height))

    return fig, ax, cax


def single_histxy(
    size: float = 3.5,
    set_axis_off: bool = True,
) -> Tuple[Figure, Axes, Axes, Axes]:
    """
    Set up a matplotlib figure with a single axis, and a hist on each axis.

    Args:
        size: The size of the plot along both axes in inches.
        set_axis_off: Should the histograms have their axes removed?

    Returns:
        The created figure containing the axes.
        The axis on which to plot.
        The axis for the x histogram.
        The axis for the y histogram.
    """
    # Create the figure
    fig: Figure = plt.figure(figsize=(size, size))

    # Define the coordinates of the axis
    left: float = 0.15
    height: float = 0.65
    bottom: float = 0.15
    width: float = 0.65

    # Set up each axis
    ax: Axes = fig.add_axes((left, bottom, width, height))  # main panel
    haxx: Axes = fig.add_axes(
        (left, bottom + height, width, 0.15)
    )  # x-axis hist panel
    haxy: Axes = fig.add_axes(
        (left + width, bottom, 0.15, height)
    )  # y-axis hist panel

    if set_axis_off:
        haxx.axis("off")
        haxy.axis("off")

    return fig, ax, haxx, haxy
