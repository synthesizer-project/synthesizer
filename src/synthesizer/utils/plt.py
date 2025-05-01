"""A module containg helper functions for working with matplotlib.

Example usage:

    fig, ax = single(3.5)
    fig, ax, haxx, haxy = single_histxy(size=3.5, set_axis_off=True)
"""

import matplotlib.pyplot as plt


def single(size: float = 3.5) -> tuple[plt.Figure, plt.Axes]:
    """Set up a matplotlib figure containing a single axis.

    Args:
        size (float):
            The size of the plot along both axes in inches.

    Returns:
        matplotlib.Figure: The created figure containing the axes.
        matplotlib.Axes: The axis on which to plot.
    """
    # Create the figure
    fig = plt.figure(figsize=size)

    # Define the coordinates of the axis
    left = 0.15
    height = 0.8
    bottom = 0.15
    width = 0.8

    # Create the single set of axes
    ax = fig.add_axes((left, bottom, width, height))

    return fig, ax


def single_wcbar_right(hsize=3.5):
    """Set up a matplotlib figure containing a single axis and a colorbar.

    Args:
        hsize (float):
            The horizontal size of the plot in inches.

    Returns:
        matplotlib.Figure: The created figure containing the axes.
        matplotlib.Axes: The axis on which to plot.
        matplotlib.Axes: The axis for the colorbar.
    """
    # Define the coordinates of the axis
    left = 0.15
    height = 0.8
    bottom = 0.15
    width = 0.65

    # Create the figure
    fig = plt.figure(figsize=(hsize, hsize * width / height))

    # Create the single set of axes and colorbar axes
    ax = fig.add_axes((left, bottom, width, height))
    cax = fig.add_axes([left + width, bottom, 0.03, height])

    return fig, ax, cax


def single_histxy(size=3.5, set_axis_off=True):
    """Get a figure containing an axis, and a histogram on each axis.

    Args:
        size (float):
            The size of the plot along both axes in inches.
        set_axis_off (bool):
            Should the histograms have their axes removed?

    Returns:
        matplotlib.Figure: The created figure containing the axes.
        matplotlib.Axes: The axis on which to plot.
        matplotlib.Axes: The axis for the x-axis histogram.
        matplotlib.Axes: The axis for the y-axis histogram.
    """
    # Create the figure
    fig = plt.figure(figsize=(size, size))

    # Define the coordinates of the axis
    left = 0.15
    height = 0.65
    bottom = 0.15
    width = 0.65

    # Set up each axis
    ax = fig.add_axes(
        (
            left,
            bottom,
            width,
            height,
        )
    )  # main panel
    haxx = fig.add_axes(
        (
            left,
            bottom + height,
            width,
            0.15,
        )
    )  # x-axis hist panel

    haxy = fig.add_axes(
        (
            left + width,
            bottom,
            0.15,
            height,
        )
    )  # y-axis hist panel

    if set_axis_off:
        haxx.axis("off")
        haxy.axis("off")

    return fig, ax, haxx, haxy
