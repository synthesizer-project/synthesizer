"""A module holding useful line ratios.

This contains both line ratios and diagnostic diagrams for emission line. As
well as including the standard line labels for common lines.

Line ids and specifically the wavelength part here are defined using the cloudy
standard, i.e. using vacuum wavelengths at <200nm and air wavelengths at
>200nm.
"""

import matplotlib.pyplot as plt

from synthesizer.emissions.utils import (
    N2,
    O1,
    O2,
    O3,
    S2,
    Ha,
    Hb,
    Ne3,
    O2b,
    O3r,
    get_diagram_labels,
)

# Define a dictionary to hold line ratios
ratios = {}

# Balmer decrement, should be [2.79--2.86] (Te, ne, dependent)
# for dust free
ratios["BalmerDecrement"] = [Ha, Hb]

# Add reference ratios
ratios["N2"] = [N2, Ha]
ratios["S2"] = [S2, Ha]
ratios["O1"] = [O1, Ha]
ratios["R2"] = [O2b, Hb]
ratios["R3"] = [O3r, Hb]
ratios["R23"] = [O3 + ", " + O2, Hb]
ratios["O32"] = [O3r, O2b]
ratios["Ne3O2"] = [Ne3, O2b]

# Create a tuple of available ratios for importing
available_ratios = tuple(ratios.keys())

# Define a dictionary to hold diagnostic diagrams
diagrams = {}

# Add reference diagrams
diagrams["OHNO"] = [ratios["R3"], [Ne3, O2]]
diagrams["BPT-NII"] = [[N2, Ha], ratios["R3"]]
diagrams["VO78-SII"] = [[S2, Ha], ratios["R3"]]
diagrams["VO78-OI"] = [[O1, Ha], ratios["R3"]]

# Create a tuple of available diagrams for importing
available_diagrams = tuple(diagrams.keys())


def get_bpt_kewley01(logNII_Ha):
    """
    BPT-NII demarcations from Kewley+2001.

    Kewley+03: https://arxiv.org/abs/astro-ph/0106324

    Demarcation defined by:

    log([OIII]/Hb) = 0.61 / (log([NII]/Ha) - 0.47) + 1.19

    Args:
        logNII_Ha (array)
            Array of log([NII]/Halpha) values to give the
            SF-AGN demarcation line

    Returns:
        array
            Corresponding log([OIII]/Hb) ratio array
    """
    return 0.61 / (logNII_Ha - 0.47) + 1.19


def plot_bpt_kewley01(
    logNII_Ha,
    fig=None,
    ax=None,
    show=True,
    xlimits=(0.01, 10),
    ylimits=(0.05, 20),
    **kwargs,
):
    """
    Plot the BPT-NII demarcations from Kewley+2001.

    Kewley+03: https://arxiv.org/abs/astro-ph/0106324

    Demarcation defined by:

    log([OIII]/Hb) = 0.61 / (log([NII]/Ha) - 0.47) + 1.19

    Args:
        logNII_Ha (array)
            Array of log([NII]/Halpha) values to give the
            SF-AGN demarcation line.
        fig (matplotlib.figure.Figure)
            Optional figure to plot on,
        ax (matplotlib.axes.Axes)
            Optional axis to plot on.
        show (bool)
            Should we show the plot?
        xlimits (tuple)
            The x-axis limits.
        ylimits (tuple)
            The y-axis limits.
        kwargs
            Any additional keyword arguments to pass to the plot.

    Returns:
        fig, ax
            The figure and axis objects
    """
    # Get the OIII/Hb ratio
    logOIII_Hb = get_bpt_kewley01(logNII_Ha)

    # Create the figure and axis if not provided
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(111)

    # Plot the demarcation line
    ax.loglog(10**logNII_Ha, 10**logOIII_Hb, **kwargs)

    # Limit the axes
    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)

    # Label the axes
    xlabel, ylabel = get_diagram_labels("BPT-NII")
    ax.set_xlabel(f"${xlabel}$")
    ax.set_ylabel(f"${ylabel}$")

    # Are we showing the plot?
    if show:
        plt.show()

    # Return the figure and axis
    return fig, ax


def get_bpt_kauffman03(logNII_Ha):
    """
    BPT-NII demarcations from Kauffman+2003.

    Kauffman+03: https://arxiv.org/abs/astro-ph/0304239

    Demarcation defined by:

    log([OIII]/Hb) = 0.61 / (log([NII]/Ha) - 0.05) + 1.3

    Args:
        logNII_Ha (array)
            Array of log([NII]/Halpha) values to give the
            SF-AGN demarcation line

    Returns:
        array
            Corresponding log([OIII]/Hb) ratio array
    """
    return 0.61 / (logNII_Ha - 0.05) + 1.3


def plot_bpt_kauffman03(
    logNII_Ha,
    fig=None,
    ax=None,
    show=True,
    xlimits=(0.01, 10),
    ylimits=(0.05, 20),
    **kwargs,
):
    """
    Plot the BPT-NII demarcations from Kauffman+2003.

    Kauffman+03: https://arxiv.org/abs/astro-ph/0304239

    Demarcation defined by:

    log([OIII]/Hb) = 0.61 / (log([NII]/Ha) - 0.05) + 1.3

    Args:
        logNII_Ha (array)
            Array of log([NII]/Halpha) values to give the
            SF-AGN demarcation line.
        fig (matplotlib.figure.Figure)
            Optional figure to plot on.
        ax (matplotlib.axes.Axes)
            Optional axis to plot on.
        show (bool)
            Should we show the plot?
        xlimits (tuple)
            The x-axis limits.
        ylimits (tuple)
            The y-axis limits.
        kwargs
            Any additional keyword arguments to pass to the plot_bpt_kewley01.

    Returns:
        fig, ax
            The figure and axis objects
    """
    # Get the OIII/Hb ratio
    logOIII_Hb = get_bpt_kauffman03(logNII_Ha)

    # Create the figure and axis if not provided
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(111)

    # Plot the demarcation line
    ax.loglog(10**logNII_Ha, 10**logOIII_Hb, **kwargs)

    # Limit the axes
    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)

    # Label the axes
    xlabel, ylabel = get_diagram_labels("BPT-NII")
    ax.set_xlabel(f"${xlabel}$")
    ax.set_ylabel(f"${ylabel}$")

    # Are we showing the plot?
    if show:
        plt.show()

    # Return the figure and axis
    return fig, ax
