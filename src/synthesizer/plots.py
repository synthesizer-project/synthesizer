""" A module containing miscellaneous plotting functions.
"""
import numpy as np
import cmasher as cmr
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def plot_log10Q(
    grid,
    ion="HI",
    hsize=3.5,
    vsize=2.5,
    cmap=cmr.sapphire,
    vmin=None,
    vmax=None,
    max_log10age=9.0,
):
    """
    Make a simple plot of the specific ionsing photon luminosity as a function
    of log10age and metallicity for a given grid and ion.

    Parameters
    ----------
    grid : str
        grid object

    ion : str
        The desired ion, most grids only have HI and HII calculated by default at present.

    hsize : float
        The horizontal size of the figure

    vsize : float
        The vertical size of the figure

    cmap : object
        A colourmap object

    vmin : float
        The minimum specific ionising luminosity used in the colourmap

    vmax : float
        The maximum specific ionising luminosity used in the colourmap

    max_log10age : float
        The maximum log10(age) to plot

    TODO: can not currently handle 3D grids.
    """

    left = 0.2
    height = 0.65
    bottom = 0.15
    width = 0.75

    if not vsize:
        vsize = hsize * width / height

    fig = plt.figure(figsize=(hsize, vsize))

    ax = fig.add_axes((left, bottom, width, height))
    cax = fig.add_axes([left, bottom + height + 0.01, width, 0.05])

    y = np.arange(len(grid.metallicity))

    # select grid for specific ion
    log10Q = grid.log10Q[ion]

    # truncate grid if max age provided
    if max_log10age:
        ia_max = grid.get_nearest_index(max_log10age, grid.log10age)
        log10Q = log10Q[:ia_max, :]
    else:
        ia_max = -1

    # if no limits supplied set a sensible range for HI ion otherwise use min max
    if ion == "HI":
        if not vmin:
            vmin = 42.5
        if not vmax:
            vmax = 47.5
    else:
        if not vmin:
            vmin = np.min(log10Q)
        if not vmax:
            vmax = np.max(log10Q)

    """ this is technically incorrect because metallicity
        is not on an actual grid."""
    ax.imshow(
        log10Q.T,
        origin="lower",
        extent=[grid.log10age[0], grid.log10age[ia_max], y[0] - 0.5, y[-1] + 0.5],
        cmap=cmap,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    cmapper.set_array([])

    # add colourbar
    fig.colorbar(cmapper, cax=cax, orientation="horizontal")
    cax.xaxis.tick_top()
    cax.xaxis.set_label_position("top")
    cax.set_xlabel(r"$\rm log_{10}(\dot{n}_{" + ion + "}/s^{-1}\ M_{\odot}^{-1})$")
    cax.set_yticks([])

    ax.set_yticks(y, grid.metallicity)
    ax.minorticks_off()
    ax.set_xlabel(mlabel("log_{10}(age/yr)"))
    ax.set_ylabel(mlabel("Z"))

    return fig, ax
