
import numpy as np
import cmasher as cmr
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from .plt import mlabel


def plot_spectra(sed):
    """
    Plot a single spectra
    """

    plt.plot(np.log10(sed.lam), np.log10(sed.lnu), lw=1, alpha=0.8, label=spec_name)
    plt.xlim([2., 4.])
    plt.ylim([18., 23])
    plt.legend(fontsize=8, labelspacing=0.0)
    plt.xlabel(r'$\rm log_{10}(\lambda/\AA)$')
    plt.ylabel(r'$\rm log_{10}(L_{\nu}/erg\ s^{-1}\ Hz^{-1} M_{\odot}^{-1})$')
    plt.show()


def plot_log10Q(grid, ion='HI', hsize=3.5, vsize=2.5, cmap=cmr.sapphire,
                vmin=42.5, vmax=47.5, max_log10age=9.):

    left = 0.2
    height = 0.65
    bottom = 0.15
    width = 0.75

    if not vsize:
        vsize = hsize*width/height

    fig = plt.figure(figsize=(hsize, vsize))

    ax = fig.add_axes((left, bottom, width, height))
    cax = fig.add_axes([left, bottom+height+0.01, width, 0.05])

    y = np.arange(len(grid.metallicities))

    log10Q = grid.log10Q[ion]

    if max_log10age:
        ia_max = grid.get_nearest_index(max_log10age, grid.log10ages)
        log10Q = log10Q[:ia_max, :]
    else:
        ia_max = -1

    """ this is technically incorrect because metallicity
        is not on an actual grid."""
    ax.imshow(log10Q.T, origin='lower', extent=[grid.log10ages[0],
              grid.log10ages[ia_max], y[0]-0.5, y[-1]+0.5], cmap=cmap,
              aspect='auto', vmin=vmin, vmax=vmax)

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    cmapper.set_array([])

    fig.colorbar(cmapper, cax=cax, orientation='horizontal')
    cax.xaxis.tick_top()
    cax.xaxis.set_label_position('top')
    cax.set_xlabel(r'$\rm log_{10}(\dot{n}_{'+ion+'}/s^{-1}\ M_{\odot}^{-1})$')
    cax.set_yticks([])

    ax.set_yticks(y, grid.metallicities)
    ax.minorticks_off()
    ax.set_xlabel(mlabel('log_{10}(age/yr)'))
    ax.set_ylabel(mlabel('Z'))

    return fig, ax
