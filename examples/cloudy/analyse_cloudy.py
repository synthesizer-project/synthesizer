
import numpy as np
import matplotlib.pyplot as plt

from synthesizer.grid import Grid
from synthesizer.cloudy_sw import read_continuum, read_lines, make_linecont
from synthesizer.plt import single
from synthesizer.sed import calculate_Q


def plot_model_spectra(models, spectra='total', show=False):

    fig, ax = single((6., 3.5))

    for model in models:
        spec_dict = read_continuum(model, return_dict=True)
        ax.plot(np.log10(spec_dict['lam']), np.log10(
            spec_dict[spectra]), lw=1, alpha=0.5)

    ax.set_xlim([3., 4.5])
    ax.set_ylim([15, 19])
    ax.legend(fontsize=8, labelspacing=0.0)
    ax.set_xlabel(r'$\rm log_{10}(\lambda/\AA)$')
    ax.set_ylabel(r'$\rm log_{10}(L_{\nu}/L_{\nu}()}^1)$')

    if show:
        plt.show()

    return fig, ax


def compare_with_grid(model, grid_name, ia=0, iZ=8, spectra='total', show=False):
    """ compare a cloudy model with the corresponding grid-spectra """

    fig, ax = single((5., 3.5))

    grid = Grid(grid_name)
    lnu = grid.spectra[spectra][ia, iZ]

    ax.plot(np.log10(grid.lam), np.log10(lnu))

    spec_dict = read_continuum(model, return_dict=True)

    log10Q_new = np.log10(calculate_Q(spec_dict['lam'], spec_dict['incident']))

    scaling = grid.log10Q[ia, iZ] - log10Q_new

    ax.plot(np.log10(spec_dict['lam']), np.log10(
        spec_dict[spectra]) + scaling, alpha=0.5)

    ax.set_xlim([3., 4.5])
    ax.set_ylim([20, 22])
    ax.legend(fontsize=8, labelspacing=0.0)
    ax.set_xlabel(r'$\rm log_{10}(\lambda/\AA)$')
    ax.set_ylabel(r'$\rm log_{10}(L_{\nu}/erg s^{-1} Hz^{-1})$')

    if show:
        plt.show()

    return fig, ax


def compare_linecont(model, show=False):
    """ build a nebular emission grid based on line lists and compare with that produced by linecont"""

    fig, ax = single((5., 3.5))

    # get linecont from continuum

    spec_dict = read_continuum(model, return_dict=True)

    ax.plot(np.log10(spec_dict['lam']), np.log10(
        spec_dict['linecont']), alpha=0.5)

    linecont = make_linecont(model, spec_dict['lam'])

    ax.plot(np.log10(spec_dict['lam']), np.log10(linecont), alpha=0.5)

    ax.set_xlim([3., 4.5])
    ax.set_ylim([15, 19])
    ax.legend(fontsize=8, labelspacing=0.0)
    ax.set_xlabel(r'$\rm log_{10}(\lambda/\AA)$')
    ax.set_ylabel(r'$\rm log_{10}(L_{\nu}/erg s^{-1} Hz^{-1})$')

    if show:
        plt.show()

    return fig, ax


def compare_model_spectra(default_model, model, spectra='total', show=False):

    fig, ax = single((6., 3.5))

    ax.axhline(c='k', lw=3, alpha=0.05)

    default_spec_dict = read_continuum(default_model, return_dict=True)
    lam = default_spec_dict['lam']
    spec_dict = read_continuum(model, return_dict=True)

    ax.plot(np.log10(lam), np.log10(
        spec_dict[spectra]/default_spec_dict[spectra]))

    ax.set_xlim([3., 4.5])
    ax.set_ylim([-0.75, 0.75])
    ax.legend(fontsize=8, labelspacing=0.0)
    ax.set_xlabel(r'$\rm log_{10}(\lambda/\AA)$')
    ax.set_ylabel(r'$\rm log_{10}(L_{\nu}^2/L_{\nu}^1)$')

    if show:
        plt.show()

    return fig, ax


def plot_lines(model, show=False, filter=None):

    ids, wavelengths, intrinsic, emergent = read_lines(model)

    if filter:
        s = emergent > emergent[ids == filter[0]]+filter[1]
        ids = ids[s]
        wavelengths = wavelengths[s]
        intrinsic = intrinsic[s]
        emergent = emergent[s]

    x = np.arange(len(ids))

    fig, ax = single((7., 3.5))

    for x_ in x:
        ax.axvline(x_, c='k', lw=1, alpha=0.05)

    ax.scatter(x, emergent, s=25)
    # ax.scatter(x, intrinsic, s = 25)

    ax.set_xlim([-0.5, x[-1]+0.5])
    # ax.set_ylim([-0.75, 0.75])

    # ax.set_ylabel(r'$\rm log_{10}(L_{\nu}^2/L_{\nu}^1)$')

    ax.set_xticks(x, ids, rotation=90, fontsize=5)

    if show:
        plt.show()

    return fig, ax


def compare_lines(default_model, model, show=False, filter=None):

    ids_, wavelengths_, intrinsic_, emergent_ = read_lines(model)
    ids, wavelengths, intrinsic, emergent = read_lines(default_model)

    if filter:
        s = emergent > emergent[ids == filter[0]]+filter[1]
        ids = ids[s]
        emergent = emergent[s]
        emergent_ = emergent_[s]

    x = np.arange(len(ids))

    fig, ax = single((7., 3.5))

    ax.axhline(0, c='k', lw=1, alpha=0.05)

    ax.scatter(x, emergent_ - emergent, s=25)

    ax.set_xlim([-0.5, x[-1]+0.5])
    ax.set_ylim([-0.75, 2.75])

    ax.set_xticks(x, ids, rotation=90, fontsize=5)

    if show:
        plt.show()

    return fig, ax


if __name__ == '__main__':

    model = 'default'

    # fig, ax = plot_lines(f'data/{model}', show = False, filter = ('HI6563', -1.5))
    # fig.savefig(f'figs/lines_{model}.pdf')
    #
    #
    # model1, model2 = 'default',  'default_alpha-0.6'
    # model1, model2 = 'default',  'default_log10U--1'
    #
    # fig, ax = compare_lines(f'data/{model1}', f'data/{model2}', show = False, filter = ('HI6563', -1.5))
    # fig.savefig(f'figs/line_comparison_{model2}.pdf')

    # --- compare with grid spectra

    # grid_name = 'bpass-v2.2.1-bin_chab-100_cloudy-v17.03_log10Uref-2'
    # compare_with_grid(f'data/{model}', grid_name)

    compare_linecont(f'data/default')
