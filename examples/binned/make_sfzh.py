

import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
from unyt import yr, Myr

from synthesizer.binned.sfzh import SFH, ZH, generate_sfzh
from synthesizer.plt import single, single_histxy, mlabel


def plot_sfh(x, sfh, log10=False):

    x_range = [x[0], x[-1]]

    sfh_ = sfzh.Binned.sfh(x, sfh, log10=log10)

    print(sfh_)

    fig, ax = single()

    ax.fill_between(x, sfh_/np.max(sfh_), step='mid', color='k', alpha=0.3)

    # --- add SFR to top of the plot
    x = np.linspace(*x_range, 1000)
    if log10:
        y = sfh.sfr(10**x)
    else:
        y = sfh.sfr(x)

    ax.plot(x, y/np.max(y))

    if log10:
        ax.set_xlabel(mlabel('log_{10}(age/yr)'))
    else:
        ax.set_xlabel(mlabel('age/yr'))

    ax.set_ylabel(mlabel('normalised\ SFR'))

    ax.set_xlim(x_range)

    plt.show()


def plot_sfhs():

    ages = np.arange(0, 2000, 1)

    # sfh_p = [1E8] # [duration/yr]
    # sfh = sfzh.SFH.Constant(*sfh_p) # constant star formation
    # plot_sfh(ages, sfh, log10 = False)
    #
    # sfh_p = [1E8, 2E8] # [tau/yr, mag_age/yr]
    # sfh = sfzh.SFH.TruncatedExponential(*sfh_p) # constant star formation
    # plot_sfh(ages, sfh, log10 = False)

    sfh_p = [700., 0.2, 1000]  # [age_peak/yr, tau, max_age/yr]
    sfh = sfzh.SFH.LogNormal(*sfh_p)  # constant star formation
    plot_sfh(ages, sfh, log10=False)

    print(sfh.sfr(10))


def plot_sfzhs():

    # --- define a age and metallicity grid. In practice these are pulled from the SPS model.
    log10ages = np.arange(6., 10.5, 0.1)
    log10metallicities = np.arange(-5., -1.5, 0.25)
    metallicities = 10**log10metallicities

    # --- define the parameters of the star formation and metal enrichment histories

    Z_p = {'log10Z': -2.5}  # can also use linear metallicity e.g. {'Z': 0.01}
    Zh = ZH.deltaConstant(Z_p)

    sfh_p = {'duration': 100 * Myr}
    sfh = SFH.Constant(sfh_p)  # constant star formation
    sfh.summary()  # print summary of the star formation history
    sfzh = generate_sfzh(log10ages, metallicities, sfh, Zh)
    sfzh.summary()
    sfzh.plot()

    print(sfzh.sfh_f.name, sfzh.sfh_f.parameters)

    # sfzh = generate_instant_sfzh(log10ages, metallicities, sfh, Zh)
    # sfzh.summary()
    # sfzh.plot()

    # sfh_p = {'tau': 100 * Myr, 'max_age': 200 * Myr}
    # sfh = SFH.TruncatedExponential(sfh_p) # constant star formation
    # sfh.summary() # print summary of the star formation history
    # sfzh = generate_sfzh(log10ages, metallicities, sfh, Zh)
    # sfzh.summary()
    # sfzh.plot()
    #
    # sfh_p = {'peak_age': 100 * Myr, 'tau': 1, 'max_age': 200 * Myr}
    # sfh = SFH.LogNormal(sfh_p) # constant star formation
    # sfh.summary() # print summary of the star formation history
    # sfzh = generate_sfzh(log10ages, metallicities, sfh, Zh)
    # sfzh.summary()
    # sfzh.plot()


if __name__ == '__main__':

    # plot_sfhs()
    plot_sfzhs()
