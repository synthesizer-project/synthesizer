import numpy as np
from synthesizer.plt import single
from synthesizer.filters import TopHatFilterCollection
from synthesizer.grid import Grid, parse_grid_id
from synthesizer.parametric.sfzh import SFH, ZH, generate_sfzh, generate_instant_sfzh
from synthesizer.galaxy.parametric import Galaxy
from unyt import yr, Myr


if __name__ == '__main__':

    # -------------------------------------------------
    # --- calcualte the EW for a given line as a function of age

    grid_dir = '/example/grid_directory/synthesizer_data/grid/'
    grid_dir = None

    models = [
        'bpass-v2.2.1-bin_chab-100_cloudy-v17.03_log10Uref-2',
        'bc03_chabrier03_cloudy-v17.03_log10Uref-2',
        # 'bc03-2016-BaSeL_chabrier03_cloudy-v17.03_log10Uref-2',
        'fsps-v3.2_Chabrier03_cloudy-v17.03_log10Uref-2',
        'maraston-rhb_kroupa_cloudy-v17.03_log10Uref-2'
    ]

    Z = 0.005
    log10Z = np.log10(Z)

    Zh = ZH.deltaConstant({'Z': Z})  # constant metallicity

    fig, ax = single()

    for model, ls in zip(models, ['-', '--', '-.', ':']):

        model_info = parse_grid_id(model)
        print(model_info)

        grid = Grid(model, grid_dir=grid_dir)

        log10durations = np.arange(0., 3.1, 0.1)

        label = rf"$\rm {model_info['sps_model'].upper()}\ {model_info['sps_model_version']} $"

        log10bb = []
        age = []

        for log10duration in log10durations:

            # --- define the functional form of the star formation and metal enrichment histories
            sfh = SFH.Constant({'duration': 10**log10duration * Myr})

            # --- get the 2D star formation and metal enrichment history for the given SPS grid. This is (age, Z).
            sfzh = generate_sfzh(grid.log10ages, grid.metallicities, sfh, Zh)

            age_ = sfzh.calculate_median_age()/1E6
            age.append(age_)
            # --- define galaxy object
            # by default this automatically calculates the pure stellar spectra
            galaxy = Galaxy(sfzh)
            galaxy.get_pacman_spectra(grid)  # adds nebular emission

            # --- get quanitities

            sed = galaxy.spectra['total']
            log10bb_ = sed.get_balmer_break()
            log10bb.append(log10bb_)

        ax.plot(np.log10(age), log10bb, lw=1, color='k', alpha=1, ls=ls, label=label)

    ax.set_xlim([0, 3.])
    ax.set_ylabel(r'$\rm log_{10}(L_{4200}/L_{3500})$')
    ax.legend(fontsize=7, labelspacing=0.1)
    ax.set_xlabel(r'$\rm log_{10}(age/Myr)$')

    fig.savefig('figs/theory_sps.pdf')
