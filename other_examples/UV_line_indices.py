"""
Example for generating the rest-frame spectrum for a parametric galaxy including
photometry. This example will:
- build a parametric galaxy (see make_sfzh)
- calculate spectral luminosity density
"""

import numpy as np
import matplotlib.pyplot as plt

from synthesizer.filters import FilterCollection
from synthesizer.grid import Grid
from synthesizer.parametric import EW
from synthesizer.parametric.sfzh import SFH, ZH, generate_sfzh
from synthesizer.galaxy.parametric import ParametricGalaxy as Galaxy
from synthesizer.plt import single, single_histxy, mlabel
from unyt import yr, Myr
from astropy.cosmology import Planck18 as cosmo
import csv

# TODO
# 1. Remove continuum and only keep UV indices 1501 and 1719.
# 3. Isolate UV indices 1501 [1496 - 1506], 1719 [1705 - 1729].
# 2. Observe changing EW with changes in metallicity Z.

grid_dir = 'C:/Users/conno/Documents/PhD ISSA/synthesizer_data/grids'  # Change this directory to your own.


def main():

    # grid_lib = ['bpass-2.3-bin_chabrier03-0,300_alpha0.2_cloudy-v17.03-resolution0.1',
    #             'bpass-2.3-bin_chabrier03-0,300_alpha0.2',
    #             'bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-v17.03-resolution0.1',
    #             'bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-alpha0.2',
    #             ]  # Change this to the appropriate .hdf5

    grid_lib = ['bpass-2.3-bin_chabrier03-0,300_alpha0.2_cloudy-v17.03-resolution0.1']
    # Change this to the appropriate .hdf5

    indices, uv_index = import_indices()

    # star_formation_history(grid_lib, indices, uv_index)

    # intrinsic_stellar(grid_lib, indices, uv_index)

    # pseudo_continuum(grid_lib, indices, uv_index)

    # alpha_enhancement(grid_lib, indices, uv_index)

    initial_mass_function(grid_lib, indices, uv_index)

    # resolution(grid_lib, indices, uv_index)


def get_ew_sfh(indices, sfh_arr, Z, grid, EqW):
    age = sfh_arr
    sfh_p = {'duration': age * Myr}

    Z_p = {'Z': Z}  # can also use linear metallicity e.g. {'Z': 0.01}
    stellar_mass = 1E8

    # --- define the functional form of the star formation and metal enrichment histories
    sfh = SFH.Constant(sfh_p)  # constant star formation
    print(sfh)  # print sfh summary

    Zh = ZH.deltaConstant(Z_p)  # constant metallicity

    # --- get 2D star formation and metal enrichment history for the given SPS grid. This is (age, Z).
    sfzh = generate_sfzh(grid.log10ages, grid.metallicities, sfh, Zh, stellar_mass=stellar_mass)

    # --- create a galaxy object
    galaxy = Galaxy(sfzh)

    # --- generate equivalent widths
    # print(grid_lib.index(grids))
    # if count == 1:
    galaxy.get_stellar_spectra(grid)
    # else:
    #     galaxy.get_intrinsic_spectra(grid, fesc=0.5)

    EqW.append(galaxy.get_equivalent_width(indices))
    return EqW


def get_ew_stellar(indices, Z, imf, grid, EqW, mode):
    sfh_p = {'duration': 100 * Myr}

    Z_p = {'Z': Z}  # can also use linear metallicity e.g. {'Z': 0.01}
    stellar_mass = imf

    # --- define the functional form of the star formation and metal enrichment histories
    sfh = SFH.Constant(sfh_p)  # constant star formation
    print(sfh)  # print sfh summary

    Zh = ZH.deltaConstant(Z_p)  # constant metallicity

    # --- get 2D star formation and metal enrichment history for the given SPS grid. This is (age, Z).
    sfzh = generate_sfzh(grid.log10ages, grid.metallicities, sfh, Zh, stellar_mass=stellar_mass)

    # --- create a galaxy object
    galaxy = Galaxy(sfzh)

    # --- generate equivalent widths
    if mode == 0:
        galaxy.get_stellar_spectra(grid)
    else:
        galaxy.get_intrinsic_spectra(grid, fesc=0.5)

    EqW.append(galaxy.get_equivalent_width(indices))
    return EqW


def get_sed(Z, grid):
    sfh_p = {'duration': 100 * Myr}

    Z_p = {'Z': Z}  # can also use linear metallicity e.g. {'Z': 0.01}
    stellar_mass = 1E8

    # --- define the functional form of the star formation and metal enrichment histories
    sfh = SFH.Constant(sfh_p)  # constant star formation
    print(sfh)  # print sfh summary

    Zh = ZH.deltaConstant(Z_p)  # constant metallicity

    # --- get 2D star formation and metal enrichment history for the given SPS grid. This is (age, Z).
    sfzh = generate_sfzh(grid.log10ages, grid.metallicities, sfh, Zh, stellar_mass=stellar_mass)

    # --- create a galaxy object
    galaxy = Galaxy(sfzh)

    return galaxy.get_stellar_spectra(grid)


def star_formation_history(grid_lib, indices, uv_index):
    for grids in grid_lib:
        grid = Grid(grids, grid_dir=grid_dir)
        Z_arr = grid.metallicities
        sfh_arr = [100, 10, 1000]

        for i in range(0, len(indices)):
            EqW = []
            for j in range(0, len(sfh_arr)):

                # --- define the parameters of the star formation and metal enrichment histories
                for k in range(0, len(Z_arr)):
                    EqW = get_ew_sfh(indices[i], sfh_arr[j], Z_arr[k], grid, EqW)

                Byrne_values = import_data_byrne(i)

                if j == 0:
                    plt.rcParams['figure.dpi'] = 200
                    plt.subplot(3, 3, i + 1)
                    plt.grid(True)

                    if i == 0 or i == 3 or i == 6:
                        plt.ylabel('EW (\u212B)', fontsize=8)
                    if i > 5:
                        plt.xlabel('Z', fontsize=8)

                    if uv_index[i] == 1501 or uv_index[i] == 1719:
                        label = 'UV_' + str(uv_index[i])
                    else:
                        label = 'F' + str(uv_index[i])

                    _, y_max = plt.ylim()
                    # plt.text(0.00025, y_max, label, va='top', fontsize=8)
                    plt.title(label, fontsize=8, transform=plt.gca().transAxes, y=0.8)

                    # plt.xlim([0.0001, 0.04])

                    if grid_lib.index(grids) == 1:
                        plt.scatter(grid.metallicities, EqW, label=label, s=10, color='white', edgecolors='grey',
                                    alpha=1.0, zorder=10, linewidth=0.5)
                        plt.semilogx(grid.metallicities, EqW, linewidth=0.75, color='grey')
                    else:
                        plt.scatter(grid.metallicities, EqW, label=label, s=10, color='blue')

                elif j == 1:
                    plt.semilogx(grid.metallicities, EqW, linewidth=0.75, color='red')
                elif j == 2:
                    plt.semilogx(grid.metallicities, EqW, linewidth=0.75, color='orange')

                # if count == len(grid_lib):
                # plt.scatter(grid.metallicities, Byrne_values, color='purple', label='Byrne +2022')
                # plt.semilogx(grid.metallicities, Byrne_values, color='grey', linewidth=0.5)

                plt.tight_layout()

                EqW.clear()
                Byrne_values.clear()

                if i == len(indices) - 1 and j == len(sfh_arr) - 1:
                    plt.show()


def intrinsic_stellar(grid_lib, indices, uv_index):
    for grids in grid_lib:
        grid = Grid(grids, grid_dir=grid_dir)

        Z = grid.metallicities
        imf = 1E8

        modes = ['Stellar', 'Intrinsic']
        for i in range(0, len(indices)):
            EqW = []
            for j in range(0, len(modes)):
                # --- define the parameters of the star formation and metal enrichment histories
                for k in range(0, len(Z)):
                    EqW = get_ew_stellar(indices[i], Z[k], imf, grid, EqW, j)

                if j == 0:
                    plt.rcParams['figure.dpi'] = 200
                    plt.subplot(3, 3, i + 1)
                    plt.grid(True)

                    if i == 0 or i == 3 or i == 6:
                        plt.ylabel('EW (\u212B)', fontsize=8)
                    if i > 5:
                        plt.xlabel('Z', fontsize=8)

                    if uv_index[i] == 1501 or uv_index[i] == 1719:
                        label = 'UV_' + str(uv_index[i])
                    else:
                        label = 'F' + str(uv_index[i])

                    _, y_max = plt.ylim()
                    # plt.text(0.00025, y_max, label, va='top', fontsize=8)

                    if uv_index[i] == 1853:
                        plt.title(label, fontsize=8, transform=plt.gca().transAxes, y=0.8, x=0.25)
                    else:
                        plt.title(label, fontsize=8, transform=plt.gca().transAxes, y=0.8)

                    plt.scatter(grid.metallicities, EqW, color='white', edgecolors='grey', alpha=1.0,
                                zorder=10, linewidth=0.5, s=10)
                    plt.semilogx(grid.metallicities, EqW, linewidth=0.75, color='grey')
                else:
                    plt.semilogx(grid.metallicities, EqW, color='blue', linewidth=0.75)
                EqW.clear()

                plt.tight_layout()

                if i == len(indices) - 1 and j == len(modes) - 1:
                    plt.show()


def pseudo_continuum(grid_lib, indices, uv_index):
    for grids in grid_lib:
        grid = Grid(grids, grid_dir=grid_dir)

        for i in range(0, len(indices)):
            Z = 0.014

            sed = get_sed(Z, grid)

            blue = []
            index = []
            red = []

            index_blue = []
            index_range = []
            index_red = []

            flux_blue = []
            flux = []
            flux_red = []

            lam_arr = np.log10(sed.lam)
            lnu_arr = np.log10(sed.lnu)

            continuum = []
            index_continuum = []
            continuum_flux = []

            plt.rcParams['figure.dpi'] = 200
            plt.subplot(3, 3, i + 1)
            plt.grid(True)

            if i == 0 or i == 3 or i == 6:
                plt.ylabel(r'$\rm log_{10}(L_{\nu}/erg\ s^{-1}\ Hz^{-1} M_{\odot}^{-1})$', fontsize=7)
            if i > 5:
                plt.xlabel(r'$\rm log_{10}(\lambda/\AA)$', fontsize=8)

            if uv_index[i] == 1501 or uv_index[i] == 1719:
                color = 'UV_' + str(uv_index[i])
            else:
                color = 'F' + str(uv_index[i])

            _, y_max = plt.ylim()
            plt.title(color, fontsize=8, transform=plt.gca().transAxes, y=0.9)

            for j in range(0, len(lam_arr)):
                plt.xlim([indices[i][2], indices[i][5]])

                if np.log10(indices[i][2]) <= lam_arr[j] <= np.log10(indices[i][5]):
                    continuum.append(lam_arr[j])
                    index_continuum.append(np.where(lam_arr == lam_arr[j]))

                if np.log10(indices[i][2]) <= lam_arr[j] <= np.log10(indices[i][3]):
                    blue.append(lam_arr[j])
                    index_blue.append(np.where(lam_arr == lam_arr[j]))

                if np.log10(indices[i][0]) <= lam_arr[j] <= np.log10(indices[i][1]):
                    index.append(lam_arr[j])
                    index_range.append(np.where(lam_arr == lam_arr[j]))

                if np.log10(indices[i][4]) <= lam_arr[j] <= np.log10(indices[i][5]):
                    red.append(lam_arr[j])
                    index_red.append(np.where(lam_arr == lam_arr[j]))

            for k in range(0, len(index_blue)):
                flux_blue.append(lnu_arr[index_blue[k]])

            for m in range(0, len(index_range)):
                flux.append(lnu_arr[index_range[m]])

            for n in range(0, len(index_red)):
                flux_red.append(lnu_arr[index_red[n]])

            for x in range(0, len(index_continuum)):
                continuum_flux.append(lnu_arr[index_continuum[x]])

            for y in range(0, len(blue)):
                blue[y] = 10 ** blue[y]

            for z in range(0, len(index)):
                index[z] = 10 ** index[z]

            for a in range(0, len(red)):
                red[a] = 10 ** red[a]

            for b in range(0, len(continuum)):
                continuum[b] = 10 ** continuum[b]

            plt.plot(continuum, continuum_flux, color='grey', lw=1, alpha=0.5)
            plt.plot(blue, flux_blue, color='blue', lw=1, alpha=0.8)
            plt.plot(index, flux, color='black', lw=1, alpha=0.8)
            plt.plot(red, flux_red, color='red', lw=1, alpha=0.8)

            plt.tight_layout()

            if i == len(indices) - 1:
                plt.show()


def alpha_enhancement(grid_lib, indices, uv_index):
    for i in range(0, len(indices)):
        for grids in grid_lib:
            grid = Grid(grids, grid_dir=grid_dir)
            Z = grid.metallicities
            imf = 1E8

            EqW = []

            # --- define the parameters of the star formation and metal enrichment histories
            for k in range(0, len(Z)):
                EqW = get_ew_stellar(indices[i], Z[k], imf, grid, EqW, 0)

            if grid_lib.index(grids) == 0:
                plt.rcParams['figure.dpi'] = 200
                plt.subplot(3, 3, i + 1)
                plt.grid(True)

                if i == 0 or i == 3 or i == 6:
                    plt.ylabel('EW (\u212B)', fontsize=8)
                if i > 5:
                    plt.xlabel('Z', fontsize=8)

                if uv_index[i] == 1501 or uv_index[i] == 1719:
                    label = 'UV_' + str(uv_index[i])
                else:
                    label = 'F' + str(uv_index[i])

                _, y_max = plt.ylim()
                # plt.text(0.00025, y_max, label, va='top', fontsize=8)

                # if uv_index[i] == 1853:
                #     # plt.title(label, fontsize=8, transform=plt.gca().transAxes, y=0.8, x=0.25)
                # else:
                plt.title(label, fontsize=8, transform=plt.gca().transAxes, y=0.8)

                plt.scatter(grid.metallicities, EqW, color='white', edgecolors='grey', alpha=1.0,
                            zorder=10, linewidth=0.5, s=10)
                plt.semilogx(grid.metallicities, EqW, linewidth=0.75, color='grey')
            else:
                plt.scatter(grid.metallicities, EqW, color='blue', linewidth=0.75, s=2)
            EqW.clear()

            plt.tight_layout()

            if i == len(indices) - 1 and grid_lib.index(grids) == len(grid_lib)-1:
                plt.show()


def initial_mass_function(grid_lib, indices, uv_index):
    for grids in grid_lib:
        grid = Grid(grids, grid_dir=grid_dir)

        Z = grid.metallicities
        imf = [1E8, 1E9, 1E10, 1E11, 1E12]

        for i in range(0, len(indices)):
            EqW = []
            for j in range(0, len(imf)):
                # --- define the parameters of the star formation and metal enrichment histories
                for k in range(0, len(Z)):
                    EqW = get_ew_stellar(indices[i], Z[k], imf[j], grid, EqW, 0)

                if j == 0:
                    plt.rcParams['figure.dpi'] = 200
                    plt.subplot(3, 3, i + 1)
                    plt.grid(True)

                    if i == 0 or i == 3 or i == 6:
                        plt.ylabel('EW (\u212B)', fontsize=8)
                    if i > 5:
                        plt.xlabel('Z', fontsize=8)

                    if uv_index[i] == 1501 or uv_index[i] == 1719:
                        label = 'UV_' + str(uv_index[i])
                    else:
                        label = 'F' + str(uv_index[i])

                    _, y_max = plt.ylim()
                    # plt.text(0.00025, y_max, label, va='top', fontsize=8)

                    if uv_index[i] == 1853:
                        plt.title(label, fontsize=8, transform=plt.gca().transAxes, y=0.8, x=0.25)
                    else:
                        plt.title(label, fontsize=8, transform=plt.gca().transAxes, y=0.8)

                    plt.scatter(grid.metallicities, EqW, color='white', edgecolors='grey', alpha=1.0,
                                zorder=10, linewidth=0.5, s=10)
                    plt.semilogx(grid.metallicities, EqW, linewidth=0.75, color='grey')
                else:
                    plt.semilogx(grid.metallicities, EqW, color='blue', linewidth=0.75)
                EqW.clear()

                plt.tight_layout()

                if i == len(indices) - 1 and j == len(imf) - 1:
                    plt.show()


def resolution(grid_lib, indices, uv_index):
    for i in range(0, len(indices)):
        color = ['yellow',  # 'bpass-2.3-bin_chabrier03-0,300_alpha0.2_cloudy-v17.03-resolution0.1',
                 'orange',  # 'bpass-2.3-bin_chabrier03-0,300_alpha0.2',
                 'red',     # 'bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-v17.03-resolution0.1',
                 'darkred']  # 'bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-alpha0.2',
        for grids in grid_lib:
            grid = Grid(grids, grid_dir=grid_dir)
            Z = grid.metallicities
            imf = 1E8

            EqW = []

            # --- define the parameters of the star formation and metal enrichment histories
            for k in range(0, len(Z)):
                EqW = get_ew_stellar(indices[i], Z[k], imf, grid, EqW, 0)

            if grid_lib.index(grids) == 0:
                plt.rcParams['figure.dpi'] = 200
                plt.subplot(3, 3, i + 1)
                plt.grid(True)

                if i == 0 or i == 3 or i == 6:
                    plt.ylabel('EW (\u212B)', fontsize=8)
                if i > 5:
                    plt.xlabel('Z', fontsize=8)

                if uv_index[i] == 1501 or uv_index[i] == 1719:
                    label = 'UV_' + str(uv_index[i])
                else:
                    label = 'F' + str(uv_index[i])

                _, y_max = plt.ylim()
                # plt.text(0.00025, y_max, label, va='top', fontsize=8)

                # if uv_index[i] == 1853:
                #     # plt.title(label, fontsize=8, transform=plt.gca().transAxes, y=0.8, x=0.25)
                # else:
                plt.title(label, fontsize=8, transform=plt.gca().transAxes, y=0.8)

                plt.scatter(grid.metallicities, EqW, color='white', edgecolors='grey', alpha=1.0,
                            zorder=10, linewidth=0.5, s=10)
                plt.semilogx(grid.metallicities, EqW, linewidth=0.75, color='grey')
            else:
                plt.semilogx(grid.metallicities, EqW, color=color[grid_lib.index(grids)], linewidth=0.75)
            EqW.clear()

            plt.tight_layout()

            if i == len(indices) - 1 and grid_lib.index(grids) == len(grid_lib)-1:
                plt.show()


def import_data_byrne(index):
    Byrne_grid = []
    temp = []

    with open('C:/Users/conno/Documents/PhD ISSA/PhD data notes/UV_indices_ContinuousStarFormation.csv', newline='') \
            as csvfile:
        csvreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        next(csvreader)
        for row in csvreader:
            Byrne_grid.append(row)
    for i in range(0, len(Byrne_grid)):
        if Byrne_grid[i][1] != '+0.200000':
            pass
        else:
            temp.append(Byrne_grid[i])

    Byrne_grid = temp

    for j in range(0, len(Byrne_grid)-1):
        for k in range(0, len(Byrne_grid[j])):
            Byrne_grid[j][k] = Byrne_grid[j][k].replace('+', '')

    float_list = [[float(element) for element in row] for row in Byrne_grid]
    Byrne_grid = float_list

    Byrne_values = [sublist[index+2] for sublist in Byrne_grid]
    return Byrne_values


def import_indices():
    import_grid = []
    temp = []
    uv_index = []

    with open('C:/Users/conno/Documents/PhD ISSA/PhD data notes/UV_Indices_Sythesizer.csv', newline='') \
            as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(csvreader)
        for row in csvreader:
            import_grid.append(row)

    for i in range(0, len(import_grid)):
        temp.append(import_grid[i][1:])
        uv_index.append(import_grid[i][0])

    temp = [[int(element) for element in row] for row in temp]
    import_grid = temp

    int_index = [int(item) for item in uv_index]
    uv_index = int_index

    return import_grid, uv_index

# def import_data():
#     import_grid = []
#     temp = []
#
#     with open('C:/Users/conno/Documents/PhD ISSA/PhD data notes/UV_Indices_Alpha_sensitivity.csv', newline='') \
#             as csvfile:
#         csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
#         next(csvreader)
#         for row in csvreader:
#             import_grid.append(row)
#     for i in range(0, len(import_grid)):
#         if '0.2' in import_grid[i] and '1370' in import_grid[i]:
#             temp.append(import_grid[i][2:])
#         else:
#             pass
#
#     import_grid = temp
#
#     for j in range(0, len(import_grid)-1):
#         for k in range(0, len(import_grid[j])):
#             import_grid[j][k] = import_grid[j][k].replace('+', '')
#
#     float_list = [[float(element) for element in row] for row in import_grid]
#     values = float_list
#
#     return values


if __name__ == '__main__':
    main()
