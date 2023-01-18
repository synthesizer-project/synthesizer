

# --- general
import h5py
import copy
import numpy as np
from scipy import integrate
from unyt import yr, erg, Hz, s, cm, angstrom

from .galaxy import BaseGalaxy
from .. import exceptions
from ..dust import power_law
from ..sed import Sed, convert_fnu_to_flam
from ..line import Line
from ..plt import single_histxy, mlabel
from ..stats import weighted_median, weighted_mean


class ParametricGalaxy(BaseGalaxy):

    """A class defining parametric galaxy objects

    """

    def __init__(self, sfzh):
        """__init__ method for ParametricGalaxy

        Parameters
        ----------
        sfzh : obj
            instance of the BinnedSFZH class containing the star formation and metal enrichment history.
        """

        self.sfzh = sfzh
        # add an extra dimension to the sfzh to allow the fast summation
        self.sfzh_ = np.expand_dims(self.sfzh.sfzh, axis=2)
        self.spectra = {}  # dictionary holding spectra
        self.lines = {}  # dictionary holding lines
        self.images = {}  # dictionary holding images

    def __str__(self):
        """Function to print a basic summary of the Galaxy object.

        Returns a string containing the total mass formed and lists of the available SEDs, lines, and images.

        Returns
        -------
        str
            Summary string containing the total mass formed and lists of the available SEDs, lines, and images.
        """

        pstr = ''
        pstr += '-'*10 + "\n"
        pstr += 'SUMMARY OF PARAMETRIC GALAXY' + "\n"
        pstr += str(self.__class__) + "\n"
        pstr += f'log10(stellar mass formed/Msol): {np.log10(np.sum(self.sfzh.sfzh))}' + "\n"
        pstr += f'available SEDs: {list(self.spectra.keys())}' + "\n"
        pstr += f'available lines: {list(self.lines.keys())}' + "\n"
        pstr += f'available images: {list(self.images.keys())}' + "\n"
        pstr += '-'*10 + "\n"
        return pstr

    def __add__(self, second_galaxy):
        """Allows two Galaxy objects to be added together.

        Parameters
        ----------
        second_galaxy : ParametricGalaxy
            A second ParametricGalaxy to be added to this one.

        NOTE: functionality for adding lines and images not yet implemented.

        Returns
        -------
        ParametricGalaxy
            New ParametricGalaxy object containing summed SFZHs, SEDs, lines, and images.
        """

        new_sfzh = self.sfzh + second_galaxy.sfzh
        new_galaxy = ParametricGalaxy(new_sfzh)

        # add together spectra
        for spec_name, spectra in self.spectra.items():
            if spec_name in second_galaxy.spectra.keys():
                new_galaxy.spectra[spec_name] = spectra + second_galaxy.spectra[spec_name]
            else:
                exceptions.InconsistentAddition(
                    'Both galaxies must contain the same spectra to be added together')

        # add together lines
        # for line_name, line in self.spectra.items():
        #     if spec_name in second_galaxy.spectra.keys():
        #         new_galaxy.lines[line_name] = line + second_galaxy.lines[line_name]
        #     else:
        #         exceptions.InconsistentAddition('Both galaxies must contain the same emission line quantities to be added together')

        # add together images
        # for img_name, image in self.images.items():
        #     if img_name in second_galaxy.images.keys():
        #         new_galaxy.images[img_name] = image + second_galaxy.image[img_name]
        #     else:
        #         exceptions.InconsistentAddition('Both galaxies must contain the same images to be added together')

        return new_galaxy

    def get_Q(self, grid):
        """ return the ionising photon luminosity (log10Q) for a given SFZH. """

        return np.sum(10**self.grid.log10Q * self.sfzh, axis=(0, 1))

    def generate_lnu(self, grid, spectra_name):

        # calculate pure stellar emission
        return np.sum(grid.spectra[spectra_name] * self.sfzh_, axis=(0, 1))

    def get_stellar_spectra(self, grid, update=True):
        """ generate the pure stellar spectra using the provided grid"""

        lnu = self.generate_lnu(grid, 'stellar')

        sed = Sed(grid.lam, lnu)

        if update:
            self.spectra['stellar'] = sed

        return sed

    def get_nebular_spectra(self, grid, fesc=0.0, update=True):

        lnu = self.generate_lnu(grid, 'nebular')

        lnu *= (1-fesc)

        sed = Sed(grid.lam, lnu)

        if update:
            self.spectra['nebular'] = sed

        return sed

    def get_intrinsic_spectra(self, grid, fesc=0.0, update=True):
        """ this generates the intrinsic spectra, i.e. not including dust but including nebular emission. It also generates the stellar and nebular spectra too. """

        stellar = self.get_stellar_spectra(grid, update=update)
        nebular = self.get_nebular_spectra(grid, fesc, update=update)

        sed = Sed(grid.lam, stellar.lnu + nebular.lnu)

        if update:
            self.spectra['intrinsic'] = sed

        return sed

    def get_screen_spectra(self, grid, tauV=None, fesc=0.0, dust_curve=power_law({'slope': -1.}), update=True):
        """
        Similar to get_intrinsic_spectra but applies a dust screen
        """

        # --- begin by calculating intrinsic spectra
        intrinsic = self.get_intrinsic_spectra(grid, fesc, update=update)

        if tauV:
            T = np.exp(-tauV) * dust_curve.T(grid.lam)
        else:
            T = 1.0

        sed = Sed(grid.lam, T * intrinsic.lnu)

        if update:
            self.spectra['attenuated'] = sed

        return sed

    def get_pacman_spectra(self, grid, fesc=0.0, fesc_LyA=1.0, tauV=None, dust_curve=power_law({'slope': -1.}), update=True):
        """ in the PACMAN model some fraction (fesc) of the pure stellar emission is assumed to completely escape the galaxy without reprocessing by gas or dust. The rest is assumed to be reprocessed by both gas and a screen of dust. """

        # --- begin by generating the pure stellar spectra
        stellar = self.get_stellar_spectra(grid, update=update)

        # --- this is the starlight that escapes any reprocessing
        self.spectra['escape'] = Sed(grid.lam, fesc * stellar.lnu)

        # --- this is the starlight after reprocessing by gas
        self.spectra['reprocessed'] = Sed(grid.lam)
        self.spectra['intrinsic'] = Sed(grid.lam)
        self.spectra['attenuated'] = Sed(grid.lam)
        self.spectra['total'] = Sed(grid.lam)

        if fesc_LyA < 1.0:
            # if Lyman-alpha escape fraction is specified reduce LyA luminosity

            # --- generate contribution of line emission alone and reduce the contribution of Lyman-alpha
            linecont = np.sum(grid.spectra['linecont'] * self.sfzh_, axis=(0, 1))
            idx = grid.get_nearest_index(1216., grid.lam)  # get index of Lyman-alpha
            linecont[idx] *= fesc_LyA  # reduce the contribution of Lyman-alpha

            nebular_continuum = np.sum(
                grid.spectra['nebular_continuum'] * self.sfzh_, axis=(0, 1))
            transmitted = np.sum(grid.spectra['transmitted'] * self.sfzh_, axis=(0, 1))
            self.spectra['reprocessed'].lnu = (
                1.-fesc) * (linecont + nebular_continuum + transmitted)

        else:
            self.spectra['reprocessed'].lnu = (
                1.-fesc) * np.sum(grid.spectra['total'] * self.sfzh_, axis=(0, 1))

        self.spectra['intrinsic'].lnu = self.spectra['escape'].lnu + \
            self.spectra['reprocessed'].lnu  # the light before reprocessing by dust

        if tauV:
            T = np.exp(-tauV) * dust_curve.T(grid.lam)
            self.spectra['attenuated'].lnu = self.spectra['escape'].lnu + \
                T*self.spectra['reprocessed'].lnu
            self.spectra['total'].lnu = self.spectra['attenuated'].lnu
        else:
            self.spectra['total'].lnu = self.spectra['escape'].lnu + self.spectra['reprocessed'].lnu

        return self.spectra['total']

    def get_CF00_spectra(tauV, p={}):
        """ add Charlot \& Fall (2000) dust """

        print('WARNING: not yet implemented')

    def get_intrinsic_line(self, grid, line_id, quantity=False, update=True):
        """ return intrinsic quantities (luminosity, EW) for a single line or line set """

        if type(line_id) is str:
            line_id = [line_id]

        luminosity_ = []
        continuum_ = []
        wavelength_ = []

        for line_id_ in line_id:
            grid_line = grid.lines[line_id_]

            wavelength_.append(grid_line['wavelength'])  # \AA
            #  continuum at line wavelength, erg/s/Hz
            continuum_.append(np.sum(grid_line['continuum'] * self.sfzh, axis=(0, 1)))
            luminosity_.append(np.sum(grid_line['luminosity'] * self.sfzh, axis=(0, 1)))

        # --- create line object

        line = Line(line_id, wavelength_, luminosity_, continuum_)

        if update:
            self.lines[line.id] = line

        return line

    # def get_intrinsic_line(self, tauV):
    #
    #     return
    #
    # def apply_dust_pacman_lines(self):
    #
    #     return
    #
    # def apply_dust_CF00_lines(self):
    #
    #     return
