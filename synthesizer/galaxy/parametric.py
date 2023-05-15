

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
from ..imaging.images import ParametricImage
from ..art import Art


class ParametricGalaxy(BaseGalaxy):

    """A class defining parametric galaxy objects

    """

    def __init__(self, sfzh, morph=None):
        """__init__ method for ParametricGalaxy

        Parameters
        ----------
        sfzh : obj
            instance of the BinnedSFZH class containing the star formation and metal enrichment history.
        morph : obj
        """

        self.sfzh = sfzh
        # add an extra dimension to the sfzh to allow the fast summation
        # **** TODO: Get rid of this expression or
        # use this throughout? 
        self.sfzh_ = np.expand_dims(self.sfzh.sfzh, axis=2)

        self.morph = morph
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
        pstr += Art.galaxy + "\n"
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
        for line_type in self.lines.keys():
            new_galaxy.lines[line_type] = {}

            if line_type not in second_galaxy.lines.keys():
                exceptions.InconsistentAddition(
                    'Both galaxies must contain the same sets of line types (e.g. intrinsic / attenuated)')
            else:
                for line_name, line in self.lines[line_type].items():
                    if line_name in second_galaxy.spectra[line_type].keys():
                        new_galaxy.lines[line_type][line_name] = line + \
                            second_galaxy.lines[line_type][line_name]
                    else:
                        exceptions.InconsistentAddition(
                            'Both galaxies must contain the same emission lines to be added together')

        # add together images
        for img_name, image in self.images.items():
            if img_name in second_galaxy.images.keys():
                new_galaxy.images[img_name] = image + second_galaxy.images[img_name]
            else:
                exceptions.InconsistentAddition(
                    'Both galaxies must contain the same images to be added together')

        return new_galaxy

    def get_Q(self, grid):
        """ return the ionising photon luminosity (log10Q) for a given SFZH. """

        return np.sum(10**grid.log10Q * self.sfzh, axis=(0, 1))

    def generate_lnu(self, grid, spectra_name, old=False, young=False):

        # calculate pure stellar emission
        if old * young:
            raise ValueError("Cannot provide old and young stars together")

        if old:
            sfzh_mask = (self.sfzh.log10ages>old)
        elif young:
            sfzh_mask = (self.sfzh.log10ages<=young)
        else:
            sfzh_mask = np.ones(len(self.sfzh.log10ages), dtype=bool)

        return np.sum(grid.spectra[spectra_name] * self.sfzh_[sfzh_mask, :, :], axis=(0, 1))

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

        sed = Sed(grid.lam, stellar._lnu + nebular._lnu)

        if update:
            self.spectra['intrinsic'] = sed

        return sed

    def get_screen_spectra(self, grid, tauV=None, dust_curve=power_law({'slope': -1.}), update=True):
        """
        Calculates dust attenuated spectra assuming a simple screen

        Parameters
        ----------
        grid : obj (Grid)
            The spectral frid
        tauV : float
            numerical value of dust attenuation
        dust_curve : obj
            instance of dust_curve

        Returns
        -------
        obj (Sed)
             A Sed object containing the dust attenuated spectra
        """

        # --- begin by calculating intrinsic spectra
        intrinsic = self.get_intrinsic_spectra(grid, update=update)

        if tauV:
            T = dust_curve.attenuate(tauV, grid.lam)
        else:
            T = 1.0

        sed = Sed(grid.lam, T * intrinsic._lnu)

        if update:
            self.spectra['attenuated'] = sed

        return sed

    def get_pacman_spectra(self, grid, fesc=0.0, fesc_LyA=1.0, tauV=None, dust_curve=power_law({'slope': -1.}), update=True):
        """
        Calculates dust attenuated spectra assuming the PACMAN dust/fesc model including variable Lyman-alpha transmission.
        In this model some fraction of the stellar emission is able to complete escape with no dust attenuation or nebular reprocessing.

        Parameters
        ----------
        grid : obj (Grid)
            The spectral frid
        fesc : float
            Lyman continuum escape fraction
        fesc_LyA : float
            Lyman-alpha escape fraction
        tauV : float
            numerical value of dust attenuation
        dust_curve : obj
            instance of dust_curve

        Returns
        -------
        obj (Sed)
             A Sed object containing the dust attenuated spectra
        """

        """ in the PACMAN model some fraction (fesc) of the pure stellar emission is assumed to completely escape the galaxy without reprocessing by gas or dust. The rest is assumed to be reprocessed by both gas and a screen of dust. """

        # --- begin by generating the pure stellar spectra
        stellar = self.get_stellar_spectra(grid, update=update)

        # --- this is the starlight that escapes any reprocessing
        self.spectra['escape'] = Sed(grid.lam, fesc * stellar._lnu)

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
            self.spectra['reprocessed']._lnu = (
                1.-fesc) * (linecont + nebular_continuum + transmitted)

        else:
            self.spectra['reprocessed']._lnu = (
                1.-fesc) * np.sum(grid.spectra['total'] * self.sfzh_, axis=(0, 1))

        self.spectra['intrinsic']._lnu = self.spectra['escape']._lnu + \
            self.spectra['reprocessed']._lnu  # the light before reprocessing by dust

        if tauV:
            T = dust_curve.attenuate(tauV, grid.lam)  # calculate dust attenuation
            self.spectra['attenuated']._lnu = self.spectra['escape']._lnu + \
                T*self.spectra['reprocessed']._lnu
            self.spectra['total']._lnu = self.spectra['attenuated']._lnu
        else:
            self.spectra['total']._lnu = self.spectra['escape']._lnu + \
                self.spectra['reprocessed']._lnu

        return self.spectra['total']

    def get_CF00_spectra(self, grid, tauV_ISM, tauV_BC, alpha_ISM=-0.7, alpha_BC=-1.3, old=False, young=False, save_young_and_old=False):
        """
        Calculates dust attenuated spectra assuming the Charlot & Fall (2000) dust model. In this model young star particles
        are embedded in a dusty birth cloud and thus feel more dust attenuation.


        Parameters
        ----------
        grid : obj (Grid)
            The spectral frid
        tauV_ISM: float
            numerical value of dust attenuation due to the ISM in the V-band
        tauV_BC: float
            numerical value of dust attenuation due to the BC in the V-band
        alpha_ISM: float
            slope of the ISM dust curve, -0.7 in MAGPHYS
        alpha_BC: float
            slope of the BC dust curve, -1.3 in MAGPHYS
        save_young_and_old: boolean
            flag specifying whether to save young and old

        Returns
        -------
        obj (Sed)
             A Sed object containing the dust attenuated spectra
        """

        # calculate intrinsic sed for young and old stars
        intrinsic_sed_young = generate_lnu(self, grid, spectra_name, old=old, young=young)
        intrinsic_sed_old = generate_lnu(self, grid, spectra_name, old=old, young=young)

        if save_young_and_old:

            self.spectra['intrinsic_young'] = intrinsic_sed_young
            self.spectra['intrinsic_old'] = intrinsic_sed_old

        # calculate dust attenuation for young and old components
        T_ISM = power_law({'slope': alpha_ISM}).attenuate(tauV_ISM, grid.lam)
        T_BC = power_law({'slope': alpha_BC}).attenuate(tauV_BC, grid.lam)

        T_young = T_ISM * T_BC
        T_old = T_ISM

        sed_young = self.intrinsic_sed_young.lnu * T_young
        sed_old = self.intrinsic_sed_old.lnu * T_old

        if save_young_and_old:

            # if integrated:
            self.spectra['attenuated_young'] = Sed(grid.lam, sed_young)
            self.spectra['attenuated_old'] = Sed(grid.lam, sed_old)

        sed = Sed(grid.lam, sed_young + sed_old)

        if update:
            self.spectra['attenuated'] = sed

        if sed_object:
            return sed
        else:
            return sed_young + sed_old


    def get_intrinsic_line(self, grid, line_ids, fesc=0.0, update=True):
        """
        Calculates **intrinsic** properties (luminosity, continuum, EW) for a set of lines.


        Parameters
        ----------
        grid : obj (Grid)
            The Grid
        line_ids : list or str
            A list of line_ids or a str denoting a single line. Doublets can be specified as a nested list or using a comma (e.g. 'OIII4363,OIII4959')
        fesc : float
            The Lyman continuum escape fraction, the fraction of ionising photons that entirely escape

        Returns
        -------
        lines : dictionaty-like (obj)
             A dictionary containing line objects.
        """

        # if only one line specified convert to a list to avoid writing a longer if statement
        if type(line_ids) is str:
            line_ids = [line_ids]

        # dictionary holding Line objects
        lines = {}

        for line_id in line_ids:

            # if the line id a doublet in string form (e.g. 'OIII4959,OIII5007') convert it to a list
            if type(line_id) is str:
                if len(line_id.split(',')) > 1:
                    line_id = line_id.split(',')

            # if the line_id is a str denoting a single line
            if isinstance(line_id, str):

                grid_line = grid.lines[line_id]
                wavelength = grid_line['wavelength']

                #  line luminosity erg/s
                luminosity = np.sum((1-fesc)*grid_line['luminosity'] * self.sfzh.sfzh, axis=(0, 1))

                #  continuum at line wavelength, erg/s/Hz
                continuum = np.sum(grid_line['continuum'] * self.sfzh.sfzh, axis=(0, 1))

                # NOTE: this is currently incorrect and should be made of the separated nebular and stellar continuum emission
                # proposed alternative
                # stellar_continuum = np.sum(
                #     grid_line['stellar_continuum'] * self.sfzh.sfzh, axis=(0, 1))  # not affected by fesc
                # nebular_continuum = np.sum(
                #     (1-fesc)*grid_line['nebular_continuum'] * self.sfzh.sfzh, axis=(0, 1))  # affected by fesc

            # else if the line is list or tuple denoting a doublet (or higher)
            elif isinstance(line_id, list) or isinstance(line_id, tuple):

                luminosity = []
                continuum = []
                wavelength = []

                for line_id_ in line_id:
                    grid_line = grid.lines[line_id_]

                    # wavelength [\AA]
                    wavelength.append(grid_line['wavelength'])

                    #  line luminosity erg/s
                    luminosity.append(
                        (1-fesc)*np.sum(grid_line['luminosity'] * self.sfzh.sfzh, axis=(0, 1)))

                    #  continuum at line wavelength, erg/s/Hz
                    continuum.append(np.sum(grid_line['continuum'] * self.sfzh.sfzh, axis=(0, 1)))

            else:
                # throw exception
                pass

            line = Line(line_id, wavelength, luminosity, continuum)
            lines[line.id] = line

        if update:
            self.lines[line.id] = line

        return lines

    def get_attenuated_line(self, grid, line_ids, fesc=0.0, tauV_nebular=None,
                            tauV_stellar=None, dust_curve_nebular=power_law({'slope': -1.}),
                            dust_curve_stellar=power_law({'slope': -1.}), update=True):
        """
        Calculates attenuated properties (luminosity, continuum, EW) for a set of lines. Allows the nebular and stellar attenuation to be set separately.

        Parameters
        ----------
        grid : obj (Grid)
            The Grid
        line_ids : list or str
            A list of line_ids or a str denoting a single line. Doublets can be specified as a nested list or using a comma (e.g. 'OIII4363,OIII4959')
        fesc : float
            The Lyman continuum escape fraction, the fraction of ionising photons that entirely escape
        tauV_nebular : float
            V-band optical depth of the nebular emission
        tauV_stellar : float
            V-band optical depth of the stellar emission
        dust_curve_nebular : obj (dust_curve)
            A dust_curve object specifying the dust curve for the nebular emission
        dust_curve_stellar : obj (dust_curve)
            A dust_curve object specifying the dust curve for the stellar emission

        Returns
        -------
        lines : dictionary-like (obj)
             A dictionary containing line objects.
        """

        # if the intrinsic lines haven't already been calcualted and saved then generate them
        if 'intrinsic' not in self.lines:
            intrinsic_lines = self.get_intrinsic_line(grid, line_ids, fesc=fesc, update=update)
        else:
            intrinsic_lines = self.lines['intrinsic']

        # dictionary holding lines
        lines = {}

        for line_id, intrinsic_line in intrinsic_lines.items():

            # calculate attenuation
            T_nebular = dust_curve_nebular.attenuate(tauV_nebular, intrinsic_line._wavelength)
            T_stellar = dust_curve_stellar.attenuate(tauV_stellar, intrinsic_line._wavelength)

            luminosity = intrinsic_line._luminosity * T_nebular
            continuum = intrinsic_line._continuum * T_stellar

            line = Line(intrinsic_line.id, intrinsic_line._wavelength, luminosity, continuum)

            # NOTE: the above is wrong and should be separated into stellar and nebular continuum components:
            # nebular_continuum = intrinsic_line._nebular_continuum * T_nebular
            # stellar_continuum = intrinsic_line._stellar_continuum * T_stellar
            # line = Line(intrinsic_line.id, intrinsic_line._wavelength, luminosity, nebular_continuum, stellar_continuum)

            lines[line.id] = line

        if update:
            self.lines['attenuated'] = lines

        return lines

    def get_screen_line(self, grid, line_ids, fesc=0.0, tauV=None, dust_curve=power_law({'slope': -1.}), update=True):
        """
        Calculates attenuated properties (luminosity, continuum, EW) for a set of lines assuming a simple dust screen (i.e. both nebular and stellar emission feels the same dust attenuation). This is a wrapper around the more general method above.

        Parameters
        ----------
        grid : obj (Grid)
            The Grid
        line_ids : list or str
            A list of line_ids or a str denoting a single line. Doublets can be specified as a nested list or using a comma (e.g. 'OIII4363,OIII4959')
        fesc : float
            The Lyman continuum escape fraction, the fraction of ionising photons that entirely escape
        tauV : float
            V-band optical depth

        dust_curve : obj (dust_curve)
            A dust_curve object specifying the dust curve for the nebular emission

        Returns
        -------
        lines : dictionary-like (obj)
             A dictionary containing line objects.
        """

        return self.get_attenuated_line(grid, line_ids, fesc=fesc, tauV_nebular=tauV, tauV_stellar=tauV, dust_curve_nebular=dust_curve, dust_curve_stellar=dust_curve)

    def make_images(self, spectra_type, resolution, npix=None, fov=None, update=True, rest_frame=True):

        images = ParametricImage(self.morph, resolution, npix=npix, fov=fov,
                                 sed=self.spectra[spectra_type], rest_frame=rest_frame)
        images.create_images()

        if update:
            self.images[spectra_type] = images

        return images
