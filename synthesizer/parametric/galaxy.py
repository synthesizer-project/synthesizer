

# --- general
import h5py
import copy
import numpy as np
from scipy import integrate
from unyt import yr, erg, Hz, s, cm, angstrom

from ..base_galaxy import BaseGalaxy
from .. import exceptions
from ..dust import power_law
from ..sed import Sed
from ..line import Line
from ..plt import single_histxy, mlabel
from ..stats import weighted_median, weighted_mean
from ..imaging.images import ParametricImage
from ..art import Art
from synthesizer.utils import fnu_to_flam


class Galaxy(BaseGalaxy):

    """A class defining parametric galaxy objects

    """

    def __init__(
            self,
            sfzh,
            morph=None,
            name="parametric galaxy",
    ):
        """__init__ method for ParametricGalaxy

        Args:
            name (string):
                name of galaxy
            sfzh (object, sfzh):
                instance of the BinnedSFZH class containing the star formation and metal enrichment history.
            morph (object)
        """

        self.name = name

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
        new_galaxy = Galaxy(new_sfzh)

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
                new_galaxy.images[img_name] = \
                    image + second_galaxy.images[img_name]
            else:
                exceptions.InconsistentAddition(
                    ('Both galaxies must contain the same'
                     ' images to be added together'))

        return new_galaxy

    def get_Q(self, grid):
        """
        Return the ionising photon luminosity (log10Q) for a given SFZH. 

        Args:
            grid (object, Grid):
                The SPS Grid object from which to extract spectra. 

        Returns:
            Log of the ionising photon luminosity over the grid dimensions
        """

        return np.sum(10**grid.log10Q['HI'] * self.sfzh.sfzh, axis=(0, 1))

    def generate_lnu(self, grid, spectra_name, old=False, young=False):
        """
        Calculate rest frame spectra from an SPS Grid.

        This is a flexible base method which extracts the rest frame spectra of
        this galaxy from the SPS grid based on the passed arguments. More
        sophisticated types of spectra are produced by the get_spectra_*
        methods on BaseGalaxy, which call this method.

        Args:
            grid (object, Grid):
                The SPS Grid object from which to extract spectra.
            spectra_name (str):
                A string denoting the desired type of spectra. Must match a
                key on the Grid.
            old (bool/float):
                Are we extracting only old stars? If so only SFZH bins with
                log10(Ages) > old will be included in the spectra. Defaults to
                False.
            young (bool/float):
                Are we extracting only young stars? If so only SFZH bins with
                log10(Ages) <= young will be included in the spectra. Defaults
                to False.

        Returns:
            The Galaxy's rest frame spectra in erg / s / Hz.
        """

        # Ensure arguments make sense
        if old * young:
            raise ValueError("Cannot provide old and young stars together")

        # Get the indices of non-zero entries in the SFZH
        non_zero_inds = np.where(self.sfzh_ > 0)

        # Make the mask for relevent SFZH bins
        if old:
            sfzh_mask = (self.sfzh.log10ages[non_zero_inds[0]] > old)
        elif young:
            sfzh_mask = (self.sfzh.log10ages[non_zero_inds[0]] <= young)
        else:
            sfzh_mask = np.ones(len(self.sfzh.log10ages[non_zero_inds[0]]),
                                dtype=bool)

        # Account for the SFZH mask in the non-zero indices 
        non_zero_inds = (non_zero_inds[0][sfzh_mask],
                         non_zero_inds[1][sfzh_mask])

        # Compute the spectra
        spectra = np.sum(
            grid.spectra[spectra_name][non_zero_inds[0], non_zero_inds[1], :]
            * self.sfzh_[non_zero_inds[0], non_zero_inds[1], :],
            axis=0
        )

        return spectra

    def get_line_intrinsic(self, grid, line_ids, fesc=0.0, update=True):
        """
        Calculates **intrinsic** properties (luminosity, continuum, EW)
        for a set of lines.


        Parameters
        ----------
        grid : obj (Grid)
            The Grid
        line_ids : list or str
            A list of line_ids or a str denoting a single line.
            Doublets can be specified as a nested list or using a comma (e.g. 'OIII4363,OIII4959')
        fesc : float
            The Lyman continuum escape fraction, the fraction of
            ionising photons that entirely escape

        Returns
        -------
        lines : dictionaty-like (obj)
             A dictionary containing line objects.
        """

        # if only one line specified convert to a list to avoid writing a
        # longer if statement
        if type(line_ids) is str:
            line_ids = [line_ids]

        # dictionary holding Line objects
        lines = {}

        for line_id in line_ids:

            # if the line id a doublet in string form
            # (e.g. 'OIII4959,OIII5007') convert it to a list
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

                # NOTE: this is currently incorrect and should be made of the 
                # separated nebular and stellar continuum emission
                # 
                # proposed alternative
                # stellar_continuum = np.sum(
                #     grid_line['stellar_continuum'] * self.sfzh.sfzh, 
                #               axis=(0, 1))  # not affected by fesc
                # nebular_continuum = np.sum(
                #     (1-fesc)*grid_line['nebular_continuum'] * self.sfzh.sfzh, 
                #               axis=(0, 1))  # affected by fesc

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

    def get_line_attenuated(self, grid, line_ids, fesc=0.0, tauV_nebular=None,
                            tauV_stellar=None, dust_curve_nebular=power_law({'slope': -1.}),
                            dust_curve_stellar=power_law({'slope': -1.}), update=True):
        """
        Calculates attenuated properties (luminosity, continuum, EW) for a set
        of lines. Allows the nebular and stellar attenuation to be set separately.

        Parameters
        ----------
        grid : obj (Grid)
            The Grid
        line_ids : list or str
            A list of line_ids or a str denoting a single line. Doublets can be
            specified as a nested list or using a comma
            (e.g. 'OIII4363,OIII4959')
        fesc : float
            The Lyman continuum escape fraction, the fraction of
            ionising photons that entirely escape
        tauV_nebular : float
            V-band optical depth of the nebular emission
        tauV_stellar : float
            V-band optical depth of the stellar emission
        dust_curve_nebular : obj (dust_curve)
            A dust_curve object specifying the dust curve
            for the nebular emission
        dust_curve_stellar : obj (dust_curve)
            A dust_curve object specifying the dust curve
            for the stellar emission

        Returns
        -------
        lines : dictionary-like (obj)
             A dictionary containing line objects.
        """

        # if the intrinsic lines haven't already been calculated and saved 
        # then generate them
        if 'intrinsic' not in self.lines:
            intrinsic_lines = self.get_line_intrinsic(grid, line_ids, fesc=fesc, update=update)
        else:
            intrinsic_lines = self.lines['intrinsic']

        # dictionary holding lines
        lines = {}

        for line_id, intrinsic_line in intrinsic_lines.items():

            # calculate attenuation
            T_nebular = dust_curve_nebular.attenuate(
                tauV_nebular, intrinsic_line._wavelength)
            T_stellar = dust_curve_stellar.attenuate(
                tauV_stellar, intrinsic_line._wavelength)

            luminosity = intrinsic_line._luminosity * T_nebular
            continuum = intrinsic_line._continuum * T_stellar

            line = Line(intrinsic_line.id, intrinsic_line._wavelength, 
                        luminosity, continuum)

            # NOTE: the above is wrong and should be separated into stellar and nebular continuum components:
            # nebular_continuum = intrinsic_line._nebular_continuum * T_nebular
            # stellar_continuum = intrinsic_line._stellar_continuum * T_stellar
            # line = Line(intrinsic_line.id, intrinsic_line._wavelength, luminosity, nebular_continuum, stellar_continuum)

            lines[line.id] = line

        if update:
            self.lines['attenuated'] = lines

        return lines

    def get_line_screen(self, grid, line_ids, fesc=0.0, tauV=None, dust_curve=power_law({'slope': -1.}), update=True):
        """
        Calculates attenuated properties (luminosity, continuum, EW) for a set 
        of lines assuming a simple dust screen (i.e. both nebular and stellar 
        emission feels the same dust attenuation). This is a wrapper around 
        the more general method above.

        Args:
            grid : obj (Grid)
                The Grid
            line_ids : list or str
                A list of line_ids or a str denoting a single line. Doublets
                can be specified as a nested list or using a comma
                (e.g. 'OIII4363,OIII4959')
            fesc : float
                The Lyman continuum escape fraction, the fraction of
                ionising photons that entirely escape
            tauV : float
                V-band optical depth
            dust_curve : obj (dust_curve)
                A dust_curve object specifying the dust curve for
                the nebular emission

        Returns:
            lines : dictionary-like (obj)
                A dictionary containing line objects.
        """

        return self.get_line_attenuated(grid, line_ids, fesc=fesc, tauV_nebular=tauV, tauV_stellar=tauV, dust_curve_nebular=dust_curve, dust_curve_stellar=dust_curve)

    def get_equivalent_width(self, index, spectra_to_plot=None):
        """ gets all equivalent widths associated with a galaxy object """
        equivalent_width = None
    
        if type(spectra_to_plot) != list:
            spectra_to_plot = list(self.spectra.keys())
    
        for sed_name in spectra_to_plot:
            sed = self.spectra[sed_name]
            lam_arr = sed.lam
            lnu_arr = sed.lnu
    
            equivalent_width = sed.calculate_ew(lam_arr, lnu_arr, index)
    
        return equivalent_width

    
    def reduce_lya(self, grid, fesc_LyA, young=False, old=False):

        # --- generate contribution of line emission alone and reduce the contribution of Lyman-alpha
        linecont = self.generate_lnu(grid, spectra_name='linecont', old=old, young=young)
        idx = grid.get_nearest_index(1216., grid.lam)  # get index of Lyman-alpha
        linecont[idx] *= fesc_LyA  # reduce the contribution of Lyman-alpha

        return linecont

    def make_images(self, resolution, fov=None, sed=None, filters=(),
                    psfs=None, depths=None, snrs=None, aperture=None,
                    noises=None, rest_frame=True, cosmo=None, redshift=None,
                    psf_resample_factor=1,
                    ):
        """
        Makes images in each filter provided in filters. Additionally an image
        can be made with or without a PSF and noise.
        NOTE: Either npix or fov must be defined.
        
        Parameters
        ----------
        resolution : float
           The size of a pixel.
           (Ignoring any supersampling defined by psf_resample_factor)
        npix : int
            The number of pixels along an axis.
        fov : float
            The width of the image in image coordinates.
        sed : obj (SED)
            An sed object containing the spectra for this image.
        filters : obj (FilterCollection)
            An imutable collection of Filter objects. If provided images are 
            made for each filter.
        psfs : dict
            A dictionary containing the psf in each filter where the key is
            each filter code and the value is the psf in that filter.
        depths : dict
            A dictionary containing the depth of an observation in each filter
            where the key is each filter code and the value is the depth in
            that filter.
        aperture : float/dict
            Either a float describing the size of the aperture in which the
            depth is defined or a dictionary containing the size of the depth
            aperture in each filter.
        rest_frame : bool
            Are we making an observation in the rest frame?
        cosmo : obj (astropy.cosmology)
            A cosmology object from astropy, used for cosmological calculations
            when converting rest frame luminosity to flux.
        redshift : float
            The redshift of the observation. Used when converting between
            physical cartesian coordinates and angular coordinates.
        psf_resample_factor : float
            The factor by which the image should be resampled for robust PSF
            convolution. Note the images after PSF application will be
            downsampled to the native pixel scale.
        Returns
        -------
        Image : array-like
            A 2D array containing the image.
        """

        # Handle a super resolution image
        if psf_resample_factor is not None:
            if psf_resample_factor != 1:
                resolution /= psf_resample_factor

        # Instantiate the Image object.
        img = ParametricImage(
            morphology=self.morph,
            resolution=resolution,
            fov=fov,
            sed=sed,
            filters=filters,
            rest_frame=rest_frame,
            redshift=redshift,
            cosmo=cosmo,
            psfs=psfs,
            depths=depths,
            apertures=aperture,
            snrs=snrs,
        )

        # Compute image
        img.get_imgs()

        if psfs is not None:

            # Convolve the image/images
            img.get_psfed_imgs()
            
            # Downsample to the native resolution if we need to.
            if psf_resample_factor is not None:
                if psf_resample_factor != 1:
                    img.downsample(1 / psf_resample_factor)
                    
        if depths is not None or noises is not None:
            
            img.get_noisy_imgs(noises)

        return img

     def get_equivalent_width(self, index, spectra_to_plot=None):
        """ gets all equivalent widths associated with a sed object """
        equivalent_width = None

        if type(spectra_to_plot) != list:
            spectra_to_plot = list(self.spectra.keys())

        for sed_name in spectra_to_plot:
            sed = self.spectra[sed_name]
            lam_arr = sed.lam
            lnu_arr = sed.lnu

            # Compute equivalent width
            equivalent_width = sed.calculate_ew(lam_arr, lnu_arr, index)

        return equivalent_width


