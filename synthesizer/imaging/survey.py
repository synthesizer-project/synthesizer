"""
Survey functionality 
"""
import synthesizer.exceptions as exceptions
from synthesizer.imaging import images, spectral_cubes
from synthesizer.galaxy.particle import ParticleGalaxy
from synthesizer.galaxy.parametric import ParametricGalaxy


class Instrument:
    """
    This class describes an instrument used to make a set of observations.

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(self, resolution, filters, psfs=None, depths=None,
                 aperture=None, resolving_power=None, lam=None):
        """
        Initialise the Observatory.

        Parameters
        ----------

        """

        # Basic metadata
        self.instrument = filter_.filter_code.split(".")[0]

        # Store some basic instrument properties
        self.resolution = resolution
        self.filters = filters
        self.psfs = psfs

        # Store some basic spectral information for this observation.
        self.resolving_power = resolving_power
        self.lams = None

        # Intilaise noise properties which can be populated by the outside.
        self.aperture = aperture
        self.depths = depths
        self.noise_sigma = None

    def _check_obs_args(self):
        """
        Ensures we have valid inputs.

        Parameters
        ----------
        
        Raises
        ------
        
        """
        pass

    def get_lam_from_R(self):
        """
        Calculates the wavelengths of a spectrum based on this observations
        resolving power.

        Parameters
        ----------
        
        Raises
        ------
        
        """
        pass

        
class Survey:
    """

    Should be both a container for information and the base object to make a
    "field" observation.

    Attributes
    ----------

    Methods
    -------

    """
    
    def __init__(self, galaxies=(), fov=None):
        """
        Initialise the Survey.

        Parameters
        ----------

        """

        # Basic information
        self.ninstruments = 0
        self.nfilters = 0

        # Information about the field being observered
        self.fov = fov

        # Observation configurations are held in a dict, initialise it.
        self.instruments = {}

        # Store the galaxies we are making images of
        self.galaxies = galaxies

        # Intialise somewhere to keep survey images, this is populated later
        self.imgs = None

    def _check_survey_args(self):
        """
        Ensures we have valid inputs.

        Parameters
        ----------
        
        Raises
        ------
        
        """
        pass

    def add_photometric_instrument(self, filters, resolution, label, psfs=None,
                                   depths=None, apertures=None):
        """
        Adds an instrument and all it's filters to the Survey.

        Parameters
        ----------
        
        Raises
        ------
        InconsistentArguments
            If the arguments do not constitute a valid combination for an
            instrument an error is thrown.
        """

        # How many filters do we have?
        nfilters = len(filters)

        # Check our inputs match the number of filters we have
        if isinstance(psfs, dict):
            if nfilters != len(psfs):
                raise exceptions.InconsistentArguments(
                    "Inconsistent number of entries in instrument dictionaries"
                    " len(filters)=%d, len(psfs)=%d)" % (nfilters, len(psfs))
                )
        if isinstance(depths, dict):
            if nfilters != len(depths):
                raise exceptions.InconsistentArguments(
                    "Inconsistent number of entries in instrument dictionaries"
                    " len(filters)=%d, len(depths)=%d)" % (nfilters,
                                                           len(depths))
                )
        if isinstance(apertures, dict):
            if nfilters != len(apertures):
                raise exceptions.InconsistentArguments(
                    "Inconsistent number of entries in instrument dictionaries"
                    " len(filters)=%d, len(apertures)=%d)" % (nfilters,
                                                              len(apertures))
                )

        # Create this observation configurations
        self.instruments[label] = Instrument(
            resolution=resolution, filters=filters, psfs=psfs,
            depths=depths, aperture=apertures
        )

        # Record that we included another insturment and count the filters
        self.ninstruments += 1
        self.nfilters += len(filters)

    def add_spectral_instrument(self, resolution, resolving_power,
                                psf=None, depth=None, aperture=None):
        pass
        

    def add_galaxies(self, galaxies):
        """
        Adds galaxies to this survey

        Parameters
        ----------
        galaxies : list
            The galaxies to include in this Survey.
        
        """

        # If we have no galaxies just add them
        if len(self.galaxies) == 0:
            self.galaxies = galaxies

        # Otherwise, we have to add them on to what we have, handling whether
        # we are adding 1 galaxy...
        elif (len(self.galaxies) > 0 and
              (isinstance(galaxies, ParticleGalaxy) or
               isinstance(galaxies, ParametricGalaxy))):

            # Double check galaxies is a list
            self.galaxies = list(self.galaxies)

            # Include the new galaxies
            self.galaxies.append(galaxies)

        # ... or multiple galaxies
        else:
            
            # Double check galaxies is a list
            self.galaxies = list(self.galaxies)

            # Include the new galaxies
            self.galaxies.extend(galaxies)


    def get_photometry(self):
        """

        Parameters
        ----------
        
        """
        pass

    def make_field_image(self, centre):
        """

        Parameters
        ----------
        
        """
        pass

    def make_images(self, img_type, spectra_type, kernel_func=None,
                    rest_frame=False, cosmo=None, igm=None):
        """

        Parameters
        ----------
        
        """

        # Make a dictionary in which to store our image objects, within
        # this dictionary imgs are stored in list ordered by galaxy.
        self.imgs = {}

        # Loop over instruments and make images for each galaxy using each
        # instrument
        for key in self.instruments:

            # Extract the instrument
            inst = self.instruments[key]

            # Create entry in images dictionary
            self.imgs[inst] = []

            # Loop over galaxies
            for gal in self.galaxies:

                # Get images of this galaxy with this instrument
                img = gal.make_image(self.resolution, fov=self.fov,
                                     img_type=img_type,
                                     sed=gal.spectra_array[spectra_type],
                                     filters=inst.filters,
                                     psfs=inst.psfs, depths=inst.depths,
                                     aperture=inst.aperture,
                                     kernel_func=kernel_func,
                                     rest_frame=rest_frame,
                                     redshift=gal.redshift,
                                     cosmo=cosmo, igm=igm)

                # Store this result
                self.imgs[inst].append(img)

        return self.imgs

            

    def make_field_ifu(self, centre):
        """

        Parameters
        ----------
        
        """
        pass

    def make_ifus(self):
        """

        Parameters
        ----------
        
        """
        pass
