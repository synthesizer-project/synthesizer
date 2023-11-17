"""
A generic blackholes class currently holding (ultimately) various blackhole 
emission models. 

TODO: this should probably be moved into components/blackholes.py
"""

import numpy as np
from unyt import deg, km, cm, s, K
from synthesizer.dust.emission import Greybody
from synthesizer.grid import Grid
from synthesizer.sed import Sed
from synthesizer.exceptions import MissingArgument


class BlackHoleEmissionModel:
    """
    Potential parent class for blackhole emission models.
    """

    pass



class Template(BlackHoleEmissionModel):

    """
    Use a template for the emission model. 

    The template is simply scaled by bolometric luminosity.
    
    """

    def __init__(self, filename=None, lam=None, lnu=None):

        """
        
        Arguments:
            filename (str)
                The filename (including full path) to a file containing the 
                template. The file should contain two columns with wavelength
                and luminosity (lnu).
            lam (array)
                Wavelength array.
            lnu (array)
                Luminosity array.
        
        """

        if filename:

            # to be implemented implement later
            pass

        if lam and lnu:
            
            # initialise a synthesizer Sed object
            sed = Sed(lam=lam, lnu=lnu)

            # normalise
            # TODO: add a method to Sed that does this.
            normalisation = sed.measure_bolometric_luminosity()
            sed.lnu /= normalisation

    def get_spectra(self, bolometric_luminosity):

        """
        
        Calculating the blackhole spectra. This is done by simply scaling the 
        normalised template by the bolometric luminosity

        Arguments:
            bolometric_luminosity (float)
                The bolometric luminosity of the blackhole(s).
        
        """

        return {'total': self.normalisation * self.sed}


class UnifiedAGN(BlackHoleEmissionModel):

    """
    The Unified AGN model.
    The combines a disc model, along with modelling of the NLR, BLR, and torus.
    """

    def __init__(self,
                 grid_dir=None,
                 disc_model=None,
                 metallicity=0.01,
                 ionisation_parameter_blr=0.1,
                 hydrogen_density_blr=1E9/cm**3,
                 covering_fraction_blr=0.1,
                 velocity_dispersion_blr=2000*km/s,
                 ionisation_parameter_nlr=0.01,
                 hydrogen_density_nlr=1E4/cm**3,
                 covering_fraction_nlr=0.1,
                 velocity_dispersion_nlr=500*km/s,
                 theta_torus=10*deg,
                 torus_emission_model=Greybody(1000*K, 1.5),
                 **kwargs,
                 ):

        """
        Arguments:
            metallicity (float)
                The metallicity of the NLR and BLR gas. The default is 0.01.
            ionisation_parameter_blr (float)
                The  ionisation parameter in the BLR. Default value
                is 0.1.
            hydrogen_density_blr (unyt.unit_object.Unit)
                The  hydrogen density in the BLR. Default value
                is 1E9/cm**3.
            covering_fraction_blr (float)
                The covering fraction of the BLR. Default value is 0.1.
            velocity_dispersion_blr (unyt.unit_object.Unit)
                The velocity disperson of the BLR. Default value is 
                2000*km/s.
            ionisation_parameter_nlr (float)
                The ionisation parameter in the BLR. Default value
                is 0.01.
            hydrogen_density_nlr (unyt.unit_object.Unit)
                The hydrogen density in the NLR. Default value
                is 1E4/cm**3.
            covering_fraction_nlr (float)
                The covering fraction of the NLR. Default value is 0.1.
            velocity_dispersion_nlr (unyt.unit_object.Unit)
                The velocity disperson of the NLR. Default value is 
                500*km/s.
            theta_torus (float)
                The opening angle of the torus component. Default value is
                10*deg.
            torus_emission_model (synthesizer.dust.emission)
                The torus emission model. A synthesizer dust emission model.
                Default is a Greybody with T=1000*K and emissivity=1.6.
            **kwargs 
                Addition parameters for the specific disc model.

        """
        self.disc_model = disc_model
        self.grid_dir = grid_dir
        # used for NLR and BLR
        self.metallicity = metallicity
        # BLR parameters
        self.ionisation_parameter_blr = ionisation_parameter_blr
        self.hydrogen_density_blr = hydrogen_density_blr
        self.covering_fraction_blr = covering_fraction_blr
        self.velocity_dispersion_blr = velocity_dispersion_blr
        # NLR parameters
        self.ionisation_parameter_nlr = ionisation_parameter_nlr
        self.hydrogen_density_nlr = hydrogen_density_nlr
        self.covering_fraction_nlr = covering_fraction_nlr
        self.velocity_dispersion_nlr = velocity_dispersion_nlr
        # Torus parameters
        self.theta_torus = theta_torus
        self.torus_emission_model = torus_emission_model

        # Open NLR and BLR grids
        # TODO: replace this by a single file.
        self.grid = {}
        for line_region in ['nlr', 'blr']:
            self.grid[line_region] = Grid(
                grid_name=f'{disc_model}_cloudy-c17.03-{line_region}',
                grid_dir=grid_dir,
                read_lines=False)
        
        # Get axes from the grid, this allows us to obtain the disc parameters.
        self.grid_parameters = self.grid['nlr'].axes[:]

        # Get disc parameters by removing line region parameters
        # TODO: replace this by saving something in the Grid file.
        self.disc_parameters = []
        for parameter in self.grid_parameters:
            if parameter not in ['metallicity',
                                 'ionisation_parameter',
                                 'hydrogen_density']:
                self.disc_parameters.append(parameter)


        # TODO: need to extract a list of torus model parameters.
        # self.torus_parameters = []

        # Get a list of parameters. Almost certainly a better way of doing this.
        self.parameters = self.disc_parameters + [
            'metallicity',
            'ionisation_parameter_blr',
            'hydrogen_density_blr',
            'covering_fraction_blr',
            'svelocity_dispersion_blr',
            'ionisation_parameter_nlr',
            'hydrogen_density_nlr',
            'covering_fraction_nlr',
            'velocity_dispersion_nlr',
            'theta_torus',
        ]

        # dictionary holding LineCollections for each (relevant) component
        self.lines = {}

        # dictionary holding Seds for each component (and sub-component)
        self.spectra = {}

    def _update_parameters(self, **kwargs):

        """
        Updates parameters.
        """
        # update parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
          
    def _get_grid_point(self, line_region, isotropic=False):
        """
        Private method used to get the grid point.

        """

        # calculate the grid point.
        parameters = []
        for parameter in self.grid_parameters:
            # for hydrogen_density and ionisation_paramter the parameters are
            # labelled e.g. 'hydrogen_density_nlr'.
            if parameter in ['hydrogen_density', 'ionisation_parameter']:
                parameter = parameter+'_'+line_region
            parameters.append(getattr(self, parameter))

        return self.grid[line_region].get_grid_point(parameters)

    def _get_spectra_disc(self, **kwargs):
        """
        Generate the disc spectra, updating the parameters if required.

        Returns
            (dict, synthesizer.sed.Sed)
                A dictionary of Sed instances including the escaping and
                transmitted disc emission.
        """

        # get grid points
        nlr_grid_point = self._get_grid_point('nlr')
        blr_grid_point = self._get_grid_point('blr')

        # calculate the incident spectra. It doesn't matter which spectra we
        # use here since we're just using the incident. Note: this assumes the 
        # NLR and BLR are not overlapping.
        self.spectra['disc_incident'] = \
            self.grid['nlr'].get_spectra(nlr_grid_point, spectra_id='incident')

        # calculate the transmitted spectra
        nlr_spectra = self.grid['nlr'].get_spectra(nlr_grid_point,
                                                   spectra_id='transmitted')
        blr_spectra = self.grid['blr'].get_spectra(blr_grid_point,
                                                   spectra_id='transmitted')
        self.spectra['disc_transmitted'] = (self.covering_fraction_nlr *
                                            nlr_spectra +
                                            self.covering_fraction_blr *
                                            blr_spectra)
            
        # calculate the escaping spectra.
        self.spectra['disc_escaped'] = (
            1 - self.covering_fraction_blr - self.covering_fraction_nlr) *\
            self.spectra['disc_incident']

        # calculate the total spectra, the sum of escaping and transmitted
        self.spectra['disc'] = self.spectra['disc_transmitted'] + self.spectra['disc_escaped']

    def _get_spectra_lr(self, line_region, **kwargs):
        """
        Generate the spectra of a generic line region.

        Arguments:
            line_region (str)
                The specific line region, i.e. 'nlr' or 'blr'.
        
        Returns:
            synthesizer.sed.Sed
                The NLR spectra
        """
        
        # get the grid point
        grid_point = self._get_grid_point(line_region)

        # covering fraction
        covering_fraction = getattr(self, f'covering_fraction_{line_region}')

        # get the nebular spectra of the line region
        sed = covering_fraction * self.grid[line_region].get_spectra(
            grid_point,
            spectra_id='nebular')

        # add line to spectra broadening here
        self.spectra[line_region] = sed

        # add lines TO BE ADDED
        self.lines[line_region] = None

    def _get_spectra_torus(self):
        """
        Generate the torus.

        Returns
            spectra (dict, synthesizer.sed.Sed)
                Dictionary of synthesizer Sed instances.
        """

        disc = self.spectra['disc']

        # calculate the bolometric dust lunminosity as the difference between
        # the intrinsic and attenuated

        torus_bolometric_luminosity = (self.theta_torus/(90*deg)) * \
            disc.measure_bolometric_luminosity()

        # get the spectrum and normalise it properly
        lnu = torus_bolometric_luminosity.to("erg/s").value * \
            self.torus_emission_model.lnu(disc.lam)

        # create new Sed object containing dust spectra
        self.spectra['torus'] = Sed(disc.lam, lnu=lnu)

    def get_spectra(self, **kwargs):
        """
        Generate the spectra, updating the parameters if required.

        Returns
            spectra (dict, synthesizer.sed.Sed)
                Dictionary of synthesizer Sed instances.
        """

        # update parameters
        self._update_parameters(**kwargs)

        self._get_spectra_disc()
        self._get_spectra_lr('nlr')
        self._get_spectra_lr('blr')
        self._get_spectra_torus()

        self.spectra['total'] = self.spectra['disc'] + \
            self.spectra['blr'] + self.spectra['nlr'] + self.spectra['torus']
    
        return self.spectra
    


