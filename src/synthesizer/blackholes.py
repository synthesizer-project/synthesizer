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


class UnifiedAGN(BlackHoleEmissionModel):

    def __init__(self,
                 grid_dir=None,
                 disc_model=None,
                 metallicity=0.01,
                 U_blr=0.1,
                 n_H_blr=1E9/cm**3,
                 cov_blr=0.1,
                 v_blr=2000*km/s,
                 U_nlr=0.01,
                 n_H_nlr=1E4/cm**3,
                 cov_nlr=0.1,
                 v_nlr=500*km/s,
                 theta_torus=10*deg,
                 torus_emission_model=Greybody(1000*K,1.5),
                 **kwargs,
                 ):

        """
        Arguments:
            metallicity (float)
                The metallicity of the NLR and BLR gas. The default is 0.01.
            U_blr (float)
                The  ionisation parameter in the BLR. Default value
                is 0.1.
            n_H_blr (unyt.unit_object.Unit)
                The  hydrogen density in the BLR. Default value
                is 1E9/cm**3.
            cov_blr (float)
                The covering fraction of the BLR. Default value is 0.1.
            v_blr (unyt.unit_object.Unit)
                The velocity disperson of the BLR. Default value is 
                2000*km/s.
            U_nlr (float)
                The ionisation parameter in the BLR. Default value
                is 0.01.
            n_H_nlr (unyt.unit_object.Unit)
                The hydrogen density in the BLR. Default value
                is 1E4/cm**3.
            cov_nlr (float)
                The covering fraction of the BLR. Default value is 0.1.
            v_nlr (unyt.unit_object.Unit)
                The velocity disperson of the BLR. Default value is 
                500*km/s.
            theta_torus (float)
                The opening angle of the torus component. Default value is
                10*deg.
            torus_emission_model (synthesizer.dust.emission)
                The torus emission model. A synthesizer dust emission model.
                Default is a Greybody with T=1000*K and emissivity=1.6.
            **kwargs 

        """
        self.disc_model = disc_model
        self.grid_dir = grid_dir
        self.metallicity = metallicity
        self.metallicity = metallicity
        self.U_blr = U_blr
        self.log10U_blr = np.log10(U_blr)
        self.n_H_blr = n_H_blr
        self.log10n_H_blr = np.log10(n_H_blr.to('1/cm**3').value)
        self.cov_blr = cov_blr
        self.v_blr = v_blr
        self.U_nlr = U_nlr
        self.log10U_nlr = np.log10(U_nlr)
        self.n_H_nlr = n_H_nlr
        self.log10n_H_nlr = np.log10(n_H_nlr.to('1/cm**3').value)
        self.cov_nlr = cov_nlr
        self.v_nlr = v_nlr
        self.theta_torus = theta_torus
        self.torus_emission_model = torus_emission_model

        # Open NLR and BLR grids
        # TODO: replace this by a single file.
        self.nlr_grid = Grid(grid_name=f'{disc_model}_cloudy-c17.03-nlr',
                             grid_dir=grid_dir,
                             read_lines=False)
        
        self.blr_grid = Grid(grid_name=f'{disc_model}_cloudy-c17.03-blr',
                             grid_dir=self.grid_dir,
                             read_lines=False)

        # Get grid parameters
        self.grid_parameters = self.nlr_grid.axes[:]

        # Replace grid parameters with renamed version.
        # TODO: to eliminate this by changing the grid naming
        if 'log10Mbh' in self.grid_parameters:
            index = self.grid_parameters.index('log10Mbh')
            self.grid_parameters[index] = 'log10mass'
        if 'log10MdotEdd' in self.grid_parameters:
            index = self.grid_parameters.index('log10MdotEdd')
            self.grid_parameters[index] = 'log10accretion_rate_eddington'
        if 'cosinc' in self.grid_parameters:
            index = self.grid_parameters.index('cosinc')
            self.grid_parameters[index] = 'inclination'

        # Get disc parameters by removing line region parameters
        # TODO: replace this by saving something in the Grid file.
        self.disc_parameters = []
        for parameter in self.grid_parameters:
            if parameter not in ['metallicity', 'log10U', 'log10n_H']:
                self.disc_parameters.append(parameter)


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

        # check that all disc parameters
        for parameter in self.disc_parameters:
            if parameter not in kwargs.keys():
                raise MissingArgument(f'{parameter} needs to be provided')
            
    def _get_grid_point(self, grid, gridt, isotropic=False):
        """
        Private method used to get the grid point.

        """

        # calculate the grid point.
        parameters = []
        for parameter in self.grid_parameters:
            # for log10nH and log10U the parameters are labelled e.g. 'log10nH_nlr'
            if parameter in ['log10n_H', 'log10U']:
                parameter = parameter+'_'+gridt
            parameters.append(getattr(self, parameter))

        return grid.get_grid_point(parameters)

    def _get_spectra_disc(self, **kwargs):
        """
        Generate the disc spectra, updating the parameters if required.

        Returns
            (dict, synthesizer.sed.Sed)
                A dictionary of Sed instances including the escaping and
                transmitted disc emission.
        """

        # get grid points
        nlr_grid_point = self._get_grid_point(self.nlr_grid, 'nlr')
        blr_grid_point = self._get_grid_point(self.blr_grid, 'blr')

        # calculate the incident spectra. It doesn't matter which spectra we
        # use here since we're just using the incident. Note: this assumes the 
        # NLR and BLR are not overlapping.
        self.spectra['disc_incident'] = \
            self.nlr_grid.get_spectra(nlr_grid_point, spectra_id='incident')

        # calculate the transmitted spectra
        nlr_spectra = self.nlr_grid.get_spectra(nlr_grid_point,
                                                spectra_id='transmitted')
        blr_spectra = self.blr_grid.get_spectra(blr_grid_point,
                                                spectra_id='transmitted')
        self.spectra['disc_transmitted'] = self.cov_nlr * nlr_spectra \
            + self.cov_blr * blr_spectra
            
        # calculate the escaping spectra. 
        self.spectra['disc_escaped'] = (1 - self.cov_blr - self.cov_nlr) *\
            self.spectra['disc_incident']

        # calculate the total spectra, the sum of escaping and transmitted
        self.spectra['disc'] = self.spectra['disc_transmitted'] + self.spectra['disc_escaped']

    def _get_spectra_lr(self, line_region, **kwargs):
        """
        Generate the spectra of a generic line region, updating the parameters if required.

        Returns:
            synthesizer.sed.Sed
                The NLR spectra
        """

        if line_region == 'nlr':
            grid = self.nlr_grid
        elif line_region == 'blr':
            grid = self.blr_grid
        else:
            print('line region not recognised')
        
        # get the grid point
        grid_point = self._get_grid_point(grid, line_region)

        # covering fraction
        cov = getattr(self, f'cov_{line_region}')

        # get spectra
        sed = cov * grid.get_spectra(grid_point, spectra_id='nebular')

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
        
        return self.spectra['total']
    


