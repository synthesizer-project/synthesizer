import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr

from .sed import Sed
from .dust import power_law
from . import exceptions
from .line import Line


class BaseGalaxy:
    """
    The base galaxy class
    """

    def __init__(self, *args, **kwargs):

        # Add some place holder attributes which are overloaded on the children
        self.spectra = {}

        raise Warning(("Instantiating a BaseGalaxy object is not "
                       "supported behaviour. Instead, you should "
                       "use one of the derived Galaxy classes:\n"
                       "`particle.galaxy.Galaxy`\n"
                       "`parametric.galaxy.Galaxy`\n")
                      )

    def generate_lnu(self, *args, **kwargs):
        raise Warning(("generate_lnu should be overloaded by child classes:\n"
                       "`particle.galaxy.Galaxy`\n"
                       "`parametric.galaxy.Galaxy`\n"
                       "You should not be seeing this!!!")
                      )
    
    def get_spectra_incident(
            self,
            grid,
            young=False,
            old=False,
            label='',
            update=True
    ):
        """
        Generate the incident (equivalent to pure stellar for stars) spectra 
        using the provided Grid.

        Args:
            grid (obj):
                spectral grid object
            update (bool):
                flag for whether to update the `stellar` spectra
                inside the galaxy object `spectra` dictionary.
                These are the combined values of young and old.
            young (bool, float):
                if not False, specifies age in Myr at which to filter
                for young star particles
            old (bool, float):
                if not False, specifies age in Myr at which to filter
                for old star particles

        Returns:
            An Sed object containing the stellar spectra
        """

        lnu = self.generate_lnu(grid, 'incident', young=young, old=old)

        sed = Sed(grid.lam, lnu)

        if update:
            self.spectra[label+'incident'] = sed

        return sed
    
    def get_spectra_transmitted(
        self,
        grid,
        fesc=0.0,
        young=False,
        old=False,
        label='',
        update=True
    ):
        """
        Generate the transmitted spectra using the provided Grid. This is the
        emission which is transmitted through the gas as calculated by the
        photoionisation code.

        Args:
            grid (obj):
                spectral grid object
            fesc (float):
                fraction of stellar emission that escapeds unattenuated from
                the birth cloud (defaults to 0.0)
            update (bool):
                flag for whether to update the `stellar` spectra
                inside the galaxy object `spectra` dictionary.
                These are the combined values of young and old.
            young (bool, float):
                if not False, specifies age in Myr at which to filter
                for young star particles
            old (bool, float):
                if not False, specifies age in Myr at which to filter
                for old star particles

        Returns:
            An Sed object containing the transmitted spectra
        """

        lnu = (1.-fesc)*self.generate_lnu(grid, 'transmitted', young=young, old=old)

        sed = Sed(grid.lam, lnu)

        if update:
            self.spectra[label+'transmitted'] = sed

        return sed

    def get_spectra_nebular(
            self,
            grid,
            fesc=0.0, 
            update=True,
            young=False,
            label='',
            old=False
    ):
        """
        Generate nebular spectra from a grid object and star particles.
        The grid object must contain a nebular component.

        Args:
            grid (obj):
                spectral grid object
            fesc (float):
                fraction of stellar emission that escapeds unattenuated from
                the birth cloud (defaults to 0.0)
            update (bool):
                flag for whether to update the `intrinsic` and `attenuated`
                spectra inside the galaxy object `spectra` dictionary.
                These are the combined values of young and old.
            young (bool, float):
                if not False, specifies age in Myr at which to filter
                for young star particles
            old (bool, float):
                if not False, specifies age in Myr at which to filter
                for old star particles

        Returns:
            An Sed object containing the nebular spectra
        """

        lnu = self.generate_lnu(grid, 'nebular', young=young, old=old)

        lnu *= (1-fesc)

        sed = Sed(grid.lam, lnu)

        if update:
            self.spectra[label+'nebular'] = sed

        return sed

    def get_spectra_reprocessed(
            self,
            grid,
            fesc=0.0,
            fesc_LyA=1.0,
            young=False,
            old=False,
            label='',
            update=True
    ):
        """
        Generates the intrinsic spectra, this is the sum of the escaping 
        radiation (if fesc>0), the transmitted emission, and the nebular 
        emission. The transmitted emission is the emission that is
        transmitted through the gas. In addition to returning the intrinsic
        spectra this saves the incident, nebular, and escaped spectra if 
        update is set to True.

        Args:
            grid (obj):
                spectral grid object
            fesc (float):
                fraction of stellar emission that escapeds unattenuated from
                the birth cloud (defaults to 0.0)
            fesc_LyA (float):
                fraction of Lyman-alpha emission that can escape unimpeded 
                by the ISM/IGM.
            update (bool):
                flag for whether to update the `intrinsic`, `stellar` and
                `nebular` spectra inside the galaxy object `spectra`
                dictionary. These are the combined values of young and old.
            young (bool, float):
                if not False, specifies age in Myr at which to filter
                for young star particles
            old (bool, float):
                if not False, specifies age in Myr at which to filter
                for old star particles

        Updates:
            incident:
            transmitted
            nebular
            reprocessed
            intrinsic

            if fesc>0:
                escaped
        
        Returns:
            An Sed object containing the intrinsic spectra.
        """

        # the incident emission 
        incident = self.get_spectra_incident(grid, update=update,
                                           young=young, old=old, label=label)
        
        # the emission which escapeds the gas
        if fesc > 0:
            escaped = Sed(grid.lam, fesc*incident._lnu)

        # the stellar emission which **is** reprocessed by the gas
        transmitted = self.get_spectra_transmitted(grid, fesc, update=update,
                                           young=young, old=old, label=label)        
        # the nebular emission 
        nebular = self.get_spectra_nebular(grid, fesc, update=update,
                                           young=young, old=old, label=label)

        
        # if the Lyman-alpha escape fraction is <1.0 suppress it.
        if fesc_LyA < 1.0:

            # get the new line contribution to the spectrum
            linecont = self.reduce_lya(grid, fesc_LyA)

            # get the nebular continuum emission
            nebular_continuum = self.generate_lnu(grid, 'nebular_continuum', young=young, old=old)

            # redefine the nebular emission
            nebular._lnu = linecont._lnu + nebular_continuum._lnu

        # the reprocessed emission, the sum of transmitted, and nebular
        reprocessed = nebular + transmitted

        # the intrinsic emission, the sum of escaped, transmitted, and nebular
        # if escaped exists other its simply the reprocessed
        if fesc > 0:
            intrinsic = reprocessed + escaped
        else:
            intrinsic = reprocessed 

        if update:
            if fesc > 0:
                self.spectra[label+'escaped'] = escaped
            self.spectra[label+'reprocessed'] = reprocessed
            self.spectra[label+'intrinsic'] = intrinsic

        return reprocessed

    def get_spectra_screen(
            self,
            grid,
            tau_v=None,
            dust_curve=power_law({'slope': -1.}),
            young=False,
            old=False,                      
            update=True
    ):
        """
        Calculates dust attenuated spectra assuming a simple screen. 
        This is disfavoured over using the pacman model but will be
        computationally faster. Note: this implicitly assumes fesc=0.0.

        Args:
            grid (object, Grid):
                The spectral grid
            tau_v (float):
                numerical value of dust attenuation
            dust_curve (object)
                instance of dust_curve
            young (bool, float):
                if not False, specifies age in Myr at which to filter
                for young star particles
            old (bool, float):
                if not False, specifies age in Myr at which to filter
                for old star particles                
            update (bool):
                flag for whether to update the `attenuated` spectra inside
                the galaxy object `spectra` dictionary. These are the
                combined values of young and old.

        Returns:
            An Sed object containing the dust attenuated spectra
        """

        # generate intrinsic spectra using full star formation and metal 
        # enrichment history or all particles
        # generates: 
        #   - incident
        #   - escaped
        #   - transmitted
        #   - nebular
        #   - reprocessed = transmitted + nebular
        #   - intrinsic = transmitted + reprocessed
        self.get_spectra_reprocessed(grid, update=update, fesc=0.0, young=young, old=old)

        if tau_v:
            T = dust_curve.attenuate(tau_v, grid.lam)
        else:
            T = 1.0

        emergent = Sed(grid.lam, T * self.spectra['intrinsic']._lnu)

        if update:
            self.spectra['emergent'] = emergent

        return emergent
    


    def get_spectra_pacman(
            self,
            grid,
            dust_curve=power_law(),
            tau_v=1.,
            alpha=-1.,
            young_old_thresh=None,
            fesc=0.0,
            fesc_LyA=1.0,
    ):
        """
        Calculates dust attenuated spectra assuming the PACMAN dust/fesc model
        including variable Lyman-alpha transmission. In this model some
        fraction (fesc) of the pure stellar emission is able to completely
        escaped the galaxy without reprocessing by gas or dust. The rest is
        assumed to be reprocessed by both gas and a screen of dust. If 
        young_old_thresh is set then the individual and combined spectra will 
        be generated for both young and old components. In this case it's 
        necessary to provide an array of tau_v and alphas dscribing the ISM
        birth cloud components respectively. The young component feels 
        attenuation from both the ISM and birth cloud while the old component
        only feels attenuation from the ISM.

        Args:
            grid  (object, Grid):
                The spectral grid
            dust_curve (object):
                instance of dust_curve
            fesc :(float):
                Lyman continuum escaped fraction
            fesc_LyA (float):
                Lyman-alpha escaped fraction
            tau_v (float):
                numerical value of dust attenuation
            alpha (float):
                numerical value of the dust curve slope
            young_old_thresh (float)
                numerical value of threshold from young to old

                

        Raises:
            InconsistentArguments:
                Errors when more than two values for tau_v and alpha is
                passed for CF00 dust screen. In case of single dust
                screen, raises error for multiple optical depths or dust
                curve slope.

        Updates:
            incident
            escaped
            transmitted
            nebular
            reprocessed
            intrinsic
            attenuated
            emergent

            if CF00:
                young_incident
                young_escaped
                young_transmitted
                young_nebular
                young_reprocessed
                young_intrinsic
                young_attenuated
                young_emergent
                old_incident
                old_escaped
                old_transmitted
                old_nebular
                old_reprocessed
                old_intrinsic
                old_attenuated
                old_emergent

        Returns:
            obj (Sed):
                A Sed object containing the emergent spectra
        """

        if young_old_thresh:
            if (len(tau_v)>2) or (len(alpha)>2):
                exceptions.InconsistentArguments(
                    ("Only 2 values for the optical depth or dust curve "
                     "slope are allowed for the CF00 model"))
        else:
            if isinstance(tau_v, (list, tuple, np.ndarray))\
                or isinstance(alpha, (list, tuple, np.ndarray)):
                exceptions.InconsistentArguments(
                    ("Only single value supported for tau_v and alpha in "
                     "case of single dust screen"))


        # initialise output spectra
        self.spectra['attenuated'] = Sed(grid.lam)
        self.spectra['emergent'] = Sed(grid.lam)

        # generate intrinsic spectra using full star formation and metal 
        # enrichment history or all particles
        # generates: 
        #   - incident
        #   - escaped
        #   - transmitted
        #   - nebular
        #   - reprocessed = transmitted + nebular
        #   - intrinsic = transmitted + reprocessed
        self.get_spectra_reprocessed(grid, fesc, fesc_LyA=fesc_LyA, young=False, old=False)

        if young_old_thresh:

            self.spectra['young_attenuated'] = Sed(grid.lam)
            self.spectra['old_attenuated'] = Sed(grid.lam)
            self.spectra['young_emergent'] = Sed(grid.lam)
            self.spectra['old_emergent'] = Sed(grid.lam)
            
            # generate the young gas reprocessed spectra
            # add a label so saves e.g. 'escaped_young' etc.
            self.get_spectra_reprocessed(grid, fesc, fesc_LyA=fesc_LyA, young=young_old_thresh, old=False, label = 'young_')

            # generate the old gas reprocessed spectra
            # add a label so saves e.g. 'escaped_old' etc.
            self.get_spectra_reprocessed(grid, fesc, fesc_LyA=fesc_LyA, young=False, old=young_old_thresh, label = 'old_')

            print(self.spectra.keys())

        
        if np.isscalar(tau_v):
            # single screen dust, no separate birth cloud attenuation
            dust_curve.params['slope'] = alpha

            # calculate dust attenuation
            T = dust_curve.attenuate(tau_v, grid.lam)
            
            # calculate the attenuated emission
            self.spectra['attenuated']._lnu = \
                T*self.spectra['reprocessed']._lnu
            
                        
        elif np.isscalar(tau_v) == False:

            """
            Apply separate attenuation to both the young and old components.  
            """

            # Two screen dust, one for diffuse other for birth cloud dust.
            if np.isscalar(alpha):
                print(("Separate dust curve slopes for diffuse and "
                        "birth cloud dust not given"))
                print(("Defaulting to alpha_ISM=-0.7 and alpha_BC=-1.4"
                        " (Charlot & Fall 2000)"))
                alpha = [-0.7, -1.4]
            
            dust_curve.params['slope']=alpha[0]
            T_ISM = dust_curve.attenuate(tau_v[0], grid.lam)
            
            dust_curve.params['slope']=alpha[1]
            T_BC = dust_curve.attenuate(tau_v[1], grid.lam)

            T_young = T_ISM * T_BC
            T_old = T_ISM

            self.spectra['young_attenuated']._lnu =  T_young * self.spectra['young_reprocessed']._lnu
            self.spectra['old_attenuated']._lnu =  T_old * self.spectra['old_reprocessed']._lnu

            self.spectra['attenuated']._lnu = self.spectra['young_attenuated']._lnu + self.spectra['old_attenuated']._lnu

            # set emergent spectra
            if not fesc > 0:
                self.spectra['young_emergent']._lnu = self.spectra['young_attenuated']._lnu
                self.spectra['old_emergent']._lnu = self.spectra['old_attenuated']._lnu       
            else:
                self.spectra['young_emergent']._lnu = self.spectra['young_escaped']._lnu + self.spectra['young_attenuated']._lnu
                self.spectra['old_emergent']._lnu = self.spectra['old_escaped']._lnu + self.spectra['old_attenuated']._lnu
            

        if not fesc > 0:
            self.spectra['emergent']._lnu = self.spectra['attenuated']._lnu
        else:
            self.spectra['emergent']._lnu = self.spectra['escaped']._lnu + self.spectra['attenuated']._lnu


        return self.spectra['emergent']
    

    def get_spectra_CharlotFall(
            self,
            grid,
            tau_v_ISM=1.,
            tau_v_BC=1.,
            alpha_ISM=-0.7,
            alpha_BC=-1.3,
            young_old_thresh=7.,
    ):
        """
        Calculates dust attenuated spectra assuming the Charlot & Fall (2000)
        dust model. In this model young star particles are embedded in a
        dusty birth cloud and thus feel more dust attenuation. This is a wrapper
        around our more generic pacman method.

        Parameters
        ----------
        grid : obj (Grid)
            The spectral frid
        tau_v_ISM: float
            numerical value of dust attenuation due to the ISM in the V-band
        tau_v_BC: float
            numerical value of dust attenuation due to the BC in the V-band
        alpha_ISM: float
            slope of the ISM dust curve, -0.7 in MAGPHYS
        alpha_BC: float
            slope of the BC dust curve, -1.3 in MAGPHYS
        young_old_thresh: float
            the threshold in log10(age/yr) for young/old stellar populations
      
        Returns
        -------
        obj (Sed)
             A Sed object containing the dust attenuated spectra
        """

        return self.get_spectra_pacman(grid, 
                                  fesc = 0,
                                  fesc_LyA=1,
                                  dust_curve=power_law(),
                                  tau_v = [tau_v_ISM, tau_v_BC], 
                                  alpha = [alpha_ISM, alpha_BC],
                                  young_old_thresh=young_old_thresh
                                  )



    def get_spectra_dust(self, emissionmodel):

        """
        Calculates dust emission spectra using the attenuated and intrinsic spectra that have already been generated and an emission model.

        Parameters
        ----------
        emissionmodel : obj 
            The spectral frid
    
        Returns
        -------
        obj (Sed)
             A Sed object containing the dust attenuated spectra
        """

        # use wavelength grid from attenuated spectra
        # NOTE: in future it might be good to allow a custom wavelength grid

        lam = self.spectra['attenuated'].lam

        # calculate the bolometric dust lunminosity as the difference between the intrinsic and attenuated

        dust_bolometric_luminosity = self.spectra['intrinsic'].get_bolometric_luminosity() - self.spectra['attenuated'].get_bolometric_luminosity()

        # get the spectrum and normalise it properly
        lnu = dust_bolometric_luminosity.to('erg/s').value * emissionmodel.lnu(lam)

        # create new Sed object containing dust spectra
        sed = Sed(lam, lnu=lnu)

        # associate that with the component's spectra dictionarity
        self.spectra['dust'] = sed
        self.spectra['total'] = self.spectra['dust'] + self.spectra['attenuated']

        return sed






    def get_line_intrinsic(
            self,
            grid,
            line_ids,
            fesc=0.0,
            update=True
    ):
        """
        Calculates **intrinsic** properties (luminosity, continuum, EW)
        for a set of lines.

        Args:
            grid (object, Grid):
                A Grid object
            line_ids (list or str):
                A list of line_ids or a str denoting a single line.
                Doublets can be specified as a nested list or using a
                comma (e.g. 'OIII4363,OIII4959')
            fesc (float):
                The Lyman continuum escaped fraction, the fraction of
                ionising photons that entirely escaped

        Returns:
            lines (dictionary-like, object):
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
                luminosity = np.sum((1-fesc)*grid_line['luminosity'] *
                                    self.sfzh.sfzh, axis=(0, 1))

                #  continuum at line wavelength, erg/s/Hz
                continuum = np.sum(grid_line['continuum'] *
                                   self.sfzh.sfzh, axis=(0, 1))

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
                        (1-fesc)*np.sum(grid_line['luminosity'] *
                                        self.sfzh.sfzh, axis=(0, 1)))

                    #  continuum at line wavelength, erg/s/Hz
                    continuum.append(np.sum(grid_line['continuum'] *
                                            self.sfzh.sfzh, axis=(0, 1)))

            else:
                # throw exception
                pass

            line = Line(line_id, wavelength, luminosity, continuum)
            lines[line.id] = line

        if update:
            self.lines[line.id] = line

        return lines

    def get_line_attenuated(
            self,
            grid,
            line_ids,
            fesc=0.0,
            tau_v_nebular=None,
            tau_v_stellar=None,
            dust_curve_nebular=power_law({'slope': -1.}),
            dust_curve_stellar=power_law({'slope': -1.}),
            update=True
    ):
        """
        Calculates attenuated properties (luminosity, continuum, EW) for a
        set of lines. Allows the nebular and stellar attenuation to be set
        separately.

        Parameters
        ----------
        grid : obj (Grid)
            The Grid
        line_ids : list or str
            A list of line_ids or a str denoting a single line. Doublets can be
            specified as a nested list or using a comma
            (e.g. 'OIII4363,OIII4959')
        fesc : float
            The Lyman continuum escaped fraction, the fraction of
            ionising photons that entirely escaped
        tau_v_nebular : float
            V-band optical depth of the nebular emission
        tau_v_stellar : float
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
            intrinsic_lines = self.get_line_intrinsic(grid, line_ids,
                                                      fesc=fesc, update=update)
        else:
            intrinsic_lines = self.lines['intrinsic']

        # dictionary holding lines
        lines = {}

        for line_id, intrinsic_line in intrinsic_lines.items():

            # calculate attenuation
            T_nebular = dust_curve_nebular.attenuate(
                tau_v_nebular, intrinsic_line._wavelength)
            T_stellar = dust_curve_stellar.attenuate(
                tau_v_stellar, intrinsic_line._wavelength)

            luminosity = intrinsic_line._luminosity * T_nebular
            continuum = intrinsic_line._continuum * T_stellar

            line = Line(intrinsic_line.id, intrinsic_line._wavelength, 
                        luminosity, continuum)

            # NOTE: the above is wrong and should be separated into stellar
            # and nebular continuum components:
            # nebular_continuum = intrinsic_line._nebular_continuum * T_nebular
            # stellar_continuum = intrinsic_line._stellar_continuum * T_stellar
            # line = Line(intrinsic_line.id, intrinsic_line._wavelength, luminosity, nebular_continuum, stellar_continuum)

            lines[line.id] = line

        if update:
            self.lines['attenuated'] = lines

        return lines

    def get_line_screen(
            self,
            grid,
            line_ids,
            fesc=0.0,
            tau_v=None,
            dust_curve=power_law({'slope': -1.}),
            update=True
    ):
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
                The Lyman continuum escaped fraction, the fraction of
                ionising photons that entirely escaped
            tau_v : float
                V-band optical depth
            dust_curve : obj (dust_curve)
                A dust_curve object specifying the dust curve for
                the nebular emission

        Returns:
            lines : dictionary-like (obj)
                A dictionary containing line objects.
        """

        return self.get_line_attenuated(
            grid, line_ids, fesc=fesc,
            tau_v_nebular=tau_v, tau_v_stellar=tau_v,
            dust_curve_nebular=dust_curve,
            dust_curve_stellar=dust_curve)   

    def T(self):
        """
        Calculate transmission as a function of wavelength

        Returns:
            transmission (array)
        """

        return self.spectra['attenuated'].lam, self.spectra['attenuated'].lnu/self.spectra['intrinsic'].lnu

    def Al(self):
        """
        Calculate attenuation as a function of wavelength
        
        Returns:
            attenuation (array)
        """

        lam, T = self.T()

        return lam, -2.5*np.log10(T)

    def A(self, l):
        """
        Calculate attenuation at a given wavelength

        Returns:
            attenuation (float)
        """

        lam, Al = self.Al()

        return np.interp(l, lam, Al)

    def AV(self):
        """
        Calculate rest-frame FUV attenuation
        
        Returns:
            attenuation at rest-frame 1500 angstrom (float)
        """

        return self.A(5500.)

    def A1500(self):
        """
        Calculate rest-frame FUV attenuation
        
        Returns:
            attenuation at rest-frame 1500 angstrom (float)
        """

        return self.A(1500.)

    def plot_spectra(self, show=False, 
                     spectra_to_plot=None, 
                     ylimits=('peak',5),
                     figsize=(3.5, 5)):
        """
        plots all spectra associated with a galaxy object
        
        Args:
            show (bool):
                flag for whether to show the plot or just return the
                figure and axes
            spectra_to_plot (None, list):
                list of named spectra to plot that are present in
                `galaxy.spectra`
            figsize (tuple):
                tuple with size 2 defining the figure size

        Returns:
            fig (object)
            ax (object)
        """

        fig = plt.figure(figsize=figsize)

        left = 0.15
        height = 0.6
        bottom = 0.1
        width = 0.8

        ax = fig.add_axes((left, bottom, width, height))

        if type(spectra_to_plot) != list:
            spectra_to_plot = list(self.spectra.keys())

        # only plot FIR if 'total' is plotted otherwise just plot UV-NIR
        if 'total' in spectra_to_plot:
            xlim = [1., 7.]
        else:
            xlim = [1., 4.5]

        ypeak = -100
        for sed_name in spectra_to_plot:
            sed = self.spectra[sed_name]
            ax.plot(np.log10(sed.lam), np.log10(sed.lnu), lw=1, alpha=0.8, label=sed_name)

            if np.max(np.log10(sed.lnu))> ypeak:
                ypeak = np.max(np.log10(sed.lnu))

        # ax.set_xlim([2.5, 4.2])

        if ylimits[0] == 'peak':
            if ypeak == ypeak:
                ylim = [ypeak-ylimits[1], ypeak+0.5]
            ax.set_ylim(ylim)

        ax.set_xlim(xlim)

        ax.legend(fontsize=8, labelspacing=0.0)
        ax.set_xlabel(r'$\rm log_{10}(\lambda/\AA)$')
        ax.set_ylabel(r'$\rm log_{10}(L_{\nu}/erg\ s^{-1}\ Hz^{-1} M_{\odot}^{-1})$')

        if show:
            plt.show()

        return fig, ax

    def plot_observed_spectra(self, cosmo, z, 
                              fc=None, 
                              show=False, 
                              spectra_to_plot=None,
                              figsize=(3.5, 5.),
                              verbose=True):
        """ 
        plots all spectra associated with a galaxy object

        Args:

        Returns:
        """

        fig = plt.figure(figsize=figsize)

        left = 0.15
        height = 0.7
        bottom = 0.1
        width = 0.8

        ax = fig.add_axes((left, bottom, width, height))
        filter_ax = ax.twinx()

        if type(spectra_to_plot) != list:
            spectra_to_plot = list(self.spectra.keys())

        for sed_name in spectra_to_plot:
            sed = self.spectra[sed_name]
            sed.get_fnu(cosmo, z)
            ax.plot(sed.obslam, sed.fnu, lw=1, alpha=0.8, label=sed_name)
            print(sed_name)

            if fc:
                sed.get_broadband_fluxes(fc, verbose=verbose)
                for f in fc:
                    wv = f.pivwv()
                    filter_ax.plot(f.lam, f.t)
                    ax.scatter(wv, sed.broadband_fluxes[f.filter_code], zorder=4)

        # ax.set_xlim([5000., 100000.])
        # ax.set_ylim([0., 100])
        filter_ax.set_ylim([3,  0])
        ax.legend(fontsize=8, labelspacing=0.0)
        ax.set_xlabel(r'$\rm log_{10}(\lambda_{obs}/\AA)$')
        ax.set_ylabel(r'$\rm log_{10}(f_{\nu}/nJy)$')

        if show:
            plt.show()

        return fig, ax # , filter_ax
