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

    def __init__(self):
        raise Warning(("Instantiating a BaseGalaxy object is not "
                       "supported behaviour. Instead, you should "
                       "use one of the derived Galaxy classes:\n"
                       "`particle.galaxy.Galaxy`\n"
                       "`parametric.galaxy.Galaxy`\n")
                      )

    def generate_lnu(self):
        raise Warning(("generate_lnu should be overloaded by child classes:\n"
                       "`particle.galaxy.Galaxy`\n"
                       "`parametric.galaxy.Galaxy`\n"
                       "You should not be seeing this!!!")
                      )

    def get_spectra_stellar(
            self,
            grid,
            young=False,
            old=False,
            update=True
    ):
        """
        Generate the pure stellar spectra using the provided Grid

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

        lnu = self.generate_lnu(grid, 'stellar', young=young, old=old)

        sed = Sed(grid.lam, lnu)

        if update:
            self.spectra['stellar'] = sed

        return sed
    
    def get_spectra_incident(
            self,
            grid,
            young=False,
            old=False,
            update=True
    ):
        """
        Generate the incident (pure stellar for stars) spectra using the provided Grid

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
            self.spectra['incident'] = sed

        return sed
    
    def get_spectra_transmitted(
        self,
        grid,
        young=False,
        old=False,
        update=True
    ):
        """
        Generate the transmitted spectra using the provided Grid

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

        lnu = self.generate_lnu(grid, 'transmitted', young=young, old=old)

        sed = Sed(grid.lam, lnu)

        if update:
            self.spectra['transmitted'] = sed

        return sed

    def get_spectra_nebular(
            self,
            grid,
            fesc=0.0, 
            update=True,
            young=False,
            old=False
    ):
        """
        Generate nebular spectra from a grid object and star particles.
        The grid object must contain a nebular component.

        Args:
            grid (obj):
                spectral grid object
            fesc (float):
                fraction of stellar emission that escapes unattenuated from
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
            self.spectra['nebular'] = sed

        return sed

    def get_spectra_intrinsic(
            self,
            grid,
            fesc=0.0,
            young=False,
            old=False,
            update=True
    ):
        """
        Generates the intrinsic spectra, i.e. the combination of the
         stellar and nebular emission, but not including dust.

        Args:
            grid (obj):
                spectral grid object
            fesc (float):
                fraction of stellar emission that escapes unattenuated from
                the birth cloud (defaults to 0.0)
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

        Returns:
            An Sed object containing the intrinsic spectra
        """

        stellar = self.get_spectra_transmitted(grid, update=update,
                                           young=young, old=old)
        nebular = self.get_spectra_nebular(grid, fesc, update=update,
                                           young=young, old=old)

        sed = Sed(grid.lam, stellar._lnu + nebular._lnu)

        if update:
            self.spectra['intrinsic'] = sed

        return sed

    def get_spectra_screen(
            self,
            grid,
             fesc=0.0,
            tauV=None,
            dust_curve=power_law({'slope': -1.}),
            young=False,
            old=False,                      
            update=True
    ):
        """
        Calculates dust attenuated spectra assuming a simple screen

        Args:
            grid (object, Grid):
                The spectral grid
            fesc (float):
                fraction of stellar emission that escapes unattenuated from
                the birth cloud (defaults to 0.0)
            tauV (float):
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

        # --- begin by calculating intrinsic spectra
        intrinsic = self.get_spectra_intrinsic(grid, update=update,
                                               fesc=fesc, young=young, old=old)

        if tauV:
            T = dust_curve.attenuate(tauV, grid.lam)
        else:
            T = 1.0

        sed = Sed(grid.lam, T * intrinsic._lnu)

        if update:
            self.spectra['attenuated'] = sed

        return sed

    def get_spectra_pacman(
            self,
            grid,
            dust_curve=power_law(),
            tauV=1.,
            alpha=-1.,
            CF00=False,
            old=7.,
            young=7.,
            save_young_and_old=False,
            spectra_name='total',
            fesc=0.0,
            fesc_LyA=1.0,
            update=True,
    ):
        """
        Calculates dust attenuated spectra assuming the PACMAN dust/fesc model
        including variable Lyman-alpha transmission. In this model some
        fraction (fesc) of the pure stellar emission is able to completely
        escape the galaxy without reprocessing by gas or dust. The rest is
        assumed to be reprocessed by both gas and a screen of dust.

        CF00 allows to toggle extra attenaution for birth clouds
        following Charlot & Fall 2000.

        Args:
            grid  (object, Grid):
                The spectral grid
            dust_curve (object):
                instance of dust_curve
            fesc :(float):
                Lyman continuum escape fraction
            fesc_LyA (float):
                Lyman-alpha escape fraction
            tauV (float):
                numerical value of dust attenuation
            alpha (float):
                numerical value of the dust curve slope
            CF00 (bool):
                Toggle Charlot and Fall 2000 dust attenuation
                Requires two values for tauV and alpha

        Raises:
            InconsistentArguments:
                Errors when more than two values for tauV and alpha is
                passed for CF00 dust screen. In case of single dust
                screen, raises error for multiple optical depths or dust
                curve slope.

        Returns:
            obj (Sed):
                A Sed object containing the dust attenuated spectra
        """

        if CF00:
            if (len(tauV)>2) or (len(alpha)>2):
                exceptions.InconsistentArguments(
                    ("Only 2 values for the optical depth or dust curve "
                     "slope are allowed for CF00"))
        else:
            if isinstance(tauV, (list, tuple, np.ndarray))\
                or isinstance(alpha, (list, tuple, np.ndarray)):
                exceptions.InconsistentArguments(
                    ("Only single value supported for tauV and alpha in "
                     "case of single dust screen"))

        # --- begin by generating the pure stellar spectra
        self.spectra['stellar'] = self.get_spectra_stellar(grid, update=update)

        # --- this is the starlight that escapes any reprocessing
        self.spectra['escape'] = \
            Sed(grid.lam, fesc * self.spectra['stellar']._lnu)

        # --- this is the starlight after reprocessing by gas
        self.spectra['reprocessed_intrinsic'] = Sed(grid.lam)
        self.spectra['reprocessed_nebular'] = Sed(grid.lam)
        self.spectra['reprocessed_attenuated'] = Sed(grid.lam)
        self.spectra['total'] = Sed(grid.lam)

        if CF00:
            # splitting contribution from old and young stars
            nebcont_old = self.generate_lnu(
                grid, spectra_name='nebular_continuum',
                old=old, young=False)

            nebcont_yng = self.generate_lnu(
                grid, spectra_name='nebular_continuum',
                old=False, young=young)

            transmitted_old = self.generate_lnu(
                grid, spectra_name='transmitted',
                old=old, young=False)

            transmitted_yng = self.generate_lnu(
                grid, spectra_name='transmitted',
                old=False, young=young)
        else:
            nebcont = np.sum(grid.spectra['nebular_continuum'] * self.sfzh_,
                             axis=(0, 1))
            transmitted = np.sum(grid.spectra['transmitted'] * self.sfzh_,
                                 axis=(0, 1))

        if fesc_LyA < 1.0:
            # if Lyman-alpha escape fraction is specified reduce LyA luminosity
            if CF00:
                linecont_old = self.reduce_lya(grid, fesc_LyA, 
                                               young=False, old=old)
                linecont_yng = self.reduce_lya(grid, fesc_LyA, 
                                               young=young, old=False)

                reprocessed_intrinsic_old = (1.-fesc) * \
                    (linecont_old + nebcont_old + transmitted_old)
                reprocessed_intrinsic_yng = (1.-fesc) * \
                    (linecont_yng + nebcont_yng + transmitted_yng)

                reprocessed_nebular_old = (1.-fesc) * \
                    (linecont_old + nebcont_old)
                reprocessed_nebular_yng = (1.-fesc) * \
                    (linecont_yng + nebcont_yng)

                self.spectra['reprocessed_intrinsic']._lnu = \
                    reprocessed_intrinsic_old + reprocessed_intrinsic_yng
                self.spectra['reprocessed_nebular']._lnu = \
                    reprocessed_nebular_old + reprocessed_nebular_yng
                
            else:
                linecont = self.reduce_lya(grid, fesc_LyA)

                self.spectra['reprocessed_intrinsic']._lnu = (
                    1.-fesc) * (linecont + nebcont + transmitted)
                self.spectra['reprocessed_nebular']._lnu = (
                    1.-fesc) * (linecont + nebcont)

        if np.isscalar(tauV):
            # single screen dust, no separate birth cloud attenuation
            dust_curve.params['slope'] = alpha

            # calculate dust attenuation
            T = dust_curve.attenuate(tauV, grid.lam)
            
            self.spectra['reprocessed_attenuated']._lnu = \
                T*self.spectra['reprocessed_intrinsic']._lnu
            
            self.spectra['total']._lnu = self.spectra['escape']._lnu + \
                self.spectra['reprocessed_attenuated']._lnu
            
            # self.spectra['total']._lnu = self.spectra['attenuated']._lnu
        
        elif np.isscalar(tauV) == False:
            # Two screen dust, one for diffuse other for birth cloud dust.
            if np.isscalar(alpha):
                print(("Separate dust curve slopes for diffuse and "
                        "birth cloud dust not given"))
                print(("Defaulting to alpha_ISM=-0.7 and alpha_BC=-1.4"
                        " (Charlot & Fall 2000)"))
                alpha = [-0.7, -1.4]

            self.spectra['reprocessed_attenuated'] = Sed(grid.lam)
            
            dust_curve.params['slope']=alpha[0]
            T_ISM = dust_curve.attenuate(tauV[0], grid.lam)
            
            dust_curve.params['slope']=alpha[1]
            T_BC = dust_curve.attenuate(tauV[1], grid.lam)

            T_yng = T_ISM * T_BC
            T_old = T_ISM

            self.spectra['reprocessed_attenuated']._lnu = \
                reprocessed_intrinsic_old * T_old + \
                reprocessed_intrinsic_yng * T_yng

            self.spectra['total']._lnu = self.spectra['escape']._lnu + \
                self.spectra['reprocessed_attenuated']._lnu

        else:
            self.spectra['total']._lnu = self.spectra['escape']._lnu + \
                self.spectra['reprocessed_intrinsic']._lnu
            
        if save_young_and_old:
            self.spectra['reprocessed_intrinsic_old'] = \
                Sed(grid.lam, reprocessed_intrinsic_old)
            self.spectra['reprocessed_intrinsic_yng'] = \
                Sed(grid.lam, reprocessed_intrinsic_yng)
            self.spectra['reprocessed_nebular_old'] = \
                Sed(grid.lam, reprocessed_nebular_old)
            self.spectra['reprocessed_nebular_yng'] = \
                Sed(grid.lam, reprocessed_nebular_yng)
        
        return self.spectra['total']
    
    def get_spectra_one_component(
            self,
            grid,
            dust_curve,
            tauV,
            alpha,
            old=False,
            young=False,
            spectra_name='total'
    ):
        """
        One component dust screen model,
        with option for optical depths and
        dust attenuation slopes

        Args:

        Returns:

        """

        # calculate intrinsic sed for young and old stars
        intrinsic_sed = self.generate_lnu(grid, spectra_name=spectra_name,
                                          old=old, young=young)

        if np.isscalar(tauV):
            dust_curve.params['slope'] = alpha
            T_total = dust_curve.attenuate(tauV, grid.lam)
        else:
            T_total = np.ones(len(intrinsic_sed))
            for _tauV, _alpha in zip(tauV, alpha):
                dust_curve.params['slope'] = _alpha
                _T = dust_curve.attenuate(_tauV, grid.lam)
                T_total *= _T

        return intrinsic_sed * T_total

    def get_spectra_CharlotFall(
            self,
            grid,
            dust_curve=power_law(),
            tauV_ISM=1.,
            tauV_BC=1.,
            alpha_ISM=-0.7,
            alpha_BC=-1.3,
            old=7.,
            young=7.,
            save_young_and_old=False,
            spectra_name='total',
            update=True
    ):
        """
        Calculates dust attenuated spectra assuming the Charlot & Fall (2000)
        dust model. In this model young star particles are embedded in a
        dusty birth cloud and thus feel more dust attenuation.

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

        if spectra_name not in grid.spectra.keys():
            print(F"{spectra_name} not in the data spectra grids\n")
            print("Available spectra_name values in grid are: ", 
                   grid.spectra.keys())
            ValueError(F"{spectra_name} not in the data spectra grids\n")

        sed_young = self.get_spectra_one_component(
            grid, dust_curve, tauV=[tauV_ISM, tauV_BC], 
            alpha=[alpha_ISM, alpha_BC], 
            old=False, young=young, spectra_name=spectra_name)
        
        sed_old = self.get_spectra_one_component(
            grid, dust_curve, tauV=tauV_ISM, alpha=alpha_ISM, 
            old=old, young=False, spectra_name=spectra_name)

        if save_young_and_old:
            # calculate intrinsic sed for young and old stars
            intrinsic_sed_young = self.generate_lnu(
                grid, spectra_name=spectra_name, old=False, young=young)
            intrinsic_sed_old = self.generate_lnu(
                grid, spectra_name=spectra_name, old=old, young=False)

            self.spectra['intrinsic_young'] = intrinsic_sed_young
            self.spectra['intrinsic_old'] = intrinsic_sed_old

            self.spectra['attenuated_young'] = Sed(grid.lam, sed_young)
            self.spectra['attenuated_old'] = Sed(grid.lam, sed_old)

        sed = Sed(grid.lam, sed_young + sed_old)

        if update:
            self.spectra['attenuated'] = sed

        return sed


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

        print(lam)

        # calculate the bolometric dust lunminosity as the difference between the intrinsic and attenuated

        dust_bolometric_luminosity = self.spectra['intrinsic'].get_bolometric_luminosity() - self.spectra['attenuated'].get_bolometric_luminosity()

        # get the spectrum and normalise it properly
        lnu = dust_bolometric_luminosity.to('erg/s').value * emissionmodel.lnu(lam)

        print(lnu)

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
                The Lyman continuum escape fraction, the fraction of
                ionising photons that entirely escape

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
            tauV_nebular=None,
            tauV_stellar=None,
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
            intrinsic_lines = self.get_line_intrinsic(grid, line_ids,
                                                      fesc=fesc, update=update)
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
            tauV=None,
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

        return self.get_line_attenuated(
            grid, line_ids, fesc=fesc,
            tauV_nebular=tauV, tauV_stellar=tauV,
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

    def A1500(self):
        """
        Calculate rest-frame FUV attenuation
        
        Returns:
            attenuation at rest-frame 1500 angstrom (float)
        """

        return self.A(1500.)

    def plot_spectra(self, show=False, 
                     spectra_to_plot=None, 
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

        for sed_name in spectra_to_plot:
            sed = self.spectra[sed_name]
            ax.plot(np.log10(sed.lam), np.log10(sed.lnu), lw=1, alpha=0.8, label=sed_name)

        # ax.set_xlim([2.5, 4.2])
        # ax.set_ylim([27., 29.5])
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
