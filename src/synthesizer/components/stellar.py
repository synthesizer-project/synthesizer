""" A module for particle and parametric stars.

This should never be directly instantiated. Instead it is the parent class for
particle.Stars and parametric.Stars and contains attributes
and methods common between them.
"""
import numpy as np
import matplotlib.pyplot as plt
from unyt import Myr, unyt_quantity

from synthesizer import exceptions
from synthesizer.dust.attenuation import PowerLaw
from synthesizer.line import Line, LineCollection
from synthesizer.sed import Sed
from synthesizer.units import Quantity


class StarsComponent:
    """
    The parent class for stellar components of a galaxy.

    This class contains the attributes and spectra creation methods which are
    common to both parametric and particle stellar components.

    This should never be instantiated directly.

    Attributes:

    """

    # Define slots
    # __slots__ = [
    #     "_ages",
    #     "metallicities",
    #     "spectra",
    #     "lines",
    # ]

    # Define quantities
    ages = Quantity()

    def __init__(
        self,
        ages,
        metallicities,
    ):
        """
        Initialise the StellarComponent.

        Args:
            ages (array-like, float)
                The age of each stellar particle/bin (particle/parametric).
            metallicities (array-like, float)
                The metallicity of each stellar particle/bin
                (particle/parametric).
        """

        # Define the spectra dictionary to hold the stellar spectra
        self.spectra = {}

        # Define the line dictionary to hold the stellar emission lines
        self.lines = {}

        # The common stellar attributes between particle and parametric stars
        self.ages = ages
        self.metallicities = metallicities

    def generate_lnu(self, *args, **kwargs):
        """
        This method is a prototype for generating spectra from SPS grids. It is
        redefined on the child classes.
        """
        raise Warning(
            (
                "generate_lnu should be overloaded by child classes:\n"
                "`particle.Stars`\n"
                "`parametric.Stars`\n"
                "You should not be seeing this!!!"
            )
        )

    def generate_line(self, *args, **kwargs):
        """
        This method is a prototype for generating lines from SPS grids. It is
        redefined on the child classes.
        """
        raise Warning(
            (
                "generate_line should be overloaded by child classes:\n"
                "`particle.Stars`\n"
                "`parametric.Stars`\n"
                "You should not be seeing this!!!"
            )
        )

    def get_spectra_linecont(
        self,
        grid,
        fesc=0.0,
        fesc_LyA=1.0,
        young=False,
        old=False,
    ):
        """
        Generate the line contribution spectra. This is only invoked if
        fesc_LyA < 1.

        Args:
            grid (obj):
                Spectral grid object.
            fesc (float):
                Fraction of stellar emission that escapeds unattenuated from
                the birth cloud (defaults to 0.0).
            fesc_LyA (float)
                Fraction of Lyman-alpha emission that can escape unimpeded
                by the ISM/IGM.
            young (bool, float):
                If not False, specifies age in Myr at which to filter
                for young star particles.
            old (bool, float):
                If not False, specifies age in Myr at which to filter
                for old star particles.

        Returns:
            numpy.ndarray
                The line contribution spectra.
        """

        # Make sure young and old in Myr, if provided
        young, old = self.check_young_old_units(young, old)

        # Generate contribution of line emission alone and reduce the
        # contribution of Lyman-alpha
        linecont = self.generate_lnu(
            grid, spectra_name="linecont", old=old, young=young
        )

        # Multiply by the Lyamn-continuum escape fraction
        linecont *= 1 - fesc

        # Get index of Lyman-alpha
        idx = grid.get_nearest_index(1216.0, grid.lam)
        linecont[idx] *= fesc_LyA  # reduce the contribution of Lyman-alpha

        return linecont

    def get_spectra_incident(self, grid, young=False, old=False, label=""):
        """
        Generate the incident (equivalent to pure stellar for stars) spectra
        using the provided Grid.

        Args:
            grid (obj):
                Spectral grid object.
            young (bool, float):
                If not False, specifies age in Myr at which to filter
                for young star particles.
            old (bool, float):
                If not False, specifies age in Myr at which to filter
                for old star particles.
            label (string)
                A modifier for the spectra dictionary key such that the
                key is label + "_incident".

        Returns:
            Sed
                An Sed object containing the stellar spectra.
        """

        # Make sure young and old in Myr, if provided
        young, old = self.check_young_old_units(young, old)

        # Get the incident spectra
        lnu = self.generate_lnu(grid, "incident", young=young, old=old)

        # Create the Sed object
        sed = Sed(grid.lam, lnu)

        # Store the Sed object
        self.spectra[label + "incident"] = sed

        return sed

    def get_spectra_transmitted(
        self,
        grid,
        fesc=0.0,
        young=False,
        old=False,
        label="",
    ):
        """
        Generate the transmitted spectra using the provided Grid. This is the
        emission which is transmitted through the gas as calculated by the
        photoionisation code.

        Args:
            grid (obj):
                Spectral grid object.
            fesc (float):
                Fraction of stellar emission that escapeds unattenuated from
                the birth cloud (defaults to 0.0).
            young (bool, float):
                If not False, specifies age in Myr at which to filter
                for young star particles.
            old (bool, float):
                If not False, specifies age in Myr at which to filter
                for old star particles.
            label (string)
                A modifier for the spectra dictionary key such that the
                key is label + "_transmitted".

        Returns:
            Sed
                An Sed object containing the transmitted spectra.
        """

        # Make sure young and old in Myr, if provided
        young, old = self.check_young_old_units(young, old)

        # Get the transmitted spectra
        lnu = (1.0 - fesc) * self.generate_lnu(
            grid, "transmitted", young=young, old=old
        )

        # Create the Sed object
        sed = Sed(grid.lam, lnu)

        # Store the Sed object
        self.spectra[label + "transmitted"] = sed

        return sed

    def get_spectra_nebular(
        self,
        grid,
        fesc=0.0,
        young=False,
        old=False,
        label="",
    ):
        """
        Generate nebular spectra from a grid object and star particles.
        The grid object must contain a nebular component.

        Args:
            grid (obj):
                Spectral grid object.
            fesc (float):
                Fraction of stellar emission that escapeds unattenuated from
                the birth cloud (defaults to 0.0).
            young (bool, float):
                If not False, specifies age in Myr at which to filter
                for young star particles.
            old (bool, float):
                If not False, specifies age in Myr at which to filter
                for old star particles.
            label (string)
                A modifier for the spectra dictionary key such that the
                key is label + "_nebular".

        Returns:
            Sed
                An Sed object containing the nebular spectra.
        """

        # Make sure young and old in Myr, if provided
        young, old = self.check_young_old_units(young, old)

        # Get the nebular emission spectra
        lnu = self.generate_lnu(grid, "nebular", young=young, old=old)

        # Apply the escape fraction
        lnu *= 1 - fesc

        # Create the Sed object
        sed = Sed(grid.lam, lnu)

        # Store the Sed object
        self.spectra[label + "nebular"] = sed

        return sed

    def get_spectra_reprocessed(
        self,
        grid,
        fesc=0.0,
        fesc_LyA=1.0,
        young=False,
        old=False,
        label="",
    ):
        """
        Generates the intrinsic spectra, this is the sum of the escaping
        radiation (if fesc>0), the transmitted emission, and the nebular
        emission. The transmitted emission is the emission that is
        transmitted through the gas. In addition to returning the intrinsic
        spectra this saves the incident, nebular, and escaped spectra.

        Args:
            grid (obj):
                Spectral grid object.
            fesc (float):
                Fraction of stellar emission that escapeds unattenuated from
                the birth cloud (defaults to 0.0).
            fesc_LyA (float)
                Fraction of Lyman-alpha emission that can escape unimpeded
                by the ISM/IGM.
            young (bool, float):
                If not False, specifies age in Myr at which to filter
                for young star particles.
            old (bool, float):
                If not False, specifies age in Myr at which to filter
                for old star particles.
            label (string)
                A modifier for the spectra dictionary key such that the
                key is label + "_transmitted".

        Updates:
            incident:
            transmitted
            nebular
            reprocessed
            intrinsic

            if fesc>0:
                escaped

        Returns:
            Sed
                An Sed object containing the intrinsic spectra.
        """

        # Make sure young and old in Myr, if provided
        young, old = self.check_young_old_units(young, old)

        # The incident emission
        incident = self.get_spectra_incident(
            grid,
            young=young,
            old=old,
            label=label,
        )

        # The emission which escapes the gas
        if fesc > 0:
            escaped = Sed(grid.lam, fesc * incident._lnu)

        # The stellar emission which **is** reprocessed by the gas
        transmitted = self.get_spectra_transmitted(
            grid, fesc, young=young, old=old, label=label
        )

        # The nebular emission
        nebular = self.get_spectra_nebular(
            grid, fesc, young=young, old=old, label=label
        )

        # If the Lyman-alpha escape fraction is <1.0 suppress it.
        if fesc_LyA < 1.0:
            # Get the new line contribution to the spectrum
            linecont = self.get_spectra_linecont(
                grid,
                fesc=fesc,
                fesc_LyA=fesc_LyA,
            )

            # Get the nebular continuum emission
            nebular_continuum = self.generate_lnu(
                grid, "nebular_continuum", young=young, old=old
            )
            nebular_continuum *= 1 - fesc

            # Redefine the nebular emission
            nebular._lnu = linecont + nebular_continuum

        # The reprocessed emission, the sum of transmitted, and nebular
        reprocessed = nebular + transmitted

        # The intrinsic emission, the sum of escaped, transmitted, and nebular
        # if escaped exists other its simply the reprocessed
        if fesc > 0:
            intrinsic = reprocessed + escaped
        else:
            intrinsic = reprocessed

        if fesc > 0:
            self.spectra[label + "escaped"] = escaped
        self.spectra[label + "reprocessed"] = reprocessed
        self.spectra[label + "intrinsic"] = intrinsic

        return reprocessed

    def get_spectra_screen(
        self,
        grid,
        tau_v,
        dust_curve=PowerLaw(slope=-1.0),
        young=False,
        old=False,
        label="",
    ):
        """
        Calculates dust attenuated spectra assuming a simple screen.
        This is disfavoured over using the pacman model but will be
        computationally faster. Note: this implicitly assumes fesc=0.0.

        Args:
            grid (object, Grid):
                The spectral grid object.
            tau_v (float):
                The V-band optical depth.
            dust_curve (object)
                Instance of a dust_curve from synthesizer.dust.attenuation.
            young (bool, float):
                If not False, specifies age in Myr at which to filter
                for young star particles.
            old (bool, float):
                If not False, specifies age in Myr at which to filter
                for old star particles.
            label (string)
                A modifier for the spectra dictionary key such that the
                key is label + "_transmitted".

        Returns:
            Sed
                An Sed object containing the dust attenuated spectra.
        """

        # Make sure young and old in Myr, if provided
        young, old = self.check_young_old_units(young, old)

        # Generate intrinsic spectra using full star formation and metal
        # enrichment history or all particles
        # generates:
        #   - incident
        #   - escaped
        #   - transmitted
        #   - nebular
        #   - reprocessed = transmitted + nebular
        #   - intrinsic = transmitted + reprocessed
        self.get_spectra_reprocessed(
            grid,
            fesc=0.0,
            young=young,
            old=old,
            label=label,
        )

        # Apply the dust screen
        emergent = self.spectra["intrinsic"].apply_attenuation(
            tau_v, dust_curve=dust_curve
        )

        # Store the Sed object
        self.spectra["emergent"] = emergent

        return emergent

    def get_spectra_pacman(
        self,
        grid,
        tau_v=1.0,
        dust_curve=PowerLaw(),
        alpha=-1.0,
        young_old_thresh=None,
        fesc=0.0,
        fesc_LyA=1.0,
        label="",
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
            grid (object, Grid):
                The spectral grid object.
            tau_v (float):
                The V-band optical depth.
            dust_curve (object)
                Instance of a dust_curve from synthesizer.dust.attenuation.
            alpha (float):
                The dust curve slope.
            young_old_thresh (float):
                The threshold in log10(age/yr) for young/old stellar
                populations.
            fesc :(float):
                Lyman continuum escaped fraction.
            fesc_LyA (float):
                Lyman-alpha escaped fraction.
            label (string)
                A modifier for the spectra dictionary key such that the
                key is label + "_transmitted".

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
            Sed
                A Sed object containing the emergent spectra.
        """

        # Ensure we have a compatible set of parameters
        if young_old_thresh:
            if (len(tau_v) > 2) or (len(alpha) > 2):
                raise exceptions.InconsistentArguments(
                    (
                        "Only 2 values for the optical depth or dust curve "
                        "slope are allowed for the CF00 model"
                    )
                )
        else:
            # If we don't have an age threshold we can't have multiple values
            # for tau_v and alpha
            if isinstance(tau_v, (list, tuple, np.ndarray)) or isinstance(
                alpha, (list, tuple, np.ndarray)
            ):
                raise exceptions.InconsistentArguments(
                    "Only singular values are supported for tau_v and alpha in "
                    "a single dust screen situation."
                )

        # Generate intrinsic spectra for young and old particles
        # separately before summing them if we have been given
        # a threshold
        if young_old_thresh is not None:
            # Generates:
            #   - young_incident
            #   - young_escaped
            #   - young_transmitted
            #   - young_nebular
            #   - young_reprocessed = young_transmitted + young_nebular
            #   - young_intrinsic = young_transmitted + young_reprocessed
            #   - old_incident
            #   - old_escaped
            #   - old_transmitted
            #   - old_nebular
            #   - old_reprocessed = old_transmitted + old_nebular
            #   - old_intrinsic = old_transmitted + old_reprocessed
            #   - incident
            #   - escaped
            #   - transmitted
            #   - nebular
            #   - reprocessed = transmitted + nebular
            #   - intrinsic = transmitted + reprocessed

            # Generate the young gas reprocessed spectra
            # add a label so saves e.g. 'escaped_young' etc.
            self.get_spectra_reprocessed(
                grid,
                fesc,
                fesc_LyA=fesc_LyA,
                young=young_old_thresh,
                old=False,
                label="young_",
            )

            # Generate the old gas reprocessed spectra
            # add a label so saves e.g. 'escaped_old' etc.
            self.get_spectra_reprocessed(
                grid,
                fesc,
                fesc_LyA=fesc_LyA,
                young=False,
                old=young_old_thresh,
                label="old_",
            )

            # Combine young and old spectra
            self.spectra["incident"] = (
                self.spectra["young_incident"] + self.spectra["old_incident"]
            )
            self.spectra["transmitted"] = (
                self.spectra["young_transmitted"] + self.spectra["old_transmitted"]
            )
            self.spectra["nebular"] = (
                self.spectra["young_nebular"] + self.spectra["old_nebular"]
            )
            self.spectra["reprocessed"] = (
                self.spectra["young_reprocessed"] + self.spectra["old_reprocessed"]
            )
            self.spectra["intrinsic"] = (
                self.spectra["young_intrinsic"] + self.spectra["old_intrinsic"]
            )
            if fesc > 0:
                self.spectra["escaped"] = (
                    self.spectra["young_escaped"] + self.spectra["old_escaped"]
                )
        else:
            # Generate intrinsic spectra for all particles
            # Generates:
            #   - incident
            #   - escaped
            #   - transmitted
            #   - nebular
            #   - reprocessed = transmitted + nebular
            #   - intrinsic = transmitted + reprocessed
            self.get_spectra_reprocessed(
                grid, fesc, fesc_LyA=fesc_LyA, young=False, old=False
            )

        if np.isscalar(tau_v):
            # Single screen dust, no separate birth cloud attenuation
            dust_curve.slope = alpha

            # Calculate the attenuated emission
            attenuated = self.spectra["reprocessed"].apply_attenuation(
                tau_v, dust_curve=dust_curve
            )
            self.spectra["attenuated"] = attenuated

        elif np.isscalar(tau_v) is False:
            # Apply separate attenuation to both the young and old components.

            # Two screen dust, one for diffuse ISM, the other for birth cloud
            # dust.
            if np.isscalar(alpha):
                print(
                    (
                        "Separate dust curve slopes for diffuse and "
                        "birth cloud dust not given"
                    )
                )
                print(("Defaulting to dust_curve.slope (-1 if unmodified)"))
                alpha = [dust_curve.slope, dust_curve.slope]

            # Overwrite Nones with the dust_curve.slope value
            if alpha[0] is None:
                alpha[0] = dust_curve.slope
            if alpha[1] is None:
                alpha[1] = dust_curve.slope

            # Calculate attenuated spectra of young stars
            dust_curve.slope = alpha[1]  # use the BC slope
            young_attenuated = self.spectra["young_reprocessed"].apply_attenuation(
                tau_v[1], dust_curve=dust_curve
            )
            dust_curve.slope = alpha[0]  # use the ISM slope
            young_attenuated = young_attenuated.apply_attenuation(
                tau_v[0], dust_curve=dust_curve
            )
            self.spectra["young_attenuated"] = young_attenuated

            # Calculate attenuated spectra of old stars
            old_attenuated = self.spectra["old_reprocessed"].apply_attenuation(
                tau_v[0], dust_curve=dust_curve
            )
            self.spectra["old_attenuated"] = old_attenuated

            # Get the combined attenuated spectra
            self.spectra["attenuated"] = young_attenuated + old_attenuated

            # Set emergent spectra based on fesc (for young and old particles)
            self.spectra["young_emergent"] = Sed(grid.lam)
            self.spectra["old_emergent"] = Sed(grid.lam)
            if fesc <= 0:
                self.spectra["young_emergent"]._lnu = self.spectra[
                    "young_attenuated"
                ]._lnu
                self.spectra["old_emergent"]._lnu = self.spectra[
                    "old_attenuated"
                ]._lnu
            else:
                self.spectra["young_emergent"]._lnu = (
                    self.spectra["young_escaped"]._lnu
                    + self.spectra["young_attenuated"]._lnu
                )
                self.spectra["old_emergent"]._lnu = (
                    self.spectra["old_escaped"]._lnu
                    + self.spectra["old_attenuated"]._lnu
                )

        # Set emergent spectra based on fesc (for all particles)
        self.spectra["emergent"] = Sed(grid.lam)
        if fesc <= 0:
            self.spectra["emergent"]._lnu = self.spectra["attenuated"]._lnu
        else:
            self.spectra["emergent"]._lnu = (
                self.spectra["escaped"]._lnu + self.spectra["attenuated"]._lnu
            )

        return self.spectra["emergent"]

    def get_spectra_CharlotFall(
        self,
        grid,
        tau_v_ISM=1.0,
        tau_v_BC=1.0,
        alpha_ISM=None,
        alpha_BC=None,
        young_old_thresh=7.0,
    ):
        """
        Calculates dust attenuated spectra assuming the Charlot & Fall (2000)
        dust model. In this model young star particles are embedded in a
        dusty birth cloud and thus feel more dust attenuation. This is a
        wrapper around our more generic pacman method.

        Args:
            grid (Grid)
                The spectral grid object.
            tau_v_ISM (float)
                The ISM optical depth in the V-band.
            tau_v_BC (float)
                The birth cloud optical depth in the V-band.
            alpha_ISM (float)
                The slope of the ISM dust curve, (defaults to dust_curve.slope=-1,
                Recommended: -0.7 from MAGPHYS)
            alpha_BC (float)
                The slope of the birth cloud dust curve, (defaults to
                dust_curve.slope=-1, Recommended: -1.3 from MAGPHYS)
            young_old_thresh (float)
                The threshold in log10(age/yr) for young/old stellar
                populations.

        Returns:
            Sed
                A Sed object containing the dust attenuated spectra.
        """

        return self.get_spectra_pacman(
            grid,
            fesc=0,
            fesc_LyA=1,
            dust_curve=PowerLaw(),
            tau_v=[tau_v_ISM, tau_v_BC],
            alpha=[alpha_ISM, alpha_BC],
            young_old_thresh=young_old_thresh,
        )

    def get_line_intrinsic(self, grid, line_ids, fesc=0.0):
        """
        Calculates the intrinsic properties (luminosity, continuum, EW)
        for a set of lines.

        Args:
            grid (Grid):
                A Grid object.
            line_ids (list/str):
                A list of line_ids or a str denoting a single line.
                Doublets can be specified as a nested list or using a
                comma (e.g. 'OIII4363,OIII4959').
            fesc (float):
                The Lyman continuum escaped fraction, the fraction of
                ionising photons that entirely escaped.

        Returns:
            LineCollection
                A dictionary like object containing line objects.
        """

        # If only one line specified convert to a list to avoid writing a
        # longer if statement
        if isinstance(line_ids, str):
            line_ids = [line_ids, ]

        # Dictionary holding Line objects
        lines = {}

        # Loop over the lines
        for line_id in line_ids:
            # If the line id a doublet in string form
            # (e.g. 'OIII4959,OIII5007') convert it to a list
            if isinstance(line_id, str):
                if len(line_id.split(",")) > 1:
                    line_id = line_id.split(",")

            # Compute the line object
            line = self.generate_line(grid=grid, line_id=line_id, fesc=fesc)

            # Store this line
            lines[line.id] = line

        # Create a line collection
        line_collection = LineCollection(lines)

        # Associate that line collection with the Stars object
        self.lines["intrinsic"] = line_collection

        return line_collection

    def get_line_attenuated(
        self,
        grid,
        line_ids,
        fesc=0.0,
        tau_v_BC=None,
        tau_v_ISM=None,
        dust_curve_BC=PowerLaw(slope=-1.0),
        dust_curve_ISM=PowerLaw(slope=-1.0),
    ):
        """
        Calculates attenuated properties (luminosity, continuum, EW) for a
        set of lines. Allows the nebular and stellar attenuation to be set
        separately.

        Args:
            grid (Grid)
                The Grid object.
            line_ids (list/str)
                A list of line_ids or a str denoting a single line. Doublets can be
                specified as a nested list or using a comma
                (e.g. 'OIII4363,OIII4959').
            fesc (float)
                The Lyman continuum escaped fraction, the fraction of
                ionising photons that entirely escaped.
            tau_v_BS (float)
                V-band optical depth of the nebular emission.
            tau_v_ISM (float)
                V-band optical depth of the stellar emission.
            dust_curve_BC (dust_curve)
                A dust_curve object specifying the dust curve.
                for the nebular emission
            dust_curve_ISM (dust_curve)
                A dust_curve object specifying the dust curve
                for the stellar emission.

        Returns:
            LineCollection
                A dictionary like object containing line objects.
        """

        # If the intrinsic lines haven't already been calculated and saved
        # then generate them
        if "intrinsic" not in self.lines:
            intrinsic_lines = self.get_line_intrinsic(
                grid, line_ids, fesc=fesc,
            )
        else:
            intrinsic_lines = self.lines["intrinsic"]

        # Dictionary holding lines
        lines = {}

        # Loop over the intrinsic lines
        for line_id, intrinsic_line in intrinsic_lines.lines.items():
            # Calculate attenuation
            T_BC = dust_curve_BC.get_transmission(
                tau_v_BC, intrinsic_line._wavelength
            )
            T_ISM = dust_curve_ISM.get_transmission(
                tau_v_ISM, intrinsic_line._wavelength
            )

            # Apply attenuation
            luminosity = intrinsic_line._luminosity * T_BC * T_ISM
            continuum = intrinsic_line._continuum * T_ISM

            # Create the line object
            line = Line(
                line_id,
                intrinsic_line._wavelength,
                luminosity,
                continuum,
            )

            # NOTE: the above is wrong and should be separated into stellar
            # and nebular continuum components:
            # nebular_continuum = intrinsic_line._nebular_continuum * T_nebular
            # stellar_continuum = intrinsic_line._stellar_continuum * T_stellar
            # line = Line(intrinsic_line.id, intrinsic_line._wavelength,
            # luminosity, nebular_continuum, stellar_continuum)

            lines[line_id] = line

        # Create a line collection
        line_collection = LineCollection(lines)

        # Associate that line collection with the Stars object
        self.lines["intrinsic"] = line_collection

        return line_collection

    def get_line_screen(
        self,
        grid,
        line_ids,
        fesc=0.0,
        tau_v=None,
        dust_curve=PowerLaw(slope=-1.0),
    ):
        """
        Calculates attenuated properties (luminosity, continuum, EW) for a set
        of lines assuming a simple dust screen (i.e. both nebular and stellar
        emission feels the same dust attenuation). This is a wrapper around
        the more general method above.

        Args:
            grid (Grid)
                The Grid object.
            line_ids (list/str)
                A list of line_ids or a str denoting a single line. Doublets can be
                specified as a nested list or using a comma
                (e.g. 'OIII4363,OIII4959').
            fesc (float)
                The Lyman continuum escaped fraction, the fraction of
                ionising photons that entirely escaped.
            tau_v (float)
                V-band optical depth.
            dust_curve (dust_curve)
                A dust_curve object specifying the dust curve.

        Returns:
            LineCollection
                A dictionary like object containing line objects.
        """

        return self.get_line_attenuated(
            grid,
            line_ids,
            fesc=fesc,
            tau_v_BC=0,
            tau_v_ISM=tau_v,
            dust_curve_nebular=dust_curve,
            dust_curve_stellar=dust_curve,
        )

    def check_young_old_units(self, young, old):
        """
        Checks whether the `young` and `old` arguments to many
        spectra generation methods are in the right units (Myr)

        Args:
            young (bool, float):
                If not False, specifies age in Myr at which to filter
                for young star particles.
            old (bool, float):
                If not False, specifies age in Myr at which to filter
                for old star particles.
        """

        if young:
            if isinstance(young, (unyt_quantity)):
                young = young.to("Myr")
            else:
                young *= Myr

        if old:
            if isinstance(old, (unyt_quantity)):
                old = old.to("Myr")
            else:
                old *= Myr

        return young, old

    def plot_spectra(
            self,
            show=False,
            spectra_to_plot=None,
            ylimits=("peak", 5),
            figsize=(3.5, 5),
    ):
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

        if not isinstance(spectra_to_plot, list):
            spectra_to_plot = list(self.spectra.keys())

        # only plot FIR if 'total' is plotted otherwise just plot UV-NIR
        if "total" in spectra_to_plot:
            xlim = [2.0, 7.0]
        else:
            xlim = [2.0, 4.5]

        ypeak = -100
        for sed_name in spectra_to_plot:
            sed = self.spectra[sed_name]
            ax.plot(
                np.log10(sed.lam), np.log10(sed.lnu), lw=1, alpha=0.8, label=sed_name
            )

            if np.max(np.log10(sed.lnu)) > ypeak:
                ypeak = np.max(np.log10(sed.lnu))

        # ax.set_xlim([2.5, 4.2])

        if ylimits[0] == "peak":
            if ypeak == ypeak:
                ylim = [ypeak - ylimits[1], ypeak + 0.5]
            ax.set_ylim(ylim)

        ax.set_xlim(xlim)

        ax.legend(fontsize=8, labelspacing=0.0)
        ax.set_xlabel(r"$\rm log_{10}(\lambda/\AA)$")
        ax.set_ylabel(r"$\rm log_{10}(L_{\nu}/erg\ s^{-1}\ Hz^{-1} M_{\odot}^{-1})$")

        if show:
            plt.show()

        return fig, ax

    def plot_observed_spectra(
        self,
        cosmo,
        z,
        fc=None,
        show=False,
        spectra_to_plot=None,
        figsize=(3.5, 5.0),
        verbose=True,
    ):
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

        if not isinstance(spectra_to_plot, list):
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
        filter_ax.set_ylim([3, 0])
        ax.legend(fontsize=8, labelspacing=0.0)
        ax.set_xlabel(r"$\rm log_{10}(\lambda_{obs}/\AA)$")
        ax.set_ylabel(r"$\rm log_{10}(f_{\nu}/nJy)$")

        if show:
            plt.show()

        return fig, ax
