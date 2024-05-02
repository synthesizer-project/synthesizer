"""A module for particle and parametric stars.

This should never be directly instantiated. Instead it is the parent class for
particle.Stars and parametric.Stars and contains attributes
and methods common between them.
"""

import numpy as np
from unyt import Lsun, Myr, unyt_quantity

from synthesizer import exceptions
from synthesizer.dust.attenuation import PowerLaw
from synthesizer.line import Line, LineCollection
from synthesizer.sed import Sed, plot_spectra
from synthesizer.units import Quantity


class StarsComponent:
    """
    The parent class for stellar components of a galaxy.

    This class contains the attributes and spectra creation methods which are
    common to both parametric and particle stellar components.

    This should never be instantiated directly.

    Attributes:

    """

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

        # Define the photometry dictionaries to hold the stellar photometry
        self.photo_luminosities = {}
        self.photo_fluxes = {}

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

    def get_spectra_nebular_continuum(
        self,
        grid,
        fesc=0.0,
        fesc_LyA=1.0,
        young=None,
        old=None,
        **kwargs,
    ):
        """
        Generate the nebular continuum spectra. This is only invoked if
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
            young (unyt_quantity):
                If not None, specifies age in Myr at which to filter
                for young star particles.
            old (unyt_quantity):
                If not None, specifies age in Myr at which to filter
                for old star particles.
            kwargs
                Any keyword arguments which can be passed to
                generate_lnu.

        Returns:
            numpy.ndarray
                The line contribution spectra.
        """

        # Make sure young and old in Myr, if provided
        young, old = self._check_young_old_units(young, old)

        # Generate contribution of line emission alone and reduce the
        # contribution of Lyman-alpha
        nebcont = self.generate_lnu(
            grid,
            spectra_name="nebular_continuum",
            old=old,
            young=young,
            **kwargs,
        )

        # Multiply by the Lyamn-continuum escape fraction
        nebcont *= 1 - fesc

        # Get index of Lyman-alpha
        idx = grid.get_nearest_index(1216.0, grid.lam)
        nebcont[idx] *= fesc_LyA  # reduce the contribution of Lyman-alpha

        return nebcont

    def get_spectra_linecont(
        self,
        grid,
        fesc=0.0,
        fesc_LyA=1.0,
        young=None,
        old=None,
        **kwargs,
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
            young (unyt_quantity):
                If not None, specifies age in Myr at which to filter
                for young star particles.
            old (unyt_quantity):
                If not None, specifies age in Myr at which to filter
                for old star particles.
            kwargs
                Any keyword arguments which can be passed to
                generate_lnu.

        Returns:
            numpy.ndarray
                The line contribution spectra.
        """

        # Make sure young and old in Myr, if provided
        young, old = self._check_young_old_units(young, old)

        # Generate contribution of line emission alone and reduce the
        # contribution of Lyman-alpha
        linecont = self.generate_lnu(
            grid,
            spectra_name="linecont",
            old=old,
            young=young,
            **kwargs,
        )

        # Multiply by the Lyamn-continuum escape fraction
        linecont *= 1 - fesc

        # Get index of Lyman-alpha
        idx = grid.get_nearest_index(1216.0, grid.lam)
        linecont[idx] *= fesc_LyA  # reduce the contribution of Lyman-alpha

        return linecont

    def get_spectra_incident(
        self,
        grid,
        young=None,
        old=None,
        label="",
        **kwargs,
    ):
        """
        Generate the incident (equivalent to pure stellar for stars) spectra
        using the provided Grid.

        Args:
            grid (obj):
                Spectral grid object.
            young (unyt_quantity):
                If not None, specifies age in Myr at which to filter
                for young star particles.
            old (unyt_quantity):
                If not None, specifies age in Myr at which to filter
                for old star particles.
            label (string)
                A modifier for the spectra dictionary key such that the
                key is label + "_incident".
            kwargs
                Any keyword arguments which can be passed to
                generate_lnu.

        Returns:
            Sed
                An Sed object containing the stellar spectra.
        """

        # Make sure young and old in Myr, if provided
        young, old = self._check_young_old_units(young, old)

        # Get the incident spectra
        lnu = self.generate_lnu(
            grid,
            "incident",
            young=young,
            old=old,
            **kwargs,
        )

        # Create the Sed object
        sed = Sed(grid.lam, lnu)

        # Store the Sed object
        self.spectra[label + "incident"] = sed

        return sed

    def get_spectra_transmitted(
        self,
        grid,
        fesc=0.0,
        young=None,
        old=None,
        label="",
        **kwargs,
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
            young (unyt_quantity):
                If not None, specifies age in Myr at which to filter
                for young star particles.
            old (unyt_quantity):
                If not None, specifies age in Myr at which to filter
                for old star particles.
            label (string)
                A modifier for the spectra dictionary key such that the
                key is label + "_transmitted".
            kwargs
                Any keyword arguments which can be passed to
                generate_lnu.

        Returns:
            Sed
                An Sed object containing the transmitted spectra.
        """

        # Make sure young and old in Myr, if provided
        young, old = self._check_young_old_units(young, old)

        # Get the transmitted spectra
        lnu = (1.0 - fesc) * self.generate_lnu(
            grid,
            "transmitted",
            young=young,
            old=old,
            **kwargs,
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
        young=None,
        old=None,
        label="",
        **kwargs,
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
            young (unyt_quantity):
                If not None, specifies age in Myr at which to filter
                for young star particles.
            old (unyt_quantity):
                If not None, specifies age in Myr at which to filter
                for old star particles.
            label (string)
                A modifier for the spectra dictionary key such that the
                key is label + "_nebular".
            kwargs
                Any keyword arguments which can be passed to
                generate_lnu.

        Returns:
            Sed
                An Sed object containing the nebular spectra.
        """

        # Make sure young and old in Myr, if provided
        young, old = self._check_young_old_units(young, old)

        # Get the nebular emission spectra
        lnu = self.generate_lnu(
            grid,
            "nebular",
            young=young,
            old=old,
            **kwargs,
        )

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
        young=None,
        old=None,
        label="",
        verbose=False,
        **kwargs,
    ):
        """
        Generates the intrinsic spectra, this is the sum of the escaping
        radiation (if fesc>0), the transmitted emission, and the nebular
        emission. The transmitted emission is the emission that is
        transmitted through the gas. In addition to returning the intrinsic
        spectra this saves the incident, nebular, and escaped spectra.

        Note, if a grid that has not been post-processed through a
        photoionisation code is provided (i.e. grid.reprocessed=False) this
        will just call `get_spectra_incident`.

        Args:
            grid (obj):
                Spectral grid object.
            fesc (float):
                Fraction of stellar emission that escapeds unattenuated from
                the birth cloud (defaults to 0.0).
            fesc_LyA (float)
                Fraction of Lyman-alpha emission that can escape unimpeded
                by the ISM/IGM.
            young (unyt_quantity):
                If not None, specifies age in Myr at which to filter
                for young star particles.
            old (unyt_quantity):
                If not None, specifies age in Myr at which to filter
                for old star particles.
            label (string):
                A modifier for the spectra dictionary key such that the
                key is label + "_transmitted".
            verbose (bool):
                verbose output flag
            kwargs
                Any keyword arguments which can be passed to
                generate_lnu.

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
        young, old = self._check_young_old_units(young, old)

        # Check if grid has been run through a photoionisation code
        if not grid.reprocessed:
            if verbose:
                print(
                    (
                        "The grid you are using has not been post-processed "
                        "through a photoionisation code. This method will "
                        "just return the incident stellar emission. Are "
                        "you sure this is the method you want to use?"
                    )
                )

            spec = self.get_spectra_incident(
                grid=grid,
                young=young,
                old=old,
                label=label,
                **kwargs,
            )

            self.spectra[f"{label}intrinsic"] = self.spectra[
                f"{label}incident"
            ]
            return spec

        # The incident emission
        incident = self.get_spectra_incident(
            grid,
            young=young,
            old=old,
            label=label,
            **kwargs,
        )

        # The emission which escapes the gas
        if fesc > 0:
            escaped = Sed(grid.lam, fesc * incident._lnu)

        # The stellar emission which **is** reprocessed by the gas
        transmitted = self.get_spectra_transmitted(
            grid,
            fesc,
            young=young,
            old=old,
            label=label,
            **kwargs,
        )

        # The nebular emission
        nebular = self.get_spectra_nebular(
            grid,
            fesc,
            young=young,
            old=old,
            label=label,
            **kwargs,
        )

        # If the Lyman-alpha escape fraction is <1.0 suppress it.
        if fesc_LyA < 1.0:
            # Get the new line contribution to the spectrum
            linecont = self.get_spectra_linecont(
                grid,
                fesc=fesc,
                fesc_LyA=fesc_LyA,
                **kwargs,
            )

            # Get the nebular continuum emission
            nebular_continuum = self.generate_lnu(
                grid,
                "nebular_continuum",
                young=young,
                old=old,
                **kwargs,
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
            self.spectra[f"{label}escaped"] = escaped
        self.spectra[f"{label}reprocessed"] = reprocessed
        self.spectra[f"{label}intrinsic"] = intrinsic

        return reprocessed

    def get_spectra_screen(
        self,
        grid,
        tau_v,
        dust_curve=PowerLaw(slope=-1.0),
        young=None,
        old=None,
        label="",
        **kwargs,
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
            young (unyt_quantity):
                If not None, specifies age in Myr at which to filter
                for young star particles.
            old (unyt_quantity):
                If not None, specifies age in Myr at which to filter
                for old star particles.
            label (string)
                A modifier for the spectra dictionary key such that the
                key is label + "_transmitted".
            kwargs
                Any keyword arguments which can be passed to
                generate_lnu.

        Returns:
            Sed
                An Sed object containing the dust attenuated spectra.
        """

        # Make sure young and old in Myr, if provided
        young, old = self._check_young_old_units(young, old)

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
            **kwargs,
        )

        # Apply the dust screen
        emergent = self.spectra[f"{label}intrinsic"].apply_attenuation(
            tau_v, dust_curve=dust_curve
        )

        # Store the Sed object
        self.spectra[f"{label}emergent"] = emergent

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
        dust_emission_model=None,
        label="",
        **kwargs,
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
            young_old_thresh (unyt_quantity):
                The threshold for young/old stellar populations with units.
            fesc :(float):
                Lyman continuum escaped fraction.
            fesc_LyA (float):
                Lyman-alpha escaped fraction.
            label (string)
                A modifier for the spectra dictionary key such that the
                key is label + "_transmitted".
            dust_emisison_model (object)
                Instance of a dust_emission_model from
                synthesizer.dust.emission.
            kwargs
                Any keyword arguments which can be passed to
                generate_lnu.

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

            if dust_emission_model provided:
                dust
                total

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

                if dust_emission_model provided:
                    young_dust
                    young_total
                    old_dust
                    old_total

        Returns:
            Sed
                A Sed object containing the emergent or total spectra depending
                on whether a dust_emission_model is provided.
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
                    "Only singular values are supported for tau_v and alpha "
                    "in a single dust screen situation."
                )

        # If grid has photoinoisation outputs, use the reprocessed outputs
        if grid.reprocessed:
            reprocessed_name = "reprocessed"
        else:  # otherwise just use the intrinsic stellar spectra
            reprocessed_name = "intrinsic"

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
                old=None,
                label="young_",
                **kwargs,
            )

            # Generate the old gas reprocessed spectra
            # add a label so saves e.g. 'escaped_old' etc.
            self.get_spectra_reprocessed(
                grid,
                fesc,
                fesc_LyA=fesc_LyA,
                young=None,
                old=young_old_thresh,
                label="old_",
                **kwargs,
            )

            # Combine young and old spectra (only if the grid has been
            # reprocessed through photoionisation code)
            if grid.reprocessed:
                self.spectra["incident"] = (
                    self.spectra["young_incident"]
                    + self.spectra["old_incident"]
                )
                self.spectra["transmitted"] = (
                    self.spectra["young_transmitted"]
                    + self.spectra["old_transmitted"]
                )
                self.spectra["nebular"] = (
                    self.spectra["young_nebular"] + self.spectra["old_nebular"]
                )
                self.spectra["reprocessed"] = (
                    self.spectra["young_reprocessed"]
                    + self.spectra["old_reprocessed"]
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
                grid,
                fesc,
                fesc_LyA=fesc_LyA,
                young=None,
                old=None,
                **kwargs,
            )

        if np.isscalar(tau_v):
            # Single screen dust, no separate birth cloud attenuation
            # Generates:
            #   - attenuated
            #   - emergent

            # if dust_emission_model:
            #   - dust
            #   - total

            dust_curve.slope = alpha

            # Calculate the attenuated emission
            attenuated = self.spectra[reprocessed_name].apply_attenuation(
                tau_v, dust_curve=dust_curve
            )
            self.spectra["attenuated"] = attenuated

            self.spectra["attenuated"].measure_bolometric_luminosity()

            # Set emergent spectra based on fesc (for all particles)
            self.spectra["emergent"] = Sed(grid.lam)
            if fesc <= 0:
                self.spectra["emergent"]._lnu = self.spectra["attenuated"]._lnu
            else:
                self.spectra["emergent"]._lnu = (
                    self.spectra["escaped"]._lnu
                    + self.spectra["attenuated"]._lnu
                )

            # Force updating of the bolometric luminosity attribute. I don't
            # know why this is necessary.
            self.spectra["emergent"].measure_bolometric_luminosity()

            if dust_emission_model is not None:
                # Calculate the dust bolometric luminosity as the difference
                # between the emergent and incident bolometric luminosities.

                dust_bolometric_luminosity = (
                    self.spectra["incident"].bolometric_luminosity
                    - self.spectra["emergent"].bolometric_luminosity
                )

                if hasattr(dust_emission_model, "template"):
                    print("Using IR template models for dust emission")
                    if dust_emission_model.ldust is None:
                        dust_emission_model.ldust = (
                            dust_bolometric_luminosity.to(Lsun)
                        )

                    self.spectra["dust"] = dust_emission_model.get_spectra(
                        grid.lam
                    )

                else:
                    print(
                        "Using dust temperature based models for "
                        "dust emission"
                    )
                    # Get normalised dust spectrum, this is an synthesizer.sed.
                    # Sed object.
                    self.spectra["dust"] = dust_emission_model.get_spectra(
                        grid.lam
                    )

                    # scale the dust spectra by the dust_bolometric_luminosity
                    self.spectra[
                        "dust"
                    ]._lnu *= dust_bolometric_luminosity.value

                # define total as the sum of emergent and dust
                self.spectra["total"] = (
                    self.spectra["dust"] + self.spectra["emergent"]
                )

        elif np.isscalar(tau_v) is False:
            # Apply separate attenuation to both the young and old components.

            # Two screen dust, one for diffuse ISM, the other for birth cloud
            # dust.

            # Generates:
            #   - young_attenuated_BC
            #   - young_attenuated
            #   - young_emergent
            #   - old_attenuated
            #   - old_emergent
            #   - attenuated
            #   - emergent

            # if dust_emission_model:
            #   - young_dust
            #   - old_dust
            #   - young_total
            #   - old_total
            #   - total

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
            young_attenuated_BC = self.spectra[
                f"young_{reprocessed_name}"
            ].apply_attenuation(tau_v[1], dust_curve=dust_curve)
            self.spectra["young_attenuated_BC"] = young_attenuated_BC

            dust_curve.slope = alpha[0]  # use the ISM slope
            young_attenuated = young_attenuated_BC.apply_attenuation(
                tau_v[0], dust_curve=dust_curve
            )
            self.spectra["young_attenuated"] = young_attenuated

            # Calculate attenuated spectra of old stars
            old_attenuated = self.spectra[
                f"old_{reprocessed_name}"
            ].apply_attenuation(tau_v[0], dust_curve=dust_curve)
            self.spectra["old_attenuated"] = old_attenuated

            # Get the combined attenuated spectra
            self.spectra["attenuated"] = young_attenuated + old_attenuated

            # Get the combined escaping spectra
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

            # Combine emergent spectra for young and old stars
            self.spectra["emergent"] = (
                self.spectra["young_emergent"] + self.spectra["old_emergent"]
            )

            # Force updating of the bolometric luminosity attribute. I don't
            # know why this is necessary.
            self.spectra["young_emergent"].measure_bolometric_luminosity()
            self.spectra["old_emergent"].measure_bolometric_luminosity()

            if dust_emission_model is not None:
                if (not isinstance(dust_emission_model, list)) and (
                    not hasattr(dust_emission_model, "template")
                ):
                    print(
                        (
                            "Separate dust emission model for diffuse and "
                            "birth cloud dust not given"
                        )
                    )

                    dust_emission_model = [
                        dust_emission_model,
                        dust_emission_model,
                    ]

                # Start with the birth cloud dust.
                dust_bolometric_luminosity = (
                    self.spectra["young_transmitted"].bolometric_luminosity
                    - self.spectra["young_attenuated_BC"].bolometric_luminosity
                )

                if hasattr(dust_emission_model, "template"):
                    ldust = dust_bolometric_luminosity.to(Lsun)
                else:
                    self.spectra["young_dust_BC"] = dust_emission_model[
                        1
                    ].get_spectra(grid.lam)

                    # Scale the dust spectra by the dust_bolometric_luminosity.
                    self.spectra[
                        "young_dust_BC"
                    ]._lnu *= dust_bolometric_luminosity.value

                # ISM dust heated by young stars. This is the difference
                # between the birth cloud and ISM attenuated spectra.
                dust_bolometric_luminosity = (
                    self.spectra["young_attenuated_BC"].bolometric_luminosity
                    - self.spectra["young_attenuated"].bolometric_luminosity
                )

                if hasattr(dust_emission_model, "template"):
                    ldust += dust_bolometric_luminosity.to(Lsun)
                else:
                    self.spectra["young_dust_ISM"] = dust_emission_model[
                        0
                    ].get_spectra(grid.lam)

                    # Scale the dust spectra by the dust_bolometric_luminosity.
                    self.spectra[
                        "young_dust_ISM"
                    ]._lnu *= dust_bolometric_luminosity.value

                    # Combine both dust components for young stars
                    self.spectra["young_dust"] = (
                        self.spectra["young_dust_BC"]
                        + self.spectra["young_dust_ISM"]
                    )

                # ISM dust heated by old stars.
                dust_bolometric_luminosity = (
                    self.spectra["old_transmitted"].bolometric_luminosity
                    - self.spectra["old_attenuated"].bolometric_luminosity
                )

                if hasattr(dust_emission_model, "template"):
                    ldust += dust_bolometric_luminosity.to(Lsun)
                    dust_emission_model.ldust = ldust

                    # making dust components from IR templates
                    # Better classified as diffuse dust and
                    # pdr dust
                    (
                        self.spectra["old_dust"],
                        self.spectra["young_dust"],
                    ) = dust_emission_model.get_spectra(
                        grid.lam, dust_components=True
                    )

                else:
                    self.spectra["old_dust"] = dust_emission_model[
                        0
                    ].get_spectra(grid.lam)

                    # Scale the dust spectra by the dust_bolometric_luminosity.
                    self.spectra[
                        "old_dust"
                    ]._lnu *= dust_bolometric_luminosity.value

                # Combine both dust components for young stars
                self.spectra["young_total"] = (
                    self.spectra["young_emergent"] + self.spectra["young_dust"]
                )

                # Combine both dust components for young stars
                self.spectra["old_total"] = (
                    self.spectra["old_emergent"] + self.spectra["old_dust"]
                )

                self.spectra["dust"] = (
                    self.spectra["young_dust"] + self.spectra["old_dust"]
                )

                self.spectra["total"] = (
                    self.spectra["young_total"] + self.spectra["old_total"]
                )

        # Return total spectra if a dust_emission_model is provided, otherwise
        # return the emergent spectra.
        if dust_emission_model is not None:
            return self.spectra["total"]
        else:
            return self.spectra["emergent"]

    def get_spectra_CharlotFall(
        self,
        grid,
        tau_v_ISM=1.0,
        tau_v_BC=1.0,
        alpha_ISM=None,
        alpha_BC=None,
        young_old_thresh=10 * Myr,
        **kwargs,
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
                The slope of the ISM dust curve, (defaults to
                dust_curve.slope=-1, Recommended: -0.7 from MAGPHYS)
            alpha_BC (float)
                The slope of the birth cloud dust curve, (defaults to
                dust_curve.slope=-1, Recommended: -1.3 from MAGPHYS)
            young_old_thresh (unyt_quantity)
                The threshold for young/old stellar populations with units.
            kwargs
                Any keyword arguments which can be passed to
                generate_lnu.

        Returns:
            Sed
                A Sed object containing the dust attenuated spectra.
        """

        return self.get_spectra_pacman(
            grid,
            fesc=0,
            fesc_LyA=1,
            tau_v=[tau_v_ISM, tau_v_BC],
            alpha=[alpha_ISM, alpha_BC],
            young_old_thresh=young_old_thresh,
            **kwargs,
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
            line_ids = [
                line_ids,
            ]

        # Dictionary holding Line objects
        lines = {}
        print(line_ids)

        # Loop over the lines
        for line_id in line_ids:
            # If the line id a doublet in string form
            # (e.g. 'O 3 4959, O 3 5007') convert it to a list
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
        tau_v_nebular=None,
        tau_v_stellar=None,
        dust_curve_nebular=PowerLaw(slope=-1.0),
        dust_curve_stellar=PowerLaw(slope=-1.0),
    ):
        """
        Calculates attenuated properties (luminosity, continuum, EW) for a
        set of lines. Allows the nebular and stellar attenuation to be set
        separately.

        Args:
            grid (Grid)
                The Grid object.
            line_ids (list/str)
                A list of line_ids or a str denoting a single line. Doublets
                can be specified as a nested list or using a comma
                (e.g. 'OIII4363,OIII4959').
            fesc (float)
                The Lyman continuum escaped fraction, the fraction of
                ionising photons that entirely escaped.
            tau_v_BS (float)
                V-band optical depth of the nebular emission.
            tau_v_stellar (float)
                V-band optical depth of the stellar emission.
            dust_curve_nebular (dust_curve)
                A dust_curve object specifying the dust curve.
                for the nebular emission
            dust_curve_stellar (dust_curve)
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
                grid,
                line_ids,
                fesc=fesc,
            )
        else:
            intrinsic_lines = self.lines["intrinsic"]

            # Ok, well are all the requested lines in it?
            for line_id in line_ids:
                if line_id not in intrinsic_lines.line_ids:
                    intrinsic_lines.concatenate(
                        self.get_line_intrinsic(grid, line_id, fesc)
                    )

            # Ensure the lines attribute holds all the intrinsic lines
            self.lines["intrinsic"] = intrinsic_lines

        # Dictionary holding lines
        lines = {}

        # Loop over the intrinsic lines
        for line_id, intrinsic_line in intrinsic_lines.lines.items():
            # Calculate attenuation
            T_nebular = dust_curve_nebular.get_transmission(
                tau_v_nebular, intrinsic_line._wavelength
            )
            T_stellar = dust_curve_stellar.get_transmission(
                tau_v_stellar, intrinsic_line._wavelength
            )

            # Apply attenuation
            luminosity = intrinsic_line._luminosity * T_nebular * T_stellar
            continuum = intrinsic_line._continuum * T_stellar

            # Create the line object
            line = Line(
                line_id,
                intrinsic_line._wavelength,
                luminosity,
                continuum,
            )

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
                A list of line_ids or a str denoting a single line. Doublets
                can be specified as a nested list or using a comma
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
            tau_v_nebular=0,
            tau_v_stellar=tau_v,
            dust_curve_nebular=dust_curve,
            dust_curve_stellar=dust_curve,
        )

    def _check_young_old_units(self, young, old):
        """
        Checks whether the `young` and `old` arguments to many
        spectra generation methods are in the right units (Myr)

        Args:
            young (unyt_quantity):
                If not None, specifies age in Myr at which to filter
                for young star particles.
            old (unyt_quantity):
                If not None, specifies age in Myr at which to filter
                for old star particles.
        """

        if young is not None:
            if isinstance(young, (unyt_quantity)):
                young = young.to("Myr")
            else:
                raise exceptions.InconsistentArguments(
                    "young must be a unyt_quantity (i.e. a value with units)"
                )
        if old is not None:
            if isinstance(old, (unyt_quantity)):
                old = old.to("Myr")
            else:
                raise exceptions.InconsistentArguments(
                    "young must be a unyt_quantity (i.e. a value with units)"
                )

        return young, old

    def get_photo_luminosities(self, filters, verbose=True):
        """
        Calculate luminosity photometry using a FilterCollection object.

        Args:
            filters (filters.FilterCollection)
                A FilterCollection object.
            verbose (bool)
                Are we talking?

        Returns:
            photo_luminosities (dict)
                A dictionary of rest frame broadband luminosities.
        """
        # Loop over spectra in the component
        for spectra in self.spectra:
            # Create the photometry collection and store it in the object
            self.photo_luminosities[spectra] = self.spectra[
                spectra
            ].get_photo_luminosities(filters, verbose)

        return self.photo_luminosities

    def get_photo_fluxes(self, filters, verbose=True):
        """
        Calculate flux photometry using a FilterCollection object.

        Args:
            filters (object)
                A FilterCollection object.
            verbose (bool)
                Are we talking?

        Returns:
            (dict)
                A dictionary of fluxes in each filter in filters.
        """
        # Loop over spectra in the component
        for spectra in self.spectra:
            # Create the photometry collection and store it in the object
            self.photo_fluxes[spectra] = self.spectra[
                spectra
            ].get_photo_fluxes(filters, verbose)

        return self.photo_fluxes

    def plot_spectra(
        self,
        spectra_to_plot=None,
        show=False,
        ylimits=(),
        xlimits=(),
        figsize=(3.5, 5),
        **kwargs,
    ):
        """
        Plots either specific spectra (specified via spectra_to_plot) or all
        spectra on the child Stars object.

        Args:
            spectra_to_plot (string/list, string)
                The specific spectra to plot.
                    - If None all spectra are plotted.
                    - If a list of strings each specifc spectra is plotted.
                    - If a single string then only that spectra is plotted.
            show (bool)
                Flag for whether to show the plot or just return the
                figure and axes.
            ylimits (tuple)
                The limits to apply to the y axis. If not provided the limits
                will be calculated with the lower limit set to 1000 (100) times
                less than the peak of the spectrum for rest_frame (observed)
                spectra.
            xlimits (tuple)
                The limits to apply to the x axis. If not provided the optimal
                limits are found based on the ylimits.
            figsize (tuple)
                Tuple with size 2 defining the figure size.
            kwargs (dict)
                Arguments to the `sed.plot_spectra` method called from this
                wrapper.

        Returns:
            fig (matplotlib.pyplot.figure)
                The matplotlib figure object for the plot.
            ax (matplotlib.axes)
                The matplotlib axes object containing the plotted data.
        """
        # Handling whether we are plotting all spectra, specific spectra, or
        # a single spectra
        if spectra_to_plot is None:
            spectra = self.spectra
        elif isinstance(spectra_to_plot, list):
            spectra = {key: self.spectra[key] for key in spectra_to_plot}
        else:
            spectra = self.spectra[spectra_to_plot]

        return plot_spectra(
            spectra,
            show=show,
            ylimits=ylimits,
            xlimits=xlimits,
            figsize=figsize,
            draw_legend=isinstance(spectra, dict),
            **kwargs,
        )
