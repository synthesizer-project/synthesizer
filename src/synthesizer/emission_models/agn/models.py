"""A submodule containing the definitions of common agn emission models.

This module contains the definitions of simple emission models that can be
used "out of the box" to generate spectra from components or as a foundation
to work from when creating more complex models.

Example usage:
    # Create a simple emission model
    model = TemplateEmission(
        template=Template(lam=lam, lnu=lnu),
    )

    # Generate the spectra
    spectra = blackholes.get_spectra(model)

"""

from synthesizer.emission_models.base_model import BlackHoleEmissionModel


class NLRIncidentEmission(BlackHoleEmissionModel):
    """
    An emission model that extracts the NLR incident emission.

    This defines the extraction of key "incident" from a NLR grid.

    This is a child of the EmisisonModel class, for a full description of the
    parameters see the BlackHoleEmissionModel class.
    """

    def __init__(self, grid, label="nlr_incident", **kwargs):
        """
        Initialise the NLRIncidentEmission model.

        Args:
            grid (Grid)
                The grid object containing the NLR incident emission.
            label (str)
                The label for the model.
            **kwargs

        """
        BlackHoleEmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            extract="incident",
            **kwargs,
        )


class BLRIncidentEmission(BlackHoleEmissionModel):
    """
    An emission model that extracts the BLR incident emission.

    This defines the extraction of key "incident" from a BLR grid.

    This is a child of the EmisisonModel class, for a full
    description of the parameters see the BlackHoleEmissionModel class.
    """

    def __init__(self, grid, label="blr_incident", **kwargs):
        """
        Initialise the BLRIncidentEmission model.

        Args:
            grid (Grid)
                The grid object containing the BLR incident emission.
            label (str)
                The label for the model.
            **kwargs

        """
        BlackHoleEmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            extract="incident",
            **kwargs,
        )


class NLRTransmittedEmission(BlackHoleEmissionModel):
    """
    An emission model that extracts the NLR transmitted emission.

    This defines the extraction of key "transmitted" from a NLR grid.

    This is a child of the EmisisonModel class, for a full
    description of the parameters see the BlackHoleEmissionModel class.
    """

    def __init__(
        self,
        grid,
        label="nlr_transmitted",
        covering_fraction=0.1,
        **kwargs,
    ):
        """
        Initialise the NLRTransmittedEmission model.

        Args:
            grid (Grid)
                The grid object containing the NLR transmitted emission.
            label (str)
                The label for the model.
            covering_fraction (float)
                The covering fraction of the NLR (Effectively the escape
                fraction of the NLR). Default is 0.1.
            **kwargs

        """
        BlackHoleEmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            fesc=covering_fraction,
            extract="transmitted",
            **kwargs,
        )


class BLRTransmittedEmission(BlackHoleEmissionModel):
    """
    An emission model that extracts the BLR transmitted emission.

    This defines the extraction of key "transmitted" from a BLR grid.

    This is a child of the EmisisonModel class, for a full
    description of the parameters see the BlackHoleEmissionModel class.
    """

    def __init__(
        self,
        grid,
        label="blr_transmitted",
        covering_fraction=0.1,
        **kwargs,
    ):
        """
        Initialise the BLRTransmittedEmission model.

        Args:
            grid (Grid)
                The grid object containing the BLR transmitted emission.
            label (str)
                The label for the model.
            covering_fraction (float)
                The covering fraction of the BLR (Effectively the escape
                fraction of the BLR). Default is 0.1.
            **kwargs

        """
        BlackHoleEmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            fesc=covering_fraction,
            extract="transmitted",
            **kwargs,
        )


class NLREmission(BlackHoleEmissionModel):
    """
    An emission model that computes the NLR emission.

    This defines the extraction of key "nebular" emission from the NLR grid.

    This is a child of the EmisisonModel class, for a full
    description of the parameters see the BlackHoleEmissionModel class.
    """

    def __init__(self, grid, label="nlr", **kwargs):
        """
        Initialise the NLREmission model.

        Args:
            grid (Grid)
                The grid object containing the NLR emission.
            label (str)
                The label for the model.
            **kwargs

        """
        BlackHoleEmissionModel.__init__(
            self,
            label=label,
            extract="nebular",
            grid=grid,
            **kwargs,
        )


class BLREmission(BlackHoleEmissionModel):
    """
    An emission model that computes the BLR emission.

    This defines the extraction of key "nebular" emission from the BLR grid.

    This is a child of the EmisisonModel class, for a full
    description of the parameters see the BlackHoleEmissionModel class.
    """

    def __init__(self, grid, label="blr", **kwargs):
        """
        Initialise the BLREmission model.

        Args:
            grid (Grid)
                The grid object containing the BLR emission.
            label (str)
                The label for the model.
            **kwargs

        """
        BlackHoleEmissionModel.__init__(
            self,
            label=label,
            extract="nebular",
            grid=grid,
            **kwargs,
        )


class DiscIncidentEmission(BlackHoleEmissionModel):
    """
    An emission model that extracts the incident disc emission.

    By definition this is just an alias to the incident NLR emission, i.e.
    the emission directly from the NLR with no reprocessing.

    This is a child of the EmisisonModel class, for a full
    description of the parameters see the BlackHoleEmissionModel class.
    """

    def __init__(self, grid, label="disc_incident", **kwargs):
        """
        Initialise the DiscIncidentEmission model.

        Args:
            grid (Grid)
                The grid object containing the incident disc emission.
            label (str)
                The label for the model.
            **kwargs

        """
        BlackHoleEmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            extract="incident",
            **kwargs,
        )


class DiscTransmittedEmission(BlackHoleEmissionModel):
    """
    An emission model that combines the transmitted disc emission.

    This will combine the transmitted NLR and BLR emission to give the
    transmitted disc emission.

    This is a child of the EmisisonModel class, for a full
    description of the parameters see the BlackHoleEmissionModel class.
    """

    def __init__(
        self,
        nlr_grid,
        blr_grid,
        label="disc_transmitted",
        covering_fraction_blr=0.1,
        covering_fraction_nlr=0.1,
        **kwargs,
    ):
        """
        Initialise the DiscTransmittedEmission model.

        Args:
            nlr_grid (Grid)
                The grid object containing the NLR transmitted emission.
            blr_grid (Grid)
                The grid object containing the BLR transmitted emission.
            label (str)
                The label for the model.
            **kwargs

        """
        # Create the child models
        nlr = NLRTransmittedEmission(
            grid=nlr_grid,
            label="nlr_transmitted",
            covering_fraction=covering_fraction_nlr,
            **kwargs,
        )
        blr = BLRTransmittedEmission(
            grid=blr_grid,
            label="blr_transmitted",
            covering_fraction=covering_fraction_blr,
            **kwargs,
        )

        BlackHoleEmissionModel.__init__(
            self,
            label=label,
            combine=(nlr, blr),
            **kwargs,
        )


class DiscEscapedEmission(BlackHoleEmissionModel):
    """
    An emission model that computes the escaped disc emission.

    This will extract the incident disc emission but apply
    fesc=1 - covering_fraction, since the escaped is the mirror of the
    transmitted emission.

    This is a child of the EmisisonModel class, for a full description of the
    parameters see the BlackHoleEmissionModel class.
    """

    def __init__(
        self,
        grid,
        label="disc_escaped",
        covering_fraction_nlr=0.1,
        covering_fraction_blr=0.1,
        **kwargs,
    ):
        """
        Initialise the DiscEscapedEmission model.

        Args:
            grid (Grid)
                The NLR grid object.
            label (str)
                The label for the model.
            covering_fraction_nlr (float)
                The covering fraction of the NLR (Effectively the escape
                fraction of the NLR). Default is 0.1.
            covering_fraction_blr (float)
                The covering fraction of the BLR (Effectively the escape
                fraction of the BLR). Default is 0.1.
            **kwargs
        """
        BlackHoleEmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            extract="incident",
            fesc=1 - covering_fraction_nlr - covering_fraction_blr,
            **kwargs,
        )


class DiscEmission(BlackHoleEmissionModel):
    """
    An emission model that computes the total disc emission.

    This will combine the tranmitted and escaped disc emission.

    This is a child of the EmisisonModel class, for a full
    description of the parameters see the BlackHoleEmissionModel class.
    """

    def __init__(self, nlr_grid, blr_grid, label="disc", **kwargs):
        """
        Initialise the DiscEmission model.

        Args:
            nlr_grid (Grid)
                The grid object containing the NLR emission.
            blr_grid (Grid)
                The grid object containing the BLR emission.
            label (str)
                The label for the model.
            **kwargs

        """
        # Create the child models
        transmitted = DiscTransmittedEmission(
            nlr_grid=nlr_grid,
            blr_grid=blr_grid,
            label="disc_transmitted",
            **kwargs,
        )
        escaped = DiscEscapedEmission(
            grid=nlr_grid,
            label="disc_escaped",
            **kwargs,
        )

        BlackHoleEmissionModel.__init__(
            self,
            label=label,
            combine=(transmitted, escaped),
            **kwargs,
        )


class TorusEmission(BlackHoleEmissionModel):
    """
    An emission model that computes the torus emission.

    This will generate the torus emission from the model.

    This is a child of the EmisisonModel class, for a full
    description of the parameters see the BlackHoleEmissionModel class.
    """

    def __init__(
        self,
        torus_emission_model,
        grid=None,
        label="torus",
        disc_incident=None,
        **kwargs,
    ):
        """
        Initialise the TorusEmission model.

        Args:
            torus_emission_model (dust.emission)
                The dust emission model to use for the torus.
            grid (Grid)
                The grid object to extract the disc incident emission from.
                Only required if disc_incident is None.
            label (str)
                The label for the model.
            disc_incident (BlackHoleEmissionModel)
                The disc incident emission model to use. If not provided
                this will be generated from the grid.
            **kwargs

        """
        # Make the disc incident model if not provided
        if disc_incident is None:
            disc_incident = DiscIncidentEmission(
                grid=grid,
                label="disc_incident",
                **kwargs,
            )
        BlackHoleEmissionModel.__init__(
            self,
            label=label,
            generator=torus_emission_model,
            dust_lum_intrinsic=disc_incident,
            scale_by="torus_fraction",
            **kwargs,
        )


class AGNIntrinsicEmission(BlackHoleEmissionModel):
    """
    An emission model that computes the intrinsic AGN emission.

    This will generate the intrinsic AGN emission from the model by combining
    all child models.

    This is a child of the EmisisonModel class, for a full
    description of the parameters see the BlackHoleEmissionModel class.
    """

    def __init__(
        self,
        nlr_grid,
        blr_grid,
        torus_emission_model,
        label="intrinsic",
        covering_fraction_nlr=0.1,
        covering_fraction_blr=0.1,
        **kwargs,
    ):
        """
        Initialise the AGNIntrinsicEmission model.

        Args:
            nlr_grid (Grid)
                The grid object containing the NLR emission.
            blr_grid (Grid)
                The grid object containing the BLR emission.
            torus_emission_model (dust.emission)
                The dust emission model to use for the torus.
            label (str)
                The label for the model.
            covering_fraction_nlr (float)
                The covering fraction of the NLR (Effectively the escape
                fraction of the NLR). Default is 0.1.
            covering_fraction_blr (float)
                The covering fraction of the BLR (Effectively the escape
                fraction of the BLR). Default is 0.1.
            **kwargs

        """
        # Create the child models
        disc = DiscEmission(
            nlr_grid=nlr_grid,
            blr_grid=blr_grid,
            label="disc",
            covering_fraction_nlr=covering_fraction_nlr,
            covering_fraction_blr=covering_fraction_blr,
            **kwargs,
        )
        torus = TorusEmission(
            torus_emission_model=torus_emission_model,
            grid=nlr_grid,
            label="torus",
            **kwargs,
        )

        BlackHoleEmissionModel.__init__(
            self,
            label=label,
            combine=(
                disc,
                torus,
            ),
            **kwargs,
        )
