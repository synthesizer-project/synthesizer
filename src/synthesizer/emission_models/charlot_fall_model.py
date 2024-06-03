"""A submodule containing the Charlot & Fall (2000) emission model.

This module contains the CharlotFall2000 class, which is a simplified version
of the BimodalPacmanEmission model. The only difference is that there is no
option to specify an escape fraction.

Example:
    To create a CharlotFall2000 model, you can use the following code:

    tau_v_ism = 0.5
    tau_v_nebular = 0.5
    dust_curve_ism = PowerLaw(...)
    dust_curve_nebular = PowerLaw(...)
    age_pivot = 7 * dimensionless
    dust_emission_ism = BlackBody(...)
    dust_emission_nebular = GreyBody(...)
    model = CharlotFall2000(
        grid,
        tau_v_ism,
        tau_v_nebular,
        dust_curve_ism,
        dust_curve_nebular,
        age_pivot,
        dust_emission_ism,
        dust_emission_nebular,
    )
"""

from synthesizer.emission_models import BimodalPacmanEmission


class CharlotFall2000(BimodalPacmanEmission):
    """
    The Charlot & Fall (2000) emission model.

    This emission model is based on the Charlot & Fall (2000) model, which
    describes the emission from a galaxy as a combination of emission from a
    young stellar population and an old stellar population. The dust
    attenuation for each population can be different, and dust emission can be
    optionally included.

    This model is a simplified version of the BimodalPacmanEmission model, so
    in reality is just a wrapper around that model. The only difference is that
    there is no option to specify an escape fraction.

    Attributes:
        grid (synthesizer.grid.Grid): The grid object.
        tau_v_ism (float): The V-band optical depth for the ISM.
        tau_v_nebular (float): The V-band optical depth for the nebular.
        dust_curve_ism (synthesizer.dust.DustCurve): The dust curve for the
            ISM.
        dust_curve_nebular (synthesizer.dust.DustCurve): The dust curve for the
            nebular.
        age_pivot (unyt.unyt_quantity): The age pivot between young and old
            populations, expressed in terms of log10(age) in Myr.
        dust_emission_ism (synthesizer.dust.DustEmissionModel): The dust
            emission model for the ISM.
        dust_emission_nebular (synthesizer.dust.DustEmissionModel): The dust
            emission model for the nebular.
    """

    def __init__(
        self,
        grid,
        tau_v_ism,
        tau_v_nebular,
        dust_curve_ism,
        dust_curve_nebular,
        age_pivot,
        dust_emission_ism=None,
        dust_emission_nebular=None,
    ):
        """
        Initialize the PacmanEmission model.

        Args:
            grid (synthesizer.grid.Grid): The grid object.
            tau_v_ism (float): The V-band optical depth for the ISM.
            tau_v_nebular (float): The V-band optical depth for the nebular.
            dust_curve_ism (synthesizer.dust.DustCurve): The dust curve for the
                ISM.
            dust_curve_nebular (synthesizer.dust.DustCurve): The dust curve for
                the nebular.
            age_pivot (unyt.unyt_quantity): The age pivot between young and old
                populations, expressed in terms of log10(age) in Myr.
            dust_emission_ism (synthesizer.dust.DustEmissionModel): The dust
                emission model for the ISM.
            dust_emission_nebular (synthesizer.dust.DustEmissionModel): The
                dust emission model for the nebular.
        """
        # Call the parent constructor to intialise the model
        BimodalPacmanEmission.__init__(
            self,
            grid,
            tau_v_ism,
            tau_v_nebular,
            dust_curve_ism,
            dust_curve_nebular,
            age_pivot,
            dust_emission_ism,
            dust_emission_nebular,
        )
