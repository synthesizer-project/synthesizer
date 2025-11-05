"""A submodule defining the Line Of Sight (LOS) emission model.

This is an emission model where the dust attenuation accounts for the
dust column density traced along the line of sight of each star. It is
therefore, by definition, a per particle model.

This model is adapted from the approach used in FLARES, first published
in:
https://ui.adsabs.harvard.edu/abs/2021MNRAS.501.3289V/abstract
"""

from unyt import Myr

from synthesizer.emission_models import (
    EmissionModel,
    NebularEmission,
    ReprocessedEmission,
    StellarEmissionModel,
    TransmittedEmission,
)
from synthesizer.emission_models.transformers.dust_attenuation import PowerLaw
from synthesizer.units import accepts


class LOSStellarEmission(EmissionModel):
    """An emission model using Line Of Sight (LOS) dust attenuation.

    This matches the approach used in FLARES, and is adapted from the model
    first published in:
    https://ui.adsabs.harvard.edu/abs/2021MNRAS.501.3289V/abstract

    This model includes:
    - Nebular emission for young stars (<=10 Myr)
    - Stellar emission transmitted via Cloudy photoionisation through the ISM
      (split by age into young and old components).
    - Birth cloud attenuation for young stars (<=10 Myr) using a PowerLaw. This
      requires the young_tau_v attribute to be defined on a galaxy given by:
        young_tau_v = Z_star / 0.01
    - Line Of Sight attenuation for all stars, again with a PowerLaw. This
      requires the tau_v attribute to have been populated by calling
      get_los_optical_depths on the galaxy prior to generating the spectra.
    - The total emergent stellar emission combining the above components.

    This requires a stellar component defines:
        - young_tau_v
        - tau_v (via get_los_optical_depths, though can be set manually)
        - ages
        - metallicities
        - Any extra axes defined on the grid.

    Attributes:
        Inherits from EmissionModel.
    """

    @accepts(young_upper_limit=Myr)
    def __init__(
        self,
        grid,
        young_upper_limit=10 * Myr,
        birth_cloud_slope=-1,
        ism_slope=-1,
    ):
        """Initialize the EmissionModel.

        The defaults are set to match the FLARES LOS model.

        Args:
            grid (Grid):
                The grid to use for the model.
            young_upper_limit (unyt_quantity):
                The upper age limit to define young stars (default: 10 Myr).
            birth_cloud_slope (float):
                The slope of the power-law dust curve for birth cloud
                attenuation (default: -1).
            ism_slope (float):
                The slope of the power-law dust curve for ISM attenuation
                (default: -1).
        """
        # Define the nebular emission models
        nebular = NebularEmission(
            grid=grid,
            label="nebular",
            mask_attr="ages",
            mask_op="<=",
            mask_thresh=young_upper_limit,
        )

        # Define the transmitted models
        young_transmitted = TransmittedEmission(
            grid=grid,
            label="young_transmitted",
            mask_attr="ages",
            mask_op="<=",
            mask_thresh=young_upper_limit,
        )
        old_transmitted = TransmittedEmission(
            grid=grid,
            label="old_transmitted",
            mask_attr="ages",
            mask_op=">",
            mask_thresh=young_upper_limit,
        )
        transmitted = StellarEmissionModel(
            grid=grid,
            label="transmitted",
            combine=[young_transmitted, old_transmitted],
        )

        # Define the reprocessed models
        young_reprocessed = ReprocessedEmission(
            grid=grid,
            label="young_reprocessed",
            transmitted=young_transmitted,
            nebular=nebular,
            mask_attr="ages",
            mask_op="<=",
            mask_thresh=young_upper_limit,
        )
        reprocessed = StellarEmissionModel(
            grid=grid,
            label="reprocessed",
            combine=[young_reprocessed, old_transmitted],
        )

        # Define the attenuated models
        young_attenuated_nebular = StellarEmissionModel(
            grid=grid,
            label="young_attenuated_nebular",
            apply_dust_to=young_reprocessed,
            tau_v="young_tau_v",
            dust_curve=PowerLaw(slope=birth_cloud_slope),
            mask_attr="ages",
            mask_op="<=",
            mask_thresh=young_upper_limit,
        )
        young_attenuated = StellarEmissionModel(
            grid=grid,
            label="young_attenuated",
            apply_dust_to=young_attenuated_nebular,
            tau_v="tau_v",
            dust_curve=PowerLaw(slope=ism_slope),
            mask_attr="ages",
            mask_op="<=",
            mask_thresh=young_upper_limit,
        )
        old_attenuated = StellarEmissionModel(
            grid=grid,
            label="old_attenuated",
            apply_dust_to=old_transmitted,
            tau_v="tau_v",
            dust_curve=PowerLaw(slope=ism_slope),
            mask_attr="ages",
            mask_op=">",
            mask_thresh=young_upper_limit,
        )

        # Finally, combine to get the emergent emission
        EmissionModel.__init__(
            self,
            grid=grid,
            label="stellar_total",
            combine=[young_attenuated, old_attenuated],
            related_models=[
                nebular,
                transmitted,
                reprocessed,
                young_attenuated_nebular,
            ],
            emitter="stellar",
        )

        self.set_per_particle(True)
