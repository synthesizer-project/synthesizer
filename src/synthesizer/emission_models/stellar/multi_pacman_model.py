"""A module defining the MultiPacman emission models.

The MultiPacman models extend the Pacman models with a second escape
fraction. As in Pacman, ``fesc`` is the fraction of the incident emission
that escapes the photoionised region without being reprocessed. In addition,
``fesc_dust`` is the fraction of the emission reaching the ISM dust that
escapes it without being attenuated.

Whether the incident emission escaping the photoionised region then sees the
ISM dust is controlled by ``attenuate_incident_escaped``. If ``True`` (the
default) the escaped incident emission joins the reprocessed emission and
``fesc_dust`` applies to both. If ``False`` the escaped incident emission
leaves the galaxy unattenuated (as in Pacman) and ``fesc_dust`` applies only
to the reprocessed emission. With ``attenuate_incident_escaped=False`` and
``fesc_dust=0`` the models reduce to their Pacman equivalents.

MultiPacmanEmission treats a single population of stars.
BimodalMultiPacmanEmission splits the stars into young and old populations,
where the young population's reprocessed emission is also attenuated by its
birth cloud before reaching the ISM. The incident emission escaping the
photoionised region also escapes the birth cloud.

Example::

    model = MultiPacmanEmission(
        grid,
        tau_v=0.33,
        fesc=0.1,
        fesc_dust=0.2,
        dust_curve=Calzetti2000(),
        dust_emission=Greybody(...),
    )
"""

from copy import deepcopy

from unyt import dimensionless, km, s

from synthesizer.emission_models.attenuation import Calzetti2000, PowerLaw
from synthesizer.emission_models.base_model import (
    EmissionModel,
    StellarEmissionModel,
)
from synthesizer.emission_models.models import (
    AttenuatedEmission,
    DustEmission,
)
from synthesizer.emission_models.stellar.models import (
    IncidentEmission,
    NebularContinuumEmission,
    NebularEmission,
    NebularLineEmission,
    ReprocessedEmission,
    TransmittedEmissionWithEscaped,
)
from synthesizer.emission_models.transformers import (
    DopplerBroadening,
    EscapedFraction,
    ProcessedFraction,
)
from synthesizer.units import accepts


@accepts(velocity_dispersion=km / s)
def _init_root(
    model, grid, label, combine, related, emitter, velocity_dispersion, kwargs
):
    """Initialise a MultiPacman root, optionally Doppler broadened.

    As in the Pacman models, broadening is applied to the final emission
    only. The unbroadened emission is kept in the tree as
    "<label>_predispersion" but is not saved.

    Args:
        model (EmissionModel):
            The model being initialised.
        grid (synthesizer.grid.Grid):
            The grid object.
        label (str):
            The label for the root model.
        combine (tuple):
            The models combined to make the root emission.
        related (tuple):
            The related models.
        emitter (str):
            The emitter of the root model.
        velocity_dispersion (unyt_quantity):
            Optional scalar velocity dispersion. None disables broadening.
        kwargs (dict):
            Additional keyword arguments to pass to the models.
    """
    if velocity_dispersion is None:
        EmissionModel.__init__(
            model,
            grid=grid,
            label=label,
            combine=combine,
            related_models=related,
            emitter=emitter,
            **kwargs,
        )
        return

    predispersion = EmissionModel(
        grid=grid,
        label=f"{label}_predispersion",
        combine=combine,
        related_models=related,
        emitter=emitter,
        **kwargs,
    )
    predispersion.set_save(False)
    EmissionModel.__init__(
        model,
        label=label,
        apply_to=predispersion,
        transformer=DopplerBroadening(sigma_v_attr="velocity_dispersion"),
        emitter=emitter,
        velocity_dispersion=velocity_dispersion,
        **kwargs,
    )


def _population_models(
    grid,
    prefix,
    fesc,
    fesc_ly_alpha,
    fesc_dust,
    tau_v_ism,
    dust_curve_ism,
    attenuate_incident_escaped,
    tau_v_birth=None,
    dust_curve_birth=None,
    mask_kwargs=None,
    **kwargs,
):
    """Build the emission models for a single stellar population.

    Args:
        grid (synthesizer.grid.Grid):
            The grid object.
        prefix (str):
            The prefix for every model label (e.g. "young_").
        fesc (float/str):
            The escape fraction from the photoionised region.
        fesc_ly_alpha (float/str):
            The Lyman alpha escape fraction.
        fesc_dust (float/str):
            The escape fraction from the ISM dust.
        tau_v_ism (float/str):
            The V-band optical depth of the ISM dust.
        dust_curve_ism (synthesizer.emission_models.Transformer):
            The ISM dust curve.
        attenuate_incident_escaped (bool):
            Whether the incident emission escaping the photoionised region
            sees the ISM dust.
        tau_v_birth (float/str):
            The V-band optical depth of the birth cloud dust. If None, there
            is no birth cloud attenuation.
        dust_curve_birth (synthesizer.emission_models.Transformer):
            The birth cloud dust curve.
        mask_kwargs (dict):
            The mask arguments selecting this population's stars.
        **kwargs:
            Additional keyword arguments to pass to the models.

    Returns:
        dict:
            The models keyed by their unprefixed label.
    """
    mask_kwargs = {} if mask_kwargs is None else mask_kwargs
    models = {}

    # The emission from the photoionised region, and the incident emission
    # escaping it
    models["incident"] = IncidentEmission(
        grid=grid,
        label=f"{prefix}incident",
        **mask_kwargs,
        **kwargs,
    )
    models["transmitted"] = TransmittedEmissionWithEscaped(
        grid=grid,
        label=f"{prefix}transmitted",
        fesc=fesc,
        incident=models["incident"],
        escaped_label=f"{prefix}escaped",
        **mask_kwargs,
        **kwargs,
    )
    models["escaped"] = models["transmitted"][f"{prefix}escaped"]
    nebular_line = NebularLineEmission(
        grid=grid,
        label=f"{prefix}nebular_line",
        fesc_ly_alpha=fesc_ly_alpha,
        **mask_kwargs,
        **kwargs,
    )
    nebular_continuum = NebularContinuumEmission(
        grid=grid,
        label=f"{prefix}nebular_continuum",
        **mask_kwargs,
        **kwargs,
    )
    models["nebular"] = NebularEmission(
        grid=grid,
        label=f"{prefix}nebular",
        nebular_line=nebular_line,
        nebular_continuum=nebular_continuum,
        fesc=fesc,
        **kwargs,
    )
    models["reprocessed"] = ReprocessedEmission(
        grid=grid,
        label=f"{prefix}reprocessed",
        nebular=models["nebular"],
        transmitted=models["transmitted"],
        **kwargs,
    )
    models["intrinsic"] = StellarEmissionModel(
        label=f"{prefix}intrinsic",
        combine=(models["reprocessed"], models["escaped"]),
        **kwargs,
    )

    # Attenuate the reprocessed emission by the birth cloud, if there is one
    if tau_v_birth is not None:
        models["attenuated_nebular"] = AttenuatedEmission(
            label=f"{prefix}attenuated_nebular",
            tau_v=tau_v_birth,
            dust_curve=dust_curve_birth,
            apply_to=models["reprocessed"],
            emitter="stellar",
            **kwargs,
        )
        reaching_ism = models["attenuated_nebular"]
    else:
        reaching_ism = models["reprocessed"]

    # Include the escaped incident emission in what reaches the ISM dust
    if attenuate_incident_escaped:
        if tau_v_birth is None:
            reaching_ism = models["intrinsic"]
        else:
            reaching_ism = StellarEmissionModel(
                label=f"{prefix}ism_incident",
                combine=(reaching_ism, models["escaped"]),
                **kwargs,
            )
            models["ism_incident"] = reaching_ism

    # Split the emission reaching the ISM into the part escaping the dust
    # and the part obscured by it, then attenuate the obscured part
    models["dust_escaped"] = StellarEmissionModel(
        label=f"{prefix}dust_escaped",
        apply_to=reaching_ism,
        transformer=EscapedFraction(fesc_attrs=("fesc_dust",)),
        fesc_dust=fesc_dust,
        **kwargs,
    )
    models["obscured"] = StellarEmissionModel(
        label=f"{prefix}obscured",
        apply_to=reaching_ism,
        transformer=ProcessedFraction(fesc_attrs=("fesc_dust",)),
        fesc_dust=fesc_dust,
        **kwargs,
    )
    models["attenuated"] = AttenuatedEmission(
        label=f"{prefix}attenuated",
        tau_v=tau_v_ism,
        dust_curve=dust_curve_ism,
        apply_to=models["obscured"],
        emitter="stellar",
        **kwargs,
    )

    # The emission leaving the population
    emerging = (models["attenuated"], models["dust_escaped"])
    if not attenuate_incident_escaped:
        emerging += (models["escaped"],)
    models["emergent"] = StellarEmissionModel(
        label=f"{prefix}emergent",
        combine=emerging,
        **kwargs,
    )

    return models


class MultiPacmanEmission(EmissionModel):
    """The MultiPacman model for a single stellar population.

    This is the Pacman model with separate escape fractions for the
    photoionised region (``fesc``) and the ISM dust (``fesc_dust``). See the
    module docstring for how they combine.

    This model will produce:
        - incident: the stellar emission incident onto the photoionised
            region.
        - transmitted: the incident emission transmitted through the
            photoionised region.
        - escaped: the incident emission escaping the photoionised region.
        - nebular: the emission from the photoionised region.
        - reprocessed: the transmitted + nebular emission.
        - intrinsic: the reprocessed + escaped emission.
        - dust_escaped: the emission escaping the ISM dust.
        - obscured: the emission obscured by the ISM dust.
        - attenuated: the obscured emission attenuated by the ISM dust.
        - emergent: the emission leaving the galaxy (attenuated +
            dust_escaped, plus escaped if attenuate_incident_escaped is
            False).

    If a dust emission model is given, it will also produce:
        - dust_emission: the emission from dust.
        - total: the emergent + dust emission.
    """

    def __init__(
        self,
        grid,
        tau_v="tau_v",
        dust_curve=PowerLaw(),
        dust_emission=None,
        fesc="fesc",
        fesc_dust="fesc_dust",
        fesc_ly_alpha="fesc_ly_alpha",
        attenuate_incident_escaped=True,
        label=None,
        stellar_dust=True,
        velocity_dispersion=None,
        **kwargs,
    ):
        """Initialise the MultiPacmanEmission model.

        Args:
            grid (synthesizer.grid.Grid):
                The grid object.
            tau_v (float):
                The V-band optical depth of the ISM dust.
            dust_curve (synthesizer.emission_models.Transformer):
                The ISM dust curve. Defaults to `PowerLaw`, with default
                parameters.
            dust_emission (synthesizer.dust.EmissionModel):
                The dust emission model. If None, no dust emission is
                produced.
            fesc (float):
                The fraction of the incident emission that escapes the
                photoionised region without being reprocessed.
            fesc_dust (float):
                The fraction of the emission reaching the ISM dust that
                escapes it without being attenuated.
            fesc_ly_alpha (float):
                The Lyman alpha escape fraction.
            attenuate_incident_escaped (bool):
                If True, the incident emission escaping the photoionised
                region then sees the ISM dust. If False, it leaves the galaxy
                unattenuated, as in Pacman.
            label (str):
                The label for the root model. If None, this will be "total"
                with dust emission and "emergent" without.
            stellar_dust (bool):
                If True, the dust emission will be treated as stellar
                emission, otherwise it will be treated as galaxy emission.
            velocity_dispersion (unyt_quantity):
                Optional scalar velocity dispersion applied to the final
                emission.
            **kwargs:
                Additional keyword arguments to pass to the models.
        """
        # Ensure the grid has been processed
        if not grid.reprocessed:
            raise ValueError(
                "The MultiPacmanEmission models require a reprocessed grid."
            )

        models = _population_models(
            grid=grid,
            prefix="",
            fesc=fesc,
            fesc_ly_alpha=fesc_ly_alpha,
            fesc_dust=fesc_dust,
            tau_v_ism=tau_v,
            dust_curve_ism=dust_curve,
            attenuate_incident_escaped=attenuate_incident_escaped,
            **kwargs,
        )

        # Without dust emission the emergent emission is the root
        if dust_emission is None:
            emergent = models.pop("emergent")
            _init_root(
                self,
                grid,
                "emergent" if label is None else label,
                emergent.combine,
                tuple(models.values()),
                "stellar",
                velocity_dispersion,
                kwargs,
            )
            return

        # Otherwise add the dust emission, balanced against the energy the
        # ISM dust absorbs
        emitter = "stellar" if stellar_dust else "galaxy"
        dust_emission = deepcopy(dust_emission)
        dust_emission.set_energy_balance(
            models["obscured"], models["attenuated"]
        )
        models["dust_emission"] = DustEmission(
            label="dust_emission",
            dust_emission_model=dust_emission,
            emitter=emitter,
            **kwargs,
        )
        _init_root(
            self,
            grid,
            "total" if label is None else label,
            (models["emergent"], models["dust_emission"]),
            tuple(models.values()),
            emitter,
            velocity_dispersion,
            kwargs,
        )


class BimodalMultiPacmanEmission(EmissionModel):
    """The MultiPacman model split into young and old populations.

    As MultiPacmanEmission, but the young population's reprocessed emission
    is first attenuated by its birth cloud. The incident emission escaping
    the photoionised region also escapes the birth cloud. Every model
    is produced for the young population (prefixed "young_"), the old
    population (prefixed "old_") and their combination (unprefixed).

    This model will produce, for each of these:
        - incident, transmitted, escaped, nebular, reprocessed, intrinsic,
            dust_escaped, obscured, attenuated and emergent, as described in
            MultiPacmanEmission.

    For the young population only:
        - young_attenuated_nebular: the reprocessed emission attenuated by
            the birth cloud.
        - young_ism_incident: the emission reaching the ISM dust (birth
            cloud attenuated + escaped), if attenuate_incident_escaped is
            True.

    If dust emission models are given, it will also produce:
        - young_dust_emission_birth: the emission from birth cloud dust.
        - young_dust_emission_ism: the emission from ISM dust heated by the
            young population.
        - young_dust_emission, old_dust_emission and dust_emission: the
            emission from dust.
        - young_total, old_total and total: the emergent + dust emission.
    """

    def __init__(
        self,
        grid,
        tau_v_ism="tau_v_ism",
        tau_v_birth="tau_v_birth",
        dust_curve_ism=Calzetti2000(),
        dust_curve_birth=Calzetti2000(),
        age_pivot=7 * dimensionless,
        dust_emission_ism=None,
        dust_emission_birth=None,
        fesc="fesc",
        fesc_dust="fesc_dust",
        fesc_ly_alpha="fesc_ly_alpha",
        attenuate_incident_escaped=True,
        label=None,
        stellar_dust=True,
        velocity_dispersion=None,
        **kwargs,
    ):
        """Initialise the BimodalMultiPacmanEmission model.

        Args:
            grid (synthesizer.grid.Grid):
                The grid object.
            tau_v_ism (float):
                The V-band optical depth of the ISM dust.
            tau_v_birth (float):
                The V-band optical depth of the birth cloud dust.
            dust_curve_ism (synthesizer.emission_models.Transformer):
                The ISM dust curve. Defaults to `Calzetti2000` with default
                parameters.
            dust_curve_birth (synthesizer.emission_models.Transformer):
                The birth cloud dust curve. Defaults to `Calzetti2000` with
                default parameters.
            age_pivot (unyt.unyt_quantity):
                The age pivot between young and old populations, expressed
                in terms of log10(age/yr).
            dust_emission_ism (synthesizer.dust.EmissionModel):
                The dust emission model for the ISM. Dust emission is only
                produced if both dust emission models are given.
            dust_emission_birth (synthesizer.dust.EmissionModel):
                The dust emission model for the birth cloud.
            fesc (float):
                The fraction of the incident emission that escapes the
                photoionised region (and birth cloud) without being
                reprocessed.
            fesc_dust (float):
                The fraction of the emission reaching the ISM dust that
                escapes it without being attenuated.
            fesc_ly_alpha (float):
                The Lyman alpha escape fraction.
            attenuate_incident_escaped (bool):
                If True, the incident emission escaping the photoionised
                region then sees the ISM dust. If False, it leaves the galaxy
                unattenuated, as in Pacman.
            label (str):
                The label for the root model. If None, this will be "total"
                with dust emission and "emergent" without.
            stellar_dust (bool):
                If True, the dust emission will be treated as stellar
                emission, otherwise it will be treated as galaxy emission.
            velocity_dispersion (unyt_quantity):
                Optional scalar velocity dispersion applied to the final
                emission.
            **kwargs:
                Additional keyword arguments to pass to the models.
        """
        # Ensure the grid has been processed
        if not grid.reprocessed:
            raise ValueError(
                "The BimodalMultiPacmanEmission models require a reprocessed "
                "grid."
            )

        # The young population always sees a birth cloud
        if tau_v_birth is None:
            raise ValueError(
                "BimodalMultiPacmanEmission requires tau_v_birth. Use "
                "MultiPacmanEmission for a model without a birth cloud."
            )

        shared = dict(
            grid=grid,
            fesc=fesc,
            fesc_ly_alpha=fesc_ly_alpha,
            fesc_dust=fesc_dust,
            tau_v_ism=tau_v_ism,
            dust_curve_ism=dust_curve_ism,
            attenuate_incident_escaped=attenuate_incident_escaped,
            **kwargs,
        )
        young = _population_models(
            prefix="young_",
            tau_v_birth=tau_v_birth,
            dust_curve_birth=dust_curve_birth,
            mask_kwargs=dict(
                mask_attr="log10ages", mask_thresh=age_pivot, mask_op="<"
            ),
            **shared,
        )
        old = _population_models(
            prefix="old_",
            mask_kwargs=dict(
                mask_attr="log10ages", mask_thresh=age_pivot, mask_op=">="
            ),
            **shared,
        )

        # Combine the populations for every model they share
        combined = {
            key: StellarEmissionModel(
                label=key,
                combine=(young[key], old[key]),
                **kwargs,
            )
            for key in old
        }
        related = (*young.values(), *old.values(), *combined.values())

        # Without dust emission the emergent emission is the root
        if dust_emission_ism is None or dust_emission_birth is None:
            emergent = combined["emergent"]
            related = tuple(m for m in related if m is not emergent)
            _init_root(
                self,
                grid,
                "emergent" if label is None else label,
                emergent.combine,
                related,
                "stellar",
                velocity_dispersion,
                kwargs,
            )
            return

        # Otherwise add the dust emission, each balanced against the energy
        # its dust absorbs
        emitter = "stellar" if stellar_dust else "galaxy"
        dust_emission_birth = deepcopy(dust_emission_birth)
        dust_emission_birth.set_energy_balance(
            young["reprocessed"], young["attenuated_nebular"]
        )
        young_dust_birth = DustEmission(
            label="young_dust_emission_birth",
            dust_emission_model=dust_emission_birth,
            emitter=emitter,
            **kwargs,
        )
        dust_emission_ism_young = deepcopy(dust_emission_ism)
        dust_emission_ism_young.set_energy_balance(
            young["obscured"], young["attenuated"]
        )
        young_dust_ism = DustEmission(
            label="young_dust_emission_ism",
            dust_emission_model=dust_emission_ism_young,
            emitter=emitter,
            **kwargs,
        )
        young_dust = EmissionModel(
            label="young_dust_emission",
            combine=(young_dust_birth, young_dust_ism),
            emitter=emitter,
            **kwargs,
        )
        dust_emission_ism_old = deepcopy(dust_emission_ism)
        dust_emission_ism_old.set_energy_balance(
            old["obscured"], old["attenuated"]
        )
        old_dust = DustEmission(
            label="old_dust_emission",
            dust_emission_model=dust_emission_ism_old,
            emitter=emitter,
            **kwargs,
        )
        dust = EmissionModel(
            label="dust_emission",
            combine=(young_dust, old_dust),
            emitter=emitter,
            **kwargs,
        )
        young_total = EmissionModel(
            label="young_total",
            combine=(young["emergent"], young_dust),
            emitter=emitter,
            **kwargs,
        )
        old_total = EmissionModel(
            label="old_total",
            combine=(old["emergent"], old_dust),
            emitter=emitter,
            **kwargs,
        )
        _init_root(
            self,
            grid,
            "total" if label is None else label,
            (young_total, old_total),
            (
                *related,
                young_dust_birth,
                young_dust_ism,
                young_dust,
                old_dust,
                dust,
            ),
            emitter,
            velocity_dispersion,
            kwargs,
        )
