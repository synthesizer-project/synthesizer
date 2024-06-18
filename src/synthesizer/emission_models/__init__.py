from synthesizer.emission_models.base_model import EmissionModel
from synthesizer.emission_models.models import (
    DustEmission,
    AttenuatedEmission,
)
from synthesizer.emission_models.stellar.models import (
    IncidentEmission,
    LineContinuumEmission,
    TransmittedEmission,
    EscapedEmission,
    NebularContinuumEmission,
    NebularEmission,
    ReprocessedEmission,
    EmergentEmission,
    TotalEmission,
)
from synthesizer.emission_models.stellar.pacman_model import (
    PacmanEmission,
    BimodalPacmanEmission,
    CharlotFall2000,
)
from synthesizer.emission_models.agn.models import (
    Template,
    NLRIncidentEmission,
    BLRIncidentEmission,
    NLRTransmittedEmission,
    BLRTransmittedEmission,
    DiscIncidentEmission,
    DiscTransmittedEmission,
    DiscEscapedEmission,
    DiscEmission,
    TorusEmission,
    AGNIntrinsicEmission,
    UnifiedAGN,
)


from synthesizer.emission_models.stellar import STELLAR_MODELS
from synthesizer.emission_models.agn import AGN_MODELS

PREMADE_MODELS = [
    "AttenuatedEmission",
    "DustEmission",
]
PREMADE_MODELS.extend(STELLAR_MODELS)
PREMADE_MODELS.extend(AGN_MODELS)
