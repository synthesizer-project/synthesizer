from synthesizer.emission_models.base_model import (
    EmissionModel,
    StellarEmissionModel,
    BlackHoleEmissionModel,
)
from synthesizer.emission_models.models import (
    DustEmission,
    AttenuatedEmission,
    TemplateEmission,
)
from synthesizer.emission_models.stellar.models import (
    IncidentEmission,
    LineContinuumEmission,
    TransmittedEmission,
    EscapedEmission,
    NebularContinuumEmission,
    NebularEmission,
    ReprocessedEmission,
    IntrinsicEmission,
    EmergentEmission,
    TotalEmission,
)
from synthesizer.emission_models.stellar.pacman_model import (
    PacmanEmission,
    BimodalPacmanEmission,
    CharlotFall2000,
)
from synthesizer.emission_models.agn.models import (
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
)
from synthesizer.emission_models.agn.unified_agn import UnifiedAGN


from synthesizer.emission_models.stellar import STELLAR_MODELS
from synthesizer.emission_models.agn import AGN_MODELS

# List of premade common models
COMMON_MODELS = [
    "AttenuatedEmission",
    "DustEmission",
    "TemplateEmission",
]

# List of premade models
PREMADE_MODELS = [
    *COMMON_MODELS,
]
PREMADE_MODELS.extend(STELLAR_MODELS)
PREMADE_MODELS.extend(AGN_MODELS)
