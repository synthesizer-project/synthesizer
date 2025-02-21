from synthesizer.emission_models.agn import AGN_MODELS
from synthesizer.emission_models.agn.models import (
    AGNIntrinsicEmission,
    BLREmission,
    BLRIncidentEmission,
    BLRTransmittedEmission,
    DiscEmission,
    DiscEscapedEmission,
    DiscIncidentEmission,
    DiscTransmittedEmission,
    NLREmission,
    NLRIncidentEmission,
    NLRTransmittedEmission,
    TorusEmission,
)
from synthesizer.emission_models.agn.unified_agn import UnifiedAGN
from synthesizer.emission_models.base_model import (
    BlackHoleEmissionModel,
    EmissionModel,
    GalaxyEmissionModel,
    StellarEmissionModel,
)
from synthesizer.emission_models.models import (
    AttenuatedEmission,
    DustEmission,
    TemplateEmission,
)
from synthesizer.emission_models.stellar import STELLAR_MODELS
from synthesizer.emission_models.stellar.models import (
    EmergentEmission,
    IncidentEmission,
    IntrinsicEmission,
    NebularContinuumEmission,
    NebularEmission,
    NebularLineEmission,
    ReprocessedEmission,
    TotalEmission,
    TransmittedEmission,
)
from synthesizer.emission_models.stellar.pacman_model import (
    BimodalPacmanEmission,
    CharlotFall2000,
    PacmanEmission,
    ScreenEmission,
)

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
