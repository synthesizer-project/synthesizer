from synthesizer.emission_models.base_model import EmissionModel
from synthesizer.emission_models.stellar.models import (
    IncidentEmission,
    LineContinuumEmission,
    TransmittedEmission,
    EscapedEmission,
    NebularContinuumEmission,
    NebularEmission,
    ReprocessedEmission,
    AttenuatedEmission,
    EmergentEmission,
    DustEmission,
    TotalEmission,
)
from synthesizer.emission_models.stellar.pacman_model import (
    PacmanEmission,
    BimodalPacmanEmission,
    CharlotFall2000,
)

PREMADE_MODELS = [
    "IncidentEmission",
    "LineContinuumEmission",
    "TransmittedEmission",
    "EscapedEmission",
    "NebularContinuumEmission",
    "NebularEmission",
    "ReprocessedEmission",
    "AttenuatedEmission",
    "EmergentEmission",
    "DustEmission",
    "TotalEmission",
    "PacmanEmission",
    "BimodalPacmanEmission",
    "CharlotFall2000",
]
