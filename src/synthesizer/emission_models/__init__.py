from synthesizer.emission_models.base_model import EmissionModel
from synthesizer.emission_models.models import (
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
from synthesizer.emission_models.charlot_fall_model import CharlotFall2000
from synthesizer.emission_models.pacman_model import (
    PacmanEmission,
    BimodalPacmanEmission,
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
    "CharlotFall2000",
    "PacmanEmission",
    "BimodalPacmanEmission",
]
