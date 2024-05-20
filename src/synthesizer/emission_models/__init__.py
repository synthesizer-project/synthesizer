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
]
