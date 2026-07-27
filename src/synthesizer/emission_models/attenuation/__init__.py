# Unpack the attenuation laws into this nice alias submodule
from synthesizer.emission_models.transformers.dust_attenuation import (
    MWN18,
    Calzetti2000,
    GrainModels,
    DraineLiGrainCurves,
    ParametricLi08,
    PowerLaw,
    SommovigoBartlett2026,
    predict_sommovigo_bartlett_2026,
)

# Unpack the IGM transformers into this nice alias submodule
from synthesizer.emission_models.transformers.igm import (
    Asada25,
    Inoue14,
    Madau96,
)
