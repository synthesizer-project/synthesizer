# Unpack the attenuation laws into this nice alias submodule
from synthesizer.emission_models.transformers.dust_attenuation import (
    MWN18,
    Calzetti2000,
    GrainsWD01,
    ParametricLi08,
    PowerLaw,
)

# Unpack the IGM transformers into this nice alias submodule
from synthesizer.emission_models.transformers.igm import (
    Inoue14,
    Madau96,
)
