"""A factory function to return the correct galaxy type.

This function means the user doesn't need to worry which galaxy they need to
import. They only need to import the galaxy function, and pass the arguments
they have to hand.

If Particles derived objects are passed a particle galaxy is initialised and
returned. If a ParametricStars object is passed then a parametric galaxy
is initialised and returned.

Example:
from synthesizer import galaxy

# Get a particle galaxy
gal = galaxy(stars=particle.Stars(...), gas=Gas(...), ...)

OR

# Get a parametric galaxy
gal = galaxy(stars=parametric.Stars(...),  ...)

"""

from typing import Optional, Union

from synthesizer.exceptions import InconsistentArguments
from synthesizer.parametric import Galaxy as ParametricGalaxy
from synthesizer.parametric import Stars as ParametricStars
from synthesizer.particle import BlackHoles, Gas
from synthesizer.particle import Galaxy as ParticleGalaxy
from synthesizer.particle import Stars as ParticleStars


def galaxy(
    stars: Optional[Union[ParticleStars, ParametricStars]] = None,
    gas: Optional[Gas] = None,
    black_holes: Optional[BlackHoles] = None,
    redshift: Optional[float] = None,
    name: str = "galaxy",
) -> Union[ParticleGalaxy, ParametricGalaxy]:
    """A factory fucntion to return the desired type of galaxy.

    This function provides a simple interface to ensure the user doesn't try to
    utilise the wrong flavour of Galaxy object.

    If Particles derived objects are passed a particle galaxy is
    initialised and returned. If a SFZH is passed then a parametric galaxy is
    initialised and returned. If an incompatible combination is passed then an
    error is raised.

    Args:
        name: A name to identify the galaxy. Only used for external labelling,
              has no internal use.
        stars: An instance of Stars containing the stellar particle data or an
               instance of Stars containing the combined star formation and
               metallicity histories. The former is only applicable to a
               particle.Galaxy while the latter is only applicable to a
               parametric.Galaxy.
        gas: An instance of Gas containing the gas particle data. Only
             applicable to a particle.Galaxy.
        black_holes: An instance of BlackHoles containing the black hole
                     particle data. Only applicable to a particle.Galaxy.
        redshift: The redshift of the galaxy.

    Returns:
            The appropriate Galaxy object based on input arguments.

    Raises:
        InconsistentArguments: If passed both particle and parametric arguments
                               an error is raised.
    """

    # Ensure the passed arguments make sense
    if isinstance(stars, ParametricStars):
        if gas is not None or black_holes is not None:
            raise InconsistentArguments(
                "A parametric Stars has been passed in conjunction with "
                "particle based gas or black hole objects. These are "
                "incompatible. Did you mean to pass a particle based Stars "
                "object?"
            )

    # Now we know we are ok, instantiate the correct object
    if isinstance(stars, ParametricStars):
        return ParametricGalaxy(
            stars=stars,
            redshift=redshift,
            black_holes=black_holes,
            name=name,
        )

    # Otherwise, we need a particle.Galaxy
    return ParticleGalaxy(
        name=name,
        stars=stars,
        gas=gas,
        black_holes=black_holes,
        redshift=redshift,
    )
