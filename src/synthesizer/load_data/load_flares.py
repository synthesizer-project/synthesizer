import h5py

from unyt import Msun, kpc, yr

from ..particle.galaxy import Galaxy
from synthesizer.load_data.utils import get_len


def load_FLARES(f, region, tag):
    """
    Load FLARES galaxies from a FLARES master file

    Args:
        f (string):
            master file location
        region (string):
            FLARES region to load
        tag (string):
            snapshot tag to load

    Returns:
        galaxies (object):
            `ParticleGalaxy` object containing stars
    """

    with h5py.File(f, "r") as hf:
        lens = hf[f"{region}/{tag}/Galaxy/S_Length"][:]
        ages = hf[f"{region}/{tag}/Particle/S_Age"][:]  # Gyr
        coods = hf[f"{region}/{tag}/Particle/S_Coordinates"][:].T
        mass = hf[f"{region}/{tag}/Particle/S_Mass"][:]  # 1e10 Msol
        imass = hf[f"{region}/{tag}/Particle/S_MassInitial"][:]  # 1e10 Msol
        # metals = hf[f'{region}/{tag}/Particle/S_Z'][:]
        metals = hf[f"{region}/{tag}/Particle/S_Z_smooth"][:]
        s_oxygen = hf[f"{region}/{tag}/Particle/S_Abundance_Oxygen"][:]
        s_hydrogen = hf[f"{region}/{tag}/Particle/S_Abundance_Hydrogen"][:]
        # ids = hf[f'{region}/{tag}/Particle/S_ID'][:]
        # index = hf[f'{region}/{tag}/Particle/S_Index'][:]
        # hf[f'{pre}/S_Vel']

    ages = ages * 1e9  # yr
    mass = mass * 1e10  # Msol
    imass = imass * 1e10  # Msol

    begin, end = get_len(lens)

    galaxies = [None] * len(begin)
    for i, (b, e) in enumerate(zip(begin, end)):
        galaxies[i] = Galaxy()
        galaxies[i].load_stars(
            mass[b:e] * Msun,
            ages[b:e] * yr,
            metals[b:e],
            s_oxygen=s_oxygen[b:e],
            s_hydrogen=s_hydrogen[b:e],
            coordinates=coods[b:e, :] * kpc,
        )

    return galaxies
