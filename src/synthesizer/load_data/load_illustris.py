import numpy as np
from astropy.cosmology import FlatLambdaCDM
from tqdm import tqdm
from unyt import Msun, kpc, yr

try:
    import illustris_python as il
except ImportError:
    print(
        "The `illustris_python` module is not installed. "
        "Please refer to the website for installation instructions: "
        "https://github.com/illustristng/illustris_python"
        "\nExiting..."
    )
    exit


from ..particle.galaxy import Galaxy


def load_IllustrisTNG(
    directory=".",
    snap_number=99,
    stellar_mass_limit=8.5e6,
    verbose=True,
    dtm=0.3,
    physical=True,
    metals=True,
):
    """
    Load IllustrisTNG particle data into galaxy objects

    Uses the `illustris_python` module, which must be installed manually.
    Loads the particles associated with each subhalo individually, rather than
    the whole simulation volume particle arrays; this can be slower for certain
    volumes, but avoids memory issues for higher resolution boxes.

    Args:
        directory (string):
            Data location of group and particle files.
        snap_number (int):
            Snapshot number.
        stellar_mass_limit (float):
            Stellar mass limit above which to load galaxies.
            In units of solar mass.
        verbose (bool):
            Verbosity flag
        dtm (float):
            Dust-to-metals ratio to apply to all gas particles
        physical (bool):
            Should the coordinates be converted to physical?
            Default: True
        metals (bool):
            Should we load individual stellar element abundances?
            Default: True. This property may not be available for
            all snapshots ( see
            https://www.tng-project.org/data/docs/specifications/#sec1a ).

    Returns:
        galaxies (list):
            List of `ParticleGalaxy` objects, each containing star and
            gas particles
        subhalo_mask (array, bool):
            Boolean array of selected galaxies from the subhalo catalogue.
    """

    # Do some simple argument
    snap_number = int(snap_number)

    if verbose:
        print("Loading header information...")

    header = il.groupcat.loadHeader(directory, snap_number)
    scale_factor = header['Time']
    redshift = header['Redshift']
    Om0 = header['Omega0']
    h = header['HubbleParam']

    if verbose:
        print("Loading subhalo catalogue...")

    fields = ['SubhaloMassType', 'SubhaloPos']
    output = il.groupcat.loadSubhalos(directory, 99, fields=fields)
    stellar_mass = output['SubhaloMassType'][:, 4]
    subhalo_mask = (stellar_mass * 1e10) > stellar_mass_limit
    subhalo_pos = output['SubhaloPos'][subhalo_mask]

    if verbose:
        print(
            f"Loaded {np.sum(subhalo_mask)} galaxies "
            f"above the stellar mass cut ({stellar_mass_limit} M_solar)"
        )

    galaxies = [None] * np.sum(subhalo_mask)

    if verbose:
        print("Loading particle information...")

    for i, (idx, pos) in tqdm(
        enumerate(
            zip(
                np.where(subhalo_mask)[0],
                subhalo_pos
            )
        ),
        total=np.sum(subhalo_mask)
    ):
        galaxies[i] = Galaxy(verbose=False)
        galaxies[i].redshift = redshift

        if physical:
            pos *= scale_factor

        galaxies[i].centre = pos * kpc

        star_fields = [
            'GFM_StellarFormationTime',
            'Coordinates',
            'Masses',
            'GFM_InitialMass',
            'GFM_Metallicity',
            'SubfindHsml',
        ]
        if metals:
            star_fields.append('GFM_Metals')

        output = il.snapshot.loadSubhalo(
            directory, snap_number, idx, 'stars', fields=star_fields
        )

        if output['count'] > 0:

            # Mask for wind particles
            mask = output['GFM_StellarFormationTime'] <= 0.0

            # filter particle arrays
            imasses = output['GFM_InitialMass'][~mask]
            form_time = output['GFM_StellarFormationTime'][~mask]
            coods = output['Coordinates'][~mask]
            metallicities = output['GFM_Metallicity'][~mask]
            masses = output['Masses'][~mask]
            hsml = output['SubfindHsml'][~mask]

            masses = (masses * 1e10) / h
            imasses = (imasses * 1e10) / h

            if metals:
                _metals = output['GFM_Metals'][~mask]
                s_oxygen = _metals[:, 4]
                s_hydrogen = 1.0 - np.sum(_metals[:, 1:], axis=1)
            else:
                s_oxygen = None
                s_hydrogen = None

            # Convert comoving coordinates to physical kpc
            if physical:
                coods *= scale_factor
                hsml *= scale_factor

            # convert formation times to ages
            cosmo = FlatLambdaCDM(H0=h * 100, Om0=Om0)
            universe_age = cosmo.age(1.0 / scale_factor - 1)
            _ages = cosmo.age(1.0 / form_time - 1)
            ages = (universe_age - _ages).value * 1e9  # yr

            if hsml is None:
                smoothing_lengths = hsml
            else:
                smoothing_lengths = hsml * kpc

            galaxies[i].load_stars(
                initial_masses=imasses * Msun,
                ages=ages * yr,
                metallicities=metallicities,
                s_oxygen=s_oxygen,
                s_hydrogen=s_hydrogen,
                coordinates=coods * kpc,
                current_masses=masses * Msun,
                smoothing_lengths=smoothing_lengths,
            )

        gas_fields = [
            'StarFormationRate',
            'Coordinates',
            'Masses',
            'GFM_Metallicity',
            'SubfindHsml',
        ]
        output = il.snapshot.loadSubhalo(
            directory, snap_number, idx, 'gas', fields=gas_fields
        )

        if output['count'] > 0:

            g_masses = output['Masses']
            g_sfr = output['StarFormationRate']
            g_coods = output['Coordinates']
            g_hsml = output['SubfindHsml']
            g_metals = output['GFM_Metallicity']

            g_masses = (g_masses * 1e10) / h
            star_forming = g_sfr > 0.0

            # Convert comoving coordinates to physical kpc
            if physical:
                coods *= scale_factor
                g_coods *= scale_factor
                g_hsml *= scale_factor

            galaxies[i].load_gas(
                coordinates=g_coods * kpc,
                masses=g_masses * Msun,
                metallicities=g_metals,
                star_forming=star_forming,
                smoothing_lengths=g_hsml * kpc,
                dust_to_metal_ratio=dtm,
            )

    return galaxies, subhalo_mask
