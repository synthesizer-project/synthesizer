"""A submodule for loading Simba data into Synthesizer.

Load Simba galaxy data from a caesar file and snapshot.

Method for loading galaxy and particle data for
the [Simba](http://simba.roe.ac.uk/) simulation
"""

import h5py
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from unyt import Mpc, Msun, km, kpc, s, yr

from synthesizer.load_data.utils import age_lookup_table, lookup_age
from synthesizer.synth_warnings import warn

from ..particle.galaxy import Galaxy
from ..particle.particles import Particles


def load_Simba(
    directory=".",
    snap_name="snap_033.hdf5",
    caesar_name="fof_subhalo_tab_033.hdf5",
    caesar_directory=None,
    object_indexes=False,
    load_halo=False,
    age_lookup=True,
    age_lookup_delta_a=1e-4,
):
    """Load Simba galaxy data from a caesar file and snapshot.

    This function can load all objects or a specific subset based on their
    indices. It uses the `slist` and `glist` datasets from the Caesar file
    to correctly map particles to their host galaxies/halos.

    Args:
        directory (string):
            Data location.
        snap_name (string):
            Snapshot filename.
        caesar_name (string):
            Caesar file name.
        caesar_directory (string):
            Optional path to the Caesar file if it's in a different
            directory from the snapshot.
        object_indexes (list or bool):
            List of object indices to load. If False, all objects are loaded.
        load_halo (bool):
            If True, loads halo objects instead of galaxy objects.
        age_lookup (bool):
            If True, creates and uses a lookup table for stellar ages for
            faster age calculation.
        age_lookup_delta_a (float):
            The resolution of the age lookup table in terms of scale factor.

    Returns:
        galaxies (list):
            A list of `synthesizer.particle.galaxy.Galaxy` objects.
    """
    # check which kind of object we're loading
    if load_halo:
        obj_str = "halo_data"
    else:
        obj_str = "galaxy_data"

    if caesar_directory is None:
        caesar_directory = directory

    # Open the Caesar file to get particle lists for each galaxy
    with h5py.File(f"{caesar_directory}/{caesar_name}", "r") as hf:
        scale_factor = hf["simulation_attributes/parameters"].attrs["Time"]

        slist_all = hf[f"{obj_str}/lists/slist"][:]
        slist_start_all = hf[f"{obj_str}/slist_start"][:]
        slist_end_all = hf[f"{obj_str}/slist_end"][:]

        glist_all = hf[f"{obj_str}/lists/glist"][:]
        glist_start_all = hf[f"{obj_str}/glist_start"][:]
        glist_end_all = hf[f"{obj_str}/glist_end"][:]

        centres_all = hf[f"{obj_str}/pos"][:] * scale_factor

    # Get the particle indices for the selected galaxies
    if object_indexes is False:
        galaxy_indices = np.arange(len(slist_start_all))

        star_particle_indices = slist_all
        gas_particle_indices = glist_all
    else:
        galaxy_indices = np.array(object_indexes)

        # Validate indices
        if np.any(galaxy_indices < 0) or np.any(
            galaxy_indices >= len(slist_start_all)
        ):
            raise ValueError(
                "Invalid object indices. Must be in range "
                "[0, {len(slist_start_all) - 1}]"
            )
        star_particle_indices = np.concatenate(
            [
                slist_all[slist_start_all[i] : slist_end_all[i]]
                for i in galaxy_indices
            ]
        )
        gas_particle_indices = np.concatenate(
            [
                glist_all[glist_start_all[i] : glist_end_all[i]]
                for i in galaxy_indices
            ]
        )

    centres = centres_all[galaxy_indices]

    with h5py.File(f"{directory}/{snap_name}", "r") as hf:
        scale_factor = hf["Header"].attrs["Time"]
        Om0 = hf["Header"].attrs["Omega0"]
        h = hf["Header"].attrs["HubbleParam"]

        # Load star data for selected particles
        form_time = hf["PartType4/StellarFormationTime"][:][
            star_particle_indices
        ]
        coods = (
            hf["PartType4/Coordinates"][:][star_particle_indices]
            * scale_factor
            / h
        )
        masses = hf["PartType4/Masses"][:][star_particle_indices]
        imasses = np.ones(len(masses)) * 0.00155
        _metals = hf["PartType4/Metallicity"][:][star_particle_indices]

        # Load gas data for selected particles
        g_masses = hf["PartType0/Masses"][:][gas_particle_indices]
        g_metals_all = hf["PartType0/Metallicity"][:][gas_particle_indices]
        g_metals = g_metals_all[:, 0]
        g_coods = (
            hf["PartType0/Coordinates"][:][gas_particle_indices]
            * scale_factor
            / h
        )
        g_hsml = hf["PartType0/SmoothingLength"][:][gas_particle_indices]
        g_dustmass = hf["PartType0/Dust_Masses"][:][gas_particle_indices]
        g_h2fraction = hf["PartType0/FractionH2"][:][gas_particle_indices]

    redshift = (1.0 / scale_factor) - 1

    # Convert units
    masses = (masses * 1e10) / h
    imasses = (imasses * 1e10) / h
    g_masses = (g_masses * 1e10) / h
    g_dustmass = (g_dustmass * 1e10) / h

    # Get individual and summed metallicity components for stars
    s_oxygen = _metals[:, 4]
    s_hydrogen = 1 - np.sum(_metals[:, 1:], axis=1)
    metallicity = _metals[:, 0]

    # Convert formation times to ages
    cosmo = FlatLambdaCDM(H0=h * 100, Om0=Om0)
    universe_age = cosmo.age(redshift)

    if age_lookup:
        scale_factors, ages_table = age_lookup_table(
            cosmo, redshift=redshift, delta_a=age_lookup_delta_a, low_lim=1e-4
        )
        _ages = lookup_age(form_time, scale_factors, ages_table)
    else:
        _ages = cosmo.age(1.0 / form_time - 1)

    ages = (universe_age - _ages).value * 1e9  # yr

    # Prepare to create Galaxy objects
    galaxies = []
    s_particle_counts = [
        slist_end_all[i] - slist_start_all[i] for i in galaxy_indices
    ]
    g_particle_counts = [
        glist_end_all[i] - glist_start_all[i] for i in galaxy_indices
    ]
    s_offsets = np.concatenate([[0], np.cumsum(s_particle_counts)[:-1]])
    g_offsets = np.concatenate([[0], np.cumsum(g_particle_counts)[:-1]])

    for i, gal_idx in enumerate(galaxy_indices):
        galaxy = Galaxy(centre=centres[i] * kpc)
        galaxy.redshift = redshift

        s_start, s_end = s_offsets[i], s_offsets[i] + s_particle_counts[i]
        g_start, g_end = g_offsets[i], g_offsets[i] + g_particle_counts[i]

        galaxy.load_stars(
            initial_masses=imasses[s_start:s_end] * Msun,
            ages=ages[s_start:s_end] * yr,
            metallicities=metallicity[s_start:s_end],
            s_oxygen=s_oxygen[s_start:s_end],
            s_hydrogen=s_hydrogen[s_start:s_end],
            coordinates=coods[s_start:s_end, :] * kpc,
            current_masses=masses[s_start:s_end] * Msun,
            centre=centres[i] * kpc,
        )

        h2_masses = g_masses[g_start:g_end] * g_h2fraction[g_start:g_end]
        galaxy.sf_gas_mass = np.sum(h2_masses) * Msun
        if galaxy.sf_gas_mass > 0:
            galaxy.sf_gas_metallicity = (
                np.sum(h2_masses * g_metals[g_start:g_end])
                / galaxy.sf_gas_mass.to(Msun).value
            )
        else:
            galaxy.sf_gas_metallicity = 0.0

        galaxy.load_gas(
            coordinates=g_coods[g_start:g_end] * kpc,
            masses=g_masses[g_start:g_end] * Msun,
            metallicities=g_metals[g_start:g_end],
            smoothing_lengths=g_hsml[g_start:g_end] * kpc,
            dust_masses=g_dustmass[g_start:g_end] * Msun,
            centre=centres[i] * kpc,
        )
        galaxies.append(galaxy)

    return galaxies


def load_Simba_slab(
    directory=".",
    snap_name="snap_033.hdf5",
    bounds=None,
    load_stars=True,
    load_gas=True,
    load_dark_matter=False,
    age_lookup=True,
    age_lookup_delta_a=1e-4,
):
    """Load a cuboid region of Simba simulation data into a Galaxy object.

    This function loads all particles within a specified cuboid region from
    a Simba snapshot, regardless of which galaxy they belong to, and creates
    a single Galaxy object containing all particles in that region.

    Args:
        directory (string):
            Data location.
        snap_name (string):
            Snapshot filename.
        bounds (array-like):
            Cuboid bounds as [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
            in physical Mpc units.
        load_stars (bool):
            If True, loads stellar particles (PartType4).
        load_gas (bool):
            If True, loads gas particles (PartType0).
        load_dark_matter (bool):
            If True, loads dark matter particles (PartType1).
        age_lookup (bool):
            If True, creates and uses a lookup table for stellar ages for
            faster age calculation.
        age_lookup_delta_a (float):
            The resolution of the age lookup table in terms of scale factor.

    Returns:
        galaxy (synthesizer.particle.galaxy.Galaxy):
            A Galaxy object containing all particles in the specified region.
    """
    if bounds is None:
        raise ValueError("bounds parameter is required")

    if not any([load_stars, load_gas, load_dark_matter]):
        raise ValueError("At least one particle type must be loaded")

    bounds = np.array(bounds)
    if bounds.shape != (3, 2):
        raise ValueError(
            "bounds must have shape (3, 2) for "
            "[[xmin, xmax], [ymin, ymax], [zmin, zmax]]"
        )

    bounds = (bounds * Mpc).to("kpc").value

    with h5py.File(f"{directory}/{snap_name}", "r") as hf:
        scale_factor = hf["Header"].attrs["Time"]
        Om0 = hf["Header"].attrs["Omega0"]
        h = hf["Header"].attrs["HubbleParam"]

        # Initialize variables
        form_time = masses = imasses = _metals = coods = None
        g_masses = g_metals_all = g_metals = g_coods = g_hsml = g_dustmass = (
            g_h2fraction
        ) = None
        dm_masses = dm_coods = None

        # Load stellar particles if requested
        if load_stars:
            star_coords_raw = hf["PartType4/Coordinates"][:]
            star_coords = (
                star_coords_raw * scale_factor / h
            )  # Convert to physical kpc

            # Select star particles within bounds
            star_mask = (
                (star_coords[:, 0] >= bounds[0, 0])
                & (star_coords[:, 0] <= bounds[0, 1])
                & (star_coords[:, 1] >= bounds[1, 0])
                & (star_coords[:, 1] <= bounds[1, 1])
                & (star_coords[:, 2] >= bounds[2, 0])
                & (star_coords[:, 2] <= bounds[2, 1])
            )

            if np.sum(star_mask) == 0:
                warn("No star particles found in specified region")

            # Load star data for selected particles
            form_time = hf["PartType4/StellarFormationTime"][:][star_mask]
            coods = star_coords[star_mask]
            masses = hf["PartType4/Masses"][:][star_mask]
            imasses = np.ones(len(masses)) * 0.00155
            _metals = hf["PartType4/Metallicity"][:][star_mask]

        # Load gas particles if requested
        if load_gas:
            gas_coords_raw = hf["PartType0/Coordinates"][:]
            gas_coords = (
                gas_coords_raw * scale_factor / h
            )  # Convert to physical kpc

            # Select gas particles within bounds
            gas_mask = (
                (gas_coords[:, 0] >= bounds[0, 0])
                & (gas_coords[:, 0] <= bounds[0, 1])
                & (gas_coords[:, 1] >= bounds[1, 0])
                & (gas_coords[:, 1] <= bounds[1, 1])
                & (gas_coords[:, 2] >= bounds[2, 0])
                & (gas_coords[:, 2] <= bounds[2, 1])
            )

            if np.sum(gas_mask) == 0:
                warn("No gas particles found in specified region")

            # Load gas data for selected particles
            g_masses = hf["PartType0/Masses"][:][gas_mask]
            g_metals_all = hf["PartType0/Metallicity"][:][gas_mask]
            g_metals = (
                g_metals_all[:, 0] if len(g_metals_all) > 0 else np.array([])
            )
            g_coods = gas_coords[gas_mask]
            g_hsml = hf["PartType0/SmoothingLength"][:][gas_mask]
            g_dustmass = hf["PartType0/Dust_Masses"][:][gas_mask]
            g_h2fraction = hf["PartType0/FractionH2"][:][gas_mask]
            g_internalenergy = hf["PartType0/InternalEnergy"][:][gas_mask]

        # Load dark matter particles if requested
        if load_dark_matter:
            dm_coords_raw = hf["PartType1/Coordinates"][:]
            dm_coords = (
                dm_coords_raw * scale_factor / h
            )  # Convert to physical kpc

            # Select dark matter particles within bounds
            dm_mask = (
                (dm_coords[:, 0] >= bounds[0, 0])
                & (dm_coords[:, 0] <= bounds[0, 1])
                & (dm_coords[:, 1] >= bounds[1, 0])
                & (dm_coords[:, 1] <= bounds[1, 1])
                & (dm_coords[:, 2] >= bounds[2, 0])
                & (dm_coords[:, 2] <= bounds[2, 1])
            )

            if np.sum(dm_mask) == 0:
                warn("No dark matter particles found in specified region")

            # Load dark matter data for selected particles
            dm_masses = hf["PartType1/Masses"][:][dm_mask]
            dm_coods = dm_coords[dm_mask]

    redshift = (1.0 / scale_factor) - 1

    # Convert units for loaded particle types
    if load_stars and masses is not None:
        masses = (masses * 1e10) / h
        imasses = (imasses * 1e10) / h
    if load_gas and g_masses is not None:
        g_masses = (g_masses * 1e10) / h
        g_dustmass = (g_dustmass * 1e10) / h
    if load_dark_matter and dm_masses is not None:
        dm_masses = (dm_masses * 1e10) / h

    # Calculate center of region
    center = np.mean(bounds, axis=1)

    # Process stellar data if we have star particles
    if load_stars and masses is not None and len(masses) > 0:
        # Get individual and summed metallicity components for stars
        s_oxygen = _metals[:, 4]
        s_hydrogen = 1 - np.sum(_metals[:, 1:], axis=1)
        metallicity = _metals[:, 0]

        # Convert formation times to ages
        cosmo = FlatLambdaCDM(H0=h * 100, Om0=Om0)
        universe_age = cosmo.age(redshift)

        if age_lookup:
            scale_factors, ages_table = age_lookup_table(
                cosmo,
                redshift=redshift,
                delta_a=age_lookup_delta_a,
                low_lim=1e-4,
            )
            _ages = lookup_age(form_time, scale_factors, ages_table)
        else:
            _ages = cosmo.age(1.0 / form_time - 1)

        ages = (universe_age - _ages).value * 1e9  # yr

    # Create Galaxy object
    galaxy = Galaxy()
    galaxy.redshift = redshift

    # Load star data if requested and available
    if load_stars and masses is not None and len(masses) > 0:
        galaxy.load_stars(
            initial_masses=imasses * Msun,
            ages=ages * yr,
            metallicities=metallicity,
            s_oxygen=s_oxygen,
            s_hydrogen=s_hydrogen,
            coordinates=coods * kpc,
            current_masses=masses * Msun,
            centre=center * kpc,
        )

    # Load gas data if requested and available
    if load_gas and g_masses is not None and len(g_masses) > 0:
        # Calculate star-forming gas properties
        h2_masses = g_masses * g_h2fraction
        galaxy.sf_gas_mass = np.sum(h2_masses) * Msun
        if galaxy.sf_gas_mass > 0:
            galaxy.sf_gas_metallicity = (
                np.sum(h2_masses * g_metals)
                / galaxy.sf_gas_mass.to(Msun).value
            )
        else:
            galaxy.sf_gas_metallicity = 0.0

        # Load gas data
        galaxy.load_gas(
            coordinates=g_coods * kpc,
            masses=g_masses * Msun,
            metallicities=g_metals,
            smoothing_lengths=g_hsml * kpc,
            dust_masses=g_dustmass * Msun,
            centre=center * kpc,
            internal_energy=g_internalenergy,
        )
    else:
        galaxy.sf_gas_mass = 0.0 * Msun
        galaxy.sf_gas_metallicity = 0.0

    # Load dark matter data if requested and available
    if load_dark_matter and dm_masses is not None and len(dm_masses) > 0:
        # Create dark matter Particles object
        # Note: DM particles don't have velocities or metallicities
        # in this context
        dm_particles = Particles(
            coordinates=dm_coods * kpc,
            velocities=np.zeros((len(dm_masses), 3))
            * (km / s),  # No velocity data
            masses=dm_masses * Msun,
            redshift=redshift,
            softening_lengths=np.zeros(len(dm_masses))
            * kpc,  # No softening data
            nparticles=len(dm_masses),
            centre=center * kpc,
        )
        galaxy.dark_matter = dm_particles

    return galaxy
