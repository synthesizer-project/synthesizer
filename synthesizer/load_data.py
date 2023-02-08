import sys
import h5py
import numpy as np

from astropy.cosmology import FlatLambdaCDM

from .galaxy import Galaxy


def load_CAMELS_SIMBA(_dir='.', snap='033'):

    with h5py.File(f'{_dir}/snap_{snap}.hdf5', 'r') as hf:
        form_time = hf['PartType4/StellarFormationTime'][:]
        coods = hf['PartType4/Coordinates'][:]
        masses = hf['PartType4/Masses'][:]  # TODO: not initial masses
        _metals = hf['PartType4/Metallicity'][:]

        scale_factor = hf['Header'].attrs[u'Time']
        Om0 = hf['Header'].attrs[u'Omega0']
        h = hf['Header'].attrs[u'HubbleParam']

    s_oxygen = _metals[:, 4]
    s_hydrogen = 1 - np.sum(_metals[:, 1:], axis=1)
    metals = _metals[:, 0]

    # convert formation times to ages
    cosmo = FlatLambdaCDM(H0=h*100, Om0=Om0)
    universe_age = cosmo.age(1. / scale_factor - 1)
    _ages = cosmo.age(1./form_time - 1)
    ages = (universe_age - _ages).value * 1e9  # yr

    with h5py.File(f'{_dir}/fof_subhalo_tab_{snap}.hdf5', 'r') as hf:
        lens = hf['Subhalo/SubhaloLenType'][:]

    begin, end = get_len(lens[:, 4])

    galaxies = [None] * len(begin)
    for i, (b, e) in enumerate(zip(begin, end)):
        galaxies[i] = Galaxy()
        # WARNING: initial masses set to current for now
        galaxies[i].load_stars(masses[b:e], ages[b:e], metals[b:e],
                               s_oxygen=s_oxygen[b:e],
                               s_hydrogen=s_hydrogen[b:e],
                               coordinates=coods[b:e, :],
                               initial_masses=masses[b:e])

    return galaxies


def load_FLARES(f, region, tag):
    with h5py.File(f, 'r') as hf:
        lens = hf[f'{region}/{tag}/Galaxy/S_Length'][:]
        ages = hf[f'{region}/{tag}/Particle/S_Age'][:]  # Gyr
        coods = hf[f'{region}/{tag}/Particle/S_Coordinates'][:].T
        mass = hf[f'{region}/{tag}/Particle/S_Mass'][:]  # 1e10 Msol
        imass = hf[f'{region}/{tag}/Particle/S_MassInitial'][:]  # 1e10 Msol
        # metals = hf[f'{region}/{tag}/Particle/S_Z'][:]
        metals = hf[f'{region}/{tag}/Particle/S_Z_smooth'][:]
        s_oxygen = hf[f'{region}/{tag}/Particle/S_Abundance_Oxygen'][:]
        s_hydrogen = hf[f'{region}/{tag}/Particle/S_Abundance_Hydrogen'][:]
        # ids = hf[f'{region}/{tag}/Particle/S_ID'][:]
        # index = hf[f'{region}/{tag}/Particle/S_Index'][:]
        # hf[f'{pre}/S_Vel']

    ages = (ages * 1e9)  # yr
    mass = mass * 1e10  # Msol
    imass = imass * 1e10  # Msol

    begin, end = get_len(lens)

    galaxies = [None] * len(begin)
    for i, (b, e) in enumerate(zip(begin, end)):
        galaxies[i] = Galaxy()
        galaxies[i].load_stars(mass[b:e], ages[b:e], metals[b:e],
                               s_oxygen=s_oxygen[b:e],
                               s_hydrogen=s_hydrogen[b:e],
                               coordinates=coods[b:e, :],
                               initial_masses=imass[b:e])

    return galaxies


def get_len(Length):
    begin = np.zeros(len(Length), dtype=np.int64)
    end = np.zeros(len(Length), dtype=np.int64)
    begin[1:] = np.cumsum(Length)[:-1]
    end = np.cumsum(Length)
    return begin, end


def main():
    # e.g.
    # '/cosma7/data/dp004/dc-love2/codes/flares/data/FLARES_30_sp_info.hdf5'
    _f = sys.argv[1]
    tag = sys.argv[2]  # '/010_z005p000/'
    load_FLARES(_f, tag)
    return None


if __name__ == "__main__":
    main()
