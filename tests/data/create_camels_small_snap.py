import numpy as np
import h5py

_dir = "."
snap_name = "LH_1_snap_031.hdf5"
fof_name = "LH_1_fof_subhalo_tab_031.hdf5"
N = 10  # number of galaxies to extract

with h5py.File(f"{_dir}/{snap_name}", "r") as hf:
    form_time = hf["PartType4/GFM_StellarFormationTime"][:]
    coods = hf["PartType4/Coordinates"][:]
    masses = hf["PartType4/Masses"][:]
    imasses = hf["PartType4/GFM_InitialMass"][:]
    _metals = hf["PartType4/GFM_Metals"][:]
    metallicity = hf["PartType4/GFM_Metallicity"][:]
    hsmls = hf["PartType4/SubfindHsml"][:]

    g_sfr = hf["PartType0/StarFormationRate"][:]
    g_masses = hf["PartType0/Masses"][:]
    g_metals = hf["PartType0/GFM_Metallicity"][:]
    g_coods = hf["PartType0/Coordinates"][:]
    g_hsml = hf["PartType0/SubfindHsml"][:]

    scale_factor = hf["Header"].attrs["Time"]
    Om0 = hf["Header"].attrs["Omega0"]
    h = hf["Header"].attrs["HubbleParam"]


with h5py.File(f"{_dir}/{fof_name}", "r") as hf:
    lens = hf["Subhalo/SubhaloLenType"][:]

lens4 = np.cumsum(lens[:N, 4])
lens0 = np.cumsum(lens[:N, 0])

with h5py.File(f"camels_snap.hdf5", "w") as hf:
    hf.require_group("PartType4")
    hf.require_group("PartType0")

    hf.require_group("Header")

    hf.create_dataset(
        "PartType4/GFM_StellarFormationTime", data=form_time[lens4[0] : lens4[-1]]
    )
    hf.create_dataset("PartType4/Coordinates", data=coods[lens4[0] : lens4[-1], :])
    hf.create_dataset("PartType4/Masses", data=masses[lens4[0] : lens4[-1]])
    hf.create_dataset("PartType4/GFM_InitialMass", data=imasses[lens4[0] : lens4[-1]])
    hf.create_dataset("PartType4/GFM_Metals", data=_metals[lens4[0] : lens4[-1]])
    hf.create_dataset(
        "PartType4/GFM_Metallicity", data=metallicity[lens4[0] : lens4[-1]]
    )
    hf.create_dataset("PartType4/SubfindHsml", data=hsmls[lens4[0] : lens4[-1]])

    hf.create_dataset("PartType0/StarFormationRate", data=g_sfr[lens0[0] : lens0[-1]])
    hf.create_dataset("PartType0/Masses", data=g_masses[lens0[0] : lens0[-1]])
    hf.create_dataset("PartType0/GFM_Metallicity", data=g_metals[lens0[0] : lens0[-1]])
    hf.create_dataset("PartType0/Coordinates", data=g_coods[lens0[0] : lens0[-1], :])
    hf.create_dataset("PartType0/SubfindHsml", data=g_hsml[lens0[0] : lens0[-1]])

    hf["Header"].attrs["Time"] = scale_factor
    hf["Header"].attrs["Omega0"] = Om0
    hf["Header"].attrs["HubbleParam"] = h

with h5py.File(f"camels_subhalo.hdf5", "w") as hf:
    hf.require_group("Subhalo")
    hf.create_dataset("Subhalo/SubhaloLenType", data=lens[1:N])
