import h5py
import numpy as np
from numpy import float64, int32
from numpy.typing import NDArray

_dir: str = "."
snap_name: str = "LH_1_snap_031.hdf5"
fof_name: str = "LH_1_fof_subhalo_tab_031.hdf5"
N: int = 10  # number of galaxies to extract
ignore_N: int = 1  # number of galaxies to ignore

with h5py.File(f"{_dir}/{snap_name}", "r") as hf:
    form_time: NDArray[float64] = hf["PartType4/GFM_StellarFormationTime"][:]
    coods: NDArray[float64] = hf["PartType4/Coordinates"][:]
    masses: NDArray[float64] = hf["PartType4/Masses"][:]
    imasses: NDArray[float64] = hf["PartType4/GFM_InitialMass"][:]
    _metals: NDArray[float64] = hf["PartType4/GFM_Metals"][:]
    metallicity: NDArray[float64] = hf["PartType4/GFM_Metallicity"][:]
    hsmls: NDArray[float64] = hf["PartType4/SubfindHsml"][:]

    g_sfr: NDArray[float64] = hf["PartType0/StarFormationRate"][:]
    g_masses: NDArray[float64] = hf["PartType0/Masses"][:]
    g_metals: NDArray[float64] = hf["PartType0/GFM_Metallicity"][:]
    g_coods: NDArray[float64] = hf["PartType0/Coordinates"][:]
    g_hsml: NDArray[float64] = hf["PartType0/SubfindHsml"][:]

    scale_factor: float = hf["Header"].attrs["Time"]
    Om0: float = hf["Header"].attrs["Omega0"]
    h: float = hf["Header"].attrs["HubbleParam"]


with h5py.File(f"{_dir}/{fof_name}", "r") as hf:
    lens: NDArray[int32] = hf["Subhalo/SubhaloLenType"][:]
    pos: NDArray[np.int64] = hf["Subhalo/SubhaloPos"][:]

lens4: NDArray[int32] = np.append(0, np.cumsum(lens[: ignore_N + N, 4]))
lens0: NDArray[int32] = np.append(0, np.cumsum(lens[: ignore_N + N, 0]))

# ignore the first `ignore_N` galaxies (often massive)
lens4 = lens4[ignore_N:]
lens0 = lens0[ignore_N:]

with h5py.File("camels_snap.hdf5", "w") as hf:
    hf.require_group("PartType4")
    hf.require_group("PartType0")

    hf.require_group("Header")

    hf.create_dataset(
        "PartType4/GFM_StellarFormationTime",
        data=form_time[lens4[0] : lens4[-1]],
    )
    hf.create_dataset(
        "PartType4/Coordinates", data=coods[lens4[0] : lens4[-1], :]
    )
    hf.create_dataset("PartType4/Masses", data=masses[lens4[0] : lens4[-1]])
    hf.create_dataset(
        "PartType4/GFM_InitialMass", data=imasses[lens4[0] : lens4[-1]]
    )
    hf.create_dataset(
        "PartType4/GFM_Metals", data=_metals[lens4[0] : lens4[-1]]
    )
    hf.create_dataset(
        "PartType4/GFM_Metallicity", data=metallicity[lens4[0] : lens4[-1]]
    )
    hf.create_dataset(
        "PartType4/SubfindHsml", data=hsmls[lens4[0] : lens4[-1]]
    )

    hf.create_dataset(
        "PartType0/StarFormationRate", data=g_sfr[lens0[0] : lens0[-1]]
    )
    hf.create_dataset("PartType0/Masses", data=g_masses[lens0[0] : lens0[-1]])
    hf.create_dataset(
        "PartType0/GFM_Metallicity", data=g_metals[lens0[0] : lens0[-1]]
    )
    hf.create_dataset(
        "PartType0/Coordinates", data=g_coods[lens0[0] : lens0[-1], :]
    )
    hf.create_dataset(
        "PartType0/SubfindHsml", data=g_hsml[lens0[0] : lens0[-1]]
    )

    hf["Header"].attrs["Time"] = scale_factor
    hf["Header"].attrs["Omega0"] = Om0
    hf["Header"].attrs["HubbleParam"] = h

with h5py.File("camels_subhalo.hdf5", "w") as hf:
    hf.require_group("Subhalo")
    hf.create_dataset(
        "Subhalo/SubhaloLenType", data=lens[ignore_N : (ignore_N + N), :]
    )
    hf.create_dataset(
        "Subhalo/SubhaloPos", data=pos[ignore_N : (ignore_N + N), :]
    )
