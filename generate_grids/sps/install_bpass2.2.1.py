"""
Download BPASS v2.2.1 and convert to HDF5 synthesizer grid.

Updated with new naming scheme.
"""

import os
import argparse
import numpy as np
import gdown
import tarfile

from hoki import load

from utils import (
    write_data_h5py,
    write_attribute,
    add_log10Q,
    get_model_filename,
)


def resolve_name(original_model_name, bin):
    """Resolve the original BPASS model name into what we need"""

    bpass_imf = original_model_name.split("imf")[-1]
    hmc = float(bpass_imf[-3:])  # high-mass cutoff

    if bpass_imf[:5] == "_chab":
        imf_type = "chabrier03"
        imf_masses = [0.1, hmc]
        imf_slopes = ""
    else:
        imf_type = "bpl"
        imf_masses = [0.1, 1.0, hmc]
        imf_slopes = [1.3, np.round(float(bpass_imf[:3]) / 100 + 1, 2)]

    model = {
        "original_model_name": original_model_name,
        "sps_name": "bpass",
        "sps_version": "2.2.1",
        "sps_variant": bin,
        "imf_type": imf_type,  # named IMF or bpl (broken power law)
        "imf_masses": imf_masses,
        "imf_slopes": imf_slopes,
        "alpha": "",
    }

    return model, bpass_imf


def download_data(model):
    """At the moment this is downloading data from a mirror I set up since I can't automate downloading the grids"""

    model_url = {}
    model_url[
        "bpass_v2.2.1_chab100"
    ] = "https://drive.google.com/file/d/1az7_hP3RDovr-BN9sXgDuaYqOZHHUeXD/view?usp=sharing"
    model_url[
        "bpass_v2.2.1_chab300"
    ] = "https://drive.google.com/file/d/1JcUM-qyOQD16RdfWjhGKSTwdNfRUW4Xu/view?usp=sharing"
    print(model_url)
    if model in model_url.keys():
        filename = gdown.download(model_url[model], quiet=False, fuzzy=True)
        return filename
    else:
        raise ValueError("ERROR: no url for that model")


def untar_data(model, filename, synthesizer_data_dir):

    model_dir = f"{synthesizer_data_dir}/input_files/bpass/{model}"
    with tarfile.open(filename) as tar:
        tar.extractall(path=model_dir)

    os.remove(filename)


def make_grid(original_model_name, bin):

    # returns a dictionary containing the sps model parameters
    model, bpass_imf = resolve_name(original_model_name, bin)

    # generate the synthesizer_model_name
    synthesizer_model_name = get_model_filename(model)

    # this is the full path to the ultimate HDF5 grid file
    out_filename = f"{synthesizer_data_dir}/grids/{synthesizer_model_name}.hdf5"

    # input director
    input_dir = f'{synthesizer_data_dir}/input_files/bpass/{model["original_model_name"]}/'

    # --- ccreate metallicity grid and dictionary
    Zk_to_Z = {
        "zem5": 0.00001,
        "zem4": 0.0001,
        "z001": 0.001,
        "z002": 0.002,
        "z003": 0.003,
        "z004": 0.004,
        "z006": 0.006,
        "z008": 0.008,
        "z010": 0.01,
        "z014": 0.014,
        "z020": 0.020,
        "z030": 0.030,
        "z040": 0.040,
    }

    Z_to_Zk = {k: v for v, k in Zk_to_Z.items()}
    Zs = np.sort(np.array(list(Z_to_Zk.keys())))
    print(f"metallicities: {Zs}")

    # get ages
    fn_ = f"starmass-{bin}-imf{bpass_imf}.{Z_to_Zk[Zs[0]]}.dat.gz"
    starmass = load.model_output(f"{input_dir}/{fn_}")
    log10ages = starmass["log_age"].values
    print(f"log10ages: {log10ages}")

    # get wavelength grid
    fn_ = f"spectra-{bin}-imf{bpass_imf}.{Z_to_Zk[Zs[0]]}.dat.gz"
    spec = load.model_output(f"{input_dir}/{fn_}")
    wavelengths = spec["WL"].values  # \AA
    nu = 3e8 / (wavelengths * 1e-10)

    # set up output arrays
    nZ = len(Zs)
    na = len(log10ages)
    stellar_mass = np.zeros((na, nZ))
    remnant_mass = np.zeros((na, nZ))
    spectra = np.zeros((na, nZ, len(wavelengths)))

    # loop over metallicity
    for iZ, Z in enumerate(Zs):

        # --- get remaining and remnant fraction
        fn_ = f"starmass-{bin}-imf{bpass_imf}.{Z_to_Zk[Z]}.dat.gz"
        starmass = load.model_output(f"{input_dir}/{fn_}")
        stellar_mass[:, iZ] = starmass["stellar_mass"].values / 1e6
        remnant_mass[:, iZ] = starmass["remnant_mass"].values / 1e6

        # --- get spectra
        fn_ = f"spectra-{bin}-imf{bpass_imf}.{Z_to_Zk[Z]}.dat.gz"
        spec = load.model_output(f"{input_dir}/{fn_}")

        for ia, log10age in enumerate(log10ages):
            spectra[ia, iZ, :] = spec[
                str(log10age)
            ].values  # Lsol AA^-1 10^6 Msol^-1

    # convert spectra to synthesizer base units
    spectra /= 1e6  # Lsol AA^-1 Msol^-1
    spectra *= 3.826e33  # erg s^-1 AA^-1 Msol^-1
    spectra *= wavelengths / nu  # erg s^-1 Hz^-1 Msol^-1

    # write out model parameters as top level attribute
    for key, value in model.items():
        # print(key, value)
        write_attribute(out_filename, "/", key, (value))

    # write out remaining stellar mass and remnant fractions
    write_data_h5py(
        out_filename, "star_fraction", data=stellar_mass, overwrite=True
    )
    write_attribute(
        out_filename,
        "star_fraction",
        "Description",
        ("Two-dimensional remaining stellar " "fraction grid, [age,Z]"),
    )

    write_data_h5py(
        out_filename, "remnant_fraction", data=remnant_mass, overwrite=True
    )
    write_attribute(
        out_filename,
        "remnant_fraction",
        "Description",
        ("Two-dimensional remaining remnant " "fraction grid, [age,Z]"),
    )

    # write out stellar spectra
    write_data_h5py(
        out_filename, "spectra/stellar", data=spectra, overwrite=True
    )
    write_attribute(
        out_filename,
        "spectra/stellar",
        "Description",
        "Three-dimensional spectra grid, [Z,Age,wavelength]",
    )
    write_attribute(out_filename, "spectra/stellar", "Units", "erg/s/Hz")

    # write out log10ages
    write_data_h5py(out_filename, "log10ages", data=log10ages, overwrite=True)
    write_attribute(
        out_filename,
        "log10ages",
        "Description",
        "Stellar population ages in log10 years",
    )
    write_attribute(out_filename, "log10ages", "Units", "dex(yr)")

    # write out metallicities
    write_data_h5py(out_filename, "metallicities", data=Zs, overwrite=True)
    write_attribute(
        out_filename, "metallicities", "Description", "raw abundances"
    )
    write_attribute(
        out_filename, "metallicities", "Units", "dimensionless [log10(Z)]"
    )

    # write out wavelength grid
    write_data_h5py(
        out_filename, "spectra/wavelength", data=wavelengths, overwrite=True
    )
    write_attribute(
        out_filename,
        "spectra/wavelength",
        "Description",
        "Wavelength of the spectra grid",
    )
    write_attribute(out_filename, "spectra/wavelength", "Units", "Angstrom")

    return out_filename


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="BPASS_2.2.1 download and grid creation"
    )
    parser.add_argument(
        "--download-data",
        default=False,
        action="store_true",
        help=(
            "download bpass data directly in current directory "
            "and untar in sunthesizer data directory"
        ),
    )

    args = parser.parse_args()

    synthesizer_data_dir = os.getenv("SYNTHESIZER_DATA")
    grid_dir = f"{synthesizer_data_dir}/grids"

    original_model_names = [
        # 'bpass_v2.2.1_imf_chab100',
        # 'bpass_v2.2.1_imf_chab300',
        "bpass_v2.2.1_imf100_300",
        "bpass_v2.2.1_imf135_300",
        "bpass_v2.2.1_imf170_300",
        "bpass_v2.2.1_imf100_100",
        "bpass_v2.2.1_imf135_100",
        "bpass_v2.2.1_imf170_100",
    ]

    for original_model_name in original_model_names:
        print("-" * 50)
        print(original_model_name)
        for bin in ["bin", "sin"]:

            out_filename = make_grid(original_model_name, bin)

            # get filename. This is useful if you want to simply add Q
            # model, bpass_imf = resolve_name(original_model_name, bin)
            # synthesizer_model_name = get_model_filename(model)
            # out_filename = f'{synthesizer_data_dir}/grids/{synthesizer_model_name}.hdf5'

            # add log10Q, can specify the desired ions with ions keyword.
            # by default calculates [HI, HeII]
            add_log10Q(out_filename)
