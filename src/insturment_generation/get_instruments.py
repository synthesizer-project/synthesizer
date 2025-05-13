"""This script will generate all instrument cache files.

This script holds the definitions for all cached insturment files available
through the synthesizer-download tool and importable from the
synthesizer.instruments module.

Note that this script requires extra dependencies not installed by default
in Synthesizer (nor listed as optional dependencies since they are
telescope specific). These include:

    - stpsf (For Webb filters)

Example usage:
    python get_instruments.py
"""

import os

import h5py
import stpsf

from synthesizer.instruments import (
    HSTACSWFC,
    HSTWFC3IR,
    HSTWFC3UVIS,
    JWSTMIRI,
    EuclidNISP,
    EuclidVIS,
    HSTACSWFCMedium,
    HSTACSWFCNarrow,
    HSTACSWFCWide,
    HSTWFC3IRMedium,
    HSTWFC3IRNarrow,
    HSTWFC3IRWide,
    HSTWFC3UVISMedium,
    HSTWFC3UVISNarrow,
    HSTWFC3UVISWide,
    JWSTNIRCam,
    JWSTNIRCamMedium,
    JWSTNIRCamNarrow,
    JWSTNIRCamWide,
)

# Get the cache file location
this_filepath = "/".join(os.path.abspath(__file__).split("/")[:-1])
INSTRUMENT_CACHE_DIR = os.path.join(
    this_filepath, "..", "synthesizer", "instruments", "instrument_cache"
)


def make_and_write_jwst_nircam():
    """Generate the JWST NIRCam instrument cache file.

    This will generate the NIRCam instrument cache file with all filters and
    PSFs.

    Note that we oversampling the PSFs by a factor of 2.
    """
    # First create the PSFs for all the NIRCam filters
    psfs = {}
    nc = stpsf.NIRCam()
    for nc_filt in JWSTNIRCam.available_filters:
        print(f"Generating PSF for {nc_filt}")
        nc.filter = nc_filt.split(".")[-1]
        psf = nc.calc_psf(oversample=2)
        psfs[nc_filt] = psf[0].data

    # Now create the NIRCam instrument with the PSFs
    nircam = JWSTNIRCam(psfs=psfs)

    # Now we can extract the psfs we need for each of the subset instruments
    # (there's no point paying the cost of generating them again)
    psfs_wide = {filt: psfs[filt] for filt in JWSTNIRCamWide.available_filters}
    psfs_medium = {
        filt: psfs[filt] for filt in JWSTNIRCamMedium.available_filters
    }
    psfs_narrow = {
        filt: psfs[filt] for filt in JWSTNIRCamNarrow.available_filters
    }

    # Now we can create the NIRCamWide, NIRCamMedium and
    # NIRCamNarrow instruments
    nircam_wide = JWSTNIRCamWide(psfs=psfs_wide)
    nircam_medium = JWSTNIRCamMedium(psfs=psfs_medium)
    nircam_narrow = JWSTNIRCamNarrow(psfs=psfs_narrow)

    # Finally we can write each instrument to disk
    with h5py.File(
        os.path.join(INSTRUMENT_CACHE_DIR, "JWST_NIRCam.hdf5"), "w"
    ) as hdf:
        nircam.to_hdf5(hdf)
    with h5py.File(
        os.path.join(INSTRUMENT_CACHE_DIR, "JWST_NIRCamWide.hdf5"), "w"
    ) as hdf:
        nircam_wide.to_hdf5(hdf)
    with h5py.File(
        os.path.join(INSTRUMENT_CACHE_DIR, "JWST_NIRCamMedium.hdf5"), "w"
    ) as hdf:
        nircam_medium.to_hdf5(hdf)
    with h5py.File(
        os.path.join(INSTRUMENT_CACHE_DIR, "JWST_NIRCamNarrow.hdf5"), "w"
    ) as hdf:
        nircam_narrow.to_hdf5(hdf)


def make_and_write_jwst_miri():
    """Generate the JWST MIRI instrument cache file.

    This will generate the MIRI instrument cache file with all filters and
    PSFs.

    Note that we oversampling the PSFs by a factor of 2.
    """
    # First create the PSFs for all the MIRI filters
    psfs = {}
    miri = stpsf.MIRI()
    for miri_filt in JWSTMIRI.available_filters:
        print(f"Generating PSF for {miri_filt}")
        miri.filter = miri_filt.split(".")[-1]
        psf = miri.calc_psf(oversample=2)
        psfs[miri_filt] = psf[0].data

    # Now create the MIRI instrument with the PSFs
    miri_inst = JWSTMIRI(psfs=psfs)

    # Finally we can write each instrument to disk
    with h5py.File(
        os.path.join(INSTRUMENT_CACHE_DIR, "JWST_MIRI.hdf5"), "w"
    ) as hdf:
        miri_inst.to_hdf5(hdf)


def make_and_write_hst_wfc3_uv():
    """Generate the HST WFC3 UVIS instrument cache file.

    This will generate the WFC3 UVIS instrument cache files.
    """
    # Now create the WFC3 UVIS instrument with the PSFs
    wfc3uv_inst = HSTWFC3UVIS()

    # Finally we can write each instrument to disk
    with h5py.File(
        os.path.join(INSTRUMENT_CACHE_DIR, "HST_WFC3_UVIS.hdf5"), "w"
    ) as hdf:
        wfc3uv_inst.to_hdf5(hdf)

    # Now generate the HST WFC3 UVIS wide, medium and narrow filters
    wfc3uv_wide = HSTWFC3UVISWide()
    wfc3uv_medium = HSTWFC3UVISMedium()
    wfc3uv_narrow = HSTWFC3UVISNarrow()

    # Finally we can write each instrument to disk
    with h5py.File(
        os.path.join(INSTRUMENT_CACHE_DIR, "HST_WFC3_UVISWide.hdf5"), "w"
    ) as hdf:
        wfc3uv_wide.to_hdf5(hdf)
    with h5py.File(
        os.path.join(INSTRUMENT_CACHE_DIR, "HST_WFC3_UVISMedium.hdf5"), "w"
    ) as hdf:
        wfc3uv_medium.to_hdf5(hdf)
    with h5py.File(
        os.path.join(INSTRUMENT_CACHE_DIR, "HST_WFC3_UVISNarrow.hdf5"), "w"
    ) as hdf:
        wfc3uv_narrow.to_hdf5(hdf)


def make_and_write_hst_wfc3_ir():
    """Generate the HST WFC3 IR instrument cache file.

    This will generate the WFC3 IR instrument cache files.
    """
    # Now create the WFC3 IR instrument with the PSFs
    wfc3ir_inst = HSTWFC3IR()

    # Finally we can write each instrument to disk
    with h5py.File(
        os.path.join(INSTRUMENT_CACHE_DIR, "HST_WFC3_IR.hdf5"), "w"
    ) as hdf:
        wfc3ir_inst.to_hdf5(hdf)

    # Now generate the HST WFC3 IR wide, medium and narrow filters
    wfc3ir_wide = HSTWFC3IRWide()
    wfc3ir_medium = HSTWFC3IRMedium()
    wfc3ir_narrow = HSTWFC3IRNarrow()

    # Finally we can write each instrument to disk
    with h5py.File(
        os.path.join(INSTRUMENT_CACHE_DIR, "HST_WFC3_IRWide.hdf5"), "w"
    ) as hdf:
        wfc3ir_wide.to_hdf5(hdf)
    with h5py.File(
        os.path.join(INSTRUMENT_CACHE_DIR, "HST_WFC3_IRMedium.hdf5"), "w"
    ) as hdf:
        wfc3ir_medium.to_hdf5(hdf)
    with h5py.File(
        os.path.join(INSTRUMENT_CACHE_DIR, "HST_WFC3_IRNarrow.hdf5"), "w"
    ) as hdf:
        wfc3ir_narrow.to_hdf5(hdf)


def make_and_write_hst_acswfc():
    """Generate the HST ACS WFC instrument cache file.

    This will generate the ACS WFC instrument cache files.
    """
    # Now create the ACS WFC instrument with the PSFs
    acswfc_inst = HSTACSWFC()

    # Finally we can write each instrument to disk
    with h5py.File(
        os.path.join(INSTRUMENT_CACHE_DIR, "HST_ACS_WFC.hdf5"), "w"
    ) as hdf:
        acswfc_inst.to_hdf5(hdf)

    # Now generate the HST ACS WFC wide, medium and narrow filters
    acswfc_wide = HSTACSWFCWide()
    acswfc_medium = HSTACSWFCMedium()
    acswfc_narrow = HSTACSWFCNarrow()

    # Finally we can write each instrument to disk
    with h5py.File(
        os.path.join(INSTRUMENT_CACHE_DIR, "HST_ACS_WFCWide.hdf5"), "w"
    ) as hdf:
        acswfc_wide.to_hdf5(hdf)
    with h5py.File(
        os.path.join(INSTRUMENT_CACHE_DIR, "HST_ACS_WFCMedium.hdf5"), "w"
    ) as hdf:
        acswfc_medium.to_hdf5(hdf)
    with h5py.File(
        os.path.join(INSTRUMENT_CACHE_DIR, "HST_ACS_WFCNarrow.hdf5"), "w"
    ) as hdf:
        acswfc_narrow.to_hdf5(hdf)


def make_and_write_euclid():
    """Generate the Euclid instrument cache file.

    This will generate the Euclid instrument cache files.
    """
    # Now create the Euclid instrument with the PSFs
    euclid_inst = EuclidNISP()

    # Finally we can write each instrument to disk
    with h5py.File(
        os.path.join(INSTRUMENT_CACHE_DIR, "Euclid_NISP.hdf5"), "w"
    ) as hdf:
        euclid_inst.to_hdf5(hdf)

    # Now generate the Euclid VIS instrument
    euclid_vis = EuclidVIS()

    # Finally we can write each instrument to disk
    with h5py.File(
        os.path.join(INSTRUMENT_CACHE_DIR, "Euclid_VIS.hdf5"), "w"
    ) as hdf:
        euclid_vis.to_hdf5(hdf)


if __name__ == "__main__":
    # Create the cache directory if it doesn't exist
    if not os.path.exists(INSTRUMENT_CACHE_DIR):
        raise FileNotFoundError(
            f"Cache directory {INSTRUMENT_CACHE_DIR} does not exist. "
            "Please create it before running this script."
        )

    # Generate the instruments
    make_and_write_jwst_nircam()
    make_and_write_jwst_miri()
    make_and_write_hst_wfc3_uv()
    make_and_write_hst_wfc3_ir()
    make_and_write_hst_acswfc()
    make_and_write_euclid()
