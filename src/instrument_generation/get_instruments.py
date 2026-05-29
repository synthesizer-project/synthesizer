"""This script will generate all instrument cache files.

This script holds the definitions for all cached instrument files available
through the synthesizer-download tool and importable from the
synthesizer.instruments module.

Note that this script requires extra dependencies not installed by default
in Synthesizer (nor listed as optional dependencies since they are
telescope specific). These include:

    - stpsf (For JWST PSFs)

Example usage:
    python get_instruments.py
"""

from pathlib import Path

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
THIS_DIR = Path(__file__).resolve().parent
INSTRUMENT_CACHE_DIR = (
    THIS_DIR / ".." / "synthesizer" / "instruments" / "instrument_cache"
).resolve()


def _write_cached_instrument(inst, inst_cls):
    """Write a premade specialised instrument to its cache filepath.

    Args:
        inst: The instrument instance to serialise.
        inst_cls: The premade instrument class defining the cache filepath.
    """
    # Resolve the output path from the specialised premade class itself so the
    # generator cannot drift out of sync with the loader expectations.
    cache_path = Path(inst_cls._instrument_cache_file)

    # Ensure the parent cache directory exists before writing the file.
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Serialise the cached instrument into its canonical HDF5 location.
    with h5py.File(cache_path, "w") as hdf:
        inst.to_hdf5(hdf)


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

    # Finally we can write each instrument to disk.
    _write_cached_instrument(nircam, JWSTNIRCam)
    _write_cached_instrument(nircam_wide, JWSTNIRCamWide)
    _write_cached_instrument(nircam_medium, JWSTNIRCamMedium)
    _write_cached_instrument(nircam_narrow, JWSTNIRCamNarrow)


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

    # Finally we can write the instrument to disk.
    _write_cached_instrument(miri_inst, JWSTMIRI)


def make_and_write_hst_wfc3_uv():
    """Generate the HST WFC3 UVIS instrument cache file.

    This will generate the WFC3 UVIS instrument cache files.
    """
    # Now create the WFC3 UVIS instrument with the PSFs
    wfc3uv_inst = HSTWFC3UVIS()

    # Finally we can write the parent instrument to disk.
    _write_cached_instrument(wfc3uv_inst, HSTWFC3UVIS)

    # Now generate the HST WFC3 UVIS wide, medium and narrow filters
    wfc3uv_wide = HSTWFC3UVISWide()
    wfc3uv_medium = HSTWFC3UVISMedium()
    wfc3uv_narrow = HSTWFC3UVISNarrow()

    # Finally we can write each subset instrument to disk.
    _write_cached_instrument(wfc3uv_wide, HSTWFC3UVISWide)
    _write_cached_instrument(wfc3uv_medium, HSTWFC3UVISMedium)
    _write_cached_instrument(wfc3uv_narrow, HSTWFC3UVISNarrow)


def make_and_write_hst_wfc3_ir():
    """Generate the HST WFC3 IR instrument cache file.

    This will generate the WFC3 IR instrument cache files.
    """
    # Now create the WFC3 IR instrument with the PSFs
    wfc3ir_inst = HSTWFC3IR()

    # Finally we can write the parent instrument to disk.
    _write_cached_instrument(wfc3ir_inst, HSTWFC3IR)

    # Now generate the HST WFC3 IR wide, medium and narrow filters
    wfc3ir_wide = HSTWFC3IRWide()
    wfc3ir_medium = HSTWFC3IRMedium()
    wfc3ir_narrow = HSTWFC3IRNarrow()

    # Finally we can write each subset instrument to disk.
    _write_cached_instrument(wfc3ir_wide, HSTWFC3IRWide)
    _write_cached_instrument(wfc3ir_medium, HSTWFC3IRMedium)
    _write_cached_instrument(wfc3ir_narrow, HSTWFC3IRNarrow)


def make_and_write_hst_acswfc():
    """Generate the HST ACS WFC instrument cache file.

    This will generate the ACS WFC instrument cache files.
    """
    # Now create the ACS WFC instrument with the PSFs
    acswfc_inst = HSTACSWFC()

    # Finally we can write the parent instrument to disk.
    _write_cached_instrument(acswfc_inst, HSTACSWFC)

    # Now generate the HST ACS WFC wide, medium and narrow filters
    acswfc_wide = HSTACSWFCWide()
    acswfc_medium = HSTACSWFCMedium()
    acswfc_narrow = HSTACSWFCNarrow()

    # Finally we can write each subset instrument to disk.
    _write_cached_instrument(acswfc_wide, HSTACSWFCWide)
    _write_cached_instrument(acswfc_medium, HSTACSWFCMedium)
    _write_cached_instrument(acswfc_narrow, HSTACSWFCNarrow)


def make_and_write_euclid():
    """Generate the Euclid instrument cache file.

    This will generate the Euclid instrument cache files.
    """
    # Now create the Euclid instrument with the PSFs
    euclid_inst = EuclidNISP()

    # Finally we can write the NISP instrument to disk.
    _write_cached_instrument(euclid_inst, EuclidNISP)

    # Now generate the Euclid VIS instrument
    euclid_vis = EuclidVIS()

    # Finally we can write the VIS instrument to disk.
    _write_cached_instrument(euclid_vis, EuclidVIS)


if __name__ == "__main__":
    # Create the cache directory tree if it does not already exist.
    INSTRUMENT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Generate the instruments.
    make_and_write_jwst_nircam()
    make_and_write_jwst_miri()
    make_and_write_hst_wfc3_uv()
    make_and_write_hst_wfc3_ir()
    make_and_write_hst_acswfc()
    make_and_write_euclid()
