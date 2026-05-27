"""Create cached instruments required by profiling scripts.

Run this on a machine with internet access before running profiling on
offline compute nodes.
"""

from __future__ import annotations

import h5py
from pipeline_test_data import INSTRUMENT_PATH

from synthesizer import __version__
from synthesizer.grid import Grid
from synthesizer.instruments.premade import JWSTNIRCamWide


def main() -> None:
    """Create the profiling instrument cache."""
    grid = Grid("test_grid")
    instrument = JWSTNIRCamWide(
        filter_lams=grid.lam,
        label="JWST.NIRCam.Wide",
    )

    with h5py.File(INSTRUMENT_PATH, "w") as hdf:
        instrument.to_hdf5(hdf)

    print(
        "Saved profiling instrument cache to "
        f"{INSTRUMENT_PATH} for synthesizer {__version__}"
    )


if __name__ == "__main__":
    main()
