"""Load simulation particle data into Synthesizer galaxy objects.

Each loader in this module reads particle data from a specific simulation
and populates :class:`~synthesizer.particle.galaxy.Galaxy` objects with
stellar, gas, and optionally black-hole components.

.. rubric:: Controlling array precision

All loader functions accept a ``dtype`` parameter (default: ``np.float64``)
that controls the floating-point precision of the particle arrays passed to
:meth:`Galaxy.load_stars <synthesizer.particle.galaxy.Galaxy.load_stars>`
and :meth:`Galaxy.load_gas <synthesizer.particle.galaxy.Galaxy.load_gas>`.

Setting ``dtype=np.float32`` reduces memory usage and is compatible with
grids opened with ``Grid(use_precision=np.float32)``.

Available loaders
-----------------
- :func:`load_camels.load_CAMELS_IllustrisTNG`
- :func:`load_camels.load_CAMELS_Astrid`
- :func:`load_camels.load_CAMELS_Simba`
- :func:`load_camels.load_CAMELS_SwiftEAGLE_subfind`
- :func:`load_illustris.load_IllustrisTNG`
- :func:`load_simba.load_Simba`
- :func:`load_simba.load_Simba_slab`
- :func:`load_flares.load_FLARES`
- :func:`load_bluetides.load_BlueTides`
- :func:`load_scsam.load_SCSAM`
- :func:`load_yt.load_yt`
"""

import numpy as np  # noqa: F401 — referenced by module docstring
