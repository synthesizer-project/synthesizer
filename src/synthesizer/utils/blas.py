"""A module for limiting the number of threads used by BLAS.

Some extensions (e.g. parametric_spectra) hand dense matrix products to an
optimised BLAS. BLAS libraries manage their own thread pools, so to respect
the number of threads requested by the user we limit that pool around the
extension call with threadpoolctl.

NOTE: This works for OpenBLAS, MKL, BLIS and FlexiBLAS (e.g. SciPy's Linux
      wheels and conda's MKL builds). Apple's Accelerate, which SciPy uses on
      macOS, can't be limited at runtime: its threading can only be set with
      the VECLIB_MAXIMUM_THREADS environment variable before it is loaded.
      Accelerate runs these products on Apple's matrix coprocessor rather than
      spreading them over the CPU cores, so this rarely matters in practice.

Example:
    with limit_blas_threads(nthreads):
        spec = compute_population_seds(...)
"""

from contextlib import nullcontext
from functools import lru_cache

from threadpoolctl import ThreadpoolController


@lru_cache(maxsize=1)
def _get_blas_controller():
    """Get a threadpoolctl controller for the BLAS libraries loaded.

    Building the controller inspects every loaded library, so we do it once.
    It is built lazily so the BLAS libraries loaded by the extensions are
    already present.

    Returns:
        ThreadpoolController:
            A controller restricted to the BLAS libraries.
    """
    return ThreadpoolController().select(user_api="blas")


def limit_blas_threads(nthreads):
    """Limit the BLAS libraries to nthreads threads within a with block.

    Args:
        nthreads (int):
            The number of threads BLAS may use. Values below 1 (e.g. -1 for
            all available threads) leave BLAS unlimited.

    Returns:
        contextlib.AbstractContextManager:
            A context manager applying the limit, or doing nothing when the
            BLAS libraries can't be limited (e.g. Apple's Accelerate).
    """
    controller = _get_blas_controller()
    if nthreads < 1 or len(controller.lib_controllers) == 0:
        return nullcontext()
    return controller.limit(limits=nthreads)
