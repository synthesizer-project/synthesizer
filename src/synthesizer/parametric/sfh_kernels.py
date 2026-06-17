"""A submodule defining covariance kernels for stochastic star formation.

These kernels define the auto-covariance of fluctuations in log10(SFR) used by
the ``SFH.Stochastic`` star formation history model. They follow the Gaussian
Process + Power Spectral Density formalism of Iyer et al. 2024
(arXiv:2208.05938), in which variability in galaxy star formation histories is
modelled as a Gaussian Process in log-SFR space whose covariance is set by a
(physically motivated) kernel.

A kernel only has to define its auto-covariance as a function of the time lag
between two epochs (the ``covariance`` method). For stationary kernels (where
the covariance depends only on the lag) the parent class then builds the full
covariance matrix on a time grid using an efficient Toeplitz construction. New
variability models (e.g. a general power-spectral-density kernel, or composite
multi-timescale kernels) are added simply by subclassing ``Kernel``.

NOTE: This module is imported as Kernels in parametric.__init__ enabling the
      syntax shown below.

Example usage:

    from synthesizer.parametric import Kernels
    from unyt import Gyr

    kernel = Kernels.DampedRandomWalk(sigma=0.3, tau=1 * Gyr)
    cov = kernel.build_covariance_matrix(tarr)  # tarr in years

"""

import numpy as np
from unyt import unyt_array, yr

from synthesizer import exceptions
from synthesizer.units import accepts

# Define a list of the available kernels
kernels = ("DampedRandomWalk",)


def _draw_from_prior(prior):
    """Draw a uniform sample from a length-2 [low, high] prior.

    Units are preserved when the prior is a unyt array.

    Args:
        prior (list/tuple/np.ndarray/unyt_array):
            A length-2 sequence defining a uniform prior [low, high].

    Returns:
        float/unyt_quantity:
            A single sample drawn uniformly between low and high.
    """
    if len(prior) != 2:
        raise exceptions.InconsistentArguments(
            "A prior must be a length 2 sequence defining a uniform prior "
            "[low, high]."
        )

    # Preserve units for unyt priors (e.g. tau=[0.5, 3] * Gyr)
    if isinstance(prior, unyt_array):
        return np.random.uniform(prior[0].value, prior[1].value) * prior.units

    return np.random.uniform(prior[0], prior[1])


class Kernel:
    """The parent class for all SFH covariance kernels.

    A kernel defines the auto-covariance of fluctuations in log10(SFR) as a
    function of the time lag between two epochs. Stationary kernels (where the
    covariance depends only on the lag) need only implement ``covariance``; the
    parent then builds the full covariance matrix on a time grid using an
    efficient symmetric Toeplitz construction. Non-stationary kernels should
    instead override ``build_covariance_matrix`` directly.

    Attributes:
        name (str):
            The name of the kernel. This is set by the child class.
        parameters (dict):
            A dictionary containing the parameters of the kernel.
    """

    def __init__(self, name, **kwargs):
        """Initialise the parent.

        Args:
            name (str):
                The name of the kernel. Set by the child class.
            **kwargs (dict):
                A dictionary containing the parameters of the kernel.
        """
        # Set the name string
        self.name = name

        # Store the kernel parameters (defined as kwargs)
        self.parameters = kwargs

    def covariance(self, delta_t):
        """Prototype for child defined auto-covariance functions.

        Args:
            delta_t (float/np.ndarray of float):
                The time lag(s) (in years) at which to evaluate the covariance.

        Returns:
            float/np.ndarray of float:
                The auto-covariance at the passed lag(s).
        """
        raise exceptions.UnimplementedFunctionality(
            "This should never be called from the parent."
            "How did you get here!?"
        )

    def build_covariance_matrix(self, tarr):
        """Build the covariance matrix on a time grid.

        This default implementation assumes a stationary kernel, i.e. the
        covariance depends only on the lag between two times. It exploits the
        resulting symmetric Toeplitz structure to fill the matrix from a single
        evaluation of ``covariance`` per lag, which makes drawing samples cheap
        once the matrix is built.

        Non-stationary kernels should override this method.

        Args:
            tarr (np.ndarray of float):
                A regularly spaced time grid (in years).

        Returns:
            np.ndarray of float:
                The (N, N) covariance matrix.
        """
        tarr = np.asarray(tarr, dtype=np.float64)
        n = tarr.size

        # The Toeplitz construction below assumes a regular grid, so verify
        # the spacing is uniform to avoid silently returning a wrong matrix.
        if n > 2:
            dt = np.diff(tarr)
            if not np.allclose(dt, dt[0]):
                raise exceptions.InconsistentArguments(
                    "build_covariance_matrix requires a regularly spaced "
                    "time grid; the spacing of tarr is non-uniform."
                )

        # Evaluate the covariance at every lag relative to the first epoch. For
        # a regular grid this contains all the unique values in the matrix.
        cov_deltat = np.asarray(
            self.covariance(tarr - tarr[0]), dtype=np.float64
        )

        # Fill the symmetric Toeplitz matrix by rolling the lag vector so that
        # entry (i, j) holds the covariance at a lag of |i - j| grid steps.
        cov_matrix = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            col = np.roll(cov_deltat, i)
            col[:i] = np.flip(cov_deltat[1 : i + 1], 0)
            cov_matrix[:, i] = col

        return cov_matrix


class DampedRandomWalk(Kernel):
    """A damped random walk (single-timescale regulator) kernel.

    This is the simplest member of the regulator/extended-regulator family of
    Iyer et al. 2024 (arXiv:2208.05938). The auto-covariance of the log10(SFR)
    fluctuations decays exponentially with the time lag:

        C(dt) = sigma**2 * exp(-|dt| / tau)

    which corresponds to a Lorentzian power spectral density with a break
    frequency set by ``tau``.

    Attributes:
        sigma (float):
            The standard deviation of the log10(SFR) fluctuations (in dex).
        tau (float):
            The correlation timescale of the fluctuations (in years).
    """

    @accepts(tau=yr)
    def __init__(self, sigma, tau):
        """Initialise the parent and this kernel.

        Args:
            sigma (float):
                The standard deviation of the log10(SFR) fluctuations (in dex).
                This sets the amplitude of the variability.
            tau (unyt_quantity):
                The correlation timescale of the fluctuations. This sets the
                timescale of the variability: large tau gives smooth, slowly
                varying SFHs while small tau gives bursty SFHs.

        Raises:
            InconsistentArguments: If sigma or tau are not positive.
        """
        # Initialise the parent
        Kernel.__init__(self, name="DampedRandomWalk", sigma=sigma, tau=tau)

        # Store the parameters in base units
        self.sigma = float(sigma)
        self.tau = tau.to("yr").value

        # Validate
        if self.sigma <= 0:
            raise exceptions.InconsistentArguments("sigma must be positive!")
        if self.tau <= 0:
            raise exceptions.InconsistentArguments("tau must be positive!")

    @classmethod
    def init_from_prior(cls, sigma, tau):
        """Initialise the kernel by drawing parameters from uniform priors.

        Each parameter is either a single fixed value or a length-2 sequence
        [low, high] defining a uniform prior. The ``tau`` prior should carry
        units (e.g. ``tau=[0.5, 3] * Gyr``).

        Args:
            sigma (float | length-2 sequence):
                Fixed value or [low, high] uniform prior for sigma (dex).
            tau (unyt_quantity | length-2 unyt_array):
                Fixed value or [low, high] uniform prior for tau.

        Returns:
            DampedRandomWalk:
                An instance with parameters drawn from the priors.
        """
        if isinstance(sigma, (list, tuple, np.ndarray)):
            sigma = _draw_from_prior(sigma)
        if isinstance(tau, (list, tuple, unyt_array)):
            tau = _draw_from_prior(tau)
        return cls(sigma=sigma, tau=tau)

    def covariance(self, delta_t):
        """Evaluate the damped random walk auto-covariance.

        Args:
            delta_t (float/np.ndarray of float):
                The time lag(s) (in years) at which to evaluate the covariance.

        Returns:
            float/np.ndarray of float:
                The auto-covariance at the passed lag(s).
        """
        return self.sigma**2 * np.exp(-np.abs(delta_t) / self.tau)
