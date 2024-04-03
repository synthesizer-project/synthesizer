"""Kernel functions for computing smoothed quantities.

Kernels can either be created using the `create_kernel` method or
by directly calling the `get_kernel` method. The `create_kernel`
method saves the computed kernel as a .npz file for easy look-up.
`get_kernel` returns the computed kernel.

Example usage:

    kernel = Kernel(name="sph_anarchy", binsize=1000)
    kernel.create_kernel()

"""

from typing import Callable, Dict

import numpy as np
from numpy.typing import NDArray
from scipy import integrate

# Define type for kernel functions
KernelFunction = Callable[[float], float]


def uniform(r: float) -> float:
    """
    Uniform kernel function.

    Args:
        r: The distance from the centre of the kernel.

    Returns:
        The kernel value at distance `r`.
    """
    if r < 1.0:
        return 1.0 / ((4.0 / 3.0) * np.pi)
    else:
        return 0.0


def sph_anarchy(r: float) -> float:
    """
    Kernel function used in the Anarchy SPH scheme.

    Args:
        r: The distance from the centre of the kernel.

    Returns:
        The kernel value at distance `r`.
    """
    if r <= 1.0:
        return (21.0 / (2.0 * np.pi)) * (
            (1.0 - r) * (1.0 - r) * (1.0 - r) * (1.0 - r) * (1.0 + 4.0 * r)
        )
    else:
        return 0.0


def gadget_2(r: float) -> float:
    """
    Gadget2 kernel function.

    Args:
        r: The distance from the centre of the kernel.

    Returns:
        The kernel value at distance `r`.
    """
    if r < 0.5:
        return (8.0 / np.pi) * (1.0 - 6 * (r * r) + 6 * (r * r * r))
    elif r < 1.0:
        return (8.0 / np.pi) * 2 * ((1.0 - r) * (1.0 - r) * (1.0 - r))
    else:
        return 0.0


def cubic(r: float) -> float:
    """
    Cubic kernel function.

    Args:
        r: The distance from the centre of the kernel.

    Returns:
        The kernel value at distance `r`.
    """
    if r < 0.5:
        return 2.546479089470 + 15.278874536822 * (r - 1.0) * r * r
    elif r < 1:
        return 5.092958178941 * (1.0 - r) * (1.0 - r) * (1.0 - r)
    else:
        return 0


def quintic(r: float) -> float:
    """
    Quintic kernel function.

    Args:
        r: The distance from the centre of the kernel.

    Returns:
        The kernel value at distance `r`.
    """
    if r < 0.333333333:
        return 27.0 * (
            6.4457752 * r * r * r * r * (1.0 - r)
            - 1.4323945 * r * r
            + 0.17507044
        )
    elif r < 0.666666667:
        return 27.0 * (
            3.2228876 * r * r * r * r * (r - 3.0)
            + 10.7429587 * r * r * r
            - 5.01338071 * r * r
            + 0.5968310366 * r
            + 0.1352817016
        )
    elif r < 1:
        return (
            27.0
            * 0.64457752
            * (
                -r * r * r * r * r
                + 5.0 * r * r * r * r
                - 10.0 * r * r * r
                + 10.0 * r * r
                - 5.0 * r
                + 1.0
            )
        )
    else:
        return 0


class Kernel:
    """
    An kernel function for computing smoothed quantities in 2D.

    Line of sight distance along a particle, l = 2*sqrt(h^2 + b^2),
    where h and b are the smoothing length and the impact parameter
    respectively. This needs to be weighted along with the kernel
    density function W(r), to calculate the los density. Integrated
    los density, D = 2 * integral(W(r)dz) from 0 to sqrt(h^2-b^2),
    where r = sqrt(z^2 + b^2), W(r) is in units of h^-3 and is a
    function of r and h. The parameters are normalized in terms of
    the smoothing length, helping us to create a look-up table for
    every impact parameter along the line-of-sight. Hence we
    substitute x = x/h and b = b/h.

    This implies
    D = h^-2 * 2 * integral(W(r) dz) for x = 0 to sqrt(1.-b^2).
    The division by h^2 is to be done separately for each particle along the
    line-of-sight.

    NOTE: the resulting kernel is integrated along the z-axis resulting in a
    2D projection of the 3D kernel.
    """

    name: str
    binsize: int
    f: KernelFunction

    def __init__(self, name: str = "sph_anarchy", binsize: int = 1000) -> None:
        """
        Initialize the kernel function.

        Args:
            name: The name of the kernel function.
            binsize: The number of bins for the kernel.
        """
        self.name = name
        self.binsize = binsize

        # Define the possible kernel functions
        kernel_functions: Dict[str, KernelFunction] = {
            "uniform": uniform,
            "sph_anarchy": sph_anarchy,
            "gadget_2": gadget_2,
            "cubic": cubic,
            "quintic": quintic,
        }

        if name in kernel_functions:
            self.f = kernel_functions[name]
        else:
            raise ValueError("Kernel name not defined")

    def _W_dz(self, z: float, b: float) -> float:
        """
        Kernel value at point.

        Args:
            z: the normalised distance.
            b: impact factor.

        Returns:
            The kernel evaluated at z and b.
        """
        return self.f(np.sqrt(z**2 + b**2))

    def _integral_func(self, ii: float) -> Callable[[float], float]:
        """
        Integral function.

        Args:
            ii: The impact factor.

        Returns:
            The integrand function.
        """
        return lambda z: self._W_dz(z, ii)

    def get_kernel(self) -> NDArray[np.float64]:
        """
        Compute the kernel.

        h^-2 * 2 * integral(W(r) dz) from x = 0 to sqrt(1.-b^2) for
        various values of `b`

        Returns:
            The kernel array.
        """
        # Define the kernel
        kernel: NDArray[np.float64] = np.zeros(self.binsize + 1)

        # Define the r/b bins
        bins: NDArray[np.float64] = np.linspace(0, 1.0, num=self.binsize + 1)

        # Get the kernel projected into 2D
        for ii in range(self.binsize):
            y: float  # Result of integration
            yerr: float  # Estimate of the absolute error in the result
            y, yerr = integrate.quad(
                self._integral_func(bins[ii]), 0, np.sqrt(1.0 - bins[ii] ** 2)
            )
            kernel[ii] = y * 2.0

        return kernel

    def create_kernel(self) -> NDArray[np.float64]:
        """
        Save the computed kernel for easy look-up as .npz file.

        Returns:
            The kernel array.
        """
        kernel: NDArray[np.float64] = self.get_kernel()
        header: NDArray[np.object_] = np.array(
            [{"kernel": self.name, "bins": self.binsize}]
        )
        np.savez(
            "kernel_{}.npz".format(self.name), header=header, kernel=kernel
        )

        print(header)

        return kernel
