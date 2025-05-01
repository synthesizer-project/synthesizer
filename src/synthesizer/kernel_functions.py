"""A module defining SPH kernels  integrated along the line-of-sight.

This module provides a class for calculating SPH kernels integrated along
the line-of-sight. The kernels are used in smoothed particle hydrodynamics
(SPH) simulations to compute the density of particles in a given volume.
The kernels are defined as functions of the distance from the center of the
kernel and are integrated along the line-of-sight to obtain the density
distribution.

Available kernels include:
    - uniform
    - sph_anarchy
    - gadget_2
    - cubic
    - quintic

Example usage:
    kernel = Kernel(name="sph_anarchy", binsize=10000)
    kernel_data = kernel.get_kernel()
"""

import numpy as np
from scipy import integrate


class Kernel:
    """A class describing a SPH kernel integrated along the line-of-sight.

    Line of sight distance along a particle, l = 2*sqrt(h^2 + b^2), where h
    and b are the smoothing length and the impact parameter respectively. This
    needs to be weighted along with the kernel density function W(r), to
    calculate the los density. Integrated los density,
        D = 2 * integral(W(r)dz) from 0 to sqrt(h^2-b^2),
    where r = sqrt(z^2 + b^2), W(r) is in units of h^-3 and is a function of
    r and h. The parameters are normalized in terms of the smoothing length,
    helping us to create a look-up table for every impact parameter along
    the line-of-sight. Hence we substitute x = x/h and b = b/h.

    This implies
        D = h^-2 * 2 * integral(W(r) dz) for x = 0 to sqrt(1.-b^2).
    The division by h^2 is to be done separately for each particle along the
    line-of-sight.
    """

    def __init__(self, name="sph_anarchy", binsize=10000):
        """Initialize the kernel class.

        Args:
            name (str): The name of the kernel to use. Options are:
                "uniform", "sph_anarchy", "gadget_2", "cubic", "quintic".
            binsize (int): The number of bins to use for the kernel. This is
                the number of elements along the radial direction at which
                the ratio between r and b is calculated. The kernel is then
                calculated at these points. The default is 10000.
        """
        self.name = name
        self.binsize = binsize

        if name == "uniform":
            self.f = uniform
        elif name == "sph_anarchy":
            self.f = sph_anarchy
        elif name == "gadget_2":
            self.f = gadget_2
        elif name == "cubic":
            self.f = cubic
        elif name == "quintic":
            self.f = quintic
        else:
            raise ValueError("Kernel name not defined")

    def W_dz(self, z, b):
        """Calculate the kernel density function W(r) as a function of z.

        Args:
            z (float): The distance along the line-of-sight.
            b (float): The impact parameter.

        Returns:
            float: The value of the kernel density function W(r).
        """
        return self.f(np.sqrt(z**2 + b**2))

    def _integral_func(self, ii):
        """Calculate W(r) as a function of z for a given impact parameter."""
        return lambda z: self.W_dz(z, ii)

    def get_kernel(self):
        """Compute the kernel.

        i.e. h^-2 * 2 * integral(W(r) dz) from x = 0 to sqrt(1.-b^2) for
        various values of `b`.

        Returns:
            np.ndarray: The kernel values for each impact parameter.
        """
        kernel = np.zeros(self.binsize + 1)

        bins = np.arange(0, 1.0, 1.0 / self.binsize)
        bins = np.append(bins, 1.0)

        for ii in range(self.binsize):
            y, yerr = integrate.quad(
                self._integral_func(bins[ii]), 0, np.sqrt(1.0 - bins[ii] ** 2)
            )
            kernel[ii] = y * 2.0

        return kernel

    def create_kernel(self):
        """Save the computed kernel for easy look-up as .npz file."""
        kernel = self.get_kernel()
        header = np.array([{"kernel": self.name, "bins": self.binsize}])
        np.savez(
            "kernel_{}.npz".format(self.name), header=header, kernel=kernel
        )

        print(header)

        return kernel


def uniform(r):
    """Calculate the uniform kernel.

    Args:
        r (float): The distance from the center of the kernel.

    Returns:
            float: The value of the uniform kernel.
    """
    if r < 1.0:
        return 1.0 / ((4.0 / 3.0) * np.pi)
    else:
        return 0.0


def sph_anarchy(r):
    """Calculate the SPH Anarchy kernel.

    Args:
        r (float): The distance from the center of the kernel.

    Returns:
            float: The value of the SPH Anarchy kernel.
    """
    if r <= 1.0:
        return (21.0 / (2.0 * np.pi)) * (
            (1.0 - r) * (1.0 - r) * (1.0 - r) * (1.0 - r) * (1.0 + 4.0 * r)
        )
    else:
        return 0.0


def gadget_2(r):
    """Calculate the Gadget-2 kernel.

    Args:
        r (float): The distance from the center of the kernel.

    Returns:
        float: The value of the Gadget-2 kernel.
    """
    if r < 0.5:
        return (8.0 / np.pi) * (1.0 - 6 * (r * r) + 6 * (r * r * r))
    elif r < 1.0:
        return (8.0 / np.pi) * 2 * ((1.0 - r) * (1.0 - r) * (1.0 - r))
    else:
        return 0.0


def cubic(r):
    """Calculate the cubic kernel.

    Args:
        r (float): The distance from the center of the kernel.

    Returns:
        float: The value of the cubic kernel.
    """
    if r < 0.5:
        return 2.546479089470 + 15.278874536822 * (r - 1.0) * r * r
    elif r < 1:
        return 5.092958178941 * (1.0 - r) * (1.0 - r) * (1.0 - r)
    else:
        return 0


def quintic(r):
    """Calculate the quintic kernel.

    Args:
        r (float): The distance from the center of the kernel.

    Returns:
        float: The value of the quintic kernel.
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
