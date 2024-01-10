import numpy as np
from scipy import integrate


class Kernel:

    """
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
    """

    def __init__(self, name="sph_anarchy", binsize=1000):
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
        """
        W(r)dz
        """

        return self.f(np.sqrt(z**2 + b**2))

    def integral_func(self, ii):
        return lambda z: self.W_dz(z, ii)

    def get_kernel(self):
        """
        h^-2 * 2 * integral(W(r) dz) from x = 0 to sqrt(1.-b^2) for
        various values of `b`
        """

        kernel = np.zeros(self.binsize + 1)

        bins = np.arange(0, 1.0, 1.0 / self.binsize)
        bins = np.append(bins, 1.0)

        for ii in range(self.binsize):
            y, yerr = integrate.quad(
                self.integral_func(bins[ii]), 0, np.sqrt(1.0 - bins[ii] ** 2)
            )
            kernel[ii] = y * 2.0

        return kernel

    def create_kernel(self):
        """
        Saves the computed kernel for easy look-up as .npz file
        """

        kernel = self.get_kernel()
        header = np.array([{"kernel": self.name, "bins": self.binsize}])
        np.savez("kernel_{}.npz".format(self.name), header=header, kernel=kernel)

        print(header)

        return kernel


def uniform(r):
    if r < 1.0:
        return 1.0 / ((4.0 / 3.0) * np.pi)
    else:
        return 0.0


def sph_anarchy(r):
    if r <= 1.0:
        return (21.0 / (2.0 * np.pi)) * (
            (1.0 - r) * (1.0 - r) * (1.0 - r) * (1.0 - r) * (1.0 + 4.0 * r)
        )
    else:
        return 0.0


def gadget_2(r):
    if r < 0.5:
        return (8.0 / np.pi) * (1.0 - 6 * (r * r) + 6 * (r * r * r))
    elif r < 1.0:
        return (8.0 / np.pi) * 2 * ((1.0 - r) * (1.0 - r) * (1.0 - r))
    else:
        return 0.0


def cubic(r):
    if r < 0.5:
        return 2.546479089470 + 15.278874536822 * (r - 1.0) * r * r
    elif r < 1:
        return 5.092958178941 * (1.0 - r) * (1.0 - r) * (1.0 - r)
    else:
        return 0


def quintic(r):
    if r < 0.333333333:
        return 27.0 * (
            6.4457752 * r * r * r * r * (1.0 - r) - 1.4323945 * r * r + 0.17507044
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
