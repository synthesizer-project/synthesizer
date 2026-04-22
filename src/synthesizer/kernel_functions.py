"""A module defining SPH kernels integrated along the line-of-sight.

This module provides a class for calculating SPH kernels integrated along the
line-of-sight. These tables are used when computing LOS column densities and
optical depths for particle data.

Available kernels include:
    - uniform
    - sph_anarchy
    - gadget_2
    - cubic
    - quintic
"""

import numpy as np
from scipy import integrate

from synthesizer.extensions.kernel import (
    compute_truncated_los_kernel,
    evaluate_kernel,
)
from synthesizer.utils.operation_timers import timed


def _call_kernel_function(kernel_name, r):
    """Evaluate a named kernel via the shared C++ implementation.

    Args:
        kernel_name (str):
            The public kernel name.
        r (float or np.ndarray):
            The dimensionless radius or radii to evaluate.

    Returns:
        float or np.ndarray:
            The kernel value(s), preserving scalar inputs as scalars.
    """
    input_array = np.asarray(r, dtype=np.float64)
    radii = np.ascontiguousarray(np.atleast_1d(input_array).ravel())
    values = evaluate_kernel(radii, kernel_name)

    if input_array.ndim == 0:
        return float(values[0])

    return values.reshape(input_array.shape)


class Kernel:
    """A class describing a SPH kernel integrated along the line-of-sight.

    The line of sight distance through a "source" particle's kernel (i.e.
    the sight line traced through an arbitrary point in an SPH kernel) is,
        l = 2*sqrt(h^2 - b^2),
    where h and b are the smoothing length and the impact parameter
    respectively. This needs to be weighted along with the kernel density
    function W(r), to calculate the LOS density. The integrated LOS density is,
        D = 2 * integral(W(r)dz) from 0 to sqrt(h^2-b^2),
    where r = sqrt(z^2 + b^2), W(r) is in units of h^-3 and is a function of
    r and h. The parameters that enter this integral are normalized in terms
    of the smoothing length, so we can create a generic look-up table for
    arbitrary source particles including every impact parameter along the
    line-of-sight. Hence we substitute z = z/h and b = b/h.

    This implies
        D = h^-2 * 2 * integral(W(r) dz) for z = 0 to sqrt(1.0 - b^2).
    The division by h^2 is to be done separately for each particle along the
    line-of-sight.

    In the case where the input particle lies inside the source kernel, the
    LOS integral must be truncated at the input particle's LOS coordinate. This
    is handled by a separate look-up table that tabulates the cumulative LOS
    integral as a function of impact parameter and support-normalized LOS
    coordinate. The truncation coordinate is
        z_trunc = (z_input - z_source) / h,
    where z_input and z_source are the LOS coordinates of the input and source
    particles respectively.
    """

    def __init__(
        self,
        name="sph_anarchy",
        binsize=10000,
        truncated_q_binsize=None,
        truncated_z_binsize=1000,
    ):
        """Initialize the kernel class.

        Args:
            name (str):
                The name of the kernel to use. Options are: "uniform",
                "sph_anarchy", "gadget_2", "cubic", "quintic".
            binsize (int):
                The number of bins to use for the projected LOS kernel table.
            truncated_q_binsize (int, optional):
                The number of bins to use along the impact-parameter axis of
                the truncated LOS kernel table. If omitted this falls back to
                ``binsize``.
            truncated_z_binsize (int):
                The number of bins to use along the LOS truncation axis of the
                truncated LOS kernel table.
        """
        self.name = name
        self.binsize = binsize
        self.truncated_q_binsize = (
            binsize if truncated_q_binsize is None else truncated_q_binsize
        )
        self.truncated_z_binsize = truncated_z_binsize

        # Set the kernel function based on the provided name.
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

        # Cache the LOS kernel tables once they have been built.
        self._projected_kernel = None
        self._truncated_los_kernel = None

    def _get_bins(self, binsize=None):
        """Get the dimensionless radial bins used for kernel lookups.

        Args:
            binsize (int, optional):
                The number of bins to generate. If omitted this uses the
                projected-kernel resolution.

        Returns:
            np.ndarray: The dimensionless bins spanning the kernel support.
        """
        if binsize is None:
            binsize = self.binsize

        bins = np.arange(0, 1.0, 1.0 / binsize)
        bins = np.append(bins, 1.0)
        return bins

    def _get_z_bins(self):
        """Get the dimensionless LOS truncation bins for the 2D lookup."""
        return np.linspace(-1.0, 1.0, self.truncated_z_binsize + 1)

    def W_dz(self, z, b):
        """Calculate the kernel density function W(r) as a function of z.

        Args:
            z (float): The distance along the line-of-sight.
            b (float): The impact parameter.

        Returns:
            float: The value of the kernel density function W(r).
        """
        return self.f(np.sqrt(z**2 + b**2))

    def _integral_func(self, impact_parameter):
        """Calculate W(r) as a function of z for a given impact parameter."""
        return lambda z: self.W_dz(z, impact_parameter)

    @timed("Kernel.get_kernel")
    def get_kernel(self):
        """Compute the projected LOS kernel table.

        This is the full LOS integral through the source kernel at each
        support-normalised impact parameter.

        Returns:
            np.ndarray: The projected kernel values for each impact parameter.
        """
        # Return the cached kernel if it has already been computed.
        if self._projected_kernel is not None:
            return self._projected_kernel.copy()

        # Get the dimensionless impact-parameter bins and set up the output.
        bins = self._get_bins()
        kernel = np.zeros(self.binsize + 1)

        # For each impact parameter, integrate the kernel through the full LOS
        # extent of the source support.
        for ii, impact_parameter in enumerate(bins[:-1]):
            value, _ = integrate.quad(
                self._integral_func(impact_parameter),
                0,
                np.sqrt(1.0 - impact_parameter**2),
            )
            kernel[ii] = value * 2.0

        # Cache it.
        self._projected_kernel = kernel

        return kernel.copy()

    @timed("Kernel.get_truncated_los_kernel")
    def get_truncated_los_kernel(self):
        """Compute the truncated LOS kernel lookup table.

        This helper tabulates the cumulative LOS integral of the kernel as a
        function of impact parameter and support-normalised LOS truncation
        coordinate. It is used when an input particle lies inside a source
        kernel and therefore only sees part of the source along the line of
        sight.

        Returns:
            tuple:
                A tuple containing the truncated kernel table and the radial
                and LOS-coordinate grids that index it.
        """
        # Return the cached kernel if it has already been computed.
        if self._truncated_los_kernel is not None:
            bins = self._get_bins(self.truncated_q_binsize)
            z_bins = self._get_z_bins()
            return self._truncated_los_kernel.copy(), bins, z_bins

        # Get the projected-separation and LOS-coordinate bins and set up the
        # output.
        bins = self._get_bins(self.truncated_q_binsize)
        z_bins = self._get_z_bins()
        kernel = compute_truncated_los_kernel(
            np.ascontiguousarray(bins, dtype=np.float64),
            np.ascontiguousarray(z_bins, dtype=np.float64),
            self.name,
        )

        # Cache it.
        self._truncated_los_kernel = kernel

        return self._truncated_los_kernel.copy(), bins, z_bins

    @timed("Kernel.create_kernel")
    def create_kernel(self):
        """Save the computed projected kernel for easy look-up as .npz file."""
        kernel = self.get_kernel()
        header = np.array([{"kernel": self.name, "bins": self.binsize}])
        np.savez(
            f"kernel_{self.name}.npz",
            header=header,
            kernel=kernel,
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
    return _call_kernel_function("uniform", r)


def sph_anarchy(r):
    """Calculate the SPH Anarchy kernel.

    Args:
        r (float): The distance from the center of the kernel.

    Returns:
        float: The value of the SPH Anarchy kernel.
    """
    return _call_kernel_function("sph_anarchy", r)


def gadget_2(r):
    """Calculate the Gadget-2 kernel.

    Args:
        r (float): The distance from the center of the kernel.

    Returns:
        float: The value of the Gadget-2 kernel.
    """
    return _call_kernel_function("gadget_2", r)


def cubic(r):
    """Calculate the cubic kernel.

    Args:
        r (float): The distance from the center of the kernel.

    Returns:
        float: The value of the cubic kernel.
    """
    return _call_kernel_function("cubic", r)


def quintic(r):
    """Calculate the quintic kernel.

    Args:
        r (float): The distance from the center of the kernel.

    Returns:
        float: The value of the quintic kernel.
    """
    return _call_kernel_function("quintic", r)
