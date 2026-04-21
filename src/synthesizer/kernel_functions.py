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

import os

import numpy as np
from scipy import integrate

from synthesizer.extensions.kernel import compute_overlap_kernel
from synthesizer.utils.operation_timers import timed, timer

# Define the default overlap-kernel grid dimensions and bounds
OVERLAP_Q_BINS = 64
OVERLAP_U_BINS = 128
OVERLAP_ETA_BINS = 48
OVERLAP_ETA_MIN = 0.1
OVERLAP_ETA_MAX = 10.0
OVERLAP_BUILD_NDIM = 16


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

        # Set the kernel function based on the provided name
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

        # Initialize the cached kernel tables. Once populated we can return
        # the results without needing to recompute. For the overlap kernel
        # in particular, the build step is expensive so this is a must
        self._projected_kernel = None
        self._radial_kernel = None
        self._truncated_los_kernel = None
        self._overlap_kernel = None
        self._overlap_q = None
        self._overlap_u = None
        self._overlap_eta = None

    def _get_bins(self):
        """Get the dimensionless radial bins used for kernel lookups.

        All kernel tables in this class are tabulated in units of the kernel
        support radius. This helper centralises the construction of those
        dimensionless bins so the projected, radial, and truncated LOS tables
        all use consistent sampling.
        """
        bins = np.arange(0, 1.0, 1.0 / self.binsize)
        bins = np.append(bins, 1.0)
        return bins

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

    def _get_radial_kernel(self):
        """Compute the private 3D radial kernel lookup table.

        This helper is used internally when constructing the overlap kernel.
        It is not part of the public LOS kernel API because the runtime only
        needs the final projected or overlap kernel tables.

        Returns:
            np.ndarray: The 3D radial kernel values for each dimensionless
                radius bin.
        """
        # Return the cached kernel if it has already been computed
        if self._radial_kernel is not None:
            return self._radial_kernel.copy()

        with timer("Building LOS radial kernel table"):
            # Get the dimensionless radius bins and set up the output
            bins = self._get_bins()
            kernel = np.zeros(self.binsize + 1)

            # Populate the kernel table by evaluating the kernel function
            # at each radius bin
            for ii, radius in enumerate(bins):
                kernel[ii] = self.f(radius)

            # Cache it
            self._radial_kernel = kernel

        return kernel.copy()

    def _get_truncated_los_kernel(self):
        """Compute the private truncated LOS kernel lookup table.

        This helper tabulates the cumulative LOS integral of the kernel as a
        function of impact parameter and support-normalised LOS coordinate. It
        is used internally to build the fast smoothed overlap table.

        Returns:
            tuple:
                A tuple containing the truncated kernel table and the
                radial and LOS-coordinate grids that index it.
        """
        # Return the cached kernel if it has already been computed
        if self._truncated_los_kernel is not None:
            bins = self._get_bins()
            z_bins = np.linspace(-1.0, 1.0, 2 * self.binsize + 1)
            return self._truncated_los_kernel.copy(), bins, z_bins

        with timer("Building truncated LOS kernel table"):
            # Get the projected-separation and LOS-coordinate bins and set up
            # the output
            bins = self._get_bins()
            z_bins = np.linspace(-1.0, 1.0, 2 * self.binsize + 1)
            kernel = np.zeros((bins.size, z_bins.size))

            # For each projected separation, integrate the kernel cumulatively
            # along the LOS coordinate
            for ii, impact_parameter in enumerate(bins):
                integrand = np.zeros_like(z_bins)
                for iz, z_value in enumerate(z_bins):
                    radius = np.sqrt(z_value**2 + impact_parameter**2)
                    if radius < 1.0:
                        integrand[iz] = self.f(radius)
                kernel[ii] = integrate.cumulative_trapezoid(
                    integrand, z_bins, initial=0.0
                )

            # Cache it
            self._truncated_los_kernel = kernel

        return self._truncated_los_kernel.copy(), bins, z_bins

    @staticmethod
    def _interpolate_truncated_kernel(kernel, q_grid, z_grid, q, z):
        """Interpolate the truncated LOS kernel on vector inputs.

        This helper evaluates the private truncated LOS kernel table for arrays
        of projected separations and LOS coordinates. It is used internally
        when working with the truncated table directly in Python.

        Args:
            kernel (np.ndarray):
                The 2D truncated LOS kernel table.
            q_grid (np.ndarray):
                The support-normalised projected-separation grid.
            z_grid (np.ndarray):
                The support-normalised LOS-coordinate grid.
            q (np.ndarray):
                The projected separations at which to evaluate the table.
            z (np.ndarray):
                The LOS coordinates at which to evaluate the table.

        Returns:
            np.ndarray: The interpolated truncated-kernel values.
        """
        # Broadcast the projected-separation and LOS-coordinate inputs onto a
        # common shape so the interpolation can be carried out element-wise.
        q, z = np.broadcast_arrays(q, z)
        values = np.zeros_like(q)

        # Only points inside the projected kernel support can contribute.
        valid = (q >= 0.0) & (q < 1.0)
        if not np.any(valid):
            return values

        # Clamp the LOS coordinate to the tabulated support range before
        # locating the interpolation cell.
        clamped_z = np.clip(z[valid], -1.0, 1.0)

        # Map the valid projected separations onto the q-axis of the table.
        scaled_q = q[valid] * (q_grid.size - 1)
        q_index = scaled_q.astype(int)
        q_next = np.minimum(q_grid.size - 1, q_index + 1)
        q_frac = scaled_q - q_index

        # Map the valid LOS coordinates onto the z-axis of the table.
        scaled_z = 0.5 * (clamped_z + 1.0) * (z_grid.size - 1)
        z_index = scaled_z.astype(int)
        z_next = np.minimum(z_grid.size - 1, z_index + 1)
        z_frac = scaled_z - z_index

        # Bilinearly interpolate the truncated kernel table.
        v00 = kernel[q_index, z_index]
        v01 = kernel[q_index, z_next]
        v10 = kernel[q_next, z_index]
        v11 = kernel[q_next, z_next]

        vz0 = v00 + z_frac * (v01 - v00)
        vz1 = v10 + z_frac * (v11 - v10)
        values[valid] = vz0 + q_frac * (vz1 - vz0)

        return values

    def _get_overlap_sample_points(self):
        """Get the sampled points used to build the overlap kernel.

        This helper prepares the sampled locations and radial-kernel weights
        used internally to average the LOS contribution across the support of
        an input particle. It is not part of the public LOS kernel API because
        the runtime only needs the final overlap table.

        Returns:
            tuple:
                A tuple containing the qx, qy, qz coordinates of the sampled
                points inside the unit support sphere and their associated
                radial-kernel weights.
        """
        with timer("Preparing overlap kernel sample points"):
            # Construct a regular grid of candidate sample points in the unit
            # cube and keep only those that fall inside the unit support sphere
            mids = np.linspace(
                -1.0 + 1.0 / OVERLAP_BUILD_NDIM,
                1.0 - 1.0 / OVERLAP_BUILD_NDIM,
                OVERLAP_BUILD_NDIM,
            )
            qx, qy, qz = np.meshgrid(mids, mids, mids, indexing="ij")
            qr2 = qx * qx + qy * qy + qz * qz
            mask = qr2 < 1.0

            qx = qx[mask]
            qy = qy[mask]
            qz = qz[mask]
            qr = np.sqrt(qr2[mask])

            # Interpolate the private radial kernel onto the sampled radii to
            # get the weights used in the overlap average
            radial_kernel = self._get_radial_kernel()
            bins = self._get_bins()
            weights = np.interp(qr, bins, radial_kernel)

        return qx, qy, qz, weights

    @timed("Building overlap kernel table")
    def _build_overlap_kernel(self):
        """Construct the smoothed LOS overlap kernel look-up table.

        The overlap kernel tabulates the kernel-averaged LOS contribution of a
        source particle to an input particle as a function of the dimensionless
        projected separation q, the dimensionless LOS offset u, and the
        smoothing-length ratio eta. The normalisation is defined in terms of
        the summed support radius, R_i + R_j, where R = h for the kernels
        used by the present LOS machinery.

        The `u` coordinate carries the full front / straddling / behind
        behaviour of the pair geometry:

        - `u <= -1` corresponds to a source particle lying entirely behind the
          input support and therefore contributes zero;
        - `-1 < u < 1` corresponds to straddling geometries where only a
          fraction of the source LOS kernel contributes;
        - `u >= 1` corresponds to a source particle lying entirely in front of
          the input support, in which case the overlap saturates to the full
          kernel-averaged LOS contribution.

        Returns:
            tuple:
                A tuple containing the overlap kernel table and the q, u, and
                eta grids that index it.

        Notes:
            The overlap table is built in two stages. Python first constructs
            the lightweight ingredients that are naturally expressed here: the
            tabulated `q`, `u`, and `eta` coordinate grids, the sampled points
            and weights used to kernel-average the input particle, and the
            truncated LOS kernel table used for source-particle foreground
            contributions. The dense numeric evaluation of the `(q, u, eta)`
            table is then delegated to the `kernel` C++ extension, which
            parallelises the outer `eta` loop with OpenMP.

            This split keeps the public API and mathematical setup readable in
            Python while moving the expensive hot loop into a lower-level
            implementation where temporary-array overhead is much smaller.
        """
        # Define the dimensionless grids used to tabulate the overlap kernel.
        # The q and u coordinates are normalised by the summed support radius,
        # while eta tracks the smoothing-length ratio between the input and
        # source particles.
        q_grid = np.linspace(0.0, 1.0, OVERLAP_Q_BINS + 1)
        u_grid = np.linspace(-1.0, 1.0, OVERLAP_U_BINS + 1)
        eta_grid = np.geomspace(
            OVERLAP_ETA_MIN, OVERLAP_ETA_MAX, OVERLAP_ETA_BINS + 1
        )

        # Get the sampled points and weights used to average the foreground LOS
        # contribution across the support of the input kernel.
        qx, qy, qz, weights = self._get_overlap_sample_points()

        # Get the truncated LOS kernel used to evaluate the foreground source
        # contribution at each sampled point.
        truncated_kernel, trunc_q, trunc_z = self._get_truncated_los_kernel()

        # Hand the prepared inputs to the C++ extension, which returns a
        # C-contiguous overlap table with shape `(q, u, eta)`. We use the local
        # CPU count here because this is a one-off build step whose cost is
        # dominated by embarrassingly parallel table evaluation rather than any
        # shared mutable state.
        kernel = compute_overlap_kernel(
            np.ascontiguousarray(q_grid, dtype=np.float64),
            np.ascontiguousarray(u_grid, dtype=np.float64),
            np.ascontiguousarray(eta_grid, dtype=np.float64),
            np.ascontiguousarray(qx, dtype=np.float64),
            np.ascontiguousarray(qy, dtype=np.float64),
            np.ascontiguousarray(qz, dtype=np.float64),
            np.ascontiguousarray(weights, dtype=np.float64),
            np.ascontiguousarray(truncated_kernel, dtype=np.float64),
            np.ascontiguousarray(trunc_q, dtype=np.float64),
            np.ascontiguousarray(trunc_z, dtype=np.float64),
            q_grid.size,
            u_grid.size,
            eta_grid.size,
            qx.size,
            trunc_q.size,
            trunc_z.size,
            os.cpu_count() or 1,
        )

        return kernel, q_grid, u_grid, eta_grid

    def get_kernel(self):
        """Compute the kernel.

        i.e. h^-2 * 2 * integral(W(r) dz) from x = 0 to sqrt(1.-b^2) for
        various values of `b`.

        Returns:
            np.ndarray: The kernel values for each impact parameter.
        """
        # Return the cached kernel if it has already been computed
        if self._projected_kernel is not None:
            return self._projected_kernel.copy()

        # Set up an array to hold the kernel values for each impact parameter
        kernel = np.zeros(self.binsize + 1)

        # Get the dimensionless impact parameter bins
        bins = self._get_bins()

        # For each impact parameter, compute the integral of W(r) along the
        # line of sight
        for ii in range(self.binsize):
            y, yerr = integrate.quad(
                self._integral_func(bins[ii]), 0, np.sqrt(1.0 - bins[ii] ** 2)
            )
            kernel[ii] = y * 2.0

        # Cache the results for future use
        self._projected_kernel = kernel

        return kernel.copy()

    @timed("Preparing overlap kernel lookup")
    def get_overlap_kernel(self):
        """Compute the overlap kernel lookup table.

        This table describes the kernel-averaged line-of-sight contribution of
        one source particle to one input particle as a function of three
        dimensionless variables:

            q = b / (R_i + R_j)
            u = (z_i - z_j) / (R_i + R_j)
            eta = h_i / h_j

        where b is the projected separation, h_i and h_j are the input and
        source smoothing lengths, and R_i and R_j are their support radii. The
        table therefore encodes the fully-behind, straddling, and fully-front
        regimes in a single look-up object for the smoothed LOS calculation.

        Returns:
            tuple:
                A tuple containing the overlap kernel table and the q, u, and
                eta grids used to index it.
        """
        # Return the cached table set if it has already been computed.
        if self._overlap_kernel is not None:
            return (
                self._overlap_kernel.copy(),
                self._overlap_q.copy(),
                self._overlap_u.copy(),
                self._overlap_eta.copy(),
            )

        # Construct the overlap kernel and cache it for future use.
        kernel, q_grid, u_grid, eta_grid = self._build_overlap_kernel()
        self._overlap_kernel = kernel
        self._overlap_q = q_grid
        self._overlap_u = u_grid
        self._overlap_eta = eta_grid

        return (
            self._overlap_kernel.copy(),
            self._overlap_q.copy(),
            self._overlap_u.copy(),
            self._overlap_eta.copy(),
        )

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
