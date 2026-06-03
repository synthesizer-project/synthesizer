"""A module defining SPH kernels integrated along the line-of-sight.

This module provides a class for calculating SPH kernels integrated along the
line-of-sight. These tables are used when computing LOS column densities and
optical depths for particle data.

Available kernels include:
    - uniform
    - sph_anarchy
    - gadget_2
    - cubic
    - quartic
    - quintic
"""

import h5py
import numpy as np

from synthesizer._version import __version__
from synthesizer.extensions.kernel import (
    compute_overlap_kernel,
    compute_projected_kernel,
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
    input_array = np.asarray(r)
    radii = np.atleast_1d(input_array).ravel()
    values = evaluate_kernel(radii, kernel_name)

    if input_array.ndim == 0:
        return float(values[0])

    return values.reshape(input_array.shape)


class Kernel:
    """A class describing a SPH kernel integrated along the line-of-sight.

    This class packages three related lookup tables used by the LOS machinery.

    1. The projected LOS kernel for point-like input particles.
    2. The truncated LOS kernel for cases where the input lies inside the
       source kernel and only part of the source contributes in front of the
       input particle.
    3. The overlap kernel for cases where the input particle itself has a
       finite smoothing length and its own kernel must be averaged over.

    For a source particle with smoothing length h, a sight line passing the
    source at impact parameter b intersects the source support over a chord of
    length

        l = 2 * sqrt(h^2 - b^2),

    provided b < h. To convert this geometric path length into a LOS column
    density we must weight the path by the SPH kernel W(r), where

        r = sqrt(z^2 + b^2).

    The full projected contribution of the source kernel is therefore

        D(b, h) = 2 * integral W(r) dz,

    with the integral running from z = 0 to z = sqrt(h^2 - b^2). Here W(r)
    has units of h^-3, so the LOS integral has units of h^-2 as required for a
    surface density kernel.

    The key simplification is that the LOS integral can be written in terms of
    support-normalized coordinates. If we define

        q = b / h,
        z_hat = z / h,

    then r / h = sqrt(z_hat^2 + q^2), and the projected kernel becomes

        D(q, h) = h^-2 * 2 * integral W_hat(sqrt(z_hat^2 + q^2)) dz_hat,

    where the integral now runs from z_hat = 0 to z_hat = sqrt(1 - q^2), and
    W_hat denotes the dimensionless kernel shape. The factor of h^-2 is applied
    separately when evaluating each particle pair, while the dimensionless part
    of the integral can be tabulated once as a 1D function of q. That is the
    projected LOS kernel returned by `get_kernel()`.

    When the input particle lies inside the source kernel, not all of the LOS
    chord contributes: only the part of the source kernel in front of the input
    particle should be counted. In that case the LOS integral must be truncated
    at the input particle's LOS coordinate. We tabulate this cumulative
    foreground contribution as a 2D function of projected separation and
    support-normalized truncation coordinate,

        z_trunc = (z_input - z_source) / h,

    where z_input and z_source are the LOS coordinates of the input and source
    particles. This truncated table is returned by
    `get_truncated_los_kernel()`.

    Finally, if the input particle is not treated as point-like, then the LOS
    signal seen by that particle must itself be averaged over the input
    particle's kernel support. The smoothed-input overlap table does exactly
    that: for each point sampled inside the input kernel, it evaluates the
    truncated source-kernel contribution and then averages those values using
    the input-kernel weights. The overlap table is indexed by three
    dimensionless quantities,

        q = b / (R_input + R_source),
        u = (z_input - z_source) / (R_input + R_source),
        eta = h_input / h_source,

    where R_input and R_source are the support radii of the input and source
    kernels. This table is returned by `get_overlap_kernel()` and is the lookup
    used by the smoothed-input LOS path.
    """

    _HDF5_FORMAT_VERSION = 1

    def __init__(
        self,
        name="sph_anarchy",
        binsize=10000,
        truncated_q_binsize=None,
        truncated_z_binsize=1000,
        overlap_q_binsize=64,
        overlap_u_binsize=128,
        overlap_eta_binsize=48,
        overlap_eta_min=0.1,
        overlap_eta_max=10.0,
        overlap_build_ndim=16,
        projected_integration_steps=256,
    ):
        """Initialize the kernel class.

        Args:
            name (str):
                The name of the kernel to use. Options are: "uniform",
                "sph_anarchy", "gadget_2", "cubic", "quartic", "quintic".
            binsize (int):
                The number of bins to use for the projected LOS kernel table.
            truncated_q_binsize (int, optional):
                The number of bins to use along the impact-parameter axis of
                the truncated LOS kernel table. If omitted this falls back to
                ``binsize``.
            truncated_z_binsize (int):
                The number of bins to use along the LOS truncation axis of the
                truncated LOS kernel table.
            overlap_q_binsize (int):
                The number of bins to use along the projected-separation axis
                of the smoothed-input overlap kernel table.
            overlap_u_binsize (int):
                The number of bins to use along the LOS-offset axis of the
                smoothed-input overlap kernel table.
            overlap_eta_binsize (int):
                The number of bins to use along the smoothing-length-ratio axis
                of the smoothed-input overlap kernel table.
            overlap_eta_min (float):
                The lower bound of the smoothing-length-ratio axis of the
                smoothed-input overlap kernel table.
            overlap_eta_max (float):
                The upper bound of the smoothing-length-ratio axis of the
                smoothed-input overlap kernel table.
            overlap_build_ndim (int):
                The number of midpoint samples per Cartesian dimension used to
                build the smoothed-input overlap kernel table.
            projected_integration_steps (int):
                The number of trapezoidal integration steps used to build the
                projected LOS kernel table.
        """
        # What kernel to use
        self.name = name

        # The resolution parameter for the 1D projected kernel lookup table
        self.binsize = binsize

        # The resolution parameters for the 2D truncated LOS kernel
        # lookup table
        self.truncated_q_binsize = (
            binsize if truncated_q_binsize is None else truncated_q_binsize
        )
        self.truncated_z_binsize = truncated_z_binsize

        # The resolution parameters for the 3D smoothed-input overlap kernel
        # lookup table
        self.overlap_q_binsize = overlap_q_binsize
        self.overlap_u_binsize = overlap_u_binsize
        self.overlap_eta_binsize = overlap_eta_binsize
        self.overlap_eta_min = overlap_eta_min
        self.overlap_eta_max = overlap_eta_max
        self.overlap_build_ndim = overlap_build_ndim
        self.projected_integration_steps = projected_integration_steps

        # Make sure we have valid look up table parameters
        if self.binsize <= 0:
            raise ValueError("binsize must be greater than 0")
        if self.truncated_q_binsize <= 0:
            raise ValueError("truncated_q_binsize must be greater than 0")
        if self.truncated_z_binsize <= 0:
            raise ValueError("truncated_z_binsize must be greater than 0")
        if self.overlap_q_binsize <= 0:
            raise ValueError("overlap_q_binsize must be greater than 0")
        if self.overlap_u_binsize <= 0:
            raise ValueError("overlap_u_binsize must be greater than 0")
        if self.overlap_eta_binsize <= 0:
            raise ValueError("overlap_eta_binsize must be greater than 0")
        if self.overlap_build_ndim <= 0:
            raise ValueError("overlap_build_ndim must be greater than 0")
        if self.overlap_eta_min <= 0.0:
            raise ValueError("overlap_eta_min must be greater than 0")
        if self.overlap_eta_min >= self.overlap_eta_max:
            raise ValueError(
                "overlap_eta_min must be less than overlap_eta_max"
            )
        if self.projected_integration_steps <= 0:
            raise ValueError(
                "projected_integration_steps must be greater than 0"
            )

        # Set the kernel function based on the provided name
        if name == "uniform":
            self.f = uniform
        elif name == "sph_anarchy":
            self.f = sph_anarchy
        elif name == "gadget_2":
            self.f = gadget_2
        elif name == "cubic":
            self.f = cubic
        elif name == "quartic":
            self.f = quartic
        elif name == "quintic":
            self.f = quintic
        else:
            raise ValueError("Kernel name not defined")

        # Cache the LOS kernel tables once they have been built.
        self._projected_kernel = None
        self._truncated_los_kernel = None
        self._overlap_kernel = None
        self._overlap_q = None
        self._overlap_u = None
        self._overlap_eta = None

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
            return self._projected_kernel

        # Get the dimensionless impact-parameter bins and set up the output.
        bins = self._get_bins()
        kernel = compute_projected_kernel(
            bins,
            self.name,
            self.projected_integration_steps,
        )

        # Cache it.
        self._projected_kernel = kernel

        return kernel

    @timed("Kernel.create_kernel")
    def create_kernel(self, filepath=None, nthreads=1):
        """Build and save the kernel lookup tables to an HDF5 file.

        Args:
            filepath (str, optional):
                The path to the HDF5 file to write. If omitted this defaults to
                ``kernel_<name>.hdf5``.
            nthreads (int):
                The number of threads to use when building the overlap table if
                it has not already been cached.

        Returns:
            np.ndarray:
                The projected LOS kernel table.
        """
        # Create the filepath if it was not provided
        if filepath is None:
            filepath = f"kernel_{self.name}.hdf5"

        # Write metadata to the Header and then save the kernel tables using
        # the HDF5 writer
        with h5py.File(filepath, "w") as hdf:
            header = hdf.create_group("Header")
            header.attrs["synthesizer_version"] = __version__
            header.attrs["type"] = "Kernel"
            header.attrs["format"] = "kernel_lookup"
            header.attrs["format_version"] = self._HDF5_FORMAT_VERSION

            self.to_hdf5(hdf.create_group("Kernel"), nthreads=nthreads)

        return self.get_kernel()

    def to_hdf5(self, group, nthreads=1):
        """Save the kernel lookup tables and metadata to an HDF5 group.

        Args:
            group (h5py.Group):
                The group in which to save the kernel tables.
            nthreads (int):
                The number of threads to use when building the overlap table if
                it has not already been cached.
        """
        projected_kernel = self.get_kernel()
        projected_bins = self._get_bins()

        truncated_kernel, truncated_q, truncated_z = (
            self.get_truncated_los_kernel()
        )
        overlap_kernel, overlap_q, overlap_u, overlap_eta = (
            self.get_overlap_kernel(nthreads=nthreads)
        )

        group.attrs["name"] = self.name
        group.attrs["binsize"] = self.binsize
        group.attrs["truncated_q_binsize"] = self.truncated_q_binsize
        group.attrs["truncated_z_binsize"] = self.truncated_z_binsize
        group.attrs["overlap_q_binsize"] = self.overlap_q_binsize
        group.attrs["overlap_u_binsize"] = self.overlap_u_binsize
        group.attrs["overlap_eta_binsize"] = self.overlap_eta_binsize
        group.attrs["overlap_eta_min"] = self.overlap_eta_min
        group.attrs["overlap_eta_max"] = self.overlap_eta_max
        group.attrs["overlap_build_ndim"] = self.overlap_build_ndim
        group.attrs["projected_integration_steps"] = (
            self.projected_integration_steps
        )
        group.attrs["format_version"] = self._HDF5_FORMAT_VERSION

        datasets = {
            "projected_kernel": projected_kernel,
            "projected_bins": projected_bins,
            "truncated_kernel": truncated_kernel,
            "truncated_q": truncated_q,
            "truncated_z": truncated_z,
            "overlap_kernel": overlap_kernel,
            "overlap_q": overlap_q,
            "overlap_u": overlap_u,
            "overlap_eta": overlap_eta,
        }
        for dataset_name, data in datasets.items():
            group.create_dataset(dataset_name, data=data, dtype=np.float64)

    @classmethod
    def _from_hdf5(cls, group):
        """Create a Kernel from an HDF5 group.

        Args:
            group (h5py.Group):
                The group containing a serialized kernel.

        Returns:
            Kernel:
                The Kernel restored from the HDF5 group.
        """
        instance = cls(
            name=group.attrs["name"],
            binsize=int(group.attrs["binsize"]),
            truncated_q_binsize=int(group.attrs["truncated_q_binsize"]),
            truncated_z_binsize=int(group.attrs["truncated_z_binsize"]),
            overlap_q_binsize=int(group.attrs["overlap_q_binsize"]),
            overlap_u_binsize=int(group.attrs["overlap_u_binsize"]),
            overlap_eta_binsize=int(group.attrs["overlap_eta_binsize"]),
            overlap_eta_min=float(group.attrs["overlap_eta_min"]),
            overlap_eta_max=float(group.attrs["overlap_eta_max"]),
            overlap_build_ndim=int(group.attrs["overlap_build_ndim"]),
            projected_integration_steps=int(
                group.attrs["projected_integration_steps"]
            ),
        )

        instance._projected_kernel = group["projected_kernel"][...]
        instance._truncated_los_kernel = group["truncated_kernel"][...]
        instance._overlap_kernel = group["overlap_kernel"][...]
        instance._overlap_q = group["overlap_q"][...]
        instance._overlap_u = group["overlap_u"][...]
        instance._overlap_eta = group["overlap_eta"][...]

        return instance

    @classmethod
    def load(cls, filepath):
        """Load a Kernel from an HDF5 file.

        Args:
            filepath (str):
                The path to the HDF5 file containing the serialized kernel.

        Returns:
            Kernel:
                The reloaded Kernel instance.
        """
        with h5py.File(filepath, "r") as hdf:
            group = hdf["Kernel"] if "Kernel" in hdf else hdf
            return cls._from_hdf5(group)

    def _get_z_bins(self):
        """Get the dimensionless LOS truncation bins for the 2D lookup."""
        return np.linspace(-1.0, 1.0, self.truncated_z_binsize + 1)

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
            return self._truncated_los_kernel, bins, z_bins

        # Get the projected-separation and LOS-coordinate bins and set up the
        # output.
        bins = self._get_bins(self.truncated_q_binsize)
        z_bins = self._get_z_bins()
        kernel = compute_truncated_los_kernel(
            bins,
            z_bins,
            self.name,
        )

        # Cache it.
        self._truncated_los_kernel = kernel

        return self._truncated_los_kernel, bins, z_bins

    @timed("Kernel._get_overlap_sample_points")
    def _get_overlap_sample_points(self):
        """Get the sampled points used to build the overlap kernel.

        Returns:
            tuple:
                The x, y, z sample coordinates inside the unit support sphere
                and their radial-kernel weights.
        """
        # Sample points at the midpoints of a regular Cartesian grid spanning
        # the unit sphere, then mask out points outside the sphere.
        mids = np.linspace(
            -1.0 + 1.0 / self.overlap_build_ndim,
            1.0 - 1.0 / self.overlap_build_ndim,
            self.overlap_build_ndim,
        )
        qx, qy, qz = np.meshgrid(mids, mids, mids, indexing="ij")
        qr2 = qx * qx + qy * qy + qz * qz
        mask = qr2 < 1.0

        # Apply the spherical mask
        qx = qx[mask]
        qy = qy[mask]
        qz = qz[mask]
        qr = np.sqrt(qr2[mask])

        # Compute the kernel weights at the sampled points based on their
        # radial coordinates
        weights = self.f(qr)

        return qx, qy, qz, weights

    @timed("Kernel._build_overlap_kernel")
    def _build_overlap_kernel(self, nthreads=1):
        """Construct the smoothed LOS overlap kernel look-up table.

        Returns:
            tuple:
                The overlap kernel table together with its q, u, and eta grids.
        """
        # Get the q, u, and eta grids that index the overlap kernel table
        q_grid = np.linspace(0.0, 1.0, self.overlap_q_binsize + 1)
        u_grid = np.linspace(-1.0, 1.0, self.overlap_u_binsize + 1)
        eta_grid = np.geomspace(
            self.overlap_eta_min,
            self.overlap_eta_max,
            self.overlap_eta_binsize + 1,
        )

        # Convert the grids into sample points inside the unit sphere and get
        # the kernel weights at those points
        qx, qy, qz, weights = self._get_overlap_sample_points()

        # Get the truncated LOS kernel table, we need this to evaluate the
        # truncated LOS contribution at each sample point inside the input
        # kernel when building the overlap table
        truncated_kernel, trunc_q, trunc_z = self.get_truncated_los_kernel()

        # Build the overlap kernel
        kernel = compute_overlap_kernel(
            q_grid,
            u_grid,
            eta_grid,
            qx,
            qy,
            qz,
            weights,
            truncated_kernel,
            trunc_q,
            trunc_z,
            q_grid.size,
            u_grid.size,
            eta_grid.size,
            qx.size,
            trunc_q.size,
            trunc_z.size,
            nthreads,
        )

        return kernel, q_grid, u_grid, eta_grid

    @timed("Kernel.get_overlap_kernel")
    def get_overlap_kernel(self, nthreads=1):
        """Compute the overlap kernel lookup table.

        Returns:
            tuple:
                The overlap kernel table together with its q, u, and eta grids.
        """
        # Return the cached kernel if it has already been computed.
        if self._overlap_kernel is not None:
            return (
                self._overlap_kernel,
                self._overlap_q,
                self._overlap_u,
                self._overlap_eta,
            )

        # Build the overlap table and cache the result.
        kernel, q_grid, u_grid, eta_grid = self._build_overlap_kernel(
            nthreads=nthreads
        )
        self._overlap_kernel = kernel
        self._overlap_q = q_grid
        self._overlap_u = u_grid
        self._overlap_eta = eta_grid

        return (
            self._overlap_kernel,
            self._overlap_q,
            self._overlap_u,
            self._overlap_eta,
        )


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


def quartic(r):
    """Calculate the SPHENIX quartic spline (M5) kernel.

    Args:
        r (float): The distance from the center of the kernel.

    Returns:
        float: The value of the quartic spline kernel.
    """
    return _call_kernel_function("quartic", r)


def quintic(r):
    """Calculate the quintic kernel.

    Args:
        r (float): The distance from the center of the kernel.

    Returns:
        float: The value of the quintic kernel.
    """
    return _call_kernel_function("quintic", r)
