"""Tests for depositing particle attributes onto meshes."""

import numpy as np
import pytest
from unyt import Msun, Myr, kpc, unyt_array

from synthesizer import exceptions
from synthesizer.kernel_functions import Kernel
from synthesizer.mesh import Meshes
from synthesizer.particle import Gas, Stars


def make_gas(n=500, seed=0, sml_range=(-1.5, 0.3), dtype=np.float64):
    """Make a gas component with smoothing lengths spanning many cells."""
    rng = np.random.default_rng(seed)
    return Gas(
        masses=rng.uniform(1e5, 2e5, n).astype(dtype) * Msun,
        metallicities=rng.uniform(0.001, 0.02, n).astype(dtype),
        coordinates=rng.normal(0, 2, (n, 3)).astype(dtype) * kpc,
        smoothing_lengths=(10 ** rng.uniform(*sml_range, n)).astype(dtype)
        * kpc,
        redshift=1,
        dust_to_metal_ratio=0.3,
    )


def make_stars(n=500, seed=1):
    """Make a stellar component without smoothing lengths."""
    rng = np.random.default_rng(seed)
    return Stars(
        initial_masses=rng.uniform(1e5, 2e5, n) * Msun,
        ages=rng.uniform(1, 1e3, n) * Myr,
        metallicities=rng.uniform(0.001, 0.02, n),
        coordinates=rng.normal(0, 1, (n, 3)) * kpc,
        current_masses=rng.uniform(5e4, 1e5, n) * Msun,
        redshift=1,
    )


def single_particle(x, h, cell=1.0):
    """Make a one particle gas component at x with smoothing length h."""
    return Gas(
        masses=np.array([1.0]) * Msun,
        metallicities=np.array([0.01]),
        coordinates=np.array([x], dtype=float) * kpc,
        smoothing_lengths=np.array([h]) * kpc,
        redshift=1,
        dust_to_metal_ratio=0.3,
    )


class TestConservation:
    """Every mode must deposit exactly the particle totals."""

    @pytest.mark.parametrize("as_points", [False, True])
    @pytest.mark.parametrize("refine", [False, True])
    def test_totals(self, as_points, refine):
        """Totals match for smoothed/CIC and uniform/refined meshes."""
        gas = make_gas()
        kwargs = {}
        if refine:
            kwargs = dict(refine_attr="masses", refine_threshold=1e6 * Msun)
        meshes = gas.get_meshes(
            0.5 * kpc,
            extensive=("masses", "dust_masses"),
            as_points=as_points,
            **kwargs,
        )
        assert meshes.refined == refine
        np.testing.assert_allclose(
            meshes.masses.sum(), gas.masses.sum(), rtol=1e-12
        )
        np.testing.assert_allclose(
            meshes.dust_masses.sum(), gas.dust_masses.sum(), rtol=1e-12
        )
        assert np.all(meshes.masses >= 0)

    @pytest.mark.parametrize("h", [1e-4, 0.3, 0.5, 1.0, 7.3])
    @pytest.mark.parametrize(
        "x", [(0.5, 0.5, 0.5), (0.0, 0.0, 0.0), (0.0, 0.37, 0.5)]
    )
    def test_single_particle(self, x, h):
        """One particle anywhere, of any size, deposits exactly its mass."""
        meshes = single_particle(x, h).get_meshes(
            1.0 * kpc, extensive="masses"
        )
        np.testing.assert_allclose(meshes.masses.sum(), 1.0, rtol=1e-13)

    def test_corner_symmetry(self):
        """A kernel centred on a shared cell corner splits evenly."""
        gas = single_particle((0.0, 0.0, 0.0), 0.8)
        meshes = gas.get_meshes(0.8 * kpc, extensive="masses")
        # The support spans exactly two cells per axis.
        assert meshes.dims == (2, 2, 2)
        np.testing.assert_allclose(meshes.masses.value, 1.0 / 8, rtol=1e-12)

    def test_kernel_profile(self):
        """A large kernel recovers the cell-averaged SPH kernel."""
        h = 5.0
        gas = single_particle((0.0, 0.0, 0.0), h)
        meshes = gas.get_meshes(1.0 * kpc, extensive="masses")

        # Brute-force cell averages of the kernel with 12^3 samples per cell.
        nsub = 12
        offsets = (np.arange(nsub) + 0.5) / nsub - 0.5
        sub = np.stack(
            np.meshgrid(offsets, offsets, offsets, indexing="ij"), axis=-1
        ).reshape(-1, 3)
        centres = meshes.cell_centres.to_value(kpc).reshape(-1, 1, 3)
        r = np.linalg.norm(centres + sub, axis=-1)
        kernel = Kernel("sph_anarchy")
        expected = kernel.f(r / h).mean(axis=1) / h**3  # cell volume 1 kpc^3

        # Well inside the kernel the deposit matches to better than 0.5%.
        inner = np.linalg.norm(centres[:, 0], axis=-1) < 0.8 * h
        np.testing.assert_allclose(
            meshes.masses.value.ravel()[inner], expected[inner], rtol=5e-3
        )


class TestGeometry:
    """The domain is set by the particles' support at the resolution."""

    def test_domain_covers_support(self):
        """The domain contains every kernel and is centred on them."""
        gas = make_gas()
        meshes = gas.get_meshes(0.3 * kpc, extensive="masses")
        pos = gas.coordinates.to_value(kpc)
        sml = gas.smoothing_lengths.to_value(kpc)
        lo = (pos - sml[:, None]).min(axis=0)
        hi = (pos + sml[:, None]).max(axis=0)
        origin = meshes.origin.to_value(kpc)
        extent = meshes.extent.to_value(kpc)
        assert np.all(origin <= lo) and np.all(extent >= hi)
        np.testing.assert_allclose(origin + extent, lo + hi, atol=1e-9)
        np.testing.assert_array_equal(
            meshes.dims, np.ceil((hi - lo) / 0.3).astype(int)
        )
        assert meshes.masses.shape == meshes.dims

    def test_cic_domain(self):
        """Cloud-in-cell pads the particle extent by half a cell."""
        stars = make_stars()
        meshes = stars.get_meshes(
            0.25 * kpc, extensive="initial_masses", as_points=True
        )
        pos = stars.coordinates.to_value(kpc)
        assert np.all(meshes.origin.to_value(kpc) <= pos.min(0) - 0.125)
        assert np.all(meshes.extent.to_value(kpc) >= pos.max(0) + 0.125)

    def test_refined_leaves_tile_domain(self):
        """Leaves exactly tile the domain and respect the refinement rule."""
        gas = make_gas(n=1000)
        threshold = 1e6
        meshes = gas.get_meshes(
            1.0 * kpc,
            extensive="masses",
            refine_attr="masses",
            refine_threshold=threshold * Msun,
            max_depth=6,
        )
        assert meshes.cell_depths.max() > 0
        domain = np.prod(meshes.extent - meshes.origin)
        np.testing.assert_allclose(meshes.cell_volumes.sum(), domain)
        assert meshes.cell_centres.shape == (meshes.ncells, 3)

        # Leaves above the threshold must have been blocked from refining by
        # the kernel floor or the depth limit (a lone contributor can only
        # exceed it if its own mass does, which it cannot here).
        over = meshes.masses.to_value(Msun) > threshold
        widths = meshes.cell_widths.to_value(kpc)
        blocked = (meshes.cell_depths[over] == 6) | (
            0.5 * widths[over] < gas.smoothing_lengths.to_value(kpc).max()
        )
        assert np.all(blocked)

    def test_like_reuses_geometry(self):
        """Stars can be deposited on the gas mesh, refinement included."""
        gas = make_gas()
        stars = make_stars()
        gas_meshes = gas.get_meshes(
            1.0 * kpc,
            extensive="masses",
            refine_attr="masses",
            refine_threshold=1e6 * Msun,
        )
        star_meshes = stars.get_meshes(
            1.0 * kpc,
            extensive="initial_masses",
            as_points=True,
            like=gas_meshes,
        )
        assert star_meshes.shape == gas_meshes.shape
        np.testing.assert_array_equal(
            star_meshes.cell_centres, gas_meshes.cell_centres
        )
        np.testing.assert_allclose(
            star_meshes.initial_masses.sum(), stars.initial_masses.sum()
        )

    def test_like_rejects_outside(self):
        """Particles outside the like domain raise."""
        stars = make_stars()
        small = stars.get_meshes(
            0.5 * kpc,
            extensive="initial_masses",
            as_points=True,
            mask=np.linalg.norm(stars.coordinates.to_value(kpc), axis=1) < 1,
        )
        with pytest.raises(exceptions.InconsistentArguments):
            Meshes.from_particles(
                stars,
                0.5 * kpc,
                extensive="initial_masses",
                as_points=True,
                like=small,
            )

    def test_mask(self):
        """Masked meshes only see (and are bounded by) masked particles."""
        stars = make_stars()
        mask = stars.coordinates.to_value(kpc)[:, 0] > 0
        meshes = stars.get_meshes(
            0.5 * kpc, extensive="initial_masses", as_points=True, mask=mask
        )
        np.testing.assert_allclose(
            meshes.initial_masses.sum(), stars.initial_masses[mask].sum()
        )
        assert meshes.origin.to_value(kpc)[0] > -0.5


class TestFields:
    """Field semantics, weighting and access."""

    def test_intensive_constant(self):
        """A constant intensive field is recovered wherever mass lands."""
        gas = make_gas()
        gas.const = np.full(gas.nparticles, 0.25)
        meshes = gas.get_meshes(0.5 * kpc, intensive="const")
        occupied = meshes.const > 0
        assert occupied.any()
        np.testing.assert_allclose(meshes.const[occupied], 0.25, rtol=1e-12)
        assert meshes.field_info["const"]["weight"] == "masses"

    def test_intensive_weighting(self):
        """Intensive fields are the weighted mean of the deposits."""
        stars = make_stars()
        meshes = stars.get_meshes(
            0.5 * kpc,
            extensive=("initial_masses", "current_masses"),
            intensive="ages",
            weights={"ages": "current_masses"},
            as_points=True,
        )
        assert meshes.field_info["ages"]["weight"] == "current_masses"

        # Recompute the weighted sum with an explicit extensive deposit.
        stars.mass_age = stars.current_masses.value * stars.ages.value
        check = Meshes.from_particles(
            stars, 0.5 * kpc, extensive="mass_age", as_points=True
        )
        occupied = meshes.current_masses.value > 0
        np.testing.assert_allclose(
            meshes.ages.value[occupied],
            check.mass_age[occupied] / meshes.current_masses.value[occupied],
            rtol=1e-10,
        )

    def test_default_star_weight(self):
        """Stars weight intensive fields by initial mass by default."""
        meshes = make_stars().get_meshes(
            0.5 * kpc, intensive="metallicities", as_points=True
        )
        assert meshes.field_info["metallicities"]["weight"] == (
            "initial_masses"
        )

    def test_access(self):
        """Fields behave like a dictionary and like attributes."""
        gas = make_gas()
        meshes = gas.get_meshes(
            0.5 * kpc, extensive="masses", intensive="metallicities"
        )
        assert gas.meshes is meshes
        assert meshes["masses"] is meshes.masses
        assert "masses" in meshes and "ages" not in meshes
        assert list(meshes) == ["masses", "metallicities"]
        assert len(meshes) == 2
        assert dict(meshes.items()).keys() == meshes.keys()
        assert str(meshes.masses.units) == "Msun"
        assert str(meshes.resolution.units) == "kpc"
        assert (
            meshes.density("masses").units.dimensions
            == (Msun / kpc**3).units.dimensions
        )
        with pytest.raises(AttributeError):
            meshes.ages
        with pytest.raises(KeyError):
            meshes["ages"]

    def test_add(self):
        """More fields can be deposited on an existing geometry."""
        gas = make_gas()
        meshes = gas.get_meshes(0.5 * kpc, extensive="masses")
        meshes.add(gas, intensive="metallicities")
        assert list(meshes) == ["masses", "metallicities"]
        assert meshes.metallicities.shape == meshes.shape

    def test_errors(self):
        """Bad requests raise informative errors."""
        gas = make_gas()
        with pytest.raises(exceptions.InconsistentArguments):
            gas.get_meshes(0.5 * kpc)
        with pytest.raises(exceptions.InconsistentArguments):
            gas.get_meshes(0.5 * kpc, extensive="not_an_attribute")
        with pytest.raises(exceptions.InconsistentArguments):
            gas.get_meshes(0.5 * kpc, extensive="masses", refine_attr="masses")
        with pytest.raises(exceptions.InconsistentArguments):
            make_stars().get_meshes(0.5 * kpc, extensive="initial_masses")
        with pytest.raises(exceptions.UnimplementedFunctionality):
            Meshes.from_parametric()


class TestLookup:
    """Point lookup, sampling and the flat cell ordering."""

    @pytest.mark.parametrize("refine", [False, True])
    def test_cell_index_matches_flat_order(self, refine):
        """Each cell centre maps back to its own flat index."""
        kwargs = {}
        if refine:
            kwargs = dict(refine_attr="masses", refine_threshold=1e6 * Msun)
        meshes = make_gas().get_meshes(0.5 * kpc, extensive="masses", **kwargs)
        centres = meshes.cell_centres.reshape(-1, 3)
        np.testing.assert_array_equal(
            meshes.cell_index(centres), np.arange(meshes.ncells)
        )
        assert meshes.flat("masses").shape == (meshes.ncells,)

    def test_flat_is_view(self):
        """Flattening a uniform field does not copy it."""
        meshes = make_gas().get_meshes(0.5 * kpc, extensive="masses")
        assert np.shares_memory(meshes.flat("masses"), meshes.masses)

    def test_outside_and_faces(self):
        """Points outside the domain get -1; the upper faces are inside."""
        meshes = make_gas().get_meshes(0.5 * kpc, extensive="masses")
        points = unyt_array(
            np.stack(
                [
                    meshes.origin.to_value(kpc) - 1.0,
                    meshes.extent.to_value(kpc) + 1.0,
                    meshes.extent.to_value(kpc),
                ]
            ),
            kpc,
        )
        idx = meshes.cell_index(points)
        assert idx[0] == -1 and idx[1] == -1
        assert idx[2] == meshes.ncells - 1

    def test_sample(self):
        """Sampling returns the containing cell's values, zero outside."""
        gas = make_gas()
        meshes = gas.get_meshes(
            0.5 * kpc,
            extensive="masses",
            intensive="metallicities",
            refine_attr="masses",
            refine_threshold=1e6 * Msun,
        )
        points = (
            np.concatenate(
                [gas.coordinates.to_value(kpc), [[100.0, 100.0, 100.0]]]
            )
            * kpc
        )
        idx = meshes.cell_index(points)
        samples = meshes.sample(points)
        assert set(samples) == {"masses", "metallicities"}
        np.testing.assert_array_equal(
            samples["masses"][:-1], meshes.flat("masses")[idx[:-1]]
        )
        assert samples["masses"][-1] == 0
        assert samples["masses"].units == meshes.masses.units
        single = meshes.sample(points, "metallicities")
        np.testing.assert_array_equal(single, samples["metallicities"])

    def test_points_need_units(self):
        """Unitless points are rejected."""
        meshes = make_gas().get_meshes(0.5 * kpc, extensive="masses")
        with pytest.raises(exceptions.MissingUnits):
            meshes.cell_index(np.zeros((1, 3)))


class TestPrecisionAndThreads:
    """Results do not depend on precision (beyond rounding) or threads."""

    def test_float32(self):
        """float32 inputs agree with float64 inputs."""
        m64 = make_gas().get_meshes(0.5 * kpc, extensive="masses")
        gas32 = make_gas(dtype=np.float32)
        m32 = gas32.get_meshes(
            0.5 * kpc, extensive="masses", out_dtype=np.float32
        )
        assert m32.masses.dtype == np.float32
        assert m32.dims == m64.dims
        np.testing.assert_allclose(
            m32.masses.value, m64.masses.value, rtol=1e-3, atol=1.0
        )

    @pytest.mark.parametrize("as_points", [False, True])
    def test_threads_bit_identical(self, as_points):
        """Any thread count gives bit-identical meshes."""
        gas = make_gas(n=2000)
        kwargs = dict(
            extensive="masses",
            intensive="metallicities",
            as_points=as_points,
            refine_attr="masses",
            refine_threshold=1e6 * Msun,
        )
        serial = Meshes.from_particles(gas, 0.5 * kpc, nthreads=1, **kwargs)
        threaded = Meshes.from_particles(gas, 0.5 * kpc, nthreads=4, **kwargs)
        np.testing.assert_array_equal(serial.masses, threaded.masses)
        np.testing.assert_array_equal(
            serial.metallicities, threaded.metallicities
        )
