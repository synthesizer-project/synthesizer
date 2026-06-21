# Self-Consistent Spatial Resampling with Deterministic Kernel Placement

## Motivation

The current `spatially_resample` implementation places child particles by
randomly sampling each parent particle's SPH kernel, then resamples
attributes independently. This has two practical limitations:

1. **Temporal noise in visualisations.** Even when the underlying galaxy
   evolves smoothly, random child placement introduces frame-to-frame
   "fizzing" in movies.

2. **Attributes ignore the reconstructed field.** A child placed in a region
   where several kernels overlap should inherit metallicity, age, velocity,
   etc. from the **local SPH field**, not just from one parent particle.

The goal of this plan is therefore:

- keep the user-facing model of `resample_factor = N children per parent`
- conserve mass exactly per parent group
- make child placement predictable rather than random
- evaluate attributes from the global SPH field at the final child positions

A new `method` parameter selects between the existing stochastic behaviour
(`"random"`) and the new deterministic field-aware behaviour (`"field"`).

`"field"` mode will perform best for temporal coherence when the input
particles are ordered consistently between snapshots, ideally by particle ID.

---

## Design overview

### `method` parameter

```
method : str, optional (default "random")
```

- `"random"` — existing per-particle kernel sampling (current behaviour,
  unchanged)
- `"field"` — deterministic child placement within each parent kernel, with
  attributes interpolated from the global SPH field

### Field-mode algorithm

For each parent particle `i`:

1. Generate exactly `resample_factor` **deterministic offsets** inside the
   parent kernel.
2. Scale those offsets by the parent smoothing length and translate by the
   parent coordinate to obtain final child positions.
3. Evaluate the global SPH field at all child positions in a **single C++
   call** via the octree, accumulating:
   ```
   rho(point)     = sum_j m_j * W(q_j) / h_j^3
   Z_weighted     = sum_j Z_j * m_j * W(q_j) / h_j^3
   age_weighted   = sum_j age_j * m_j * W(q_j) / h_j^3
   vel_weighted   = sum_j v_j * m_j * W(q_j) / h_j^3
   ```
4. Compute attributes at the chosen positions:
   - `metallicity = Z_weighted / rho`
   - `age = 10**(log_age_weighted / rho)` if ages are interpolated in log-space
   - `velocity = vel_weighted / rho`
   - apply `attr_modes` scatter only where it is still physically sensible
5. Set masses to `parent_mass / resample_factor`.
6. Set smoothing lengths by scaling the parent smoothing length as in the
   current implementation, preserving the existing volume-conservation
   convention.

This keeps the mass assignment parent-owned and deterministic while making the
thermodynamic / chemical / kinematic properties field-consistent.

### Why deterministic placement instead of field-weighted random placement?

Using the field value `rho(point)` to *randomly* choose child positions still
introduces Monte Carlo noise and can over-emphasise overlap regions when each
parent contributes a fixed number of children. Deterministic placement avoids
that failure mode and directly addresses the movie-fizzing problem.

---

## Deterministic kernel offsets

### Requirements

The deterministic child-placement rule should:

- place exactly `resample_factor` children per parent
- stay inside the parent kernel support
- sample the kernel volume smoothly and predictably
- avoid obvious lattice artefacts
- depend only on `resample_factor` and parent ordering

### Proposed implementation

Add a helper that generates a fixed set of offsets in the **unit kernel**:

```python
def deterministic_kernel_offsets(kernel, n_samples):
    """Return a fixed set of kernel-sampled offsets in the unit kernel.

    Args:
        kernel: Kernel instance.
        n_samples (int): Number of child particles per parent.

    Returns:
        np.ndarray: (n_samples, 3) float64 offsets with support radius <= 1.
    """
```

Recommended construction:

1. Use a low-discrepancy sequence in `[0, 1)^3` (Sobol or Halton).
2. Map one coordinate through the kernel radial inverse CDF.
3. Map the remaining two coordinates to a uniform direction on the sphere.
4. Convert to Cartesian unit-kernel offsets.

This gives deterministic, well-distributed child positions without needing a
random number generator in `"field"` mode.

### Optional per-parent phase rotation

If needed later, a small deterministic rotation or phase shift can be applied
per parent to reduce coherent directional structure. For now, that is not
required. The first implementation can simply reuse the same ordered offset set
for every parent.

---

## New C++ extension: `sph_density`

### File: `src/synthesizer/extensions/sph_density.cpp`

A new extension module (`synthesizer.extensions.sph_density`) that performs
batched SPH field evaluation at query points using the existing octree.

### Data structures

Uses the existing `struct particle` from `octree.h` (`pos[3]`, `sml`, `index`)
with no modifications. The `index` field is used to look up masses and any
attribute arrays passed from Python.

### Core function: `evaluate_sph_density`

Python signature:

```
evaluate_sph_density(
    query_positions,       # (N_query, 3) float64
    particle_pos,          # (N_part, 3) float64
    smoothing_lengths,     # (N_part,) float64
    masses,                # (N_part,) float64
    attribute_arrays,      # tuple of (N_part,) float64 arrays
    kernel_name,           # str
    maxdepth,              # int (default 16)
    min_count,             # int (default 8)
)
-> (density, *weighted_attrs)
```

### Algorithm

1. Build the octree from `particle_pos` and `smoothing_lengths` via the
   existing `construct_cell_tree`.
2. For each query point, walk the tree:
   - skip any branch whose minimum distance to the query point exceeds the
     maximum kernel reach stored for that branch
   - at leaf cells, loop over particles and accumulate
     ```
     q = |r_query - r_particle| / h_particle
     if q >= 1: continue
     w = W(q) / h^3 * m
     density += w
     attr_weighted += w * attr_value
     ```
3. Return `density` and all weighted attribute sums.
4. Free the tree via `cleanup_cell_tree`.

The octree is built once per call and reused across all child positions.

### Kernel function dispatch

Reuse `get_kernel_function` from `kernel_functions.h`.

### Python bindings

Expose a single `evaluate_sph_density` entry point. Inputs are extracted using
the existing numpy helpers and outputs are wrapped as numpy arrays.

### Registration in `setup.py`

```python
create_extension(
    "synthesizer.extensions.sph_density",
    [
        "src/synthesizer/extensions/sph_density.cpp",
        "src/synthesizer/extensions/octree.cpp",
        "src/synthesizer/extensions/property_funcs.cpp",
        "src/synthesizer/extensions/numpy_init.cpp",
        "src/synthesizer/extensions/timers.cpp",
    ],
)
```

---

## Python wrapper: `sph_density.py`

### File: `src/synthesizer/particle/sph_density.py`

A thin units-aware wrapper that:

1. accepts `unyt_array` inputs
2. strips units to `float64` ndarrays
3. calls the C++ extension
4. reattaches units to returned density and weighted attributes

```python
def evaluate_sph_density(
    query_positions,
    particle_positions,
    smoothing_lengths,
    masses,
    attributes,
    kernel_name,
    maxdepth=16,
    min_count=8,
):
```

---

## New utilities in `resample_utils.py`

### `deterministic_kernel_offsets`

```python
def deterministic_kernel_offsets(kernel, n_samples):
    """Generate deterministic offsets inside the unit kernel support."""
```

This replaces the need for random candidate generation in `"field"` mode.

### `resample_coordinates_field`

```python
def resample_coordinates_field(
    coordinates,
    smoothing_lengths,
    kernel,
    resample_factor,
):
    """Generate deterministic child coordinates for field mode.

    Uses a shared deterministic unit-kernel offset pattern for all parents,
    scaled by each particle's smoothing length.
    """
```

Returns:

- `new_coordinates`: `(N * resample_factor, 3)`
- `parent_indices`: `(N * resample_factor,)`

### `resample_attributes_field`

```python
def resample_attributes_field(
    density,
    attr_weighted,
    parent_indices,
    parent_masses,
    resample_factor,
    attr_modes,
    rng,
):
    """Compute field-interpolated child attributes.

    Intensive quantities are derived from attr_weighted / density.
    Mass-like quantities remain parent-conserved by construction.
    """
```

### `resample_smoothing_lengths_field`

```python
def resample_smoothing_lengths_field(
    parent_smoothing_lengths,
    resample_factor,
):
    """Scale parent smoothing lengths exactly as in current resampling."""
```

This should match the current `resample_smoothing_lengths` behaviour:

```python
h_new = h_parent / resample_factor**(1/3)
```

so that `"field"` mode preserves the existing smoothing-length convention.

---

## API changes to `Gas.spatially_resample`

### New parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | `str` | `"random"` | `"random"` or `"field"` |
| `field_kernel` | `str` | `None` | Kernel used for SPH field evaluation |
| `tree_maxdepth` | `int` | `16` | Maximum octree depth |
| `tree_min_count` | `int` | `8` | Minimum particles per leaf cell |

### Control flow

```
spatially_resample(method="random"):
    # existing code path, unchanged

spatially_resample(method="field"):
    1. Validate inputs as usual
    2. Split by mask if provided
    3. Generate deterministic child coordinates
    4. Call evaluate_sph_density once for all final positions
    5. Compute field-interpolated attributes
    6. Scale smoothing lengths using the current rule
    7. Apply velocity dispersion / attr_modes where appropriate
    8. Combine with no_resample arrays
    9. Construct and return new Gas object
```

### Default attribute behaviour in field mode

| Attribute | Default behaviour |
|-----------|-------------------|
| `masses` | forced `"proportional"` |
| `metallicities` | SPH-weighted mean + optional scatter |
| `star_forming` | keep parent value by default |
| `dust_masses` | forced `"proportional"` |
| `dust_to_metal_ratio` | SPH-weighted mean |
| `tau_v` | SPH-weighted mean if per-particle, else unchanged |
| `softening_lengths` | scaled by `resample_factor^(-1/3)` |
| scalar attrs | unchanged |

For categorical or flag-like quantities, defaulting to the parent value is
preferable to a noisy neighbourhood vote.

---

## API changes to `Stars.spatially_resample`

### Same new parameters as Gas

`method`, `field_kernel`, `tree_maxdepth`, `tree_min_count`.

### SFZH / SFH / MetalDist interaction with field mode

When `method="field"` and `sfzh` / `sfh` / `metal_dist` is provided:

1. **Spatial positions** are still generated deterministically from the parent
   kernels.
2. **Ages and metallicities** are sampled from the provided SFZH / SFH /
   MetalDist inputs, as in the current specialised path.
3. **Masses** remain parent-conserved using the current logic.
4. **All other attributes** are interpolated from the SPH field at the final
   child positions.

When `method="field"` and none of those are provided:

- ages and metallicities are SPH-weighted means, optionally followed by
  scatter from `attr_modes`

### Default attribute behaviour in field mode

| Attribute | Default behaviour |
|-----------|-------------------|
| `initial_masses` | forced `"proportional"` |
| `current_masses` | forced `"proportional"` |
| `ages` | SPH-weighted mean, or SFZH / SFH override |
| `metallicities` | SPH-weighted mean, or SFZH / MetalDist override |
| `alpha_enhancement` | SPH-weighted mean |
| `s_oxygen` | SPH-weighted mean |
| `s_hydrogen` | SPH-weighted mean |
| `fesc`, `fesc_ly_alpha` | SPH-weighted mean |
| scalar attrs | unchanged |

---

## Files changed

| File | Action | Purpose |
|------|--------|---------|
| `src/synthesizer/extensions/sph_density.cpp` | **Create** | batched octree-based SPH field evaluation |
| `src/synthesizer/particle/sph_density.py` | **Create** | units-aware Python wrapper |
| `src/synthesizer/particle/resample_utils.py` | **Edit** | deterministic field-mode coordinate and attribute helpers |
| `src/synthesizer/particle/gas.py` | **Edit** | add `method` parameter and field-mode path |
| `src/synthesizer/particle/stars.py` | **Edit** | add `method` parameter and field-mode path |
| `src/synthesizer/particle/particles.py` | **Edit** | update base class docs |
| `setup.py` | **Edit** | register `sph_density` extension |

---

## Performance

| Operation | Cost |
|-----------|------|
| Octree build | `O(N_part log N_part)` once per call |
| Query evaluation | `O(N_query log N_part + N_query * n_overlap)` |
| Total queries | `N_part * resample_factor` |

Compared to the previous oversampled-candidate design, this approach evaluates
the field only at the final child positions, so it is both cheaper and more
predictable.

OpenMP parallelisation over query points can be added later if needed.

---

## Implementation order

| Step | Description | Dependencies |
|------|-------------|-------------|
| 1 | Create `sph_density.cpp` for batched SPH field evaluation | None |
| 2 | Register in `setup.py` and verify compilation | Step 1 |
| 3 | Create `sph_density.py` wrapper | Step 2 |
| 4 | Add `deterministic_kernel_offsets` to `resample_utils.py` | None |
| 5 | Add `resample_coordinates_field` to `resample_utils.py` | Step 4 |
| 6 | Add `resample_attributes_field` to `resample_utils.py` | Step 3 |
| 7 | Add `resample_smoothing_lengths_field` to `resample_utils.py` | None |
| 8 | Add `method` and field path to `Gas.spatially_resample` | Steps 5-7 |
| 9 | Add `method` and field path to `Stars.spatially_resample` | Steps 5-7 |
| 10 | Write C++ unit tests | Step 1 |
| 11 | Write Python integration tests | Steps 8-9 |
| 12 | Add example scripts comparing `random` and `field` modes | Steps 8-9 |

## Implementation checklist

- [x] Add deterministic unit-kernel offset generation in `resample_utils.py`
- [x] Add field-mode coordinate generation helper
- [x] Add SPH field-evaluation helper
- [x] Add field-mode attribute post-processing helper
- [x] Add `method="field"` to `Gas.spatially_resample`
- [x] Add `method="field"` to `Stars.spatially_resample`
- [x] Preserve the existing random mode unchanged by default
- [x] Add deterministic repeatability tests
- [x] Add field-mode mass-conservation tests
- [x] Add overlap/interpolation regression tests

---

## Testing strategy

1. **Determinism tests**
   - same input produces identical child coordinates across repeated runs
   - `method="field"` does not depend on `seed` for positions

2. **C++ unit tests**
   - single isolated particle
   - two overlapping particles
   - point at kernel boundary

3. **Python integration tests**
   - total mass is conserved exactly
   - each parent produces exactly `resample_factor` children
   - field-interpolated metallicity in an overlap region matches the expected
     SPH-weighted mean
   - `field` positions are stable under rerun

4. **Regression tests**
   - all existing `test_spatial_resampling.py` tests still pass because
     `method="random"` remains the default

5. **Visual tests**
   - compare movies or snapshot sequences from `random` and `field` modes
   - verify that `field` mode removes obvious stochastic fizzing while
     preserving smooth galaxy structure

---

## Edge cases

- **Particles without smoothing lengths** — raise `InconsistentArguments`
- **No particles** — return empty copy as current behaviour does
- **All particles masked out** — return deepcopy
- **Single particle** — `field` mode becomes deterministic kernel splitting of
  one parent, with field interpolation reducing to self-contribution only
- **Zero or negative masses** — reject in validation
- **NaN attributes** — propagate cleanly through interpolation
- **`resample_factor < 2`** — raise `ValueError`
- **Changing particle order between snapshots** — still deterministic for a
  given input ordering, but temporal coherence is best when particles are kept
  in a stable order, ideally by particle ID
