# Synthesizer Profiling System

This directory contains profiling scripts for Synthesizer, organized into three categories:

1. **Pipeline Profiling** (`pipeline/`) - Profile timing and memory use across implementations using the `Pipeline` interface to compare to "real-world" performance.
2. **Scaling Analysis** (`scaling/`) - Strong scaling of shared memory parallelism (OpenMP) implementations.
3. **General Performance** (`general/`) - General performance scaling (particles, wavelength array size, etc.) for core algorithms.

---

## Pipeline Profiling (`pipeline/`)

Profile timing and memory performance of full `Pipeline` runs. This can be used to compare performance across branches, precision settings, or other configurations. These scripts use the real `Pipeline` object and provide the option to run all available operations (LOS optical depths, SFZH/SFH, spectra, photometry, lines, imaging, data cubes, spectroscopy) in both rest-frame and observer-frame variants.

### Quick Start: Compare Branches

To use the pipeline profiling tools to compare two branches (e.g. `main` vs `feature-branch`), run the following commands:

```bash
# Profile main branch
git checkout main
<INSTALL OPTIONS> pip install -e .
python profiling/pipeline/profile_timing.py --basename "main_1000p"
python profiling/pipeline/profile_memory.py --basename "main_1000p"

# Profile feature branch
git checkout feature-branch
<INSTALL OPTIONS> pip install -e .
python profiling/pipeline/profile_timing.py --basename "feature_1000p"
python profiling/pipeline/profile_memory.py --basename "feature_1000p"

# Compare timings
python profiling/pipeline/analyze_timing.py \
  --inputs profiling/outputs/timing/main_1000p/timing.csv \
           profiling/outputs/timing/feature_1000p/timing.csv \
  --labels "main" "feature"

# Compare memory profiles
python profiling/pipeline/analyze_memory.py \
  --inputs profiling/outputs/memory/main_1000p/memory.csv \
           profiling/outputs/memory/feature_1000p/memory.csv \
  --labels "main" "feature"


# Validate numerical precision of outputs
python profiling/pipeline/validate_results.py \
  --inputs profiling/outputs/timing/main_1000p/output.h5 \
           profiling/outputs/timing/feature_1000p/output.h5 \
  --labels "main" "feature" \
  --tolerance default
```

### Core Profiling Tools

#### `profile_timing.py` - Measure Execution Time

Profile `Pipeline` execution time. Runs setup (grid load, galaxy/instrument build, emission model) + `pipeline.run()` for all operations, then extracts timing data from `pipeline._op_timing`.

**Usage:**

```bash
python profiling/pipeline/profile_timing.py --basename "run_name" \
  [--nparticles 1000] [--ngalaxies 10] [--fov-kpc 60] \
  [--include-observer-frame]
```

**Arguments:**

- `--basename` (required): Name for this profiling run
- `--nparticles` (optional, default 1000): Stellar particles per galaxy
- `--ngalaxies` (optional, default 10): Number of galaxies
- `--seed` (optional, default 42): Random seed
- `--fov-kpc` (optional, default 60): Field of view for imaging in kpc
- `--include-observer-frame`: Include observer-frame/flux operations
  (get_observed_spectra, get_photometry_fluxes, get_observed_lines,
  get_images_flux, get_data_cubes_fnu, get_spectroscopy_fnu)

**Output:**

- `profiling/outputs/timing/{basename}/timing.csv`

#### `profile_memory.py` - Memory Sampling

Profile memory usage with 1000 Hz continuous sampling during full `Pipeline`
run (setup + execution). Samples RSS memory from setup through
`pipeline.run()` completion.

**Usage:**

```bash
python profiling/pipeline/profile_memory.py --basename "run_name" \
  [--nparticles 1000] [--ngalaxies 10] [--fov-kpc 60] \
  [--include-observer-frame]
```

**Arguments:**

- Same as `profile_timing.py` above

**Output:**

- `profiling/outputs/memory/{basename}/memory.csv` (all raw 1000 Hz samples)

### Analysing Profiling Tools

#### `analyze_timing.py` - Compare Timing Results

Compare execution times across 2 or more runs.

**Usage:**

```bash
python profiling/pipeline/analyze_timing.py \
  --inputs timing1.csv timing2.csv [timing3.csv ...] \
  [--labels "label1" "label2" ...] \
  [--output-dir output_dir]
```

**Output:**

- Stacked bar chart: `timing_comparison.png`
- Summary statistics: `timing_summary.txt`

#### `analyze_memory.py` - Compare Memory Profiles

Compare memory usage across 2 or more runs.

**Usage:**

```bash
python profiling/pipeline/analyze_memory.py \
  --inputs memory1.csv memory2.csv [memory3.csv ...] \
  [--labels "label1" "label2" ...] \
  [--output-dir output_dir]
```

**Output:**

- Memory timelines overlay: `memory_comparison.png`
- Summary statistics: `memory_summary.txt`

#### `validate_results.py` - Numerical Precision Validation

Compare HDF5 output files from `Pipeline.write()` and validate numerical
precision. Auto-discovers datasets from the structured HDF5 layout
(Galaxies/Spectra/..., Galaxies/Stars/..., etc.) and compares all numeric
datasets across runs.

**Usage:**

```bash
python profiling/pipeline/validate_results.py \
  --inputs output1.h5 output2.h5 [output3.h5 ...] \
  [--labels "label1" "label2" ...] \
  [--tolerance default|loose|tight] \
  [--datasets path1 path2 ...]
```

**Arguments:**

- `--inputs`: HDF5 files to compare (from `Pipeline.write()`)
- `--labels`: Optional labels for each file (default: use filenames)
- `--tolerance`: Comparison tolerance (default: rtol=1e-6, atol=1e-8)
- `--datasets`: Specific HDF5 dataset paths to compare (default:
  auto-discover all numeric datasets from first file)

**Tolerance Levels:**

- `default`: rtol=1e-6, atol=1e-8
- `loose`: rtol=1e-4, atol=1e-6
- `tight`: rtol=1e-7, atol=1e-9

---

## Scaling Analysis (`scaling/`)

Strong scaling analysis showing how performance scales with thread count and parallelism. Useful for understanding OpenMP efficiency and optimization opportunities.

### Available Profilers

| Script                            | Purpose                                     | Output                            |
| --------------------------------- | ------------------------------------------- | --------------------------------- |
| `profile_thread_scaling.py`       | Thread count strong scaling                 | `plots/*_performance_threads.png` |
| `spectral_cube_strong_scaling.py` | Spectral cube strong scaling with threads   | `plots/scaling_*.png`             |
| `strong_scaling_images.py`        | Image generation strong scaling             | `plots/scaling_*.png`             |
| `strong_scaling_int_spectra.py`   | Integrated spectra strong scaling           | `plots/scaling_*.png`             |
| `strong_scaling_los_col_den.py`   | Line-of-sight column density strong scaling | `plots/scaling_*.png`             |
| `strong_scaling_part_spectra.py`  | Particle spectra strong scaling             | `plots/scaling_*.png`             |
| `strong_scaling_photometry.py`    | Photometry strong scaling                   | `plots/scaling_*.png`             |

### Usage Examples

```bash
# Profile thread scaling (strong scaling)
python profiling/scaling/profile_thread_scaling.py --max_threads 8 --nstars 100000

# Profile photometry scaling
python profiling/scaling/strong_scaling_photometry.py --max_threads 8 --nstars 100000 --nfilters 10

# Run all strong scaling tests
python profiling/scaling/strong_scaling_images.py
python profiling/scaling/strong_scaling_int_spectra.py
python profiling/scaling/strong_scaling_part_spectra.py
```

### Output

Plots are saved to `profiling/plots/` and are suitable for inclusion in documentation (titles removed for caption flexibility).

---

## General Performance (`general/`)

General performance scaling analysis showing how performance scales with problem size (particle count, wavelength array size, etc.).

### Available Profilers

| Script                          | Purpose                             | Output                                      |
| ------------------------------- | ----------------------------------- | ------------------------------------------- |
| `profile_nparticles_scaling.py` | Particle count scaling (10³ to 10⁵) | `plots/nparticles_performance_*.png`        |
| `profile_nparticles_memory.py`  | Particle count memory scaling       | `plots/nparticles_performance_memory_*.png` |
| `profile_wavelength_scaling.py` | Wavelength array size timing impact | `plots/wavelength_performance_*.png`        |
| `profile_wavelength_memory.py`  | Wavelength array size memory impact | `plots/wavelength_performance_memory_*.png` |

### Usage Examples

```bash
# Profile particle count scaling
python profiling/general/profile_nparticles_scaling.py

# Profile wavelength scaling
python profiling/general/profile_wavelength_scaling.py

# Memory scaling analyses
python profiling/general/profile_nparticles_memory.py
python profiling/general/profile_wavelength_memory.py
```

### `make_all_plots.py`

Generates documentation plots from profiling results. Used to create publication-ready figures from profiling data.

```bash
python profiling/general/make_all_plots.py
```

---

## Output Directory Structure

```
profiling/
├── pipeline/               # Pipeline comparison tools
│   ├── __init__.py
│   ├── profile_timing.py
│   ├── profile_memory.py
│   ├── analyze_timing.py
│   ├── analyze_memory.py
│   ├── validate_results.py
│   └── compare_precision_builds.sh
│
├── scaling/                # Strong scaling analysis tools
│   ├── __init__.py
│   ├── profile_thread_scaling.py
│   ├── strong_scaling_*.py
│   └── spectral_cube_strong_scaling.py
│
├── general/                # General performance scaling
│   ├── __init__.py
│   ├── profile_nparticles_scaling.py
│   ├── profile_nparticles_memory.py
│   ├── profile_wavelength_*.py
│   └── make_all_plots.py
│
├── outputs/                # Output data from pipeline profiling
│   ├── timing/             # timing.csv files
│   ├── memory/             # memory.csv files
│   ├── timing_analysis/    # comparison plots & summaries
│   ├── memory_analysis/    # comparison plots & summaries
│   └── precision_validation/  # validation plots
│
└── plots/                  # Output plots from scaling analysis
```
