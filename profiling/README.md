# Synthesizer Profiling System

This directory contains profiling scripts for Synthesizer, organized into three categories:

1. **Pipeline Profiling** (`pipeline/`) - Compare timing and memory across implementations
2. **Scaling Analysis** (`scaling/`) - Strong scaling with respect to parallelism (threads)
3. **General Performance** (`general/`) - General performance scaling (particles, wavelength, etc.)

---

## Pipeline Profiling (`pipeline/`)

Compare timing and memory performance across branches, precision settings, or other configurations. Use these to measure performance impact of code changes across various operations include in a `Pipeline` run.

### Quick Start: Compare Branches

```bash
# Profile main branch
git checkout main
pip install -e .
python profiling/pipeline/profile_timing.py --basename "main_1000p"
python profiling/pipeline/profile_memory.py --basename "main_1000p"

# Profile feature branch
git checkout feature-branch
pip install -e .
python profiling/pipeline/profile_timing.py --basename "feature_1000p"
python profiling/pipeline/profile_memory.py --basename "feature_1000p"

# Compare
python profiling/pipeline/analyze_timing.py \
  --inputs profiling/outputs/timing/main_1000p/timing.csv \
           profiling/outputs/timing/feature_1000p/timing.csv \
  --labels "main" "feature"
```

### Core Scripts

#### `profile_timing.py` - Measure Execution Time

Profile pipeline execution time on current branch.

**Usage:**

```bash
python profiling/pipeline/profile_timing.py --basename "run_name" [--nparticles 1000] [--ngalaxies 10]
```

**Arguments:**

- `--basename` (required): Name for this profiling run
- `--nparticles` (optional, default 1000): Number of particles per galaxy
- `--ngalaxies` (optional, default 10): Number of galaxies
- `--seed` (optional, default 42): Random seed

**Output:**

- `profiling/outputs/timing/{basename}/timing.csv`

#### `profile_memory.py` - Memory Sampling

Profile memory usage with 1000 Hz continuous sampling.

**Usage:**

```bash
python profiling/pipeline/profile_memory.py --basename "run_name" [--nparticles 1000] [--ngalaxies 10]
```

**Output:**

- `profiling/outputs/memory/{basename}/memory.csv` (all raw 1000 Hz samples)

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

#### `validate_precision.py` - Numerical Precision Validation

Compare HDF5 output files and validate numerical precision.

**Usage:**

```bash
python profiling/pipeline/validate_precision.py \
  --inputs output1.h5 output2.h5 [output3.h5 ...] \
  [--labels "label1" "label2" ...] \
  [--tolerance default|loose|tight]
```

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

### Usage Examples

```bash
# Profile thread scaling (strong scaling)
python profiling/scaling/profile_thread_scaling.py --max_threads 8 --nstars 100000

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

## Profiling Directory Structure

```
profiling/
├── pipeline/               # Pipeline comparison tools
│   ├── __init__.py
│   ├── profile_timing.py
│   ├── profile_memory.py
│   ├── analyze_timing.py
│   ├── analyze_memory.py
│   ├── validate_precision.py
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
