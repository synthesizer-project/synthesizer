# Synthesizer Profiling Suite

This directory contains simplified profiling scripts for Synthesizer, designed to produce clean plots for documentation.

## Core Scripts

The suite consists of two main scripts that target the primary performance dimensions: particle count scaling and thread scaling.

### 1. Particle Count Scaling (`profile_nparticles_scaling.py`)
This script measures how various operations scale with the number of particles (from $10^3$ to $10^6$).

**Operations profiled:**
- Particle Spectra Generation
- Integrated Spectra Generation
- Particle Photometry
- Smoothed Imaging

**Usage:**
```bash
python profiling/profile_nparticles_scaling.py
```

**Output:**
- `profiling/plots/particle_performance_nparticles.png`

### 2. Thread Scaling (`profile_thread_scaling.py`)
This script measures the strong scaling (speedup) of core operations as the number of threads increases.

**Operations profiled:**
- Integrated Spectra
- Particle Spectra
- Smoothed Imaging

**Usage:**
```bash
python profiling/profile_thread_scaling.py --max_threads 8 --nstars 100000
```

**Arguments:**
- `--max_threads`: Maximum number of threads to test (default: 8).
- `--nstars`: Number of stars to use for the test (default: 10^5).
- `--average_over`: Number of runs to average for each data point (default: 3).

**Outputs:**
- `profiling/plots/integrated_performance_threads.png`
- `profiling/plots/particle_performance_threads.png`
- `profiling/plots/particle_imaging_performance_threads.png`

## Design Principles

1. **Clean Plots:** Titles are removed from plots to make them suitable for inclusion in documentation where captions are used instead.
2. **Clear Naming:** Filenames explicitly distinguish between **particle** and **integrated** approaches.
3. **Reproducibility:** All scripts use a fixed random seed (42) and standard test grids.

## Requirements

- **Test grid:** Scripts require `test_grid` to be available.
- Download via: `synthesizer-download --test-grids`
- **Dependencies:** `numpy`, `matplotlib`, `unyt`.