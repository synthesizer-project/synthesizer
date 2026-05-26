"""Run all general profiling scripts and collect their plots.

This script executes the following profiling scripts:
1. profile_nparticles_scaling.py (Time vs N_particles)
2. profile_nparticles_memory.py (Memory vs N_particles)
3. profile_wavelength_scaling.py (Time vs Wavelengths)
4. profile_wavelength_memory.py (Memory vs Wavelengths)

Generated plots are saved to the requested output directory.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_script(
    script_name,
    project_root,
    nthreads=1,
    n_averages=3,
    output_dir=Path("profiling/plots"),
):
    """Run a python script and check for errors."""
    print(
        f"Running {script_name} (nthreads={nthreads}, "
        f"n_averages={n_averages})..."
    )
    # script_name is relative to profiling/general dir
    script_path = Path("profiling/general") / script_name
    try:
        cmd = [
            sys.executable,
            str(script_path),
            "--nthreads",
            str(nthreads),
            "--n_averages",
            str(n_averages),
            "--output_dir",
            str(output_dir),
        ]

        subprocess.run(
            cmd,
            check=True,
            cwd=project_root,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        sys.exit(1)


def make_all_plots(
    nthreads=1,
    n_averages=3,
    output_dir=Path("profiling/plots"),
):
    """Run all profiling scripts and collect plots in one directory."""
    # Define directories
    general_dir = Path(__file__).parent  # profiling/general/
    profiling_dir = general_dir.parent  # profiling/
    project_root = profiling_dir.parent  # repo root
    output_dir = (project_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # List of scripts to run (relative to profiling dir)
    scripts = [
        "profile_nparticles_scaling.py",
        "profile_nparticles_memory.py",
        "profile_wavelength_scaling.py",
        "profile_wavelength_memory.py",
    ]

    # Run each script
    for script in scripts:
        run_script(
            script,
            project_root,
            nthreads=nthreads,
            n_averages=n_averages,
            output_dir=output_dir,
        )

    print(f"\nAll plots generated in {output_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nthreads", type=int, default=1)
    parser.add_argument("--n_averages", type=int, default=3)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("profiling/plots"),
    )
    args = parser.parse_args()

    make_all_plots(
        nthreads=args.nthreads,
        n_averages=args.n_averages,
        output_dir=args.output_dir,
    )
