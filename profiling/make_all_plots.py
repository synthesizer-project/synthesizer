"""Run all profiling scripts and copy plots to the documentation directory.

This script executes the following profiling scripts:
1. profile_nparticles_scaling.py (Time vs N_particles)
2. profile_nparticles_memory.py (Memory vs N_particles)
3. profile_wavelength_scaling.py (Time vs Wavelengths)
4. profile_wavelength_memory.py (Memory vs Wavelengths)

Generated plots are saved to `profiling/plots/` and then copied to
`docs/source/performance/plots/`.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run_script(
    script_name, project_root, nthreads=1, n_averages=3, mem_interval=0.01
):
    """Run a python script and check for errors."""
    print(
        f"Running {script_name} (nthreads={nthreads}, "
        f"n_averages={n_averages}, mem_interval={mem_interval})..."
    )
    # script_name is relative to profiling dir
    script_path = Path("profiling") / script_name
    try:
        cmd = [
            sys.executable,
            str(script_path),
            "--nthreads",
            str(nthreads),
            "--n_averages",
            str(n_averages),
        ]
        # Only add mem_interval if the script supports it (memory scripts)
        if "memory" in script_name:
            cmd.extend(["--mem_interval", str(mem_interval)])

        subprocess.run(
            cmd,
            check=True,
            cwd=project_root,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        sys.exit(1)


def make_all_plots(nthreads=1, n_averages=3, mem_interval=0.01):
    """Run all profiling scripts and move plots."""
    # Define directories
    profiling_dir = Path(__file__).parent
    project_root = profiling_dir.parent
    source_dir = project_root / "profiling/plots"
    dest_dir = project_root / "docs/source/performance/plots"

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
            mem_interval=mem_interval,
        )

    # Create destination if it doesn't exist
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCopying plots from {source_dir} to {dest_dir}...")

    # Copy all png files from source to destination
    for src in source_dir.glob("*.png"):
        filename = src.name
        dst = dest_dir / filename
        shutil.copy2(src, dst)
        print(f"Copied {filename}")

    print("\nAll plots generated and copied successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--nthreads", type=int, default=1)

    parser.add_argument("--n_averages", type=int, default=3)

    parser.add_argument("--mem_interval", type=float, default=0.01)

    args = parser.parse_args()

    make_all_plots(
        nthreads=args.nthreads,
        n_averages=args.n_averages,
        mem_interval=args.mem_interval,
    )
