"""Measure MPI weak or strong scaling of the Pipeline.

Each rank builds its own fake galaxies and the whole Pipeline (the same
operations as the pipeline profiling scripts) is run across all ranks.

    - Weak scaling: every rank gets --ngalaxies galaxies, so the total work
      grows with the number of ranks. Ideally the run time stays constant.
    - Strong scaling: --ngalaxies galaxies in total are split as evenly as
      possible across the ranks. Ideally the run time halves when the number
      of ranks doubles.

Only Pipeline.run is timed, as the slowest rank's wall-clock time. Rank 0
appends one row to --out_csv per invocation, so run this once per rank count
and plot the results with analyse_mpi_scaling.py.

Usage:
    mpirun -np 4 python profiling/mpi/pipeline_mpi_scaling.py --mode weak \
        --ngalaxies 25 --nparticles 10000 --nthreads 16 \
        --out_csv profiling/outputs/mpi_weak/mpi_weak.csv
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI

from synthesizer import set_default_out_dtype
from synthesizer.grid import Grid
from synthesizer.pipeline import Pipeline

# Add pipeline profiling to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
from pipeline_test_data import (
    add_test_operations,
    build_test_galaxies,
    get_test_emission_model,
)


def local_galaxy_count(mode, ngalaxies, rank, size):
    """Get the number of galaxies this rank should build.

    Args:
        mode (str): "weak" or "strong".
        ngalaxies (int): Galaxies per rank (weak) or in total (strong).
        rank (int): This rank.
        size (int): The number of ranks.

    Returns:
        int: The number of galaxies for this rank.
    """
    if mode == "weak":
        return ngalaxies
    return ngalaxies // size + (1 if rank < ngalaxies % size else 0)


def main():
    """Run one MPI scaling measurement and record it."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mode", choices=("weak", "strong"), required=True)
    parser.add_argument(
        "--ngalaxies",
        type=int,
        required=True,
        help="Galaxies per rank (weak) or in total (strong).",
    )
    parser.add_argument("--nparticles", type=int, default=10000)
    parser.add_argument("--nthreads", type=int, default=16)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument(
        "--grid-precision",
        choices=("float32", "float64"),
        default="float64",
        help="Precision to load the grid arrays at.",
    )
    parser.add_argument(
        "--out-dtype",
        choices=("float32", "float64"),
        default=None,
        help="Requested output precision. Defaults to the global default.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Set the global output precision if one was requested
    if args.out_dtype is not None:
        set_default_out_dtype(np.dtype(args.out_dtype))

    # Build this rank's galaxies, seeded per rank so they differ
    grid = Grid("test_grid", use_precision=args.grid_precision)
    nlocal = local_galaxy_count(args.mode, args.ngalaxies, rank, size)
    galaxies = build_test_galaxies(
        grid, args.nparticles, nlocal, seed=args.seed + rank
    )

    # Set up the Pipeline with the standard profiling operations
    pipeline = Pipeline(
        emission_model=get_test_emission_model(grid),
        nthreads=args.nthreads,
        verbose=0,
        comm=comm,
    )
    pipeline.add_galaxies(galaxies)
    add_test_operations(pipeline, grid, include_observer_frame=True)

    # Time the run, starting every rank together
    comm.Barrier()
    start = time.perf_counter()
    pipeline.run()
    elapsed = time.perf_counter() - start

    # The run takes as long as its slowest rank; the fastest shows imbalance
    slowest = comm.reduce(elapsed, op=MPI.MAX, root=0)
    fastest = comm.reduce(elapsed, op=MPI.MIN, root=0)
    ntotal = comm.reduce(nlocal, op=MPI.SUM, root=0)

    if rank == 0:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        new_file = not args.out_csv.exists()
        with open(args.out_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(
                    [
                        "mode",
                        "nranks",
                        "nthreads",
                        "ngalaxies_total",
                        "nparticles",
                        "run_time_max",
                        "run_time_min",
                    ]
                )
            writer.writerow(
                [
                    args.mode,
                    size,
                    args.nthreads,
                    ntotal,
                    args.nparticles,
                    f"{slowest:.6f}",
                    f"{fastest:.6f}",
                ]
            )
        print(
            f"{args.mode}: {size} ranks x {args.nthreads} threads, "
            f"{ntotal} galaxies, run {slowest:.2f}s (fastest rank "
            f"{fastest:.2f}s)"
        )


if __name__ == "__main__":
    main()
