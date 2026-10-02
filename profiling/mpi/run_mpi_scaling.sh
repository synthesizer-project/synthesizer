#!/bin/bash
#
# Run the MPI weak or strong scaling test of the Pipeline over a range of
# rank counts and plot the results.
#
# Each rank is pinned to its own block of cores (by default 16, one NUMA
# domain on a COSMA8 node, with 8 ranks per node). Run this from the
# repository root inside a job allocation large enough for the largest rank
# count, e.g. 4 COSMA8 nodes for 32 ranks of 16 threads.
#
# Requirements:
#   - An MPI implementation loaded (e.g. module load openmpi/5.0.3) and
#     mpi4py installed in the Python environment
#   - Synthesizer and the profiling test data as for the other profiling
#     scripts
#
# Usage:
#   bash profiling/mpi/run_mpi_scaling.sh --mode weak --ngalaxies 25
#   bash profiling/mpi/run_mpi_scaling.sh --mode strong --ngalaxies 1000

set -e

MODE=""
NGALAXIES=""
NPARTICLES=10000
NTHREADS=16
RANKS_PER_NODE=8
RANKS="1 2 4 8 16 32"
GRID_PRECISION=float64
OUTPUT_ROOT="profiling/outputs"

while [[ $# -gt 0 ]]; do
	case $1 in
	--mode)
		MODE="$2"
		shift 2
		;;
	--ngalaxies)
		NGALAXIES="$2"
		shift 2
		;;
	--nparticles)
		NPARTICLES="$2"
		shift 2
		;;
	--nthreads)
		NTHREADS="$2"
		shift 2
		;;
	--ranks-per-node)
		RANKS_PER_NODE="$2"
		shift 2
		;;
	--ranks)
		RANKS="$2"
		shift 2
		;;
	--grid-precision)
		GRID_PRECISION="$2"
		shift 2
		;;
	--output-dir)
		OUTPUT_ROOT="$2"
		shift 2
		;;
	-h | --help)
		echo "Usage: $0 --mode weak|strong --ngalaxies N [OPTIONS]"
		echo ""
		echo "Options:"
		echo "  --mode weak|strong      Weak (per-rank work) or strong (fixed total)"
		echo "  --ngalaxies N           Galaxies per rank (weak) or in total (strong)"
		echo "  --nparticles N          Stellar particles per galaxy (default: 10000)"
		echo "  --nthreads N            Threads per rank (default: 16)"
		echo "  --ranks-per-node N      Ranks per node (default: 8)"
		echo "  --ranks \"LIST\"          Rank counts to run (default: \"1 2 4 8 16 32\")"
		echo "  --grid-precision DTYPE  float32 or float64 (default: float64)"
		echo "  --output-dir PATH       Output root (default: profiling/outputs)"
		exit 0
		;;
	*)
		echo "Unknown option: $1"
		echo "Use --help for usage information"
		exit 1
		;;
	esac
done

if [[ "$MODE" != "weak" && "$MODE" != "strong" ]] || [ -z "$NGALAXIES" ]; then
	echo "Error: --mode weak|strong and --ngalaxies are required"
	exit 1
fi

OUT_DIR="$OUTPUT_ROOT/mpi_$MODE"
CSV="$OUT_DIR/mpi_$MODE.csv"
mkdir -p "$OUT_DIR"

# Start a fresh set of measurements
rm -f "$CSV"

for nranks in $RANKS; do
	echo "Running MPI $MODE scaling on $nranks ranks x $NTHREADS threads..."
	mpirun -np "$nranks" \
		--map-by "ppr:$RANKS_PER_NODE:node:pe=$NTHREADS" \
		--bind-to core \
		-x OMP_NUM_THREADS="$NTHREADS" \
		-x LD_LIBRARY_PATH \
		python profiling/mpi/pipeline_mpi_scaling.py \
		--mode "$MODE" \
		--ngalaxies "$NGALAXIES" \
		--nparticles "$NPARTICLES" \
		--nthreads "$NTHREADS" \
		--grid-precision "$GRID_PRECISION" \
		--out-dtype "$GRID_PRECISION" \
		--out_csv "$CSV"
done

python profiling/mpi/analyse_mpi_scaling.py \
	--input "$CSV" \
	--output "$OUT_DIR/mpi_${MODE}_scaling.png"
