#!/bin/bash
#
# Generate ALL profiling plots for documentation
#
# This script runs the complete profiling suite and generates all plots used in
# the performance documentation:
#   - Pipeline profiling (timing and memory)
#   - Particle and wavelength scaling
#   - Strong scaling (thread count)
#
# Can be run from anywhere:
#     bash profiling/run_doc_profiling_plots.sh  (from repo root)
#     ./run_doc_profiling_plots.sh               (from profiling/ directory)
#
# Requirements:
#   - Test grids downloaded (synthesizer-download --test-grids)
#   - Python environment with synthesizer installed
#   - At least 16 GB RAM recommended for larger particle counts
#   - HPC system with 32+ threads recommended for strong scaling tests
#
# Runtime: ~30-45 minutes on AMD EPYC 7542 (8-32 threads)
#   - Pipeline profiling: ~15-20 minutes
#   - Particle/wavelength scaling: ~10-15 minutes
#   - Strong scaling: ~5-10 minutes
#

set -e # Exit on error

# Default values
PIPELINE_THREADS=8
SCALING_THREADS=8
STRONG_THREADS=32
STRONG_AVERAGES=10

# Parse command line arguments
while [[ $# -gt 0 ]]; do
	case $1 in
	--pipeline-threads)
		PIPELINE_THREADS="$2"
		shift 2
		;;
	--scaling-threads)
		SCALING_THREADS="$2"
		shift 2
		;;
	--strong-threads)
		STRONG_THREADS="$2"
		shift 2
		;;
	--strong-averages)
		STRONG_AVERAGES="$2"
		shift 2
		;;
	-h | --help)
		echo "Usage: $0 [OPTIONS]"
		echo ""
		echo "Options:"
		echo "  --pipeline-threads N    Number of threads for pipeline profiling (default: 8)"
		echo "  --scaling-threads N     Number of threads for particle/wavelength scaling (default: 8)"
		echo "  --strong-threads N      Max threads for strong scaling tests (default: 32)"
		echo "  --strong-averages N     Number of averages for strong scaling (default: 10)"
		echo "  -h, --help             Show this help message"
		echo ""
		echo "Examples:"
		echo "  # Run with default settings (8 threads for pipeline, 32 for strong scaling)"
		echo "  bash profiling/run_doc_profiling_plots.sh"
		echo ""
		echo "  # Run with 16 threads for pipeline profiling"
		echo "  bash profiling/run_doc_profiling_plots.sh --pipeline-threads 16"
		echo ""
		echo "  # Run with 32 threads everywhere"
		echo "  bash profiling/run_doc_profiling_plots.sh --pipeline-threads 32 --scaling-threads 32"
		exit 0
		;;
	*)
		echo "Unknown option: $1"
		echo "Use --help for usage information"
		exit 1
		;;
	esac
done

# Determine the script directory and repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to repository root
cd "$REPO_ROOT"

echo "========================================"
echo "Configuration"
echo "========================================"
echo "Working directory: $REPO_ROOT"
echo "Pipeline threads: $PIPELINE_THREADS"
echo "Scaling threads: $SCALING_THREADS"
echo "Strong scaling max threads: $STRONG_THREADS"
echo "Strong scaling averages: $STRONG_AVERAGES"
echo ""

# Check we're in the right place
if [ ! -d "profiling" ] || [ ! -d "docs" ]; then
	echo "Error: Cannot find profiling/ and docs/ directories"
	echo "This script must be in the profiling/ directory of the synthesizer repository"
	exit 1
fi

echo "========================================"
echo "Pipeline Profiling - Timing"
echo "========================================"

# Run Pipeline timing profiling for different particle counts
for npart in 100 500 1000 5000 10000; do
	echo "Running timing profiling for ${npart} particles..."
	python profiling/pipeline/profile_timing.py \
		--basename "npart_${npart}" \
		--nparticles ${npart} \
		--ngalaxies 10 \
		--nthreads $PIPELINE_THREADS \
		--include-observer-frame
done

echo ""
echo "========================================"
echo "Pipeline Profiling - Memory"
echo "========================================"

# Run Pipeline memory profiling with appropriate sampling frequencies
# Higher frequencies for shorter runs to get good temporal resolution
python profiling/pipeline/profile_memory.py \
	--basename "npart_100" \
	--nparticles 100 \
	--ngalaxies 10 \
	--nthreads $PIPELINE_THREADS \
	--sample-freq 5000 \
	--include-observer-frame

python profiling/pipeline/profile_memory.py \
	--basename "npart_500" \
	--nparticles 500 \
	--ngalaxies 10 \
	--nthreads $PIPELINE_THREADS \
	--sample-freq 3000 \
	--include-observer-frame

python profiling/pipeline/profile_memory.py \
	--basename "npart_1000" \
	--nparticles 1000 \
	--ngalaxies 10 \
	--nthreads $PIPELINE_THREADS \
	--sample-freq 2000 \
	--include-observer-frame

python profiling/pipeline/profile_memory.py \
	--basename "npart_5000" \
	--nparticles 5000 \
	--ngalaxies 10 \
	--nthreads $PIPELINE_THREADS \
	--sample-freq 1000 \
	--include-observer-frame

python profiling/pipeline/profile_memory.py \
	--basename "npart_10000" \
	--nparticles 10000 \
	--ngalaxies 10 \
	--nthreads $PIPELINE_THREADS \
	--sample-freq 500 \
	--include-observer-frame

echo ""
echo "========================================"
echo "Analysing Results and Generating Plots"
echo "========================================"

# Analyse timing results
echo "Generating timing analysis plots..."
python profiling/pipeline/analyse_timing.py \
	--inputs \
	profiling/outputs/timing/npart_100/timing.csv \
	profiling/outputs/timing/npart_500/timing.csv \
	profiling/outputs/timing/npart_1000/timing.csv \
	profiling/outputs/timing/npart_5000/timing.csv \
	profiling/outputs/timing/npart_10000/timing.csv \
	--labels "100" "500" "1000" "5000" "10000" \
	--output-dir profiling/outputs/timing_analysis

# Analyse memory results
echo "Generating memory analysis plots..."
python profiling/pipeline/analyse_memory.py \
	--inputs \
	profiling/outputs/memory/npart_100/memory.csv \
	profiling/outputs/memory/npart_500/memory.csv \
	profiling/outputs/memory/npart_1000/memory.csv \
	profiling/outputs/memory/npart_5000/memory.csv \
	profiling/outputs/memory/npart_10000/memory.csv \
	--labels "100" "500" "1000" "5000" "10000" \
	--output-dir profiling/outputs/memory_analysis

echo ""
echo "========================================"
echo "Particle and Wavelength Scaling"
echo "========================================"

# Run the general profiling scripts using make_all_plots.py
echo "Running particle and wavelength scaling profiling..."
python profiling/general/make_all_plots.py --nthreads $SCALING_THREADS --n_averages 3

echo ""
echo "========================================"
echo "Strong Scaling (Thread Count)"
echo "========================================"

# Run strong scaling tests
echo "Running integrated spectra strong scaling..."
python profiling/scaling/strong_scaling_int_spectra.py \
	--basename docs \
	--out_dir profiling/plots \
	--max_threads $STRONG_THREADS \
	--nstars 1000000 \
	--average_over $STRONG_AVERAGES \
	--low_thresh 0.01

echo "Running particle spectra strong scaling..."
python profiling/scaling/strong_scaling_part_spectra.py \
	--basename docs \
	--out_dir profiling/plots \
	--max_threads $STRONG_THREADS \
	--nstars 10000 \
	--average_over $STRONG_AVERAGES \
	--low_thresh 0.01

echo "Running LOS column density strong scaling..."
python profiling/scaling/strong_scaling_los_col_den.py \
	--basename docs \
	--out_dir profiling/plots \
	--max_threads $STRONG_THREADS \
	--nstars 1000000 \
	--ngas 1000000 \
	--average_over $STRONG_AVERAGES \
	--low_thresh 0.01

echo "Running imaging strong scaling..."
python profiling/scaling/strong_scaling_images.py \
	--basename test \
	--out_dir profiling/plots \
	--max_threads $STRONG_THREADS \
	--nstars 10000 \
	--average_over $STRONG_AVERAGES \
	--low_thresh 0.01

echo ""
echo "========================================"
echo "Copying All Plots to Documentation"
echo "========================================"

# Copy Pipeline profiling plots
echo "Copying Pipeline profiling plots..."
cp profiling/outputs/timing_analysis/timing_comparison.png \
	docs/source/performance/plots/pipeline_timing_scaling.png

cp profiling/outputs/memory_analysis/memory_comparison_normalized.png \
	docs/source/performance/plots/pipeline_memory_normalized.png

cp profiling/outputs/memory_analysis/memory_comparison_scaling.png \
	docs/source/performance/plots/pipeline_memory_scaling.png

# Copy particle/wavelength scaling plots (generated by make_all_plots.py)
echo "Copying particle and wavelength scaling plots..."
cp profiling/plots/nparticles_performance_*.png docs/source/performance/plots/
cp profiling/plots/wavelength_performance_*.png docs/source/performance/plots/

# Copy strong scaling plots
echo "Copying strong scaling plots..."
cp profiling/plots/docs_int_spectra_cic_totThreads${STRONG_THREADS}_nstars1000000.png \
	docs/source/performance/plots/docs_int_spectra_cic_totThreads32_nstars1000000.png
cp profiling/plots/docs_part_spectra_cic_totThreads${STRONG_THREADS}_nstars10000.png \
	docs/source/performance/plots/docs_part_spectra_cic_totThreads32_nstars10000.png
cp profiling/plots/docs_los_column_density_totThreads${STRONG_THREADS}_nstars1000000_ngas1000000.png \
	docs/source/performance/plots/docs_los_column_density_totThreads32_nstars1000000_ngas1000000.png
cp profiling/plots/test_images_totThreads${STRONG_THREADS}_nstars10000.png \
	docs/source/performance/plots/test_images_totThreads32_nstars10000.png

echo ""
echo "========================================"
echo "All Profiling Complete!"
echo "========================================"
echo ""
echo "Generated plots:"
echo "  Pipeline Profiling ($PIPELINE_THREADS threads):"
echo "    - pipeline_timing_scaling.png"
echo "    - pipeline_memory_normalized.png"
echo "    - pipeline_memory_scaling.png"
echo "  Particle/Wavelength Scaling ($SCALING_THREADS threads):"
echo "    - nparticles_performance_*.png (6 plots)"
echo "    - wavelength_performance_*.png (2 plots)"
echo "  Strong Scaling (up to $STRONG_THREADS threads, $STRONG_AVERAGES averages):"
echo "    - docs_int_spectra_cic_totThreads32_nstars1000000.png"
echo "    - docs_part_spectra_cic_totThreads32_nstars10000.png"
echo "    - docs_los_column_density_totThreads32_nstars1000000_ngas1000000.png"
echo "    - test_images_totThreads32_nstars10000.png"
echo ""
echo "All plots have been copied to docs/source/performance/plots/"
echo ""
