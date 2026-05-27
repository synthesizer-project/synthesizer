#!/bin/bash

set -e

OUTPUT_ROOT="profiling/outputs"
DOCS_PLOTS_DIR="docs/source/performance/plots"
STRONG_THREADS=32

while [[ $# -gt 0 ]]; do
	case $1 in
	--output-dir)
		OUTPUT_ROOT="$2"
		shift 2
		;;
	--strong-threads)
		STRONG_THREADS="$2"
		shift 2
		;;
	-h | --help)
		echo "Usage: $0 [OPTIONS]"
		echo ""
		echo "Options:"
		echo "  --output-dir PATH       Profiling output root (default: profiling/outputs)"
		echo "  --strong-threads N      Strong scaling thread count in filenames (default: 32)"
		exit 0
		;;
	*)
		echo "Unknown option: $1"
		exit 1
		;;
	esac
done

cp "$OUTPUT_ROOT/pipeline_timing_scaling.png" \
	"$DOCS_PLOTS_DIR/pipeline_timing_scaling.png"
cp "$OUTPUT_ROOT/pipeline_memory_normalized.png" \
	"$DOCS_PLOTS_DIR/pipeline_memory_normalized.png"
cp "$OUTPUT_ROOT/pipeline_memory_scaling.png" \
	"$DOCS_PLOTS_DIR/pipeline_memory_scaling.png"

cp "$OUTPUT_ROOT"/nparticles_performance_*.png "$DOCS_PLOTS_DIR/"
cp "$OUTPUT_ROOT"/wavelength_performance_*.png "$DOCS_PLOTS_DIR/"

cp "$OUTPUT_ROOT/exclusive_docs_int_spectra_cic_totThreads${STRONG_THREADS}_nstars1000000.png" \
	"$DOCS_PLOTS_DIR/exclusive_docs_int_spectra_cic_totThreads${STRONG_THREADS}_nstars1000000.png"
cp "$OUTPUT_ROOT/exclusive_docs_part_spectra_cic_totThreads${STRONG_THREADS}_nstars10000.png" \
	"$DOCS_PLOTS_DIR/exclusive_docs_part_spectra_cic_totThreads${STRONG_THREADS}_nstars10000.png"
cp "$OUTPUT_ROOT/exclusive_docs_los_column_density_totThreads${STRONG_THREADS}_nstars1000000_ngas1000000.png" \
	"$DOCS_PLOTS_DIR/exclusive_docs_los_column_density_totThreads${STRONG_THREADS}_nstars1000000_ngas1000000.png"
cp "$OUTPUT_ROOT/exclusive_docs_images_totThreads${STRONG_THREADS}_nstars10000.png" \
	"$DOCS_PLOTS_DIR/exclusive_docs_images_totThreads${STRONG_THREADS}_nstars10000.png"

echo "Copied profiling plots from $OUTPUT_ROOT to $DOCS_PLOTS_DIR"
