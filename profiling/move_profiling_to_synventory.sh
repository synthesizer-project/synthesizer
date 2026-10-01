#!/bin/bash
#
# Copy the documentation profiling plots into a synventory checkout, from
# where the docs link to them. The plots are grouped in subdirectories of the
# profiling output root (see run_doc_profiling_plots.sh and
# mpi/run_mpi_scaling.sh), which are mirrored in synventory's profiling/.

set -e

OUTPUT_ROOT="profiling/outputs"
SYNVENTORY_DIR="../synventory"

while [[ $# -gt 0 ]]; do
	case $1 in
	--output-dir)
		OUTPUT_ROOT="$2"
		shift 2
		;;
	--synventory)
		SYNVENTORY_DIR="$2"
		shift 2
		;;
	-h | --help)
		echo "Usage: $0 [OPTIONS]"
		echo ""
		echo "Options:"
		echo "  --output-dir PATH       Profiling output root (default: profiling/outputs)"
		echo "  --synventory PATH       synventory checkout (default: ../synventory)"
		exit 0
		;;
	*)
		echo "Unknown option: $1"
		exit 1
		;;
	esac
done

if [ ! -d "$SYNVENTORY_DIR/profiling" ]; then
	echo "Error: $SYNVENTORY_DIR/profiling not found, pass --synventory"
	exit 1
fi

# Copy each group of plots that has been generated
for group in pipeline problem_size thread_scaling mpi_weak mpi_strong; do
	if compgen -G "$OUTPUT_ROOT/$group/*.png" >/dev/null; then
		mkdir -p "$SYNVENTORY_DIR/profiling/$group"
		cp "$OUTPUT_ROOT/$group"/*.png "$SYNVENTORY_DIR/profiling/$group/"
		echo "Copied $group plots"
	fi
done

echo "Copied profiling plots from $OUTPUT_ROOT to $SYNVENTORY_DIR/profiling"
echo "Commit and push them in synventory to update the documentation."
