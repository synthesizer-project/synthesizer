#!/bin/bash
# Package the SVO filter cache for CI distribution.
#
# After running the full test suite, docs build, and examples locally,
# this script tars the populated cache directory so it can be uploaded
# to the data server for CI consumption.
#
# Usage:
#   ./package_svo_cache.sh [output_path]
#
# Args:
#   output_path: Path for the output tarball (default: svo_filter_cache.tar.gz)

set -e

OUTPUT="${1:-svo_filter_cache.tar.gz}"

# Resolve the cache directory from Python
CACHE_DIR=$(python -c "from synthesizer import SVO_FILTER_CACHE_DIR; print(SVO_FILTER_CACHE_DIR)")

# Count cached filters
EXISTING=$(ls "$CACHE_DIR"/*.hdf5 2>/dev/null | wc -l)

if [ "$EXISTING" -eq 0 ]; then
    echo "No cached filters found in $CACHE_DIR"
    echo "Run the tests, docs build, and examples first to populate the cache."
    exit 1
fi

echo "SVO filter cache: $CACHE_DIR ($EXISTING filter files)"
echo "Packaging into $OUTPUT..."
tar -czf "$OUTPUT" -C "$CACHE_DIR" .
echo "Done: $OUTPUT"
