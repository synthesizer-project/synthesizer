#!/bin/bash
#
# Format all C/C++ extension source files with clang-format.
#
# Usage:
#   ./format_extensions.sh           # format in-place
#   ./format_extensions.sh --check   # dry-run, exit 1 if any file would change

set -euo pipefail

REQUIRED_VERSION="22.1.4"

HERE="$(cd "$(dirname "$0")" && pwd)"
CLANG_FORMAT="${CLANG_FORMAT:-clang-format}"

if ! command -v "$CLANG_FORMAT" &>/dev/null; then
  echo "Error: $CLANG_FORMAT not found."
  echo "Install clang-format $REQUIRED_VERSION with:"
  echo "  brew install clang-format"
  echo "  pip install clang-format==$REQUIRED_VERSION"
  exit 1
fi

VERSION=$("$CLANG_FORMAT" --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
echo "Using $("$CLANG_FORMAT" --version)"

if [[ "$VERSION" != "$REQUIRED_VERSION" ]]; then
  echo "Warning: expected clang-format $REQUIRED_VERSION, got $VERSION."
  echo "Install the correct version with:"
  echo "  brew install clang-format"
  echo "  pip install clang-format==$REQUIRED_VERSION"
  echo "Or set CLANG_FORMAT to point to a specific binary."
fi

MODE="${1:--i}"

case "$MODE" in
-i)
  echo "Formatting all C/C++ files in-place..."
  ;;
--check)
  echo "Checking formatting of all C/C++ files..."
  ;;
*)
  echo "Usage: $0 [-i | --check]"
  echo "  -i        Format files in-place (default)"
  echo "  --check   Dry-run; exit 1 if any file would be changed"
  exit 1
  ;;
esac

collect_files() {
  find "$HERE/src" \( -name '*.cpp' -o -name '*.h' \) -print0
}

case "$MODE" in
-i)
  collect_files | xargs -0 "$CLANG_FORMAT" -i
  echo "Done."
  ;;
--check)
  errors=0
  while IFS= read -r -d '' file; do
    if ! "$CLANG_FORMAT" -n --Werror "$file" 2>/dev/null; then
      echo "Would reformat: $file"
      ((errors++))
    fi
  done < <(collect_files)
  if ((errors > 0)); then
    echo "$errors file(s) would be reformatted."
    exit 1
  fi
  echo "All files are correctly formatted."
  ;;
esac
