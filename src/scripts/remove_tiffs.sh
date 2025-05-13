#!/usr/bin/env bash
#
# list_and_remove_tiffs.sh
#
# For each immediate subdirectory:
#   1. List all *.tiff files (no recursion).
#   2. Remove those files.

# Enable nullglob so that *.tiff expands to an empty list if there are no matches
shopt -s nullglob

for dir in */; do
  # Only proceed if it really is a directory
  [ -d "$dir" ] || continue

  echo "Processing directory: $dir"

  # Gather all .tiff files in this directory (maxdepth = 1)
  tiff_files=( "$dir"/*.tif )

  if [ ${#tiff_files[@]} -gt 0 ]; then
    echo "  Found the following .tif files:"
    for file in "${tiff_files[@]}"; do
      echo "    $file"
    done

    # Now remove them (the -v flag makes rm verbose)
    rm -v "${tiff_files[@]}"
  else
    echo "  No .tiff files found."
  fi
done
