#!/usr/bin/env bash
set -euo pipefail

# Base directory (defaults to current dir; change if needed)
BASE_DIR="${1:-.}"

# Find all suite2p/plane0 directories under BASE_DIR
find "$BASE_DIR" -type d -path "*/suite2p/plane0" | while read -r plane_dir; do
  # parent_dir is the "suite2p" folder
  parent_dir="$(dirname "$plane_dir")"
  dest_dir="$parent_dir/suite2p_cyto_final"

  # create the destination if it doesn't exist
  mkdir -p "$dest_dir"

  # list of files to copy
  for file in stat.npy iscell.npy ops.npy F.npy; do
    src="$plane_dir/$file"
    if [[ -f "$src" ]]; then
      cp "$src" "$dest_dir/"
    else
      echo "Warning: not found: $src" >&2
    fi
  done
done

echo "Done copying .npy files to suite2p_cyto_final folders."
