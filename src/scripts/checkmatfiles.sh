#!/usr/bin/env bash
# check_behaviordata.sh
#
# Usage:
#   1. Make executable: chmod +x check_behaviordata.sh
#   2. Run: ./check_behaviordata.sh /path/to/parent_dir
#      If you omit the path, it defaults to the current directory.

# Parent directory containing your 30 folders (default: current dir)
# PARENT_DIR="${1:-.}"
PARENT_DIR="/Users/savani/Downloads/2p_data"
# Counters
found=0
missing=0

# Iterate over each item in PARENT_DIR
for folder in "$PARENT_DIR"/*; do
  # Only care about directories
  if [ -d "$folder" ]; then
    # Use basename to strip path, and drop any trailing slash
    folder_name=$(basename "$folder")
    # Construct the expected file path
    target="$folder/suite2p/plane0/behaviordata.mat"

    if [ -f "$target" ]; then
      echo "✔️  Found in: $folder_name"
      ((found++))
    else
      echo "❌  Missing in: $folder_name"
      ((missing++))
    fi
  fi
done

echo
echo "Summary:"
echo "  Present: $found"
echo "  Missing: $missing"