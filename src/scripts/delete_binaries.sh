#!/usr/bin/env bash
#
# delete_data_bin.sh
#
# Recursively find, list, and delete all files named "data.bin"

set -euo pipefail

echo "Scanning for files named 'data.bin' in $(pwd)..."

# Grab the list of matches
files=$(find . -type f -name 'data.bin')

# If none found, exit
if [ -z "$files" ]; then
  echo "No files named 'data.bin' found."
  exit 0
fi

# Print what will be deleted
echo "The following 'data.bin' files will be deleted:"
printf '%s\n' "$files"

# Now delete them
echo "Deleting files..."
printf '%s\n' "$files" | while IFS= read -r f; do
  rm -v "$f"
done

echo "Done."
