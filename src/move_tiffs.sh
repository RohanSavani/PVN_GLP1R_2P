#!/bin/bash

# Set the base directory
BASE_DIR="/Users/savani/Downloads/2p_data"

# Create the training_data directory if it doesn't exist
mkdir -p "$BASE_DIR/training_data"

# Find and copy each mean_image_enhanced.tif file
find "$BASE_DIR" -type f -path "*/suite2p/plane0/mean_image_enhanced.tif" | while read -r filepath; do
    # Extract parent folder name for unique naming
    parent_dir=$(basename "$(dirname "$(dirname "$(dirname "$filepath")")")")
    cp "$filepath" "$BASE_DIR/training_data/${parent_dir}_mean_image_enhanced.tif"
done

echo "All mean_image_enhanced.tif files have been copied to training_data."
