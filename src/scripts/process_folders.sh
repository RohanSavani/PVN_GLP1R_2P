#!/bin/bash

# Loop through all folders in the current directory
for exp_dir in */ ; do
    # Define the path to the plane0 directory
    plane0_dir="${exp_dir}suite2p/plane0"
    
    # Check if plane0 directory exists
    if [ -d "$plane0_dir" ]; then
        # Define the destination directory
        manual_dir="${exp_dir}suite2p_manual"
        
        # Create the destination directory if it doesn't exist
        mkdir -p "$manual_dir"
        
        # Move specific files
        mv "$plane0_dir"/iscell.npy "$manual_dir"/
        mv "$plane0_dir"/F.npy "$manual_dir"/
        mv "$plane0_dir"/Fneu.npy "$manual_dir"/
        # mv "$plane0_dir"/mean_image_with_filled_rois.png "$manual_dir"/
        # mv "$plane0_dir"/mean_image_enhanced_with_filled_rois.png "$manual_dir"/
        
        # Copy specific files
        cp "$plane0_dir"/stat.npy "$manual_dir"/
        cp "$plane0_dir"/stat_orig.npy.npy "$manual_dir"/
        cp "$plane0_dir"/ops.npy "$manual_dir"/
        
        echo "Processed $exp_dir"
    else
        echo "No suite2p/plane0 directory in $exp_dir, skipping."
    fi
done

 
