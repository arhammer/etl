#!/bin/bash

# Script to execute a command with each file from a list, wait after each execution, and sleep between jobs.
# Files are assumed to be in the same directory.

# Define the directory where the files are located
directory="/home/arhammer/Documents/VLTL/etl/etl_test_cases"  # Modify this to the actual directory path

# List of file names to process (without the directory path)
#file_list=("mse_reach" "mse_reach_avoid" "l1_reach" "l1_reach_avoid" "mse_avoid" "l1_avoid" "cd_reach" "cd_reach_avoid" "cd_avoid")  
file_list=("mse_reach_avoid" "l1_reach" "l1_reach_avoid" "mse_avoid" "l1_avoid" "cd_reach" "cd_reach_avoid" "cd_avoid")  

# file extension
file_ext="yaml"

# Loop over each file in the list
for file in "${file_list[@]}"; do
    # Combine the directory path with the file name
    full_path="$file.$file_ext"

    echo "Starting planning for $full_path..."
    
    # Run plan.py with the given  config file
    python plan.py --config-name "$full_path" |& tee $file.output
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Planning for $full_path completed successfully."
    else
        echo "Planning for $full_path failed. Exiting script."
        exit 1  # Exit the script if the command fails
    fi
    
    # Sleep for 15 seconds before processing the next file
    sleep 15
done

echo "All jobs completed successfully."
