
import os
import shutil

# Define the paths to directories A and B
dir_A = "/efs/benchmark/high_resolution_validation_data_ble_backup3"
dir_B = "/efs/fim-data/hand_fim/test_cases/ble_test_cases/validation_data_ble"

# Iterate over the subdirectories in directory A
for mag in ['100yr', '500yr']:
    for subdir_A_name in os.listdir(dir_A):
        subdir_A_path = os.path.join(dir_A, subdir_A_name)
        
        # Check if it is indeed a directory
        if os.path.isdir(subdir_A_path):
            # Construct the file name based on the subdirectory name
            file_name = f"ble_huc_{subdir_A_name}_flows_{mag}.csv"

            # Construct the source and destination file paths
            src_file_path = os.path.join(dir_B, subdir_A_name, mag, file_name)
            dest_file_path = os.path.join(dir_A, subdir_A_name, mag, file_name)

            # Copy the file from B to A if it exists in B
            if os.path.exists(src_file_path):
                shutil.copy2(src_file_path, dest_file_path)
                print(f"Copied: {file_name} to {subdir_A_path}")
            else:
                print(f"File not found in source: {src_file_path}")
