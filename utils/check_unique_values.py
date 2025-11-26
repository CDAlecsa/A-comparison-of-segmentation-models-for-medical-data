# Load modules
import os
import numpy as np
from PIL import Image


# Run the script from the root folder "main_codes" using the -m flag: "python -m utils.check_unique_values"
if __name__ == '__main__':

    root_dir = "./Vascular_pathologies_data_cropped_128_128/" 
    root_dir = os.path.abspath(root_dir)

    target_folders = {"png_lumen", "png_mask", "png_mdc", "png_raw", "png_plaque"}
    unique_values = {folder: set() for folder in target_folders}

    for split in ["train_val", "test"]:
        split_path = os.path.join(root_dir, split)
        if not os.path.isdir(split_path):
            continue

        for patient in os.listdir(split_path):
            patient_path = os.path.join(split_path, patient)
            if not os.path.isdir(patient_path):
                continue

            for folder in target_folders:
                folder_path = os.path.join(patient_path, folder)
                if not os.path.isdir(folder_path):
                    continue

                for fname in os.listdir(folder_path):
                    fpath = os.path.join(folder_path, fname)

                    # skip non-image files
                    if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp")):
                        continue

                    try:
                        img = np.array(Image.open(fpath))
                        unique_values[folder].update(np.unique(img))
                    except Exception as e:
                        print(f"Error reading {fpath}: {e}")

    
    print("\nUnique pixel values per folder type...\n")
    for folder, vals in unique_values.items():
        vals_sorted = sorted(vals)
        print(f"{folder}:")
        print(f"Number of unique pixel values: {len(vals_sorted)}")
        print(f"Unique pixel values: {vals_sorted}\n")