# Load modules
import os, argparse
from utils.utils import (visualize_lumen_mask_plaque, 
                         load_image_as_tensor)  



# Run the script from the root folder "main_codes" using the -m flag: "python -m utils.visualize_targets"
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = "Visualize targets")
    parser.add_argument("--data-type", type = str, required = True, choices = ["train_val", "test"], 
                        help = "The type of data, i.e., `train_val` or `test`.")
    parser.add_argument("--patient-name", type = str, required = True, help = "Name of the patient")
    parser.add_argument("--patient-id", type = str, required = True, help = "ID of the patient")
    args = parser.parse_args()

    # Select identifier    
    patient_base = args.patient_id.rsplit('_', 1)[0]  
    patient_number = args.patient_id.rsplit('_', 1)[1]

    # Paths
    base_data_path = "./Vascular_pathologies_data_cropped_128_128/" 
    base_data_path = os.path.abspath(base_data_path)

    data_path = os.path.join(base_data_path, f'{args.data_type}\{args.patient_name}')
    save_dir = os.path.abspath('./targets_visualization')
    os.makedirs(save_dir, exist_ok = True)

    # Ground-truth visualization
    lumen_path = os.path.join(data_path, 'png_lumen', f'{patient_base}_lumen_{patient_number}.png')
    mask_path = os.path.join(data_path, 'png_mask', f'{patient_base}_mask_{patient_number}.png')
    plaque_path = os.path.join(data_path, 'png_plaque', f'{patient_base}_plaque_{patient_number}.png')

    lumen = load_image_as_tensor(lumen_path)
    mask = load_image_as_tensor(mask_path)
    plaque = load_image_as_tensor(plaque_path)

    save_path = os.path.join(save_dir, f'post_processed_targets---{args.patient_id}.png')
    visualize_lumen_mask_plaque(lumen = lumen,
                                mask = mask,
                                plaque = plaque,
                                padding = 5,
                                pad_value = 0.5,
                                save_path = save_path)
