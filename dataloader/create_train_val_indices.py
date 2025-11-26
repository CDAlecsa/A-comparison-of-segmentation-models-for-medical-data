# Load modules
import os, cv2, torch
import numpy as np
import pandas as pd

from collections import Counter
from skmultilearn.model_selection import iterative_train_test_split

from .dataset import MedicalSegmentationDataset
from utils.utils import post_process_targets



# Run the script from the root folder "main_codes" using the -m flag: "python -m dataloader.create_train_val_indices"
if __name__ == '__main__':

    # Define parameters
    val_size = 0.2
    background_value = 0

    script_path = os.path.dirname(os.path.abspath(__file__))

    data_path = "../Vascular_pathologies_data_cropped_128_128"
    data_path = os.path.join(script_path, data_path)
    data_path = os.path.abspath(data_path)
    data_path = os.path.join(data_path, 'train_val')
    print(f'\nDATA PATH: [{data_path}]\n')
    print(f'val_size = {val_size}\n') 

    # Collect patient folders
    patient_folders = [p for p in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, p))]
    print(f"Found {len(patient_folders)} patients\n")

    # Create the dataset correspoding to [train + val]
    full_dataset = MedicalSegmentationDataset(input_folder = data_path,
                                              background_value = background_value, 
                                              transform = None,
                                              verbose = False)
    
    # Compute patient-level features for multi-stratification
    patient_features = {}

    for patient_id in patient_folders:
        slices = [item 
                    for item in full_dataset.data 
                        if os.path.basename(os.path.dirname(os.path.dirname(item['raw_path']))) == patient_id]

        num_slices = len(slices)
        post_lumen_slices = []

        for item in slices:

            # Load lumen and plaque masks
            lumen = cv2.imread(item['lumen_path'], cv2.IMREAD_GRAYSCALE)
            plaque = cv2.imread(item['plaque_path'], cv2.IMREAD_GRAYSCALE)
            
            # Convert to torch tensors
            lumen_tensor = torch.from_numpy(lumen).long()
            plaque_tensor = torch.from_numpy(plaque).long()
            
            # Post-process lumen (lumen minus plaque)
            lumen_post = post_process_targets(lumen = lumen_tensor, plaque = plaque_tensor)  # (H, W)
            post_lumen_slices.append(lumen_post)

        # Compute proportion of foreground pixels on post-processed lumen
        all_slices = torch.stack(post_lumen_slices)                                          # (N, H, W)
        prop_foreground = torch.mean(all_slices.float()).item()                              # scalar
        patient_features[patient_id] = (num_slices, prop_foreground)

    # Create DataFrame for patients
    patient_df = pd.DataFrame.from_dict(patient_features, 
                                        orient = 'index',
                                        columns = ['num_slices', 'prop_foreground'])

    # Create multi-label matrix for iterative stratification
    patient_df['high_slices'] = (patient_df['num_slices'] >= patient_df['num_slices'].median()).astype(int)
    patient_df['high_foreground'] = (patient_df['prop_foreground'] >= patient_df['prop_foreground'].median()).astype(int)

    print(patient_df, '\n\n')    
    
    # Split patients with stratification
    all_patients = list(patient_df.index)

    X = np.array(all_patients).reshape(-1, 1)
    Y = patient_df[['high_slices', 'high_foreground']].values

    X_train, Y_train, X_val, Y_val = iterative_train_test_split(X, Y, test_size = val_size)

    train_patients = [i[0] for i in X_train]
    val_patients = [i[0] for i in X_val]

    print("X_train:", X_train)
    print("X_val:", X_val)
    
    print("TRAIN patients:", train_patients, '\n\n')
    print("VAL patients:", val_patients, '\n\n')
    print("Missing patients: ", patient_df.loc[~patient_df.index.isin(np.array(train_patients + val_patients))])
    
    # Map patients to dataset indices
    train_indices, val_indices = [], []

    for (idx, item) in enumerate(full_dataset.data):
        patient_id = os.path.basename(os.path.dirname(os.path.dirname(item['raw_path'])))

        if patient_id in train_patients:
            train_indices.append(idx)
        elif patient_id in val_patients:
            val_indices.append(idx)
        else:
            raise ValueError(f"Patient [{patient_id}] must be either in `train` or in `validation`.")

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)

    # Slice-level check
    assert set(train_indices).isdisjoint(set(val_indices)), "Train & validation indices overlap."

    # Patient-level check
    train_patient_ids = {os.path.basename(os.path.dirname(os.path.dirname(full_dataset.data[i]['raw_path']))) for i in train_indices}
    val_patient_ids = {os.path.basename(os.path.dirname(os.path.dirname(full_dataset.data[i]['raw_path']))) for i in val_indices}
    assert train_patient_ids.isdisjoint(val_patient_ids), "Patient leakage detected between train and val!"

    # Prints
    print(f"\nTotal train slices: {len(train_indices)}")
    print(f"Total val slices:   {len(val_indices)}")

    print(f"True `val_size` ratio:   {len(val_indices) / (len(train_indices) + len(val_indices))}")

    print("\nTrain slice counts per patient:")
    print(Counter([os.path.basename(os.path.dirname(os.path.dirname(full_dataset.data[i]['raw_path']))) for i in train_indices]))

    print("\nVal slice counts per patient:")
    print(Counter([os.path.basename(os.path.dirname(os.path.dirname(full_dataset.data[i]['raw_path']))) for i in val_indices]))


    # Save train & validation indices for splitting
    save_path = '../dataloader'
    save_path = os.path.join(script_path, save_path)
    save_path = os.path.abspath(save_path)

    np.save( os.path.join(save_path, 'train_indices.npy'), train_indices )
    np.save( os.path.join(save_path, 'val_indices.npy'), val_indices )

