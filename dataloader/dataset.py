# Load modules
import os, cv2, torch
import numpy as np

from pathlib import Path
from functools import partial
from collections import Counter

from typing import (Optional, Union, Tuple, Dict, Callable)
from torch.utils.data import (Subset, Dataset, DataLoader)

from torchvision import transforms
from utils.utils import (worker_init_fn, logger)
from .augmentations import DataAugmenter



class MedicalSegmentationDataset(Dataset):
    def __init__(self, 
                 input_folder: str, 
                 background_value: Optional[int],
                 transform: Callable = None,
                 verbose: bool = True
        ) -> None:
        """
            Args:
                input_folder (str): Path to the directory containing data subfolders.
                background (Optional[int]): The pixe value corresponding to the background.
                    If it is `None`, then we will choose the label with the highest count as background.
                transform (callable, optional): Optional transform to be applied on a sample.
                verbose (bool): Option for printing with global logger.
        """

        self.input_folder = input_folder
        self.transform = transform
        self.background_value = background_value
        self.verbose = verbose

        self.data = []
        self._scan_directories()
        
        self.label_encoder = None
        self.apply_lumen_encoding()


    # Ensure background (usually 0) is first
    # We will need `background = 0` in the function `validate_model` belonging to `network_pipeline/inference.py`
    def apply_lumen_encoding(self) -> None:
        
        # Gather label counts
        label_counts = {}

        for item in self.data:
            lumen = cv2.imread(item['lumen_path'], cv2.IMREAD_GRAYSCALE)
            unique, counts = np.unique(lumen, return_counts = True)
            for (val, count) in zip(unique, counts):
                label_counts[val] = label_counts.get(val, 0) + count

        # If background value is not given, then choose the label with the highest count as background
        if self.background_value is None:
            sorted_labels = sorted(label_counts.items(), key = lambda x: x[1], reverse = True)
            background_label = sorted_labels[0][0]
        else:
            background_label = self.background_value

        # Reorder labels
        all_labels = sorted(label_counts.keys())
        all_labels.remove(background_label)
        all_labels = [background_label] + all_labels

        # Make label encoding: (k, v) = (original, encoded) 
        self.label_encoder = {label: idx for (idx, label) in enumerate(all_labels)}
        assert self.label_encoder[background_label] == 0, "Background must be mapped to class 0"

        if self.verbose:
            logger.info(f'The background value is set to `{background_label}`.')
            logger.info(f'The `lumen` labels are `{all_labels}`.')
            logger.info(f'The label encoding is the following `{self.label_encoder}`.\n')


    def _scan_directories(self):
        
        # Walk through each patient folder
        for patient_folder in os.listdir(self.input_folder):
            
            # Retrieve subfolders corresponding to different patients
            patient_path = os.path.join(self.input_folder, patient_folder)

            if not os.path.isdir(patient_path):
                continue
            
            # Find the corresponding subfolders
            raw_folder = os.path.join(patient_path, 'png_raw')
            lumen_folder = os.path.join(patient_path, 'png_lumen')
            plaque_folder = os.path.join(patient_path, 'png_plaque')

            # Get all filenames in the raw folder
            filenames = os.listdir(raw_folder)

            # Loop over the filenames belonging to the current subfolder
            for raw_img_filename in filenames:
                
                if raw_img_filename.endswith('.png'):
                   
                    lumen_filename = raw_img_filename.replace('raw', 'lumen')
                    plaque_filename = raw_img_filename.replace('raw', 'plaque')

                    raw_path = os.path.join(raw_folder, raw_img_filename)
                    lumen_path = os.path.join(lumen_folder, lumen_filename)
                    plaque_path = os.path.join(plaque_folder, plaque_filename)

                    self.data.append({
                                        'raw_path': raw_path,
                                        'lumen_path': lumen_path,
                                        'plaque_path': plaque_path,
                                        'case_name': Path(raw_img_filename.replace('raw', '')).stem
                                    })


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):

        item = self.data[idx]

        # Load images
        raw_image = cv2.imread(item['raw_path'], cv2.IMREAD_UNCHANGED)
        lumen = cv2.imread(item['lumen_path'], cv2.IMREAD_GRAYSCALE)
        plaque = cv2.imread(item['plaque_path'], cv2.IMREAD_GRAYSCALE)

        # Normalize image
        raw_image = raw_image.astype(np.float32) / 255.0

        # Reshape image
        if len(raw_image.shape) == 2:
            raw_image = np.expand_dims(raw_image, axis = 0)                                     # shape: (1, H, W)
        else:
            raw_image = np.transpose(raw_image, (2, 0, 1))                                      # shape: (1, H, W)

        # Get encoding function
        label_map_func = np.vectorize(self.label_encoder.get)

        # Make `lumen` binary encoding
        lumen_flatten = lumen.flatten()                                                         # shape: (H * W, )
        lumen_encoded = label_map_func(lumen_flatten)                                           # shape: (H * W, )
        lumen_encoded = lumen_encoded.reshape(lumen.shape).astype(np.uint8)                     # shape: (H, W)

        # Make `plaque` binary encoding
        plaque_flatten = plaque.flatten()                                                       # shape: (H * W, )
        plaque_encoded = label_map_func(plaque_flatten)                                         # shape: (H * W, )
        plaque_encoded = plaque_encoded.reshape(plaque.shape).astype(np.uint8)                  # shape: (H, W)

        # Convert to torch tensors
        raw_image_tensor = torch.from_numpy(raw_image).float()                                  # shape: (1, H, W)
        lumen_tensor = torch.from_numpy(lumen_encoded).long()                                   # shape: (H, W)
        plaque_tensor = torch.from_numpy(plaque_encoded).long()                                 # shape: (H, W)

        sample = {'raw_image': raw_image_tensor, 
                  'lumen': lumen_tensor,
                  'plaque': plaque_tensor, 
                  'case_name': item['case_name']}

        if self.transform:
            sample = self.transform(sample)

        return sample



def get_base_dataset(dataset: Union[Dataset, Subset]) -> Dataset:
    """
        Recursively retrieves the base PyTorch `Dataset` from a potentially nested `torch.utils.data.Subset`.

        This is used when needing to access attributes of the original dataset
            that are not directly available through a Subset wrapper.

        Args:
            dataset (Union[Dataset, Subset]): A PyTorch `Dataset` or `Subset` instance.

        Returns:
            Dataset: The underlying base dataset.
    """
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset



def create_train_val_dataloaders(data_config: Dict,
                                 train_aug_config: Dict
            ) -> Tuple[DataLoader]:
    """
        Creates the `train` & `validation` DataLoaders for the `MedicalSegmentationDataset`.

        Args:
            data_config (Dict): Dictionary containing the dataset configuration.
            train_aug_config (Optional[Dict]): Augmentation configuration to be passed to DataAugmenter.

        Returns:
            dataloaders (Tuple[DataLoader]): The train & val PyTorch DataLoaders.
    """

    # Define the train + val path from which we will load the data
    data_path = os.path.join(data_config['data_path'], 'train_val')

    # Define worker reproducibility
    if data_config['random_state'] is None:
        data_worker_init = None
    else:
        data_worker_init = partial(worker_init_fn, base_seed = data_config['random_state'])

    # Define the background encoding value
    background_value = data_config['background_value']

    # Define the augmentation transforms which will be used for the training dataset
    train_transform = transforms.Compose([DataAugmenter(train_aug_config)])
    
    # Retrieve train/val splitting indices (which will be eventually used for all the models)
    logger.info(f'Loading `train indices` ...')
    train_file = os.path.join(data_config['train_val_indices_path'], 'train_indices.npy')
    train_indices = np.load(train_file)

    logger.info(f'Loading `val indices` ...')
    val_file = os.path.join(data_config['train_val_indices_path'], 'val_indices.npy')
    val_indices = np.load(val_file)
    
    # Slice-level check
    assert set(train_indices).isdisjoint(set(val_indices)), "Train & validation indices overlap."
    print("\n")
    
    # Create the corresponding train & val datasets
    logger.info(f'TRAIN Dataset ...')
    train_dataset = MedicalSegmentationDataset(input_folder = data_path,
                                               background_value = background_value, 
                                               transform = train_transform)
    
    logger.info(f'VAL Dataset ...')
    val_dataset = MedicalSegmentationDataset(input_folder = data_path,
                                             background_value = background_value, 
                                             transform = None)
    
    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)

    # Patient-level check
    def get_patient_id(dataset, idx):
        item = dataset.dataset.data[dataset.indices[idx]]
        return os.path.basename(os.path.dirname(os.path.dirname(item['raw_path'])))

    train_patient_ids = {get_patient_id(train_dataset, i) for i in range(len(train_dataset))}
    val_patient_ids = {get_patient_id(val_dataset, i) for i in range(len(val_dataset))}

    overlap = train_patient_ids.intersection(val_patient_ids)
    assert not overlap, f"Patient-level overlap found: {overlap}"

    # Per-patient slice counts
    train_patient_counts = Counter(get_patient_id(train_dataset, i) for i in range(len(train_dataset)))
    val_patient_counts = Counter(get_patient_id(val_dataset, i) for i in range(len(val_dataset)))

    # Prints
    print(f"\nTrain slices: {len(train_dataset)}")
    print(f"Val slices:   {len(val_dataset)}")

    print(f"\nTrain patients ({len(train_patient_ids)}): {train_patient_ids}")
    print(f"Val patients   ({len(val_patient_ids)}): {val_patient_ids}")

    # Create the underlying train & val dataloaders
    train_dataloader = DataLoader(train_dataset,
                                  shuffle = True,
                                  pin_memory = True,
                                  batch_size = data_config['batch_size'],
                                  num_workers = data_config.get('num_workers', 4),
                                  worker_init_fn = data_worker_init
                        )
    
    val_dataloader = DataLoader(val_dataset,
                                shuffle = False,
                                batch_size = 1,
                                num_workers = 1,
                                worker_init_fn = data_worker_init
                        )
    
    # Return the results
    split_ids = {
                    "train": {
                                "patient_ids": list(train_patient_ids),
                                "slice_counts": dict(train_patient_counts)
                    },
                    "val": {
                                "patient_ids": list(val_patient_ids),
                                "slice_counts": dict(val_patient_counts)
                    }
                }  
    
    return train_dataloader, val_dataloader, split_ids



def create_test_dataloader(data_config: Dict) -> DataLoader:
    """
        Creates the `test` DataLoader for the `MedicalSegmentationDataset`.

        Args:
            data_config (Dict): Dictionary containing the dataset configuration.
            
        Returns:
            dataloader (DataLoader): The test PyTorch DataLoaders.
    """

    # Define the test path from which we will load the data
    data_path = os.path.join(data_config['data_path'], 'test')

    # Define worker reproducibility
    if data_config['random_state'] is None:
        data_worker_init = None
    else:
        data_worker_init = partial(worker_init_fn, base_seed = data_config['random_state'])

    # Define the background encoding value
    background_value = data_config['background_value']

    # Create the test dataset
    logger.info(f'TEST Dataset ...')
    logger.info(f'Data path: [{data_path}]')
    
    test_dataset = MedicalSegmentationDataset(input_folder = data_path,
                                              background_value = background_value, 
                                              transform = None)

    # Create the underlying test dataloaders    
    test_dataloader = DataLoader(test_dataset,
                                 shuffle = False,
                                 batch_size = 1,
                                 num_workers = 1,
                                 worker_init_fn = data_worker_init
                        )
    
    return test_dataloader
