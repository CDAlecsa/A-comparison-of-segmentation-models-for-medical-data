# Load modules
from utils.utils import visualize_samples
from dataloader.dataset import create_train_val_dataloaders



# Run the script from the root folder "main_codes" using the -m flag: "python -m pipeline_validation.validate_dataloader"
if __name__ == '__main__':

    data_config = {
                    'data_path': './Vascular_pathologies_data_cropped_128_128',
                    'batch_size': 8,
                    'num_workers': 4,
                    'background_value': 0,
                    'val_size': 0.2,
                    'random_state': None
                }
    
    data_path = data_config['data_path']

    # Example augmentation config, or None for no augmentation
    aug_config = {
                    'gaussian_blur': {'prob': 0.5, 'sigma': [0.2, 0.75]},
                    'brightness': {'prob': 0.4, 'range': [0.8, 1.2]},
                    'mirror': {'prob': 0.5},
                    'affine': {'prob': 1.0, 'rotate': 20, 'translate': 0.2}
                 }

    train_dataloader, val_dataloader, _ = create_train_val_dataloaders(data_config = data_config,
                                                                       train_aug_config = aug_config)
    
    visualize_samples(train_dataloader, 'train', num_samples = 8)
    visualize_samples(val_dataloader, 'val', num_samples = 8)
