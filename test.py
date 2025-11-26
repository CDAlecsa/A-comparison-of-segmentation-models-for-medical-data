# Load modules
import os
from network_pipeline.training import model_inference



# Training of segmentation models
if __name__ == "__main__":

    experiments_path = os.path.abspath('experiments')

    experiments = [
                    name for name in os.listdir(experiments_path)
                        if os.path.isdir(os.path.join(experiments_path, name))
                  ]

    for config_filename in experiments:
        print('\n')
        print(40 * '..')
        print(f'EXPERIMENT: {config_filename}')
        model_inference(
                        config_filename = config_filename,
                        inference_dataloader = None,
                        data_type = 'test'
                    )

