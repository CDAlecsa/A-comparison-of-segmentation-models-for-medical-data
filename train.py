# Load modules
import argparse, warnings
warnings.filterwarnings("ignore")

from network_pipeline.training import model_training



# Training of segmentation models
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', 
                        type = str,
                        help = 'the file name containing the configuration settings')    
    args = parser.parse_args()

    model_training(config_filename = args.config_file)

