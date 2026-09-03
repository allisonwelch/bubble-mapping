# Main script to run preprocessing, training or prediction tasks.
# Convert this to a notebook if you are using Jupyter.

# This is where you can change which config to use
from anyio import sleep
import config.configUnetxPS2 as configuration

# For running on differnt Modalities, adjust the modality in the config file.

# INIT
config = configuration.Configuration().validate()

# Project modules
import preprocessing
import tuning
import training
import evaluation

# USE PYTORCH ENV FOR THIS PROJECT

if __name__ == "__main__":
    
    # PREPROCESSING
    #preprocessing.preprocess_all(config)

    # HYPERPARAMETER TUNING
    #best = tuning.tune_UNet(config)

    # TRAINING
    for i in range(5):

       print(f"\n =========== Starting training iteration {i+1}/5 ===========\n")
       training.train_UNet(config)
    # EVLAUATION
    #evaluation.evaluate_unet(config)
