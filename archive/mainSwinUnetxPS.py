# Main script to run preprocessing, training or prediction tasks.
# Convert this to a notebook if you are using Jupyter.

# This is where you can change which config to use
# For running on differnt Modalities, adjust the modality in the config file.
import config.configSwinUnetxPS as configuration

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

    # TUNING
    #best = tuning.tune_SwinUNetPP(config)

    # TRAINING
    #for i in range(10):
    #    print(f" ========== Starting training iteration {i+1}/10 =========== ")
    #    training.train_SwinUNetPP(config)

    # EVALUATION
    evaluation.evaluate_SwinUNetPP(config)