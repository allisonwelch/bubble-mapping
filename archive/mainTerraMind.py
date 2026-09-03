# Main script to run preprocessing, training or prediction tasks.
# Convert this to a notebook if you are using jupyter.

# For running on differnt Modalities, adjust the modality in the config file.

# This is where you can change which config to use
import config.configTerraMindxS2 as configuration

# INIT
config = configuration.Configuration().validate()
import preprocessing
import training
import tuning
import evaluation 


if __name__ == "__main__":

    # PREPROCESSING
    #preprocessing.preprocess_all(config)

    # TUNING (HYPERPARAMETER SEARCH)
    #best = tuning.tune_TerraMind(config)

    # TRAINING
    #for i in range(10):

    #    print(f"=========== Starting training iteration {i+1}/10 ============")
    #    training.train_TerraMind(config)

    # EVALUATION
    evaluation.evaluate_TerraMind(config)
    





