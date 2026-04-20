#Python file to validate configuration

import config.configUnetxAE as configuration

config = configuration.Configuration().validate()
print("Configuration loaded:", config.modality)