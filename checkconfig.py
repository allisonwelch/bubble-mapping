#Python file to validate configuration

import config.configSwinUnet as configuration

config = configuration.Configuration().validate()
print("Configuration loaded:", config.modality)