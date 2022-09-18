from enum import Enum


class DataTypes(Enum):
    PNG = '.png'
    NUMPY = '.npy'


"""
light -> test exec
medium -> test fonctionnement
high -> test performance
"""

AUTOENCODER_TRAINING_CONFIG_LIGHT = {
    'latent_dim': 32,
    'dropout_rate': 0.2,
    'epochs': 200,
    'batch_size': 128,
    'learning_rate': 1e-3
}

AUTOENCODER_TRAINING_CONFIG_MEDIUM = {
    'latent_dim': 32,
    'dropout_rate': 0.2,
    'epochs': 300,
    'batch_size': 128,
    'learning_rate': 1e-3
}

AUTOENCODER_TRAINING_CONFIG_HIGH = {
    'latent_dim': 64,
    'dropout_rate': 0.2,
    'epochs': 1000,
    'batch_size': 128,
    'learning_rate': 1e-3
}
