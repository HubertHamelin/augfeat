"""
light -> test exec
medium -> test fonctionnement
high -> test performance
"""

AUTOENCODER_TRAINING_CONFIG_LIGHT = {
    'latent_dim': 32,
    'dropout_rate': 0.2,
    'epochs': 3,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'reconstruction_limit_factor': 1
}

AUTOENCODER_TRAINING_CONFIG_MEDIUM = {
    'latent_dim': 128,
    'dropout_rate': 0.2,
    'epochs': 500,
    'batch_size': 64,
    'learning_rate': 1e-3,
    'reconstruction_limit_factor': 0.5
}

AUTOENCODER_TRAINING_CONFIG_HIGH = {
    'latent_dim': 256,
    'dropout_rate': 0.2,
    'epochs': 1000,
    'batch_size': 128,
    'learning_rate': 1e-3,
    'reconstruction_limit_factor': 0.25
}
