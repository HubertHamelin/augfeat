import os
import numpy as np
from src.augfeat.balancer import Balancer
from src.augfeat.autoencoder import AutoEncoder
from matplotlib import pyplot as plt
from src.augfeat.config import DataTypes, AUTOENCODER_TRAINING_CONFIG_LIGHT


# Format the data as expected by the library
def format_data():
    time = np.arange(0, 10, 0.1)
    amplitude = np.sin(time)
    os.mkdir('../test')
    os.mkdir('sinusoid')
    np.save('sinusoid/1.npy', amplitude)
    np.save('sinusoid/2.npy', amplitude + 1)


dataset_path = '../test'
balancer = Balancer(dataset_path, DataTypes.NUMPY)

# augfeat.balance(4, AUTOENCODER_TRAINING_CONFIG_MEDIUM)
inputs = balancer.dataset['sinusoid']
test_autoencoder = AutoEncoder(inputs, balancer.timesteps, balancer.n_features, AUTOENCODER_TRAINING_CONFIG_LIGHT)
test_autoencoder.train()
test_autoencoder.evaluate()

# plot original and reconstructed vectors
for original in inputs:
    reconstructed = test_autoencoder.autoencoder.predict(np.array([test_autoencoder.scale(original)]))
    reconstructed = test_autoencoder.reverse_scale(reconstructed[0])
    original = original.squeeze()
    reconstructed = reconstructed.squeeze()
    x_axis = np.arange(original.shape[0])
    plt.plot(x_axis, original, c='blue')
    plt.plot(x_axis, reconstructed, c='red')
    plt.show()

# augfeat.dataset_interface.get_random_elem_from_dataset('sinusoid', 1)
# augfeat.dataset_interface.get_random_elem_from_augmented_data('sinusoid', 1)
