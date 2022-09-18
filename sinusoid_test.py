import os
import numpy as np
from balancer import Balancer
from autoencoder import AutoEncoder
from matplotlib import pyplot as plt
from config import DataTypes, AUTOENCODER_TRAINING_CONFIG_LIGHT, AUTOENCODER_TRAINING_CONFIG_MEDIUM


# Format the data as expected by the library
def format_data():
    time = np.arange(0, 10, 0.1)
    amplitude = np.sin(time)
    os.mkdir('./test')
    os.mkdir('./test/sinusoid')
    np.save('./test/sinusoid/1.npy', amplitude)
    np.save('./test/sinusoid/2.npy', amplitude + 0.5)


dataset_path = './test'
balancer = Balancer(dataset_path, DataTypes.NUMPY)

# balancer.balance(4, AUTOENCODER_TRAINING_CONFIG_MEDIUM)
inputs = balancer.dataset['sinusoid']
test_autoencoder = AutoEncoder(inputs, balancer.n_features, AUTOENCODER_TRAINING_CONFIG_LIGHT)
test_autoencoder.train()
test_autoencoder.evaluate()

# plot original and reconstructed vectors
original = inputs[1]
reconstructed = test_autoencoder.autoencoder.predict(np.array([original]))
original = original.squeeze()
reconstructed = reconstructed.squeeze()
x_axis = np.arange(original.shape[0])
plt.plot(x_axis, original, c='blue')
plt.plot(x_axis, reconstructed, c='red')
plt.show()

# balancer.dataset_interface.get_random_elem_from_dataset('sinusoid', 1)
# balancer.dataset_interface.get_random_elem_from_augmented_data('sinusoid', 1)
