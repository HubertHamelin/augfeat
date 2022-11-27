import os
import idx2numpy
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from balancer import Balancer
from config import DataTypes, AUTOENCODER_TRAINING_CONFIG_LIGHT, AUTOENCODER_TRAINING_CONFIG_MEDIUM
import tensorflow as tf

train_images_path = '../Experiments/train-images.idx3-ubyte'
train_labels_path = '../Experiments/train-labels.idx1-ubyte'
train_images = idx2numpy.convert_from_file(train_images_path)
train_labels = idx2numpy.convert_from_file(train_labels_path)


# Example image
def example_image():
    plt.imshow(train_images[4], cmap=plt.cm.binary)
    plt.show()


# Format the data as expected by the AugFeat lib
def format_data():
    os.mkdir('../Experiments/train')
    for i in range(10):
        os.mkdir(f'../Experiments/train/{i}')

    j = 0
    for label, image in tqdm(zip(train_labels, train_images), total=len(train_labels)):
        np.save(f'../Experiments/train/{label}/{label}_{j}.npy', image)
        j += 1

"""
with tf.device('cpu:0'):
    # Balancer Class test
    dataset_path = '../Experiments/train'
    balancer = Balancer(dataset_path, DataTypes.NUMPY)
    balancer.augment_class('5', 10, AUTOENCODER_TRAINING_CONFIG_MEDIUM)
"""

# Results visualisation
class_name = '5'
augmented_path = f'../Experiments/augmented_dataset/{class_name}'
fig, axs = plt.subplots(3, 3)
for file_name, ax in zip(os.listdir(augmented_path), axs.flat):
    array = np.load(os.path.join(augmented_path, file_name))
    ax.imshow(array, cmap=plt.cm.binary)
plt.show()


# Test the reconstruction of original data using the trained autoencoder (visualize the evaluation in a grid).
"""
inputs = balancer.dataset['9']
test_autoencoder = AutoEncoder(inputs, balancer.timesteps, balancer.n_features, AUTOENCODER_TRAINING_CONFIG_LIGHT)
test_autoencoder.train()
test_autoencoder.evaluate()

for img in inputs[:1]:
    reconstructed = test_autoencoder.autoencoder.predict(np.array([test_autoencoder.scale(img)]))
    reconstructed = test_autoencoder.reverse_scale(reconstructed[0])

    img = balancer.dataset_interface.reverse_transform_numpy_data_to_numpy(img.squeeze())
    reconstructed = balancer.dataset_interface.reverse_transform_numpy_data_to_numpy(reconstructed.squeeze())

    plt.imshow(img, cmap=plt.cm.binary)
    plt.show()
    plt.imshow(reconstructed, cmap=plt.cm.binary)
    plt.show()
"""
