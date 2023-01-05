"""
In this file I show you an example of how to use the augfeat library with the very well known MNIST dataset.
It takes some classes from the dataset, generates augmented data and plot the results for you to see.
Test with: python mnist.py
"""

import os
import idx2numpy
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import tensorflow as tf

from augfeat.balancer import Balancer
from augfeat.custom_types import DataTypes


def format_data() -> None:
    """
    Format the MNIST data as expected by the AugFeat lib (see README for more details).
    :return: None
    """
    train_images_path = 'mnist-train-images.idx3-ubyte'
    train_labels_path = 'mnist-train-labels.idx1-ubyte'
    train_images = idx2numpy.convert_from_file(train_images_path)
    train_labels = idx2numpy.convert_from_file(train_labels_path)

    os.mkdir('')
    for i in range(10):
        os.mkdir(f'./mnist/{i}')

    j = 0
    for label, image in tqdm(zip(train_labels, train_images), total=len(train_labels)):
        np.save(f'./mnist/{label}/{label}_{j}.npy', image)
        j += 1


# The configuration defines the augmentation complexity and capability. See README for more details.
autoencoder_config = {
    'latent_dim': 128,
    'dropout_rate': 0.2,
    'epochs': 200,
    'batch_size': 128,
    'learning_rate': 1e-3,
    'reconstruction_limit_factor': 0.25
}

# Format the mnist dataset as expected by the augfeat library (just the first time).
if not os.path.isdir(''):
    format_data()

# Everything you need to set up: source and target paths, a volume target, that's all !
base_path = '..'
dataset_path = base_path + '/mnist'
target_path = base_path + '/mnist_augmented_dataset'
augmentation_target = 10
balancer = Balancer(dataset_path, target_path, DataTypes.NUMPY)
classes = ['7', '8', '9']

# Augment the data for each test class.
for class_name in classes:
    with tf.device('cpu:0'):
        balancer.augment_class(class_name, augmentation_target, autoencoder_config)

# Results visualisation : 3x10 : 3 classes with 10 examples of each.
fig, axs = plt.subplots(3, 10)
cursor = 0
for class_name in classes:
    augmented_path = target_path + f'/{class_name}'
    for file_name, ax in zip(os.listdir(augmented_path), axs.flat[cursor:]):
        array = np.load(os.path.join(augmented_path, file_name))
        ax.imshow(array, cmap=plt.cm.binary)
        ax.axis('off')
    cursor += len(os.listdir(augmented_path))

plt.show()
