import os
import idx2numpy
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from balancer import Balancer
from config import DataTypes, AUTOENCODER_TRAINING_CONFIG_LIGHT, AUTOENCODER_TRAINING_CONFIG_MEDIUM

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


dataset_path = '../Experiments/train'
balancer = Balancer(dataset_path, DataTypes.NUMPY)
# balancer.dataset_interface.get_labels_distribution()
balancer.balance(510, AUTOENCODER_TRAINING_CONFIG_MEDIUM)
# balancer.dataset_interface.get_random_elem_from_dataset('9', 2, True)
balancer.dataset_interface.get_random_elem_from_augmented_data('9', 2, True)

# Test the reconstruction of original data using the trained autoencoder (visualize the evaluation in a grid).
