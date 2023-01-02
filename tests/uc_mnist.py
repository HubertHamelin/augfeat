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
    Format the MNIST data as expected by the AugFeat lib (see README for more details)
    :return: None
    """
    train_images_path = '../../Experiments/train-images.idx3-ubyte'
    train_labels_path = '../../Experiments/train-labels.idx1-ubyte'
    train_images = idx2numpy.convert_from_file(train_images_path)
    train_labels = idx2numpy.convert_from_file(train_labels_path)

    os.mkdir('../../Experiments/train')
    for i in range(10):
        os.mkdir(f'../Experiments/train/{i}')

    j = 0
    for label, image in tqdm(zip(train_labels, train_images), total=len(train_labels)):
        np.save(f'../Experiments/train/{label}/{label}_{j}.npy', image)
        j += 1


autoencoder_config = {
    'latent_dim': 128,
    'dropout_rate': 0.2,
    'epochs': 200,
    'batch_size': 128,
    'learning_rate': 1e-3
}

# Results visualisation : 3x10 : 3 classes with 10 examples of each
base_path = '/Users/hubert/Documents/Lib project/Python project/Experiments'
dataset_path = base_path + '/train'
target_path = base_path + '/augmented_dataset'
augmentation_target = 10
balancer = Balancer(dataset_path, target_path, DataTypes.NUMPY)
classes = ['4', '5', '6']

for class_name in classes:
    with tf.device('cpu:0'):
        balancer.augment_class(class_name, augmentation_target, autoencoder_config)

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
