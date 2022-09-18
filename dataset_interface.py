import os
import numpy as np
from matplotlib import pyplot as plt
from config import DataTypes

# TODO: incorporate the most classical formats from Kaggle competitions
# TODO: infer data type by default if not specified


class DatasetInterface:

    def __init__(self, path: str, data_type: DataTypes):
        self.path = path
        self.labels_mapper = {}
        self.og_dims = []  # list the original dimensions of all elements in the dataset.
        self.og_dim = None  # if all original dims are identical, dimension is saved here.
        self.dataset = {}
        self.og_data_type = data_type
        self.augmented_data = None
        self.augmented_dataset = None
        self.n_features = None

    # Depending on dataset size, or user choice, persist a copy of npy dataset (progressive) rather than a full load
    # in memory => performance optimisation.
    # 1st use case: MNIST in memory. Expected format is one directory per class
    def load_dataset(self):

        n_features = []

        # TODO: load npy arrays in memory when augment/downsize is called
        for class_ in os.listdir(self.path):

            inputs = []

            # For each class, map the actual label with its numerical counterpart
            label = len(self.labels_mapper.keys())
            self.labels_mapper[label] = class_
            class_path = os.path.join(self.path, class_)

            if not os.path.isdir(class_path):
                continue

            for element in os.listdir(class_path):
                element_path = os.path.join(class_path, element)
                input_ = self.transform_numpy_data_to_numpy(element_path)
                n_features.append(len(input_))
                input_ = input_.reshape(1, (len(input_)))
                inputs.append(input_)

            # TODO: manage inputs per classes directly.
            inputs = np.asarray(inputs)
            print(f'{class_} inputs dim: {inputs.shape}')
            self.dataset[class_] = inputs[:500]

        self.n_features = n_features[0]
        self.og_dim = self.get_inputs_original_dimension()
        print(f'original dimensions: {self.og_dim}')
        print(f'n_features: {self.n_features}')

    def get_inputs_original_dimension(self):
        # TODO: does og_dims really needs to be a class attribute ?
        common_dim = self.og_dims[0]
        for dim in self.og_dims[1:]:
            if dim != common_dim:
                raise Exception(f'All elements in the dataset do not have identical dimensions: {dim} != {common_dim}')
        return common_dim

    # In case the user just want to test the library fast, he has a solution to test with a raw dataset
    # without doing the full preprocessing of his own.
    def default_preprocessing_for_images(self):
        pass

    # 1er cas simple: formattage des données MNIST vers matrices npy
    # NUMPY ARRAY => NUMPY VECTOR
    def transform_numpy_data_to_numpy(self, path: str):
        # TODO: manage loading exceptions, and same type exceptions (between all elements)
        input_ = np.load(path)
        self.og_dims.append(input_.shape)
        input_ = input_.flatten()
        # TODO: if all images do not have same dimensions, raise an error (1st version)
        return input_

    # NUMPY VECTOR (Lib standard format) => ORIGINAL IMAGE FORMAT
    def reverse_transform_numpy_data_to_image(self, npy_vector: np.array):
        pass

    # TODO: to test !
    def reverse_transform_numpy_data_to_numpy(self, npy_vector: np.array):
        output_ = np.reshape(npy_vector, self.og_dim)
        return output_

    # 1er cas simple: reverse transform du preprocess (npy -> image)
    # maybe this should be in the Balancer class
    # TODO: a proof for the exclusion of poorly reconstructed vectors will be needed for the functionality
    def postprocess_data(self):
        pass

    def get_augmented_data_only(self):
        return self.augmented_data

    def get_augmented_dataset(self):
        return self.augmented_dataset

    def save_augmented_dataset(self):
        pass

    def get_labels_distribution(self):
        # labels distribution check
        labels = []
        count = []
        for label, inputs in self.dataset.items():
            print(f'{label} : {inputs.shape[0]}')
            labels.append(label)
            count.append(inputs.shape[0])
        plt.bar(labels, count)
        plt.show()

    def get_random_elem_from_dataset(self, class_: str, nb_elements: int, plot_as_image: bool = False):
        # Get N random elements from the original dataset specified class
        class_indices = np.arange(self.dataset[class_].shape[0])
        np.random.seed(42)
        np.random.shuffle(class_indices)
        # Show each picked element
        # TODO: show elements as a grid ?
        for indice in class_indices[:nb_elements]:
            element = self.dataset[class_][indice]
            og_element = self.reverse_transform_numpy_data_to_numpy(element)
            if self.og_data_type is DataTypes.NUMPY and plot_as_image:
                plt.imshow(og_element, cmap=plt.cm.binary)
                plt.show()
            elif self.og_data_type is DataTypes.NUMPY and not plot_as_image:
                plt.plot(np.arange(og_element.shape[0]), og_element)
                plt.show()
            # TODO: handle missing data type (else)

    # TODO: Mutualiser les 2 méthodes de random_elem
    def get_random_elem_from_augmented_data(self, class_: str, nb_elements: int, plot_as_image: bool = False):
        # Get N random elements from the augmented data
        class_indices = np.arange(len(self.augmented_data[class_]))
        np.random.seed(42)
        np.random.shuffle(class_indices)
        # Show each picked element
        for indice in class_indices[:nb_elements]:
            element = self.augmented_data[class_][indice]
            og_element = self.reverse_transform_numpy_data_to_numpy(element)
            if self.og_data_type is DataTypes.NUMPY and plot_as_image:
                plt.imshow(og_element, cmap=plt.cm.binary)
                plt.show()
            elif self.og_data_type is DataTypes.NUMPY and not plot_as_image:
                plt.plot(np.arange(og_element.shape[0]), og_element)
                plt.show()
