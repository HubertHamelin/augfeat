import os.path
import logging
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

# Local imports from the augfeat library
from augfeat.config import AUTOENCODER_TRAINING_CONFIG_MEDIUM as MEDIUM
from augfeat.custom_types import CustomClass
from augfeat.custom_types import DataTypes
from augfeat.autoencoder import AutoEncoder


class Balancer:

    def __init__(self, origin_path: str, target_path: str, data_type: DataTypes, lambda_: float = 0.5,
                 nb_nearest_neighbours: int = 2, verbose: int = 0):
        """
        Balancer instance initialisation, read about its configuration below.
        :param origin_path: path of the original dataset (global, not specific to a class).
        :param target_path: path where the augmented data will be stored.
        :param data_type: type of the data from the original dataset.
        :param lambda_: controls the extrapolation level: higher means more extensive, but also devient.
        :param nb_nearest_neighbours: controls quality of extrapolation, higher -> extensive, but devient.
        """
        self.path = origin_path
        self.target_path = target_path
        self.data_type = data_type
        self.lambda_ = lambda_
        self.nb_nearest_neighbours = nb_nearest_neighbours
        self.verbose = verbose

    def augment_class(self, class_name: str, target_size: int, autoencoder_config: dict = MEDIUM) -> None:
        """
        Augment a dataset so that the specified class has <target_size> elements
        :param class_name: name of the class to be augmented.
        :param target_size: the number of elements to generate for the given class.
        :param autoencoder_config: configuration for the autoencoder training.
        :return: None
        """
        # Initial parameters
        custom_class = CustomClass(self.path, class_name, self.data_type)
        inputs = custom_class.load_inputs_from_elements()
        timesteps = custom_class.timesteps
        n_features = custom_class.nb_features

        logging.info(f'\n===== Augmenting data for class : {class_name} =====\n')

        # Flatten inputs
        flatten_inputs = []
        original_shape = (timesteps, n_features)
        for input_ in inputs:
            input_ = np.reshape(input_, (1, timesteps * n_features))
            flatten_inputs.append(input_)
        inputs = np.array(flatten_inputs)
        n_features = timesteps * n_features
        timesteps = 1

        # Train the autoencoder to rebuild each elements of the current class
        logging.info(f'Training the autoencoder on {inputs.shape[0]} inputs')
        autoencoder = AutoEncoder(inputs, timesteps, n_features, autoencoder_config, self.verbose)
        autoencoder.train()

        # Keeps only best reconstructed vectors from the training set using norm evaluation
        logging.info('Cleaning training set...')
        inputs = autoencoder.evaluate()

        # Steps to generate new elements for the given class.
        logging.info('Augmenting data...')
        context_vectors = self.__compute_context_vectors(inputs, autoencoder)
        nearest_neighbours = self.__find_nearest_neighbours(context_vectors, self.nb_nearest_neighbours)
        extrapolated_vectors = self.__extrapolate_context_vectors(context_vectors, nearest_neighbours, target_size)
        decoded_vectors = self.__decode_extrapolated_vectors(extrapolated_vectors, autoencoder)

        # Unflatten outputs
        unflatten_decoded_vectors = []
        for vector in decoded_vectors:
            vector = np.reshape(vector, original_shape)
            unflatten_decoded_vectors.append(vector)
        decoded_vectors = unflatten_decoded_vectors

        # Saving the augmented data into the target path specified by the user.
        logging.info('Saving augmented data...')
        self.__save_class_augmented_data(custom_class, decoded_vectors)

        logging.info('Done.')

    def __save_class_augmented_data(self, custom_class: CustomClass, decoded_vectors: list) -> None:
        """
        Saves the augmented data into the target directory passed by the user to the class instance.
        :param custom_class: handler of a specific class from the dataset.
        :param decoded_vectors: augmented data from a specific class from the dataset.
        :return: None
        """
        augmented_dataset_path = self.target_path
        if not os.path.isdir(augmented_dataset_path):
            os.mkdir(augmented_dataset_path)

        class_dir_path = os.path.join(augmented_dataset_path, custom_class.name)
        if not os.path.isdir(class_dir_path):
            os.mkdir(class_dir_path)

        for i, vector in enumerate(decoded_vectors):
            new_path = os.path.join(class_dir_path, f'aug_{str(i)}')
            custom_class.create_element(new_path, vector)

    @staticmethod
    def __compute_context_vectors(inputs: np.array, autoencoder: AutoEncoder) -> np.array:
        """
        Computes context vectors from the original inputs of the dataset class.
        It is the result of the encoding step (by the autoencoder).
        :param inputs: elements of a specific class from the dataset.
        :param autoencoder: trained autoencoder able to rebuild inputs from the given class of the dataset.
        :return: vectors encoded by the class autoencoder.
        """
        logging.info('Computing context vectors')
        context_vectors = []
        for input_ in tqdm(inputs):
            input_ = autoencoder.scale(input_)
            input_ = input_[np.newaxis, :, :]
            context_vector = autoencoder.encoder.predict(input_, verbose=0)
            context_vector = context_vector.squeeze()
            context_vectors.append(context_vector)
        context_vectors = np.asarray(context_vectors)
        return context_vectors

    def __extrapolate_context_vector(self, cj: np.array, ck: np.array) -> np.array:
        """
        Computes an extrapolated vector given two vectors. The intensity of the extrapolation is controlled
        by the lambda parameter. In the reference paper, 0.5 is pointed out as a good default value.
        :param cj: context vector
        :param ck: another context vector
        :return: extrapolated vector
        """
        extrapolated_cj = (cj - ck) * self.lambda_ + cj
        return extrapolated_cj

    @staticmethod
    def __find_nearest_neighbours(context_vectors: np.array, nb_nearest_neighbours: int) -> NearestNeighbors:
        """
        Finds the K nearest neighbours for each context vectors.
        :param context_vectors: context vectors within which the nearest neighbours are to be found.
        :param nb_nearest_neighbours: K nearest neighbours to be found for each context vector.
        :return: K nearest neighbours for each context vectors.
        """
        logging.info(f'Finding nearest neighbours with factor k={nb_nearest_neighbours - 1}')
        nn = NearestNeighbors(n_neighbors=nb_nearest_neighbours, algorithm='ball_tree').fit(context_vectors)
        return nn

    def __extrapolate_context_vectors(self, context_vectors: np.array, nearest_neighbours: NearestNeighbors,
                                      target_size: int) -> list:
        """
        Applies extrapolation operation for each pair of nearest neighbours encoded vectors.
        The first one is always excluded (itself). Farther neighbours will be used last to meet the target size.
        :param context_vectors: encoded vectors used for the extrapolation in features space.
        :param nearest_neighbours: K Nearest Neighbours for each context vector.
        :param target_size: number of augmented elements to generate from the features space.
        :return: list of <target_size> augmented elements, still encoded in features space.
        """
        logging.info('Extrapolating context vectors')
        extrapolated_vectors = []
        iteration = 0
        while len(extrapolated_vectors) < target_size:
            for context_vector in context_vectors:
                if len(extrapolated_vectors) >= target_size:
                    break
                indices = nearest_neighbours.kneighbors(context_vector.reshape(1, -1), return_distance=False)
                indice = indices[0, 1 + iteration]
                extrapolated_vector = self.__extrapolate_context_vector(context_vector, context_vectors[indice])
                extrapolated_vector = extrapolated_vector[np.newaxis, :]
                extrapolated_vectors.append(extrapolated_vector)
            iteration += 1
        return extrapolated_vectors

    @staticmethod
    def __decode_extrapolated_vectors(extrapolated_vectors: list, autoencoder: AutoEncoder) -> list:
        """
        Decodes extrapolated vectors from features space to original space of the dataset.
        :param extrapolated_vectors: list of <target_size> extrapolated vectors
        :param autoencoder: trained autoencoder used to decode the extrapolated vectors.
        :return: list of <target_size> decoded extrapolated vectors.
        """
        logging.info('Decoding context vectors')
        decoded_vectors = []
        for vector in tqdm(extrapolated_vectors, total=len(extrapolated_vectors)):
            decoded_vector = autoencoder.decoder.predict(vector, verbose=0)
            decoded_vector = autoencoder.reverse_scale(decoded_vector[0])
            decoded_vectors.append(decoded_vector)
        return decoded_vectors
