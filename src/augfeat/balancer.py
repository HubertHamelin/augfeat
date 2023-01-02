import os.path
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

# Local imports from the augfeat library
from augfeat.config import AUTOENCODER_TRAINING_CONFIG_LIGHT as LIGHT
from augfeat.custom_types import CustomClass
from augfeat.custom_types import DataTypes
from augfeat.autoencoder import AutoEncoder

# TODO: translate all docs and comments in english
# TODO: implement proper logging and log levels


class Balancer:

    def __init__(self, origin_path: str, target_path: str, data_type: DataTypes, lambda_: float = 0.5,
                 norm_threshold: int = 100, nb_nearest_neighbours: int = 2):

        # Parameters
        # TODO: group all parameters in an 'autoencoder_config' file with a default setting
        self.path = origin_path
        self.target_path = target_path
        self.data_type = data_type
        self.lambda_ = lambda_
        self.norm_threshold = norm_threshold
        self.nb_nearest_neighbours = nb_nearest_neighbours

    def augment_class(self, class_name: str, target_size: int, autoencoder_config: dict = LIGHT) -> None:
        """
        Augment a dataset so that each class has <target_size> elements
        :param class_name:
        :param target_size:
        :param autoencoder_config:
        :return:
        """
        # Initial parameters
        custom_class = CustomClass(self.path, class_name, self.data_type)
        inputs = custom_class.load_inputs_from_elements()
        timesteps = custom_class.timesteps
        n_features = custom_class.nb_features

        print(f'\n===== Augmenting data for class : {class_name} =====\n')

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
        print(f'Training the autoencoder on {inputs.shape[0]} inputs')
        autoencoder = AutoEncoder(inputs, timesteps, n_features, autoencoder_config)
        autoencoder.train()

        # Keeps only best reconstructed vectors from the training set using norm evaluation
        print('Cleaning training set...')
        inputs = autoencoder.evaluate()

        print('Augmenting data...')

        context_vectors = self.__compute_context_vectors(inputs, autoencoder, target_size)
        nearest_neighbours = self.__find_nearest_neighbours(context_vectors, self.nb_nearest_neighbours)
        extrapolated_vectors = self.__extrapolate_context_vectors(context_vectors, nearest_neighbours, target_size)
        decoded_vectors = self.__decode_extrapolated_vectors(extrapolated_vectors, autoencoder)

        # Unflatten outputs
        unflatten_decoded_vectors = []
        for vector in decoded_vectors:
            vector = np.reshape(vector, original_shape)
            unflatten_decoded_vectors.append(vector)
        decoded_vectors = unflatten_decoded_vectors

        print('Saving augmented data...')
        self.__save_class_augmented_data(custom_class, decoded_vectors)

        print('Done.')

    def __save_class_augmented_data(self, custom_class: CustomClass, decoded_vectors: list) -> None:
        # Check that the augmented directory already exists
        augmented_dataset_path = self.target_path
        if not os.path.isdir(augmented_dataset_path):
            os.mkdir(augmented_dataset_path)

        class_dir_path = os.path.join(augmented_dataset_path, custom_class.name)
        if not os.path.isdir(class_dir_path):
            os.mkdir(class_dir_path)

        for i, vector in enumerate(decoded_vectors):
            new_path = os.path.join(class_dir_path, f'aug_{str(i)}')
            custom_class.create_element(new_path, vector)

    def __compute_context_vectors(self, inputs: np.array, autoencoder: AutoEncoder, delta_nb_vectors: int) -> np.array:
        print('Computing context vectors')
        context_vectors = []
        if delta_nb_vectors > inputs.shape[0]:
            nb_vectors_max = inputs.shape[0]
        else:
            nb_vectors_max = delta_nb_vectors
        for input_ in tqdm(inputs[:nb_vectors_max], total=nb_vectors_max):
            input_ = autoencoder.scale(input_)
            input_ = input_[np.newaxis, :, :]
            context_vector = autoencoder.encoder.predict(input_, verbose=0)
            context_vector = context_vector.squeeze()
            context_vectors.append(context_vector)
        context_vectors = np.asarray(context_vectors)
        return context_vectors

    def __extrapolate_context_vector(self, cj: np.array, ck: np.array) -> np.array:
        extrapolated_cj = (cj - ck) * self.lambda_ + cj
        return extrapolated_cj

    def __find_nearest_neighbours(self, context_vectors: np.array, nb_nearest_neighbours: int) -> NearestNeighbors:
        # Recherche des K plus proches voisins (espace encodé)
        #nb_nearest_neighbours = int((df_train.shape[0] - incidents.shape[0]) / incidents.shape[0])
        print('Finding nearest neighbours with factor k={}'.format(nb_nearest_neighbours))
        nn = NearestNeighbors(n_neighbors=self.nb_nearest_neighbours, algorithm='ball_tree').fit(context_vectors)
        return nn

    def __extrapolate_context_vectors(self, context_vectors: np.array, nearest_neighbours: NearestNeighbors,
                                      delta_nb_vectors: int) -> list:
        print('Extrapolating context vectors')
        extrapolated_vectors = []
        iteration = 0
        while len(extrapolated_vectors) < delta_nb_vectors:
            for context_vector in context_vectors:
                if len(extrapolated_vectors) >= delta_nb_vectors:
                    break
                indices = nearest_neighbours.kneighbors(context_vector.reshape(1, -1), return_distance=False)
                indice = indices[0, 1 + iteration]
                # Application des transformations. Pour chaque paire de vecteurs encodés voisins
                # (on exclut le 1er qui est lui-même)
                extrapolated_vector = self.__extrapolate_context_vector(context_vector, context_vectors[indice])
                extrapolated_vector = extrapolated_vector[np.newaxis, :]
                extrapolated_vectors.append(extrapolated_vector)
            iteration += 1
        return extrapolated_vectors

    def __decode_extrapolated_vectors(self, extrapolated_vectors: list, autoencoder: AutoEncoder) -> list:
        print('Decoding context vectors')
        decoded_vectors = []
        for vector in tqdm(extrapolated_vectors, total=len(extrapolated_vectors)):
            decoded_vector = autoencoder.decoder.predict(vector, verbose=0)
            decoded_vector = autoencoder.reverse_scale(decoded_vector[0])
            decoded_vectors.append(decoded_vector)
        return decoded_vectors

    def print_config(self):
        pass
