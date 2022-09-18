import numpy as np
from tqdm import tqdm
from config import DataTypes
from autoencoder import AutoEncoder
from dataset_interface import DatasetInterface
from sklearn.neighbors import NearestNeighbors
from config import AUTOENCODER_TRAINING_CONFIG_LIGHT as LIGHT


# TODO: translate all docs and comments in english
# TODO: implement proper logging and log levels


class Balancer:

    def __init__(self, path: str, data_type: DataTypes, lambda_: float = 0.5, norm_threshold: int = 100,
                 nb_nearest_neighbours: int = 2):

        # Parameters
        self.lambda_ = lambda_
        self.norm_threshold = norm_threshold
        self.nb_nearest_neighbours = nb_nearest_neighbours

        # Lib objects init
        self.dataset_interface = DatasetInterface(path, data_type)
        self.dataset_interface.load_dataset()
        self.dataset = self.dataset_interface.dataset
        self.n_features = self.dataset_interface.n_features

    # Augment a dataset so that each class has <target_size> elements
    def balance(self, target_size: int, autoencoder_config: dict = LIGHT):

        # TODO: target_size default

        augmented_data = {}

        for class_, inputs in self.dataset.items():

            print(f'\n===== Class : {class_} =====\n')

            # Train the autoencoder to rebuild each elements of the current class
            print(f'Training the autoencoder on {inputs.shape[0]} inputs')
            autoencoder = AutoEncoder(inputs, self.n_features, autoencoder_config)
            autoencoder.train()
            autoencoder.evaluate()

            # If there are less elements in the class than the target size passed by the user
            if len(inputs) < target_size:
                print('Augmenting data...')
                delta_nb_vectors = target_size - inputs.shape[0]
                context_vectors = self.compute_context_vectors(inputs, autoencoder, delta_nb_vectors)
                nearest_neighbours = self.find_nearest_neighbours(context_vectors, self.nb_nearest_neighbours)
                extrapolated_vectors = self.extrapolate_context_vectors(context_vectors, nearest_neighbours,
                                                                        delta_nb_vectors)
                decoded_vectors = self.decode_extrapolated_vectors(extrapolated_vectors, autoencoder)
                augmented_data[class_] = decoded_vectors

            break  # TO BE REMOVED AFTER TESTS

        self.dataset_interface.augmented_data = augmented_data

    def augment(self, inputs: np.array, autoencoder: AutoEncoder):
        pass

    def downsize(self, target_size: int):
        pass

    # Augment only a specific class in a dataset to <target_size> elements
    # TODO: add the functionality, but not as overloading

    def compute_context_vectors(self, inputs: np.array, autoencoder: AutoEncoder, delta_nb_vectors: int):
        print('Computing context vectors')
        context_vectors = []
        if delta_nb_vectors > inputs.shape[0]:
            nb_vectors_max = inputs.shape[0]
        else:
            nb_vectors_max = delta_nb_vectors
        for input_ in tqdm(inputs[:nb_vectors_max], total=nb_vectors_max):
            input_ = input_[np.newaxis, :, :]
            context_vector = autoencoder.encoder.predict(input_, verbose=0)
            context_vector = context_vector.squeeze()
            context_vectors.append(context_vector)
        context_vectors = np.asarray(context_vectors)
        return context_vectors

    def extrapolate_context_vector(self, cj: np.array, ck: np.array):
        extrapolated_cj = (cj - ck) * self.lambda_ + cj
        return extrapolated_cj

    def find_nearest_neighbours(self, context_vectors: np.array, nb_nearest_neighbours: int):
        # Recherche des K plus proches voisins (espace encodé)
        #nb_nearest_neighbours = int((df_train.shape[0] - incidents.shape[0]) / incidents.shape[0])
        print('nearest neighbours factor K: {}'.format(nb_nearest_neighbours))
        nn = NearestNeighbors(n_neighbors=self.nb_nearest_neighbours, algorithm='ball_tree').fit(context_vectors)
        return nn

    def extrapolate_context_vectors(self, context_vectors: np.array, nearest_neighbours: NearestNeighbors,
                                    delta_nb_vectors: int):
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
                extrapolated_vector = self.extrapolate_context_vector(context_vector, context_vectors[indice])
                extrapolated_vectors.append(extrapolated_vector)
            iteration += 1
        return extrapolated_vectors

    def decode_extrapolated_vectors(self, extrapolated_vectors: list, autoencoder: AutoEncoder):
        print('Decoding context vectors')
        decoded_vectors = []
        for vector in tqdm(extrapolated_vectors, total=len(extrapolated_vectors)):
            vector = vector[np.newaxis, :]
            decoded_vector = autoencoder.decoder.predict(vector, verbose=0)
            decoded_vectors.append(decoded_vector)
        return decoded_vectors

    def print_config(self):
        pass
