import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.optimizers import Adam
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


# TODO: Harmonize CAPITALS ... (see packages conventions)
# TODO: define properly attributes and methods scope (private/public)


class AutoEncoder:

    DROPOUT_RATE = None
    LATENT_DIM = None
    N_FEATURES = None
    TIMESTEPS = None
    EPOCHS = None
    BATCH_SIZE = None
    LEARNING_RATE = None

    def __init__(self, inputs: np.array, timesteps: int, n_features: int, config: dict):
        """

        :param inputs: sequence format (samples, timesteps, features)
        :param config:
        """

        # TODO: 1st version -> assumption data is already correctly preprocessed by the user
        # TODO: 2nde assumption -> any correction to missing data has been made. each input is considered full (or drop)
        self.inputs = inputs

        # Parameters defined by users
        self.DROPOUT_RATE = config['dropout_rate']
        self.LATENT_DIM = config['latent_dim']
        self.N_FEATURES = n_features
        self.TIMESTEPS = timesteps
        self.EPOCHS = config['epochs']
        self.BATCH_SIZE = config['batch_size']
        self.LEARNING_RATE = config['learning_rate']

        # Initialize the LSTM autoencoder
        self.encoder = self.init_encoder()
        self.decoder = self.init_decoder()
        self.autoencoder = self.init_autoencoder()

        # Preprocessing
        self.scaler = self.__get_scaler()

    def init_encoder(self):

        # TODO: each hidden layer has the same number of units
        # TODO: 1e-3 reduced by half if no improvement in validation_set for 10 epochs
        # TODO: reverse the order of input sequences
        # TODO: Cho et al., 2014

        # Encoder
        """
        encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.TIMESTEPS, self.N_FEATURES)),
            tf.keras.layers.LSTM(units=self.LATENT_DIM, return_sequences=True),
            tf.keras.layers.Dropout(self.DROPOUT_RATE),
            tf.keras.layers.LSTM(units=self.LATENT_DIM),
            tf.keras.layers.Dropout(self.DROPOUT_RATE)
        ])
        """
        encoder = Sequential([
            Input(shape=(self.TIMESTEPS, self.N_FEATURES)),
            LSTM(units=self.LATENT_DIM, activation='relu', return_sequences=True),
            Dropout(self.DROPOUT_RATE),
            LSTM(units=self.LATENT_DIM, activation='relu', return_sequences=False),
            Dropout(self.DROPOUT_RATE)
        ])
        return encoder

    def init_decoder(self):
        # Decoder
        """
        decoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=self.LATENT_DIM),
            tf.keras.layers.RepeatVector(self.TIMESTEPS),
            tf.keras.layers.LSTM(units=self.LATENT_DIM, return_sequences=True),
            tf.keras.layers.Dropout(self.DROPOUT_RATE),
            tf.keras.layers.LSTM(units=self.LATENT_DIM, return_sequences=True),
            tf.keras.layers.Dropout(self.DROPOUT_RATE),
            tf.keras.layers.Dense(units=self.N_FEATURES)
        ])
        """
        decoder = Sequential([
            Input(shape=self.LATENT_DIM),
            RepeatVector(self.TIMESTEPS),
            LSTM(units=self.LATENT_DIM, activation='relu', return_sequences=True),
            Dropout(self.DROPOUT_RATE),
            LSTM(units=self.LATENT_DIM, activation='relu', return_sequences=True),
            Dropout(self.DROPOUT_RATE),
            TimeDistributed(Dense(units=self.N_FEATURES))
        ])
        return decoder

    def init_autoencoder(self):
        # Implémentation de l'autoencoder
        """
        autoencoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.TIMESTEPS, self.N_FEATURES)),
            self.encoder,
            self.decoder
        ])
        """
        autoencoder = Sequential([
            Input(shape=(self.TIMESTEPS, self.N_FEATURES)),
            self.encoder,
            self.decoder
        ])
        # autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.LEARNING_RATE), loss='mse')
        autoencoder.compile(optimizer=Adam(learning_rate=self.LEARNING_RATE, clipnorm=1), loss='mse')
        return autoencoder

    def __get_scaler(self):
        # Fit a scaler on the input dataset
        scaler = StandardScaler()
        # TODO: flatten as [(n_features), ()...] from (k, timesteps, n_features) to (k * timesteps, n_features)
        inputs_as_list = np.reshape(self.inputs, (self.inputs.shape[0] * self.inputs.shape[1], self.inputs.shape[2]))
        scaler.fit(inputs_as_list)
        return scaler

    def scale(self, data: np.array):
        # Scale input data before training
        return self.scaler.transform(data)

    def reverse_scale(self, predicted_data: np.array):
        # Scale predicted data back to its original format
        return self.scaler.inverse_transform(predicted_data)

    def train(self):
        # Data preprocessing
        inputs = []
        for input_ in tqdm(self.inputs):
            inputs.append(self.scale(input_))
        inputs = np.asarray(inputs)

        # Entraînement de l'autoencoder
        history = self.autoencoder.fit(inputs, inputs, epochs=self.EPOCHS, verbose=1, shuffle=False,
                                       batch_size=self.BATCH_SIZE)

        # Historique de l'entraînement
        #  TODO: implement a debug mode
        plt.plot(history.history['loss'])
        plt.title('autoencoder loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()

    def evaluate(self):
        """

        :return:
        """
        norms = []
        for i in range(0, self.inputs.shape[0], self.BATCH_SIZE):
            inputs = np.asarray([self.scale(input_) for input_ in self.inputs[i:i+self.BATCH_SIZE]])
            reconstructed_vectors = self.autoencoder.predict(inputs)
            reconstructed_vectors = np.asarray([self.reverse_scale(vector) for vector in reconstructed_vectors])
            true_inputs = self.inputs[i:i+self.BATCH_SIZE]
            for reconstructed_vector, true_input in zip(reconstructed_vectors, true_inputs):
                reconstructed_vector = reconstructed_vector.squeeze()
                true_input = true_input.squeeze()
                norm = np.linalg.norm(true_input - reconstructed_vector)
                norms.append(norm)

        # Find 25% best reconstructed vectors from the training set
        norms = np.asarray(norms)
        ordered_indexes = np.argsort(norms)
        ordered_indexes = ordered_indexes[:int(0.25*ordered_indexes.shape[0])]
        best_reconstructed_inputs = [self.inputs[i] for i in ordered_indexes]
        best_norms = [norms[i] for i in ordered_indexes]

        plt.hist(norms, bins=100, color='g')
        plt.hist(best_norms, bins=100, color='r')
        plt.show()

        return np.array(best_reconstructed_inputs)
