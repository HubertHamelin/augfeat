import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf


# TODO: Harmonize CAPITALS ... (see packages conventions)

class AutoEncoder:

    # TODO: define properly attributes and methods scope (private/public)
    DROPOUT_RATE = None
    LATENT_DIM = None
    N_FEATURES = None
    EPOCHS = None
    BATCH_SIZE = None
    LEARNING_RATE = None

    def __init__(self, inputs: np.array, n_features: int, config: dict):

        # TODO: 1st version -> assumption data is already correctly preprocessed by the user
        # TODO: 2nde assumption -> any correction to missing data has been made. each input is considered full (or drop)
        self.inputs = inputs

        # Parameters defined by users
        self.DROPOUT_RATE = config['dropout_rate']
        self.LATENT_DIM = config['latent_dim']
        self.N_FEATURES = n_features
        self.EPOCHS = config['epochs']
        self.BATCH_SIZE = config['batch_size']
        self.LEARNING_RATE = config['learning_rate']

        # Initialize the LSTM autoencoder
        self.encoder = self.init_encoder()
        self.decoder = self.init_decoder()
        self.autoencoder = self.init_autoencoder()

    def init_encoder(self):

        # TODO: each hidden layer has the same number of units
        # TODO: 1e-3 reduced by half if no improvement in validation_set for 10 epochs
        # TODO: reverse the order of input sequences
        # TODO: Cho et al., 2014

        # Encoder
        encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(1, self.N_FEATURES)),
            tf.keras.layers.LSTM(units=self.LATENT_DIM, return_sequences=True),
            tf.keras.layers.Dropout(self.DROPOUT_RATE),
            tf.keras.layers.LSTM(units=self.LATENT_DIM),
            tf.keras.layers.Dropout(self.DROPOUT_RATE)
        ])
        return encoder

    def init_decoder(self):
        # Decoder
        decoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=self.LATENT_DIM),
            tf.keras.layers.RepeatVector(1),
            tf.keras.layers.LSTM(units=self.LATENT_DIM, return_sequences=True),
            tf.keras.layers.Dropout(self.DROPOUT_RATE),
            tf.keras.layers.LSTM(units=self.LATENT_DIM, return_sequences=True),
            tf.keras.layers.Dropout(self.DROPOUT_RATE),
            tf.keras.layers.Dense(units=self.N_FEATURES)
        ])
        return decoder

    def init_autoencoder(self):
        # Implémentation de l'autoencoder
        autoencoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(1, self.N_FEATURES)),
            self.encoder,
            self.decoder
        ])
        autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.LEARNING_RATE), loss='mse')
        return autoencoder

    def train(self):
        # Entraînement de l'autoencoder
        history = self.autoencoder.fit(self.inputs, self.inputs, epochs=self.EPOCHS, verbose=1, shuffle=False,
                                       batch_size=self.BATCH_SIZE)

        # Historique de l'entraînement
        #  TODO: implement a debug mode
        """
        plt.plot(history.history['loss'])
        plt.title('autoencoder loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()
        """

    def evaluate(self):
        # Evaluation de l'autoencoder
        print(f'Evaluating autoencoder ...')
        norms = []
        for i in range(0, self.inputs.shape[0], self.BATCH_SIZE):
            """
            if i + self.BATCH_SIZE >= self.inputs.shape[0]:
                break
            """
            reconstructed_vectors = self.autoencoder.predict(self.inputs[i:i+self.BATCH_SIZE])
            true_inputs = self.inputs[i:i+self.BATCH_SIZE]
            for reconstructed_vector, true_input in zip(reconstructed_vectors, true_inputs):
                reconstructed_vector = reconstructed_vector.squeeze()
                true_input = true_input.squeeze()
                norm = np.linalg.norm(true_input - reconstructed_vector)
                norms.append(norm)

        """
        norms = np.asarray(norms)
        plt.hist(norms, bins=100)
        plt.show()
        """
        print(norms)
