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


class AutoEncoder:

    def __init__(self, inputs: np.array, timesteps: int, n_features: int, config: dict, verbose: int = 0):
        self.inputs = inputs
        self.dropout_rate = config['dropout_rate']
        self.latent_dim = config['latent_dim']
        self.n_features = n_features
        self.timesteps = timesteps
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']
        self.reconstruction_limit_factor = config['reconstruction_limit_factor']
        self.verbose = verbose

        # Initialize the LSTM autoencoder
        self.encoder = self.init_encoder()
        self.decoder = self.init_decoder()
        self.autoencoder = self.init_autoencoder()

        # Preprocessing
        self.scaler = self.__get_scaler()

    def init_encoder(self):
        encoder = Sequential([
            Input(shape=(self.timesteps, self.n_features)),
            LSTM(units=self.latent_dim, activation='relu', return_sequences=True),
            Dropout(self.dropout_rate),
            LSTM(units=self.latent_dim, activation='relu', return_sequences=False),
            Dropout(self.dropout_rate)
        ])
        return encoder

    def init_decoder(self):
        decoder = Sequential([
            Input(shape=self.latent_dim),
            RepeatVector(self.timesteps),
            LSTM(units=self.latent_dim, activation='relu', return_sequences=True),
            Dropout(self.dropout_rate),
            LSTM(units=self.latent_dim, activation='relu', return_sequences=True),
            Dropout(self.dropout_rate),
            TimeDistributed(Dense(units=self.n_features))
        ])
        return decoder

    def init_autoencoder(self):
        autoencoder = Sequential([
            Input(shape=(self.timesteps, self.n_features)),
            self.encoder,
            self.decoder
        ])
        autoencoder.compile(optimizer=Adam(learning_rate=self.learning_rate, clipnorm=1), loss='mse')
        return autoencoder

    def __get_scaler(self):
        # Fit a scaler on the input dataset
        scaler = StandardScaler()
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

        # Autoencoder training
        history = self.autoencoder.fit(inputs, inputs, epochs=self.epochs, verbose=self.verbose, shuffle=False,
                                       batch_size=self.batch_size)

        # Plot training history
        if self.verbose > 0:
            plt.plot(history.history['loss'])
            plt.title('autoencoder loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.show()

    def evaluate(self):
        norms = []
        for i in range(0, self.inputs.shape[0], self.batch_size):
            inputs = np.asarray([self.scale(input_) for input_ in self.inputs[i:i+self.batch_size]])
            reconstructed_vectors = self.autoencoder.predict(inputs)
            reconstructed_vectors = np.asarray([self.reverse_scale(vector) for vector in reconstructed_vectors])
            true_inputs = self.inputs[i:i+self.batch_size]
            for reconstructed_vector, true_input in zip(reconstructed_vectors, true_inputs):
                reconstructed_vector = reconstructed_vector.squeeze()
                true_input = true_input.squeeze()
                norm = np.linalg.norm(true_input - reconstructed_vector)
                norms.append(norm)

        # Finds and keeps only the X% best reconstructed vectors from the training set for the augmentation.
        norms = np.asarray(norms)
        ordered_indexes = np.argsort(norms)
        ordered_indexes = ordered_indexes[:int(self.reconstruction_limit_factor * ordered_indexes.shape[0])]
        best_reconstructed_inputs = [self.inputs[i] for i in ordered_indexes]
        best_norms = [norms[i] for i in ordered_indexes]

        if self.verbose > 0:
            plt.hist(norms, bins=20, color='g')
            plt.hist(best_norms, bins=5, color='r')
            plt.show()

        return np.array(best_reconstructed_inputs)
