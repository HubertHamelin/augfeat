import unittest
import numpy as np
import os
from balancer import Balancer
from config import DataTypes, AUTOENCODER_TRAINING_CONFIG_LIGHT
from autoencoder import AutoEncoder


class TestDatasetInterfaceMethods(unittest.TestCase):

    def setUp(self) -> None:
        data_1 = np.asarray([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ])

        data_2 = data_1 + 1

        if not os.path.isdir('./unit_test'):
            os.mkdir('./unit_test')
        if not os.path.isdir('./unit_test/class_test'):
            os.mkdir('./unit_test/class_test')

        np.save('./unit_test/class_test/data_1.npy', data_1)
        np.save('./unit_test/class_test/data_2.npy', data_2)

        # TODO: delete it from disk after all unit tests

    def test_numpy_to_numpy_transformation(self):
        """
        Test the data format transformation and reverse transformation.
        :return:
        """

        pass

    def test_scale_unscale_data(self):
        """
        Test the data scaling and reverse scaling.
        :return:
        """

        dataset_path = './unit_test'
        balancer = Balancer(dataset_path, DataTypes.NUMPY)
        inputs = balancer.dataset['class_test']
        test_autoencoder = AutoEncoder(inputs, balancer.timesteps, balancer.n_features,
                                       AUTOENCODER_TRAINING_CONFIG_LIGHT)
        for input_ in inputs:
            scaled_data = test_autoencoder.scale(input_)
            reverse_scaled_data = test_autoencoder.reverse_scale(scaled_data)
            self.assertTrue(np.array_equal(input_, reverse_scaled_data))

    def test_autoencoder_performance(self):
        """
        Assert that autoencoder configurations get satisfying results.
        :return:
        """

        dataset_path = './unit_test'
        balancer = Balancer(dataset_path, DataTypes.NUMPY)
        inputs = balancer.dataset['class_test']
        test_autoencoder = AutoEncoder(inputs, balancer.timesteps, balancer.n_features,
                                       AUTOENCODER_TRAINING_CONFIG_LIGHT)
        test_autoencoder.train()
        norms = test_autoencoder.evaluate()
        for norm in norms:
            self.assertTrue(norm <= 1)


if __name__ == '__main__':
    unittest.main()
