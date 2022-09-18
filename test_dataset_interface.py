import unittest
import numpy as np


class TestDatasetInterfaceMethods(unittest.TestCase):

    def setUp(self) -> None:
        og_data = np.asarray([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ])

        # TODO: save it on disk
        # TODO: delete it from disk after all unit tests

    def test_numpy_to_numpy_transformation(self):
        """
        Test the data format transformation and reverse transformation.
        :return:
        """

        pass
