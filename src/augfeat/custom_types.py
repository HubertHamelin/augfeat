import os
import numpy as np
from typing import List
from enum import Enum

# Local import from the augfeat library.
from augfeat.custom_data_type_interface import CustomDataTypeInterface


class DataTypes(Enum):
    PNG = '.png'
    NUMPY = '.npy'


# TODO: add comments to raised errors

class CustomTypeNumpy(CustomDataTypeInterface):

    def __init__(self, path: str):
        self.path = path

    def check_file_type(self) -> bool:
        if self.path[-4:] == '.npy':
            return True
        else:
            return False

    def set_shape(self) -> None:
        value = self.load_from_file()
        self.shape = value.shape

    def load_from_file(self) -> np.array:
        value = np.load(self.path)
        return value

    def save(self, element: np.array) -> None:
        np.save(self.path, element)

    def transform_to_numpy(self, target_shape: tuple) -> None:
        pass

    def reverse_transform(self, vector: np.array) -> np.array:
        element = vector
        return element

    def plot(self, shape: tuple) -> None:
        # if dim=1: plot curve
        # if dim=2: plot grayscale image
        # if dim=(x,y,3): plot RGB image
        # if dim >=4: unable to plot
        pass


class CustomClass:

    def __init__(self, path: str, class_name: str, data_type: DataTypes):
        self.path = os.path.join(path, class_name)
        self.name = class_name
        self.data_type = data_type

        self.elements: List[CustomDataTypeInterface] = self.__load_element_from_type()

        self.timesteps, self.nb_features = self.__check_and_get_inputs_format()
        self.shape = (self.timesteps, self.nb_features)

    def __load_element_from_type(self) -> list:
        """
        Creates a custom element for each target file in the class instance path.
        :return:
        """
        elements = []
        for file_name in os.listdir(self.path):
            if self.data_type == DataTypes.NUMPY:
                element = CustomTypeNumpy(os.path.join(self.path, file_name))
                if element.check_file_type():
                    element.set_shape()
                    elements.append(element)
            else:
                raise TypeError('Unhandled data type.')
        return elements

    def load_inputs_from_elements(self) -> np.array:
        """
        Transforms every element of the class instance into a numpy array input.
        :return:
        """
        inputs = []
        for element in self.elements:
            input_ = element.load_from_file()
            inputs.append(input_)
        return np.array(inputs)

    def __check_and_get_inputs_format(self) -> tuple:
        """
        Checks that all transformed inputs have the exact same format, and return it.
        :return:
        """
        shape_ = self.elements[0].shape
        for element in self.elements[1:]:
            if shape_ != element.shape:
                raise TypeError(f'shapes {shape_} and {element.shape} are different, unable to proceed.')
        return shape_

    def create_element(self, path: str, decoded_vector: np.array) -> None:
        """
        Creates and saves a new element in the original dataset type from a decoded augmented vector.
        :param path:
        :param decoded_vector:
        :return:
        """
        if self.data_type == DataTypes.NUMPY:
            element = CustomTypeNumpy(path + '.npy')
            original_data = element.reverse_transform(decoded_vector)
            element.save(original_data)
        else:
            raise TypeError('Unhandled data type.')
