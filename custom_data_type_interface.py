from abc import ABC, abstractmethod
import numpy as np
from matplotlib import pyplot as plt


# TODO: Comment every methods
# expected types: numpy arrays, images, pandas Dataframes
# TODO: explain for each expected type the format_ meaning


class CustomDataTypeInterface(ABC):

    path: str
    shape: tuple

    @abstractmethod
    def check_file_type(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def set_shape(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_from_file(self) -> object:
        raise NotImplementedError

    @abstractmethod
    def save(self, element: object) -> None:
        raise NotImplementedError

    @abstractmethod
    def reverse_transform(self, decoded_vector: np.array) -> object:
        raise NotImplementedError

    @abstractmethod
    def plot(self, shape: tuple) -> None:
        raise NotImplementedError
