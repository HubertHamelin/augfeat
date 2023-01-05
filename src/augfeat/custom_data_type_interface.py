from abc import ABC, abstractmethod
import numpy as np


class CustomDataTypeInterface(ABC):
    """
    The point of this interface is to free the Balancer class from managing the data. The Balancer class will work
    with instances of subclasses implementing this interface, and the subclasses will take care of data compatibility.
    """

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
    def transform_to_numpy(self, target_shape: tuple) -> np.array:
        raise NotImplementedError
