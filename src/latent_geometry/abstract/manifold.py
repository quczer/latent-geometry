from abc import ABC, abstractmethod

import numpy as np


class AbstractManifold(ABC):
    @staticmethod
    @abstractmethod
    def dimensionality() -> int:
        ...

    @abstractmethod
    def inner_product(self, zs: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        z : (N, D) ndarray
            Points on the manifold.
        Returns
        ----------
        matrix : (N, D, D) ndarray
            Array of matrices representing inner products around corresponding zs.
        """
