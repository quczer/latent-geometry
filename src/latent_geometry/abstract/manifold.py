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
        zs : (N, D) ndarray
            Points on the manifold.
        Returns
        ----------
        matrices : (N, D, D) ndarray
            Matrices representing inner products around corresponding zs.
        """
