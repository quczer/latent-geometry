from abc import ABC, abstractmethod

import numpy as np


class AbstractMapping(ABC):
    @abstractmethod
    def __call__(self, zs: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        zs : (N, D) ndarray
            Points from the domain - usually latent space.

        Returns
        ----------
        xs : (N, D') ndarray
            Points from the codomain - usually interpretable space.
        """

    @abstractmethod
    def jacobian(self, zs: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        zs : (N, D) ndarray
            Points from the domain - usually latent space.

        Returns
        ----------
        jacobians : (N, D, D) ndarray
            Jacobians of the transformation computed near zs.
        """
