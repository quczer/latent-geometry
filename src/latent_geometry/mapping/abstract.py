from abc import ABC, abstractmethod

import numpy as np


class Mapping(ABC):
    @abstractmethod
    def __call__(self, z: np.ndarray) -> np.ndarray:
        """Apply the mapping function.

        Parameters
        ----------
        z : (D,) ndarray
            Point from the domain - usually latent space.

        Returns
        -------
        x : (D',) ndarray
            Point from the codomain - usually ambient space.
        """

    @abstractmethod
    def jacobian(self, z: np.ndarray) -> np.ndarray:
        """Compute mapping's jacobian matrix.

        Parameters
        ----------
        z : (D,) ndarray
            Point from the domain - usually latent space.

        Returns
        -------
        jacobian : (D', D) ndarray
            Jacobian of the mapping computed at z.
        """

    @abstractmethod
    def metric_matrix_derivative(self, z: np.ndarray) -> np.ndarray:
        r"""Given the jacobian J of the mapping compute jacobian of J^T @ J wrt. to z.

        Parameters
        ----------
        z : (D,) ndarray
            Point from the domain - usually latent space.

        Returns
        -------
        (D, D, D) ndarray
            Derivative of the inner-product matrix, where the index
            k of the derivation is last: math:`mat_{ijk} = \partial_k g_{ij}`.
        """
