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
        r"""Compute mapping's jacobian matrix.

        Parameters
        ----------
        z : (D,) ndarray
            Point from the domain - usually latent space.

        Returns
        -------
        J : (D', D) ndarray
            Jacobian of the mapping computed at z, where index
            of the derivation is put second.

            :math: `J_{ij} = \partial_j g_i`.
        """

    @abstractmethod
    def hessian(self, z: np.ndarray) -> np.ndarray:
        r"""Compute mapping's hessian tensor.

        Parameters
        ----------
        z : (D,) ndarray
            Point from the domain - usually latent space.

        Returns
        -------
        H : (D', D, D) ndarray
            Hessian of the mapping computed at z, where indices
            of the derivation are put last.

            :math: `H_{ijk} = \partial_{jk} g_i`.
        """
