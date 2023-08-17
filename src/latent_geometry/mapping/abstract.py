from abc import ABC, abstractmethod
from typing import Union

import numpy as np


class BaseMapping(ABC):
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
            Mapped point from the codomain - usually ambient space.
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

    @property
    @abstractmethod
    def in_dim(self) -> int:
        ...

    @property
    @abstractmethod
    def out_dim(self) -> int:
        ...


class DerivativeMapping(BaseMapping, ABC):
    @abstractmethod
    def second_derivative(self, z: np.ndarray) -> np.ndarray:
        r"""Compute mapping's second derivative tensor.

        Parameters
        ----------
        z : (D,) ndarray
            Point from the domain - usually latent space.

        Returns
        -------
        H : (D', D, D) ndarray
            The second derivative of the mapping computed at z,
            where indices of the derivation are put last.

            :math: `H_{ijk} = \partial_{jk} g_i`.
        """


class MatrixMapping(BaseMapping, ABC):
    @abstractmethod
    def metric_matrix_derivative(
        self, z: np.ndarray, ambient_metric_matrix: np.ndarray
    ) -> np.ndarray:
        r"""Compute mapping's second derivative tensor.

        Parameters
        ----------
        z : (D,) ndarray
            Point from the domain - usually latent space.

        ambient_metric_matrix : (D', D') ndarray
            Metric matrix from the co-domain.

        Returns
        -------
        dM: (D, D, D) ndarray
            Derivative of the inner-product matrix of the domain, where the index
            k of the derivation is last: math:`mat_{ijk} = \partial_k g_{ij}`

            Let `J` be the jacobian of the mapping, `A := ambient_metric_matrix` then:
            `dM_{ijk} = \partial_k (J.T @ A @ J)_{ij}`
        """


Mapping = Union[DerivativeMapping, MatrixMapping]
