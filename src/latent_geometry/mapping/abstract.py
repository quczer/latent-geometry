from abc import ABC, abstractmethod
from typing import Union

import numpy as np


class BaseMapping(ABC):
    @abstractmethod
    def __call__(self, zs: np.ndarray) -> np.ndarray:
        """Apply the mapping function.

        Parameters
        ----------
        zs : (B, D) array
            Batch of points from the domain - usually latent space.

        Returns
        -------
        xs : (B, D') array
            Mapped batch of points from the codomain - usually ambient space.
        """

    @abstractmethod
    def jacobian(self, zs: np.ndarray) -> np.ndarray:
        r"""Compute mapping's jacobian matrix.

        Parameters
        ----------
        zs : (B, D) array
            Batch of points from the domain - usually latent space.

        Returns
        -------
        Js : (B, D', D) array
            Jacobian of the mapping computed at zs, where index
            of the derivation is put last.

            :math: `Js_{bij} = \partial_j g_{bi}`.
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
    def second_derivative(self, zs: np.ndarray) -> np.ndarray:
        r"""Compute mapping's second derivative tensor.

        Parameters
        ----------
        zs : (B, D) array
            Batch of points from the domain - usually latent space.

        Returns
        -------
        Hs : (B, D', D, D) array
            The second derivative of the mapping computed at zs,
            where indices of the derivation are put last.

            :math: `H_{bijk} = \partial_{jk} g_{bi}`.
        """


class MatrixMapping(BaseMapping, ABC):
    @abstractmethod
    def metric_matrix_derivative(
        self, zs: np.ndarray, ambient_metric_matrices: np.ndarray
    ) -> np.ndarray:
        r"""
        Parameters
        ----------
        zs : (B, D) array
            Batch of points from the domain - usually latent space.

        ambient_metric_matrices : (B, D', D') array
            Batch of metric matrices from the co-domain.

        Returns
        -------
        dMs: (B, D, D, D) array
            Derivative of the inner-product matrices of the domain, where the index
            k of the derivation is last: math:`mat_{bijk} = \partial_k g_{bij}`

            Let `J` be the jacobian of the mapping, `A := ambient_metric_matrix` then:
            `dM_{ijk} = \partial_k (J.T @ A @ J)_{ij}` (dMs is a batch of dM)
        """


class EuclideanMatrixMapping(BaseMapping, ABC):
    @abstractmethod
    def euclidean_metric_matrix_derivative(self, zs: np.ndarray) -> np.ndarray:
        r"""Compute :py:meth:`MatrixMapping.metric_matrix_derivative` for the special case
        of the Euclidean ambient space.

        Parameters
        ----------
        zs : (B, D) array
            Batch of points from the domain - usually latent space.

        Returns
        -------
        dMs: (B, D, D, D) array
            Derivative of the inner-product matrices of the domain, where the index
            k of the derivation is last: math:`mat_{bijk} = \partial_k g_{bij}`

            Let `J` be the jacobian of the mapping, `A := I` then:
            `dM_{ijk} = \partial_k (J.T @ A @ J)_{ij} = \partial_k (J.T @ J)` (dMs is a batch of dM)
        """


class ChristoffelsDerivativeMapping(BaseMapping, ABC):
    @abstractmethod
    def euclidean_christoffels_derivative(self, zs: np.ndarray) -> np.ndarray:
        r"""Compute mapping's derivative of the Christoffel symbols for the special case
        of the Euclidean ambient space.

        It may be done by the following formula:
        :math:` \partial_l \Gamma^k_{ij} = \frac{1}{2} \partial_l g^{pk}(
        \partial_i g_{jp} + \partial_j g_{pi} - \partial_p g_{ij})`, where

        - `J_{ij} = \partial_i f_j` is the Jacobian of the mapping `f`
        - `g_{ij} = J_{ik} J_{jk}` is the metric tensor of the mapping
        - `g^{ij}` is the inverse of the metric tensor `g_{ij}`

        Parameters
        ----------
        zs : (B, D) array
            Batch of points from the domain - usually latent space.

        Returns
        -------
        dGamma : (B, D, D, D, D) array
            Christoffel symbols derivative, where the contravariant index is second
            and the derivation index is last.

            `dGamma(bijkl) == \partial_l \Gamma^i_{jk}`
        """


Mapping = Union[DerivativeMapping, MatrixMapping]
