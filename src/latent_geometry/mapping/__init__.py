from latent_geometry.mapping.abstract import (
    DerivativeMapping,
    EuclideanMatrixMapping,
    Mapping,
    MatrixMapping,
)
from latent_geometry.mapping.identity import IdentityMapping
from latent_geometry.mapping.linear import LinearMapping
from latent_geometry.mapping.torch import BaseTorchModelMapping, TorchModelMapping
from latent_geometry.mapping.toy import (
    create_northern_hemisphere_mapping,
    create_sphere_immersion,
)
