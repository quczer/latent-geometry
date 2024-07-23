# Latent Geometry
Master's thesis python package allowing for the exploration of latent spaces of generative models through Riemannian geometry.

By employing a pull-back metric from the observation space one can reveal nuanced geometrical structures in hidden spaces. The framework is agnostic to the automatic differetiation backend e.g. PyTorch, TensorFlow. It works even with custom, hand-made differentiable mappings.

[Master Thesis PDF](https://smallpdf.com/file#s=5a36cf0c-5886-4cc8-8056-f71159916b62)

[![CI - Test](https://github.com/quczer/latent-geometry/actions/workflows/tests.yaml/badge.svg)](https://github.com/quczer/latent-geometry/actions/workflows/tests.yaml)
[![PyPI Latest Release](https://img.shields.io/pypi/v/latent-geometry.svg)](https://pypi.org/project/latent-geometry/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/latent-geometry.svg)](https://pypi.org/project/latent-geometry/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/latent-geometry)](https://pypi.org/project/latent-geometry/)
[![License - GPLv3](https://img.shields.io/pypi/l/latent-geometry.svg)](https://github.com/quczer/latent-geometry/blob/master/LICENSE)

# Installation
```cosole
pip install latent-geometry
```

# Usage
## Geodesics

```python
import numpy as np
import torch.nn as nn # just for the sake of example

your_neural_net: nn.Module = YourPyTorchNet(latent_dim=8)

# create Mapping
from latent_geometry.mapping import Mapping, TorchModelMapping

mapping: Mapping = TorchModelMapping(
    model=your_neural_net,
    in_shape=(-1, 8), # dimensionality of the domain w/ -1 for the batch size
    out_shape=(-1, 3, 32, 32), # dimensionality of the co-domain w/ -1 for the batch size
    batch_size=batch_size,
    call_fn=your_neural_net.forward, # optional
)

# define your favourite metric for the observation space
from latent_geometry.metric import EuclideaMetric

ambient_metric = EuclideanMetric()

# create the manifold spanned by your latent space with the pulled-back ambient metric
from latent_geometry.manifold import LatentManifold

latent_manifold = LatentManifold(
    mapping=latent_mapping,
    ambient_metric=ambient_metric,
)

# calculate the geodesic starting from z_0 with velocity v_0
from latent_geometry.path import ManifoldPath

z_0 = np.zeros(8)
v_0 = np.ones_like(z_0)

geodesic: ManifoldPath = latent_manifold.geodesic(z=z_0, velocity_vec=v_0)
# geodesic(0) == z_0

# calculate the the shortest path between z_a and z_b
z_a = np.zeros(8)
z_b = np.zeros(8) + 3

shortest_path: ManifoldPath = latent_manifold.shortest_path(z_a=z_a, z_b=z_b)
# shortest_path(0.0), shortest_path(1.0) == z_a, z_b

```
If your mapping is not based on PyTorch you need to implement one of two possible interfaces

```python
# you can implement first and second derivative of output wrt. to input
from latent_geometry.mapping import DerivativeMapping

class YourMappingWrapper(DerivativeMapping):
    def __init__(self, your_mapping: Callable) -> None: ...
    def jacobian(self, zs: np.ndarray) -> np.ndarray: ...
    def second_derivative(self, zs: np.ndarray) -> np.ndarray: ...

# or follow the other interface (for speed-up purpuses)
from latent_geometry.mapping import MatrixMapping

class YourMappingWrapper(MatrixMapping):
    def __init__(self, your_mapping: Callable) -> None: ...
    def jacobian(self, zs: np.ndarray) -> np.ndarray: ...
    def metric_matrix_derivative(
        self, zs: np.ndarray, ambient_metric_matrices: np.ndarray
    ) -> np.ndarray: ...
```
## Riemannian optimizer
only for PyTorch right now
```python
import torch
import torch.nn as nn
from latent_geometry.optim import TorchMetric, InputGDOptimizer

your_neural_net: nn.Module = YourPyTorchNet()
loss_fn = lambda x: x.mean()
example_input = torch.zeros(8, requires_grad=True)

optimizer = InputDGOptimizer(
    param=example_input,
    metric=TorchMetric(mapping=your_neural_net),
    lr=0.001,
    gradient_type="geometric" # may also be "standard", "retractive"
)

for _ in range(1_000):
    optimizer.zero_grad()
    x = your_neural_net(example_input)
    loss = loss_fn(x)

    loss.backward()
    optimizer.step()

```
