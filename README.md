# latent-geometry
Master thesis package for exploring latent spaces by introduction of riemannian geometry.

## Required software

Latent-geometry was tested under Linux running python 3.9. Other python versions might work as well but it is not guaranteed. All required packages are enclosed in `setup.py`. For developer installation run:

```console
pip install -e .[dev]
```
Then install pre-commit hooks. This will ensure that all developers stick to the same code standards. Hooks are defined in `.pre-commit-config.yaml`. Run:

```console
pre-commit install
```

For use-only purposes installation run:
```console
pip install .
```
