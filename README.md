# latent-geometry
Master thesis package for exploring latent spaces by introducing riemannian geometry.

# Required software

Latent-geometry was tested under Linux running python 3.9. Other python versions might work as well but it is not guaranteed. All required packages are enclosed in `setup.py`.

# Installation
## users
- run the following in the console to install the package
```console
pip install .
```
## developers
- python package installation
```console
pip install -e .[dev]
```
- git hooks

Pre-commit hooks will ensure that all developers stick to the same code standards. They are defined in `.pre-commit-config.yaml`.
```console
pre-commit install
```

- testing:

    - to run all tests (recommended):
    ```console
    pytest
    ```
    - to run all tests in `test/test_file.py`:
    ```console
    pytest test/test_file.py
    ```
    - to run a single `test_function` in `test/test_file.py`:
    ```console
    pytest test/test_something.py::test_function
    ```
