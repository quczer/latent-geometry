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

- type checking (using `mypy`):
```console
mypy src
```
# Data
Tu pull data and saved models simply run
```console
dvc pull
```

If you are a developer (with proper access) and want to share changes in data run either
```console
dvc add data
```
or
```console
dvc add models
```
and then commit changes using `git`.

For more details please visit https://dvc.org/doc/start/data-management/data-versioning.
