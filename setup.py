from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="latent-geometry",
    version="1.0.0a2",
    description="Python package for latent space exploration using Riemannian geometry.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Michal Kucharczyk",
    author_email="kucharczi@gmail.com",
    license="GPLv3",
    license_file="LICENSE",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries",
    ],
    packages=find_packages("src", exclude=("viz",)),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.22.0",
        "scipy>=1.10.1",
        "torchvision",
    ],
    python_requires=">=3.9",
    extras_require={
        "test": ["pytest", "mypy"],
        "dev": [
            "black[jupyter]~=22.3.0",
            "flake8",
            "isort",
            "mypy",
            "pre-commit",
            "pytest",
            "twine",
            "build",
        ],
        "data": [
            "ipykernel",
            "matplotlib",
            "dvc[gdrive]",
            "pandas",
            "plotly",
            "clearml",
            "nbconvert",
        ],
    },
)
