from setuptools import find_packages, setup

setup(
    name="latent-geometry",
    version="0.0.0.dev0",
    description="Python package for latent space exploration using Riemann geometry.",
    author="Michal Kucharczyk",
    author_email="kucharczi@gmail.com",
    license="GPLv3",
    license_file="LICENSE",
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.22.0",
        "scipy>=1.10.1",
    ],
    python_requires=">=3.9",
    extras_require={
        "test": ["pytest", "mypy"],
        "dev": [
            "black[jupyter]>=23.7.0",
            "flake8",
            "isort",
            "mypy",
            "ipykernel",
            "pre-commit",
            "pytest",
            "matplotlib",
            "dvc[gdrive]",
            "pandas",
            "plotly",
            "torchvision"
        ],
    },
)
