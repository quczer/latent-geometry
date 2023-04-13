from setuptools import find_packages, setup

setup(
    name="latent-geometry",
    version="0.0.0.dev0",
    description="Python package for latent space exploration using Riemann geometry.",
    author="Michal Kucharczyk, Jacek Rutkowski",
    author_email="kucharczi@gmail.com",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    packages=find_packages(),
    package_dir={"": "src"},
    install_requires=[
        # "geomstats[backends] @ git+https://github.com/geomstats/geomstats@master",
        "torch>=2.0.0",
        "numpy>=1.24.2",
    ],
    python_requires=">=3.9",
    extras_require={
        "dev": [
            "black[jupyter]",
            "flake8",
            "isort",
            "mypy",
            "ipykernel",
            "pre-commit",
            "pytest",
        ]
    },
)
