from setuptools import find_packages
from setuptools import setup

setup(
    name="latent-geometry",
    version="0.0.0.dev0",
    description="Python package for latent space exploration",
    author="Michal Kucharczyk",
    author_email="kucharczi@gmail.com",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3.9",
    ],
    packages=find_packages(),
    package_dir={"": "src"},
    install_requires=[],
    python_requires='>=3.8',
    setup_requires=["wheel"],
    extras_require={
        "dev": [
            "black[jupyter]",
            "flake8",
            "isort",
            "mypy",
        ]
    },
)
