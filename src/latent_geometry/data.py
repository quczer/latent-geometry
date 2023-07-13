from typing import Literal

import torchvision
import torchvision.transforms as transforms

from latent_geometry.config import DATA_DIR


def load_mnist_dataset(
    split: Literal["train", "test"]
) -> torchvision.datasets.mnist.MNIST:
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )
    dataset = torchvision.datasets.MNIST(
        root=DATA_DIR / "mnist", train=split == "train", transform=transform
    )
    return dataset
