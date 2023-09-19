from typing import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from latent_geometry.config import MODELS_DIR

_EPS = 1e-4


class EncoderVAE(nn.Module):
    def __init__(self, init_channels: int = 8, latent_dim: int = 2):
        super().__init__()
        self.convs = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        nn.Conv2d(
                            in_channels=1,
                            out_channels=init_channels,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                        ),
                    ),
                    ("tanh1", nn.Tanh()),
                    (
                        "conv2",
                        nn.Conv2d(
                            in_channels=init_channels,
                            out_channels=init_channels * 2,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                        ),
                    ),
                    ("tanh2", nn.Tanh()),
                    (
                        "conv3",
                        nn.Conv2d(
                            in_channels=init_channels * 2,
                            out_channels=init_channels * 4,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                        ),
                    ),
                    ("tanh3", nn.Tanh()),
                    (
                        "conv4",
                        nn.Conv2d(
                            in_channels=init_channels * 4,
                            out_channels=64,
                            kernel_size=4,
                            stride=2,
                            padding=0,
                        ),
                    ),
                    ("tanh4", nn.Tanh()),
                ]
            )
        )
        self.fcs = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(64, 128)),
                    ("tanh", nn.Tanh()),
                    ("fc2", nn.Linear(128, latent_dim + 1)),
                ]
            )
        )

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        """Sample from posterior distribution"""
        mu, std = self.forward(x)
        noise = torch.randn_like(std)
        sample = mu + (noise * std)
        return sample

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, std), where std is a batch of scalars"""
        if x.dim() < 4:
            x = x.unsqueeze(0)
        x = self.convs(x)
        x = x.squeeze(2, 3)
        hidden = self.fcs(x)

        mu, std_out = hidden[..., :-1], hidden[..., [-1]]
        std = F.softplus(std_out) + _EPS
        return mu, std


class DecoderVAE(nn.Module):
    def __init__(self, init_channels: int = 8, latent_dim: int = 2):
        super().__init__()

        self.fcs = nn.Sequential(
            OrderedDict([("fc", nn.Linear(latent_dim, 64)), ("tanh", nn.Tanh())])
        )
        self.deconvs = nn.Sequential(
            OrderedDict(
                [
                    (
                        "deconv1",
                        nn.ConvTranspose2d(
                            in_channels=64,
                            out_channels=init_channels * 8,
                            kernel_size=4,
                            stride=2,
                            padding=0,
                        ),
                    ),
                    ("tanh1", nn.Tanh()),
                    (
                        "deconv2",
                        nn.ConvTranspose2d(
                            in_channels=init_channels * 8,
                            out_channels=init_channels * 4,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                        ),
                    ),
                    ("tanh2", nn.Tanh()),
                    (
                        "deconv3",
                        nn.ConvTranspose2d(
                            in_channels=init_channels * 4,
                            out_channels=init_channels * 2,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                        ),
                    ),
                    ("tanh3", nn.Tanh()),
                    (
                        "deconv4",
                        nn.ConvTranspose2d(
                            in_channels=init_channels * 2,
                            out_channels=1,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                        ),
                    ),
                    # ("sigmoid", nn.Sigmoid()),
                ]
            )
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return logits"""
        if z.dim() == 1:
            z = z.unsqueeze(0)
        z = self.fcs(z)
        z = z.reshape((*z.shape, 1, 1))
        z = self.deconvs(z)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Return probabilities"""
        x = self.forward(z)
        return F.sigmoid(x)


def load_decoder(
    device: torch.device = torch.device("cpu"), name: str = "decoder.pt"
) -> DecoderVAE:
    state_dict = torch.load(MODELS_DIR / "mnist" / name, map_location=device)
    decoder = DecoderVAE(init_channels=8, latent_dim=2)
    decoder.load_state_dict(state_dict)
    decoder.to(device)
    decoder.eval()
    return decoder


def load_encoder(
    device: torch.device = torch.device("cpu"), name: str = "encoder.pt"
) -> EncoderVAE:
    state_dict = torch.load(MODELS_DIR / "mnist" / name, map_location=device)
    encoder = EncoderVAE(init_channels=8, latent_dim=2)
    encoder.load_state_dict(state_dict)
    encoder.to(device)
    encoder.eval()
    return encoder


def save_model(model: nn.Module, name: str) -> None:
    torch.save(model.state_dict(), MODELS_DIR / "mnist" / name)
