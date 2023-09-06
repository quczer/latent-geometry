from typing import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from latent_geometry.config import MODELS_DIR


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

    # def sample(self, x: torch.Tensor) -> torch.Tensor:
    #     """sample from posterior distribution"""
    #     mu, std = self.forward(x)
    #     noise = torch.randn_like(std)
    #     sample = mu + (noise * std)
    #     return sample

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, std), where std is a batch of scalars"""
        if x.dim() < 4:
            x = x.unsqueeze(0)
        x = self.convs(x)
        x = x.squeeze(2, 3)
        hidden = self.fcs(x)

        mu, std_out = hidden[..., :-1], hidden[..., [-1]]
        std = F.softplus(std_out)
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
                    ("sigmoid", nn.Sigmoid()),
                ]
            )
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 1:
            z = z.unsqueeze(0)
        z = self.fcs(z)
        z = z.reshape((*z.shape, 1, 1))
        z = self.deconvs(z)
        return z


# def load_decoder(device: torch.device = torch.device("cpu")) -> DecoderVAE:
#     return torch.load(MODELS_DIR / "mnist" / "decoder.pt", map_location=device)


# def load_encoder(device: torch.device = torch.device("cpu")) -> EncoderVAE:
#     return torch.load(MODELS_DIR / "mnist" / "encoder.pt", map_location=device)
