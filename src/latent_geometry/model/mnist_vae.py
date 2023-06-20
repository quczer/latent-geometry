import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderVAE(nn.Module):
    def __init__(
        self, init_channels: int, latent_dim: int, kernel_size: int, image_channels: int
    ):
        super(EncoderVAE, self).__init__()

        # encoder
        self.enc1 = nn.Conv2d(
            in_channels=image_channels,
            out_channels=init_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=1,
        )
        self.enc2 = nn.Conv2d(
            in_channels=init_channels,
            out_channels=init_channels * 2,
            kernel_size=kernel_size,
            stride=2,
            padding=1,
        )
        self.enc3 = nn.Conv2d(
            in_channels=init_channels * 2,
            out_channels=init_channels * 4,
            kernel_size=kernel_size,
            stride=2,
            padding=1,
        )
        self.enc4 = nn.Conv2d(
            in_channels=init_channels * 4,
            out_channels=64,
            kernel_size=kernel_size,
            stride=2,
            padding=0,
        )
        # fully connected layers for learning representations
        self.fc1 = nn.Linear(64, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling
        return sample

    def forward(self, x):
        # encoding
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        batch, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        hidden = self.fc1(x)
        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var


class DecoderVAE(nn.Module):
    def __init__(
        self, init_channels: int, latent_dim: int, kernel_size: int, image_channels: int
    ):
        super(DecoderVAE, self).__init__()

        # decoder
        self.fc2 = nn.Linear(latent_dim, 64)

        self.dec1 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=init_channels * 8,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
        )
        self.dec2 = nn.ConvTranspose2d(
            in_channels=init_channels * 8,
            out_channels=init_channels * 4,
            kernel_size=kernel_size,
            stride=2,
            padding=1,
        )
        self.dec3 = nn.ConvTranspose2d(
            in_channels=init_channels * 4,
            out_channels=init_channels * 2,
            kernel_size=kernel_size,
            stride=2,
            padding=1,
        )
        self.dec4 = nn.ConvTranspose2d(
            in_channels=init_channels * 2,
            out_channels=image_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=1,
        )

    def decode(self, z):
        # decoding
        z = self.fc2(z)
        z = z.view(-1, 64, 1, 1)
        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        reconstruction = torch.sigmoid(self.dec4(x))
        return reconstruction  # , mu, log_var

    def forward(self, z):
        # z = self.encode(x)
        return self.decode(z)
