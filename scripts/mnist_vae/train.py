import argparse
import itertools
from typing import Optional

import clearml
import numpy as np
import plotly.express as px
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from latent_geometry.data import load_mnist_dataset
from latent_geometry.model.mnist_vae import DecoderVAE, EncoderVAE, save_model


def report_metrics(
    logger: clearml.Logger,
    series: str,
    epoch: int,
    kl_loss: float,
    bce_loss: float,
    image_samples: list[tuple[str, np.ndarray]],
    stds: Optional[np.ndarray] = None,
    latent_distr: Optional[tuple[np.ndarray, np.ndarray]] = None,
):
    logger.report_scalar(title="kl_loss", series=series, value=kl_loss, iteration=epoch)
    logger.report_scalar(
        title="bce_loss", series=series, value=bce_loss, iteration=epoch
    )
    logger.report_scalar(
        title="total_loss",
        series=series,
        value=kl_loss + bce_loss,
        iteration=epoch,
    )

    for title, samples in image_samples:
        for i, sample in enumerate(samples):
            logger.report_image(
                title=title,
                series=f"{series}_sample_{i}",
                iteration=epoch,
                image=sample,
            )

    if latent_distr is not None:
        mu, label = latent_distr
        fig = px.scatter(
            x=mu[:, 0],
            y=mu[:, 1],
            color=label,
            labels=label,
            title="latent space",
            opacity=0.3,
        )

        logger.report_plotly(
            title="latent space", series=series, iteration=epoch, figure=fig
        )

    if stds is not None:
        logger.report_scalar(
            title="std_mean",
            series=series,
            value=stds.mean(),
            iteration=epoch,
        )
        logger.report_scalar(
            title="std_std",
            series=series,
            value=stds.std(),
            iteration=epoch,
        )


def train(
    encoder: nn.Module,
    decoder: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    logger: Optional[clearml.Logger],
    epoch: int,
    beta: float,
):
    bce_loss_fun = nn.BCEWithLogitsLoss(reduction="sum")
    encoder.to(device).train()
    decoder.to(device).train()

    kl_loss_sum = 0.0
    bce_loss_sum = 0.0

    stds = []
    for batch in dataloader:
        imgs = batch[0].to(device)
        optimizer.zero_grad()

        # autoencoding
        mu, std = encoder(imgs)
        z = mu + torch.randn_like(mu) * std
        sampled_img_logits = decoder(z)

        # losses
        bce_loss = bce_loss_fun(sampled_img_logits, imgs)
        kl_div = -0.5 * torch.sum(1 + 2 * std.log() - mu.pow(2) - std.pow(2))
        loss = (bce_loss + beta * kl_div) / len(batch)

        bce_loss_sum += bce_loss.item()
        kl_loss_sum += kl_div.item()

        # optimizer
        loss.backward()
        optimizer.step()

        # reporting
        stds.append(std.detach().cpu().numpy())

    if logger is not None:
        std = np.concatenate(stds, axis=0)
        len_data = len(dataloader)
        report_metrics(
            logger,
            "train",
            epoch,
            kl_loss_sum / len_data,
            bce_loss_sum / len_data,
            image_samples=[],
            stds=std,
            latent_distr=None,
        )


def test(
    encoder: nn.Module,
    decoder: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    logger: Optional[clearml.Logger],
    epoch: int,
    debug_z: np.ndarray,
) -> float:
    bce_loss_fun = nn.BCEWithLogitsLoss(reduction="sum")
    encoder.to(device).eval()
    decoder.to(device).eval()

    kl_loss_sum = 0.0
    bce_loss_sum = 0.0
    mus, labels, stds = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            imgs, label = batch[0].to(device), batch[1]

            # autoencoding
            mu, std = encoder(imgs)
            z = mu + torch.randn_like(mu) * std
            sampled_img_logits = decoder(z)

            # losses
            bce_loss = bce_loss_fun(sampled_img_logits, imgs)
            kl_div = -0.5 * torch.sum(1 + 2 * std.log() - mu.pow(2) - std.pow(2))

            bce_loss_sum += bce_loss.item()
            kl_loss_sum += kl_div.item()

            # reporting
            mus.append(mu.cpu().numpy())
            labels.append(label.cpu().numpy())
            stds.append(std.detach().cpu().numpy())
            if i == 0:
                shot_img_samples = F.sigmoid(sampled_img_logits).cpu().numpy()
                const_samples = (
                    decoder.decode(
                        torch.tensor(debug_z, dtype=torch.float32).to(device)
                    )
                    .detach()
                    .cpu()
                    .numpy()
                ).reshape(-1, 32, 32)

    if logger is not None:
        shot_samples = shot_img_samples[:8].reshape(8, 32, 32)
        mu = np.concatenate(mus, axis=0)
        label = np.concatenate(labels, axis=0)
        std = np.concatenate(stds, axis=0)
        len_data = len(dataloader)
        report_metrics(
            logger,
            "test",
            epoch,
            kl_loss_sum / len_data,
            bce_loss_sum / len_data,
            image_samples=[("shot", shot_samples), ("const", const_samples)],
            stds=std,
            latent_distr=(mu, label),
        )
    return bce_loss_sum + kl_loss_sum


def main(args: argparse.Namespace):
    DEBUG_Z = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 0.0],
            [-1.0, -1.0],
            [0.0, -1.0],
        ]
    )

    # Connecting ClearML with the current process,
    # from here on everything is logged automatically
    if args.clearml_run_name is not None:
        task = clearml.Task.init(
            project_name="MNIST VAE", task_name=args.clearml_run_name
        )
        task.set_parameters(vars(args))
        logger = task.logger
    else:
        logger = None

    # init
    torch_device = torch.device(args.device)
    torch.manual_seed(args.seed)
    train_loader = DataLoader(
        load_mnist_dataset("train"), batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        load_mnist_dataset("test"), batch_size=args.test_batch_size, shuffle=False
    )

    encoder = EncoderVAE()
    decoder = DecoderVAE()
    optimizer = torch.optim.Adam(
        params=itertools.chain(
            encoder.parameters(),
            decoder.parameters(),
        ),
        lr=args.lr,
    )

    best_loss = 1e18
    for epoch in tqdm(range(args.epochs)):
        train(
            encoder,
            decoder,
            train_loader,
            torch_device,
            optimizer,
            logger,
            epoch,
            args.beta,
        )
        if epoch % args.eval_every == 0:
            loss = test(
                encoder,
                decoder,
                test_loader,
                torch_device,
                logger,
                epoch,
                debug_z=DEBUG_Z,
            )
            if loss < best_loss and args.save_best_model:
                best_loss = loss
                save_model(encoder, "best_encoder.pt")
                save_model(decoder, "best_decoder.pt")

    task.close()


if __name__ == "__main__":
    # Settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Arguments")

    parser.add_argument(
        "--clearml-run-name",
        type=str,
        metavar="C",
        help="Name of ClearML run (optional)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        metavar="N",
        help="input batch size for training",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        metavar="N",
        help="input batch size for testing",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        metavar="N",
        help="number of epochs to train",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--device", default="cuda", choices=("cpu", "cuda"), help="device to train on"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before evaluating on validation split",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        metavar="B",
        help="Coefficient of the KL loss part",
    )
    parser.add_argument(
        "--save-best-model",
        action="store_true",
        default=True,
        help="Save best performing model on validation split",
    )
    args = parser.parse_args()
    main(args)
