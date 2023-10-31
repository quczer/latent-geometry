import argparse
from functools import partial
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from scipy.stats.distributions import binom, norm, uniform
from tqdm import tqdm
from utils import allign_arrays, get_img_from_fig

from latent_geometry.config import FIGURES_DIR
from latent_geometry.model.mnist_vae import load_decoder
from latent_geometry.optim.metric import TorchMetric
from latent_geometry.optim.torch import InputSGDOptimizer


def optimize(
    z: torch.Tensor,
    iters: int,
    optimizer,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    map_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    path = []
    for _ in range(iters):
        optimizer.zero_grad()
        loss = loss_fn(map_fn(z))

        # optimizer
        loss.backward()
        path.append(z.unsqueeze(0).clone())
        optimizer.step()
    return torch.concat(path)


def create_path_in_ambient_fig(
    path: torch.Tensor,
    path_name: str,
    map_fn: Callable[[torch.Tensor], torch.Tensor],
    n_points: int = 9,
):
    fig, axes = plt.subplots(2, n_points, figsize=(1.5 * n_points, 3))
    step = path.shape[0] // n_points
    path_pts = path[::step, 0, ...]
    with torch.no_grad():
        imgs = map_fn(path_pts).cpu().numpy().reshape(-1, 32, 32)
    VMAX = 0.3
    for i, (img, ax) in enumerate(zip(imgs, axes[0, :])):
        ax.imshow(img, vmin=0, vmax=1)
        ax.set_title(
            f"{i+1} / {n_points}",
            fontsize=8,
        )
        ax.set_axis_off()

    axes[1, 0].set_axis_off()
    for img, img_next, ax in zip(imgs[:-1], imgs[1:], axes[1, 1:]):
        img_diff = (img_next - img).reshape(32, 32)

        diff_mse = np.sqrt((img_diff**2).sum())
        ax.imshow(img_diff, cmap="PiYG", vmin=-VMAX, vmax=VMAX)
        ax.set_title(
            (f"MSE: {diff_mse: .2f}"),
            fontsize=8,
        )
        ax.set_axis_off()

    fig.suptitle(
        f"Ambient mid-points on the {path_name} path",
        fontsize=13,
    )
    fig.tight_layout()
    return fig


def summarize_path(
    path: torch.Tensor,
    metric: TorchMetric,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
) -> pd.DataFrame:
    data = []
    x_len, z_euc_len, z_man_len = 0.0, 0.0, 0.0
    for i, (z, z_next) in enumerate(zip(path[:-1], path[1:])):
        x, x_next = metric.mapping(z), metric.mapping(z_next)
        loss, loss_next = loss_fn(x).item(), loss_fn(x_next).item()
        dz = z_next - z
        dx = x_next - x
        dx_len = metric.euclidean_length(dx, x).item()
        x_len += dx_len
        dz_man_len = metric.manifold_length(dz, z).item()
        z_man_len += dz_man_len
        dz_euc_len = metric.euclidean_length(dz, z).item()  # ?
        z_euc_len += dz_euc_len

        data.append(
            [
                i,
                x_len,
                z_man_len,
                z_euc_len,
                loss,
                loss_next - loss,
                z[0, 0].item(),
                z[0, 1].item(),
                dx_len,
                dz_man_len,
                dz_euc_len,
            ]
        )
    df = pd.DataFrame(
        columns=[
            "iter",
            "x_len",
            "z_man_len",
            "z_euc_len",
            "loss",
            "dloss",
            "z0",
            "z1",
            "dx_len",
            "dz_man_len",
            "dz_euc_len",
        ],
        data=data,
    )
    df["gamma"] = df.x_len
    return df


def create_summary_fig(df: pd.DataFrame, n_path_points: int, loss_fn_name: str):
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    z_start = np.array(
        [df.loc[df["iter"].idxmin(), "z0"], df.loc[df["iter"].idxmin(), "z1"]]
    )
    step = df["iter"].max() // n_path_points
    sns.scatterplot(
        df[::step],
        x="z0",
        y="z1",
        hue="method",
        marker=".",
        alpha=0.8,
        linestyle="-",
        zorder=1,
        ax=axs[0, 0],
    )
    axs[0, 0].set_title("Optimizer path trace in the latent space")

    sns.lineplot(
        df,
        x="iter",
        y="loss",
        hue="method",
        alpha=0.5,
        lw=2,
        ax=axs[0, 1],
    )
    axs[0, 1].set_ylabel("loss")

    sns.lineplot(
        df,
        x="gamma",
        y="x_len",
        hue="method",
        alpha=0.5,
        lw=2,
        ax=axs[0, 2],
    )
    axs[0, 2].set_title(f"gamma := length of the path in the ambient space")
    axs[0, 2].set_ylabel("euclidean length (ambient)")

    sns.lineplot(
        df,
        x="iter",
        y="gamma",
        hue="method",
        alpha=0.5,
        lw=2,
        ax=axs[1, 0],
    )
    sns.lineplot(
        df,
        x="gamma",
        y="loss",
        hue="method",
        alpha=0.5,
        lw=2,
        ax=axs[1, 1],
    )
    sns.lineplot(
        df,
        x="z_euc_len",
        y="loss",
        hue="method",
        alpha=0.5,
        lw=2,
        ax=axs[1, 2],
    )
    axs[1, 2].set_xlabel("euclidean length (latent)")

    fig.suptitle(
        f"Latent paths optimizing {loss_fn_name} starting in {z_start.round(2)}",
        fontsize=13,
    )
    return fig


def run_setup(
    z: torch.Tensor,
    setup: list[tuple[str, float]],
    n_iter: int,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    map_fn: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[pd.DataFrame, list[torch.Tensor]]:
    metric = TorchMetric(mapping=map_fn)
    dfs, paths = [], []
    for gt, lr in setup:
        z_optim = z.clone().requires_grad_()
        optimizer = InputSGDOptimizer(
            z_optim,
            metric=metric,
            lr=lr,
            gradient_type=gt,
        )
        path = optimize(z_optim, n_iter, optimizer, loss_fn=loss_fn, map_fn=map_fn)
        paths.append(path)
        df = summarize_path(path, metric, loss_fn=loss_fn)
        df["method"] = gt
        dfs.append(df)
    df = pd.concat(dfs).reset_index()
    return df, paths


def run(
    z_start: torch.Tensor,
    n_iter: int,
    loss: tuple[str, Callable[[torch.Tensor], torch.Tensor]],
    optim_setup: list[tuple[str, float]],
    map_fn: Callable[[torch.Tensor], torch.Tensor],
) -> Image.Image:
    loss_name, loss_fn = loss
    df, paths = run_setup(
        z_start, optim_setup, loss_fn=loss_fn, map_fn=map_fn, n_iter=n_iter
    )
    fig = create_summary_fig(df, 50, loss_name)
    img_arrs = [get_img_from_fig(fig)]
    plt.close(fig)
    for (optim_name, lr), path in zip(optim_setup, paths):
        fig = create_path_in_ambient_fig(
            path, f"Path for {optim_name} optimizer, {lr = :.3f}", map_fn=map_fn
        )
        img_arrs.append(get_img_from_fig(fig))
        plt.close(fig)
    img = allign_arrays(img_arrs)
    return img
