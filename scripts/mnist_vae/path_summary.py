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
from latent_geometry.manifold import LatentManifold, Manifold
from latent_geometry.mapping import TorchModelMapping
from latent_geometry.metric import EuclideanMetric, ManifoldMetric
from latent_geometry.model.mnist_vae import load_decoder
from latent_geometry.path import ManifoldPath
from latent_geometry.solver import SolverFailedException


def create_straight_path(
    from_: np.ndarray, to_: np.ndarray, metric: ManifoldMetric
) -> ManifoldPath:
    def x_fun(t: float) -> np.ndarray:
        return from_ + (to_ - from_) * t

    return ManifoldPath(x_fun, metric)


def create_path_in_ambient_fig(
    path: ManifoldPath,
    path_name: str,
    n_points: int = 9,
):
    fig, axes = plt.subplots(2, n_points, figsize=(1.5 * n_points, 3))
    ts = np.linspace(0, 1, n_points)
    ambient_path = path.ambient_path
    VMAX = 0.3
    for t, ax in zip(ts, axes[0, :]):
        img = ambient_path(t).reshape(32, 32)
        ax.imshow(img, vmin=0, vmax=1)
        ax.set_title(
            f"t = {t:.1f}",
            fontsize=8,
        )
        ax.set_axis_off()

    axes[1, 0].set_axis_off()
    for t, t_next, ax in zip(ts[:-1], ts[1:], axes[1, 1:]):
        img, img_next = ambient_path(t), ambient_path(t_next)
        img_diff = (img_next - img).reshape(32, 32)

        latent_dist = path.manifold_length(t, t_next)
        euclidean_dist = path.euclidean_length(t, t_next)
        ambient_dist = ambient_path.euclidean_length(t, t_next, dt=0.01)
        diff_mse = np.sqrt((img_diff**2).sum())
        ax.imshow(img_diff, cmap="PiYG", vmin=-VMAX, vmax=VMAX)
        ax.set_title(
            (
                f"l - Euc: {euclidean_dist:.2f}, P-B: {latent_dist: .2f}\n"
                f"a - Euc: {ambient_dist:.2f}, MSE: {diff_mse: .2f}"
            ),
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
    path: ManifoldPath,
    n_points: int = 100,
) -> pd.DataFrame:
    data = []
    ts = np.linspace(0, 1, n_points + 2)
    x_len, z_euc_len, z_man_len = 0.0, 0.0, 0.0
    t_cum, i = 0.0, 0
    for t, t_next in zip(ts[:-1], ts[1:]):
        t_cum += t_next - t
        z, z_next = path(t), path(t_next)
        dz = z_next - z
        dz_euc_len = path.euclidean_length(t, t_next)
        dz_man_len = path.manifold_length(t, t_next)
        dx_len = path.ambient_path.euclidean_length(t, t_next, dt=0.01)
        x_len += dx_len
        z_man_len += dz_man_len
        z_euc_len += dz_euc_len

        data.append(
            [
                t_cum,
                x_len,
                z_man_len,
                z_euc_len,
                z[0],
                z[1],
                dx_len,
                dz_man_len,
                dz_euc_len,
                i,
            ]
        )
        i += 1

    df = pd.DataFrame(
        columns=[
            "t",
            "x_len",
            "z_man_len",
            "z_euc_len",
            "z0",
            "z1",
            "dx_len",
            "dz_man_len",
            "dz_euc_len",
            "i",
        ],
        data=data,
    )

    return df


def create_summary_fig(df: pd.DataFrame):
    fig, axs = plt.subplots(2, 2, figsize=(11, 10))
    z_start = np.array([df.loc[df["i"].idxmin(), "z0"], df.loc[df["i"].idxmin(), "z1"]])
    z_end = np.array([df.loc[df["i"].idxmax(), "z0"], df.loc[df["i"].idxmax(), "z1"]])
    sns.scatterplot(
        df,
        x="z0",
        y="z1",
        hue="path",
        marker=".",
        alpha=0.8,
        linestyle="-",
        zorder=1,
        ax=axs[0, 0],
    )
    axs[0, 0].set_title("Path trace in the latent space")

    sns.lineplot(
        df,
        x="t",
        y="z_euc_len",
        hue="path",
        # marker='.',
        alpha=0.8,
        lw=2,
        ax=axs[0, 1],
    )
    axs[0, 1].set_title("Path length (euclidean) in the latent space")
    axs[0, 1].set_ylabel("")

    sns.lineplot(
        df,
        x="t",
        y="z_man_len",
        hue="path",
        alpha=0.8,
        lw=2,
        ax=axs[1, 0],
    )
    axs[1, 0].set_title("Path length (pull-back) in the latent space")
    axs[1, 0].set_ylabel("")

    sns.lineplot(
        df,
        x="t",
        y="x_len",
        hue="path",
        alpha=0.8,
        lw=2,
        ax=axs[1, 1],
    )
    axs[1, 1].set_title("Path length in the ambient space")
    axs[1, 1].set_ylabel("")
    fig.suptitle(
        f"Latent path {z_start.round(1)} -> {z_end.round(1)}",
        fontsize=13,
    )
    return fig


def run(start: np.ndarray, end: np.ndarray, manifold: Manifold) -> Image.Image:
    geodesic_path = manifold.geodesic(start, end)
    straight_path = create_straight_path(
        geodesic_path(0), geodesic_path(1), manifold.metric
    )
    paths = [
        (straight_path, "straight"),
        (geodesic_path, "geodesic"),
    ]
    img_arrs = []
    dfs = []
    for path, name in paths:
        df = summarize_path(path, n_points=100)
        df["path"] = name
        dfs.append(df)
        fig = create_path_in_ambient_fig(path, n_points=7, path_name=name)
        img_arrs.append(get_img_from_fig(fig))
        plt.close(fig)

    df = pd.concat(dfs).reset_index(drop=True)
    summ_fig = create_summary_fig(df)
    img_arrs.append(get_img_from_fig(summ_fig))
    plt.close(summ_fig)
    img = allign_arrays(img_arrs[::-1])
    return img


def sample_randn_uni(std: float, tau: float) -> tuple[np.ndarray, np.ndarray]:
    start = norm.rvs(0, std, (2,))
    dx = uniform(loc=tau, scale=tau).rvs() * (binom(p=0.5, n=1).rvs() * 2 - 1)
    dy = uniform(loc=tau, scale=tau).rvs() * (binom(p=0.5, n=1).rvs() * 2 - 1)
    end = start + np.array([dx, dy])
    return start, end


def sample_randn_randn(std0: float, std1: float) -> tuple[np.ndarray, np.ndarray]:
    start = norm.rvs(0, std0, (2,))
    end = norm.rvs(0, std1, (2,))
    return start, end


def main(
    model_name: str,
    latent_dim: int,
    n_iter: int,
    tag: str,
    sample_fn: Callable[[], tuple[np.ndarray, np.ndarray]],
    device: torch.device = torch.device("cuda"),
    solver_tols: tuple[float, ...] = (0.01,),
    bvp_mesh_nodes: int = 1_000,
    batch_size: int = 1_000,
) -> None:
    decoder = load_decoder(device, f"{model_name}_decoder.pt", latent_dim=latent_dim)

    ambient_metric = EuclideanMetric()
    latent_mapping = TorchModelMapping(
        decoder,
        (latent_dim,),
        (1, 32, 32),
        batch_size=batch_size,
        call_fn=decoder.decode,
    )
    manifold_mnist = LatentManifold(
        latent_mapping,
        ambient_metric,
        bvp_n_mesh_nodes=bvp_mesh_nodes,
    )

    for i in tqdm(range(n_iter)):
        start, end = sample_fn()
        for tol in solver_tols:
            try:
                manifold_mnist.set_solver_tols(tol)
                img = run(start, end, manifold_mnist)
            except SolverFailedException:
                print(
                    f"Solver failed for {tol=}, {tag = }, {i = }: {start.round(1)} -> {end.round(1)}"
                )
                continue
            else:
                dir_ = (
                    FIGURES_DIR
                    / "mnist"
                    / "images"
                    / f"{latent_dim}dim"
                    / "paths"
                    / tag
                )
                dir_.mkdir(exist_ok=True, parents=True)
                img.save(dir_ / f"v{i}_tol{tol:.2f}.png", "PNG")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iter", type=int)
    parser.add_argument("--sampler", choices=["randn_randn", "randn_uni"])
    parser.add_argument("--tag", type=str)
    parser.add_argument("--model_name", type=str, default="beta_10")
    parser.add_argument("--latent_dim", type=int, default=2)
    args = parser.parse_args()

    tag = args.tag
    if args.sampler == "randn_randn":
        sample_fn = partial(sample_randn_randn, std0=1, std2=1)
    elif args.sampler == "randn_uni":
        sample_fn = partial(sample_randn_uni, std=0, tau=0.4)

    main(args.model_name, args.latent_dim, args.n_iter, tag=tag, sample_fn=sample_fn)
