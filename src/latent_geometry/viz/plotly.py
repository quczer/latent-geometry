from typing import Callable, Optional

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.basedatatypes import BaseTraceType

import latent_geometry.viz.config as C
from latent_geometry.path import Path

_PIC_WIDTH = 32


def create_dot_background(
    mus: np.ndarray, labels: np.ndarray, opacity: float
) -> BaseTraceType:
    cmap = np.array(px.colors.qualitative.G10)
    colors = cmap[labels]
    return go.Scatter(
        x=mus[:, 0],
        y=mus[:, 1],
        customdata=labels,
        mode="markers",
        marker=dict(color=colors, opacity=opacity),
        name="mnist dataset",
        hovertemplate="digit=%{customdata}<br>x=%{x:.3}<br>y=%{y:.3}<extra></extra>",
        showlegend=True,
    )


def create_scalar_field(
    scalar_fn: Callable[[np.ndarray], np.ndarray],
    num: int,
    opacity: float = 1.0,
    field_title: str = "scalar field",
    cmap: str = "RdBu",
) -> go.Heatmap:
    xs, ys = np.meshgrid(
        np.linspace(*C.AXES_RANGE, num=num),
        np.linspace(*C.AXES_RANGE, num=num),
    )
    pts = np.vstack((xs.reshape(-1), ys.reshape(-1))).T
    zs = scalar_fn(pts)
    return go.Heatmap(
        x=pts[:, 0],
        y=pts[:, 1],
        z=zs,
        showscale=False,
        colorscale=cmap,
        opacity=opacity,
        name=field_title,
        showlegend=True,
        legendgroup=field_title,
    )


def draw_spiders(
    spiders: list[list[Path]],
    background_trace: Optional[go.Scatter] = None,
) -> go.Figure:
    traces: list[go.Scatter] = []
    for spider in spiders:
        x0, y0 = spider[0](0.0)
        legend_group = f"spider_({x0:.1f},{y0:.1f})"
        traces.extend(
            _path_to_trace(path, legend_group=legend_group, show_legend=i == 0)
            for i, path in enumerate(spider)
        )
    if background_trace:
        traces.append(background_trace)
    fig = plot_traces(traces)
    return fig


def draw_balls(
    balls: list[Path],
    background_trace: Optional[go.Scatter] = None,
) -> go.Figure:
    traces: list[go.Scatter] = []
    for i, path in enumerate(balls):
        traces.append(_path_to_trace(path, legend_group="balls", show_legend=i == 0))
    if background_trace:
        traces.append(background_trace)
    fig = plot_traces(traces)
    return fig


def plot_traces(traces: list[BaseTraceType], force_layout: bool = True) -> go.Figure:
    fig = go.Figure(data=traces)
    if force_layout:
        fig.update_layout(
            **{
                "width": C.FIGURE_WIDTH,
                "height": C.FIGURE_HEIGHT,
                "margin": {
                    "b": C.MARGIN,
                    "l": C.MARGIN,
                    "r": C.MARGIN,
                    "t": C.MARGIN,
                },
                "paper_bgcolor": "#fff",
                "yaxis": {"range": C.AXES_RANGE},
                "xaxis": {"range": C.AXES_RANGE},
            }
        )
    return fig


def create_digit_background(
    num: int,
    mapping: Callable[[np.ndarray], np.ndarray],
    opacity: float,
) -> list[go.Heatmap]:
    PIC_WIDTH, BG_WIDTH = 32, 3 * num / (num + 1)
    xs, ys = np.meshgrid(
        np.linspace(-BG_WIDTH, BG_WIDTH, num=num),
        np.linspace(-BG_WIDTH, BG_WIDTH, num=num),
    )
    xs_latent = np.vstack((xs.reshape(-1), ys.reshape(-1))).T
    imgs = mapping(xs_latent).reshape(-1, PIC_WIDTH, PIC_WIDTH)
    heatmaps = []
    for i, (img, (x, y)) in enumerate(zip(imgs, xs_latent)):
        hmap = _create_img_heatmap(
            img,
            x,
            y,
            dx=2 * BG_WIDTH / PIC_WIDTH / (num - 1),
            opacity=opacity,
            show_legend=i == 0,
        )
        heatmaps.append(hmap)
    return heatmaps


def _path_to_trace(
    path: Path,
    n_points: int = 30,
    color: str = "black",
    legend_group: Optional[str] = None,
    show_legend: bool = False,
    opacity: Optional[float] = None,
) -> go.Scatter:
    timestamps = np.linspace(0.0, 1.0, n_points)
    points = np.vstack([path(t) for t in timestamps])
    return go.Scatter(
        x=points[:, 0],
        y=points[:, 1],
        mode="lines",
        name=legend_group,
        line={"color": color, "width": C.LINE_WIDTH},
        opacity=opacity or C.LINE_OPACITY,
        legendgroup=legend_group,
        showlegend=show_legend,
    )


def _create_img_heatmap(
    img: np.ndarray,
    x_centre: float,
    y_centre: float,
    dx: float,
    opacity: float = 1.0,
    colorscale: str = "gray_r",
    name: str = "ambient images",
    show_legend: bool = False,
):
    x0 = x_centre - _PIC_WIDTH / 2 * dx
    y0 = y_centre - _PIC_WIDTH / 2 * dx
    return go.Heatmap(
        z=np.flip(img.reshape(_PIC_WIDTH, _PIC_WIDTH), axis=0),
        x0=x0,
        dx=dx,
        y0=y0,
        dy=dx,
        showscale=False,
        colorscale=colorscale,
        opacity=opacity,
        name=name,
        showlegend=show_legend,
        legendgroup=name,
    )
