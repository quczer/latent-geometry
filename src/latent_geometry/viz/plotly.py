from typing import Callable

import numpy as np
import plotly.graph_objects as go

from latent_geometry.manifold import Manifold
from latent_geometry.viz.calc import create_circles, create_lines
from latent_geometry.viz.config import (
    FIGURE_HEIGHT,
    FIGURE_WIDTH,
    LINE_OPACITY,
    LINE_WIDTH,
)


def create_scatter_object_given_path(
    path: Callable[[float], np.ndarray], n_points: int = 30, color: str = "black"
) -> go.Scatter:
    timestamps = np.linspace(0.0, 1.0, n_points)
    points = np.vstack([path(t) for t in timestamps])
    return go.Scatter(
        x=points[:, 0],
        y=points[:, 1],
        mode="lines",
        name="",
        line={"color": color, "width": LINE_WIDTH},
        opacity=LINE_OPACITY,
    )


def create_topology_fig(
    centres: list[np.ndarray],
    manifold: Manifold,
    background_trace: go.Scatter,
    num_lines: int,
    num_circles: int,
    line_length: float = 2.5,
    show_lines: bool = True,
    show_circles: bool = True,
) -> go.Figure:
    paths = []
    for centre in centres:
        lines = create_lines(centre, num_lines, manifold, length=line_length)
        if show_lines:
            paths.extend(lines)
        if show_circles:
            paths.extend(create_circles(lines, num_circles))
    return draw_paths(background_trace, paths)


def draw_paths(
    background_trace: go.Scatter, paths: list[Callable[[float], np.ndarray]]
) -> go.Figure:
    traces = [create_scatter_object_given_path(path) for path in paths] + [
        background_trace
    ]
    fig = go.Figure(layout={"width": FIGURE_WIDTH, "height": FIGURE_HEIGHT})
    for trace in traces:
        fig.add_trace(trace)
    return fig
