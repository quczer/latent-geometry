from typing import Callable, List

import numpy as np
import plotly.graph_objects as go

from latent_geometry.manifold import Manifold
from latent_geometry.path import ManifoldPath
from latent_geometry.visual.calc import get_circles, get_geodesics, get_lines
from latent_geometry.visual.config import (
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
    centre: np.ndarray,
    manifold: Manifold,
    background_trace: go.Scatter,
    num_lines: int,
    num_circles: int,
    line_length: float = 2.5,
    show_lines: bool = True,
    show_circles: bool = True,
) -> go.Figure:
    lines = get_lines(centre, num_lines, manifold, length=line_length)
    circles = get_circles(lines, num_circles)

    paths = []
    if show_lines:
        paths.extend(lines)
    if show_circles:
        paths.extend(circles)
    return draw_paths(background_trace, paths)


def draw_paths(
    background_trace: go.Scatter, paths: List[Callable[[float], np.ndarray]]
) -> go.Figure:
    traces = [background_trace] + [
        create_scatter_object_given_path(path) for path in paths
    ]
    fig = go.Figure(layout={"width": FIGURE_WIDTH, "height": FIGURE_HEIGHT})
    for trace in traces:
        fig.add_trace(trace)
    return fig


def create_topology_fig_geodesics(
    centers1: List[np.ndarray],
    centers2: List[np.ndarray],
    manifold: Manifold,
    background_trace: go.Scatter,
) -> go.Figure:
    geodesics = get_geodesics(centers1, centers2, manifold)
    traces = [background_trace]
    traces.extend(
        [create_scatter_object_given_path(geodesic) for geodesic in geodesics]
    )
    fig = go.Figure(layout={"width": FIGURE_WIDTH, "height": FIGURE_HEIGHT})
    for trace in traces:
        fig.add_trace(trace)
    return fig
